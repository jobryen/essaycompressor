#encoder and decoder imports identical
from PIL import Image
import numpy as np
import pywt
import struct
import os
import argparse

# SSIM and PSNR analysis functions, ideally build my own
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# ---------------------------------- ENCODER ------------------------------------

#thresholds and quantises the DWT

def treshold_quantise(coeffs, threshold_fraction, quant_bits):
    all_coeffs = [coeffs[0]]  
    for detail in coeffs[1:]:
        all_coeffs.extend(detail) 
    max_coeff = max(np.max(np.abs(arr)) for arr in all_coeffs)
    threshold_value = threshold_fraction * max_coeff

    coeffs_thresh = [None] * len(coeffs)
    cA = coeffs[0].copy()
    cA[np.abs(cA) < threshold_value] = 0
    coeffs_thresh[0] = cA

    for i, detail in enumerate(coeffs[1:], start=1):
        cH, cV, cD = detail
        cH_th = cH.copy(); cH_th[np.abs(cH_th) < threshold_value] = 0
        cV_th = cV.copy(); cV_th[np.abs(cV_th) < threshold_value] = 0
        cD_th = cD.copy(); cD_th[np.abs(cD_th) < threshold_value] = 0
        coeffs_thresh[i] = (cH_th, cV_th, cD_th)

    total_coeffs = sum(arr.size for arr in all_coeffs)
    nonzero_coeffs = sum(np.count_nonzero(arr) for arr in [coeffs_thresh[0]] + [d for detail in coeffs_thresh[1:] for d in detail])
    print(f'Total coefficients: {total_coeffs}, Non-zeros after threshold: {nonzero_coeffs}')

    quant_levels = 2**quant_bits

    # min and max to scale
    min_val = min(np.min(arr) for arr in all_coeffs)
    max_val = max(np.max(arr) for arr in all_coeffs)
    if max_val == min_val:
        step_size = 1.0
    else:
        step_size = (max_val - min_val) / (quant_levels - 1)

    # uniform quantise
    quantized_coeffs = [None] * len(coeffs_thresh)

    cA_th = coeffs_thresh[0]
    qA = np.rint((cA_th - min_val) / step_size).astype(np.int16)
    qA[cA_th == 0] = 0  # needa to check that i havent actaully set the threshold coefficients to zero twice lol
    qA = np.clip(qA, 0, quant_levels-1)
    quantized_coeffs[0] = qA


    for i, detail in enumerate(coeffs_thresh[1:], start=1):
        cH_th, cV_th, cD_th = detail
        qH = np.rint((cH_th - min_val) / step_size).astype(np.int16)
        qV = np.rint((cV_th - min_val) / step_size).astype(np.int16)
        qD = np.rint((cD_th - min_val) / step_size).astype(np.int16)
        # second time being here haha ------------------------------------------
        qH[cH_th == 0] = 0
        qV[cV_th == 0] = 0
        qD[cD_th == 0] = 0
        qH = np.clip(qH, 0, quant_levels-1)
        qV = np.clip(qV, 0, quant_levels-1)
        qD = np.clip(qD, 0, quant_levels-1)
        quantized_coeffs[i] = (qH, qV, qD)

    print('Quantization step size:', step_size)
    print('Example quantized coeff range:', np.min(qA), 'to', np.max(qA))

    return quantized_coeffs, min_val, step_size

#compresses long runs of zeros (should replace with entropy or huffman soon)

def rle_encode(values, sentinel=0xFFFF):
    '''Basic run-length encoding for a list of integers. 
    Uses `sentinel` followed by count to encode runs of zeros (length >=3).'''
    encoded = []
    i = 0
    n = len(values)
    while i < n:
        if values[i] != 0:
            # Non-zero value: encode it directly
            encoded.append(values[i])
            i += 1
        else:
            # Zero run detected
            j = i
            while j < n and values[j] == 0:
                j += 1
            run_len = j - i  # length of this zero-run
            if run_len < 3:
                # For short runs, output literal zeros (no compression gain for 1 or 2 zeros)
                encoded.extend([0] * run_len)
            else:
                # Use sentinel to encode a long run of zeros
                full_runs = run_len // 0xFFFF
                remainder = run_len % 0xFFFF
                # If run is very long, split into multiple sentinel segments
                for _ in range(full_runs):
                    encoded.append(sentinel)
                    encoded.append(0xFFFF)  # maximum count for a run in one segment
                if remainder > 0:
                    encoded.append(sentinel)
                    encoded.append(remainder)
            i = j
    return encoded

#checks if name if available and if not adds a suffix

def name_available(path: str, sep: str = '_') -> str:
    folder, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)

    candidate = path
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f'{stem}{sep}{i}{ext}')
        i += 1
    return candidate

#encodes each individual R G B array same way we do with the greyscale ones. returns data for dict

def encode_array(image_array, wavelet_name, L, threshold_fraction, quant_bits):

    image_array = np.asarray(image_array, dtype=np.float32)
    H, W = image_array.shape
    print(f'Loaded channel: {W} x {H} pixels')

    coeffs = pywt.wavedec2(image_array, wavelet_name, level=L)
    quantized_coeffs, min_val, step_size = treshold_quantise(coeffs, threshold_fraction, quant_bits)

    flat_coeffs = []
    flat_coeffs.extend(quantized_coeffs[0].ravel().tolist())
    for qH, qV, qD in quantized_coeffs[1:]:
        flat_coeffs.extend(qH.ravel().tolist())
        flat_coeffs.extend(qV.ravel().tolist())
        flat_coeffs.extend(qD.ravel().tolist())

    total_values = len(flat_coeffs)
    encoded_values = rle_encode(flat_coeffs)

    return {
        'W': int(W),
        'H': int(H),
        'min_val': float(min_val),
        'step_size': float(step_size),
        'total_values': int(total_values),
        'encoded_values': encoded_values,
    }

#creats dictionary / header then produces the .job file. returns filename (might be useful?)

def write_file(image_path, image_array_rgb, L, wavelet_name, y_thresh, c_thresh, y_quant, c_quant, subsamp, channels):
    # filename (make threshold filename-safe)

    if subsamp == 0:
        chroma = '4-4-4'
    elif subsamp == 1:
        chroma = '4-2-2'
    elif subsamp == 2:
        chroma = '4-2-0'

    out_path = (
        os.path.splitext(image_path)[0]
        + f'_{wavelet_name}_L{L}_yt{str(y_thresh)}_ct{str(c_thresh)}_yq{str(y_quant)}_cq{c_quant}_{chroma}.job'
    )
    output_filename = name_available(out_path)

    H, W = image_array_rgb.shape[:2]

    with open(output_filename, 'wb') as f:
        f.write(b'RGB0')  # or change this to b'YCC0' if you truly store Y,Cb,Cr

        f.write(struct.pack('<B', subsamp))
        f.write(struct.pack('<H', W))
        f.write(struct.pack('<H', H))
        f.write(struct.pack('<B', L))

        wavelet_bytes = wavelet_name.encode('ascii')
        f.write(struct.pack('<B', len(wavelet_bytes)))
        f.write(wavelet_bytes)

        f.write(struct.pack('<B', y_quant))
        f.write(struct.pack('<B', c_quant))
        f.write(struct.pack('<f', float(y_thresh)))
        f.write(struct.pack('<f', float(c_thresh)))

        if len(channels) != 3:
            raise ValueError('Expected 3 channel payloads')

        for ch in channels:
            chW = int(ch['W'])
            chH = int(ch['H'])
            min_val = float(ch['min_val'])
            step_size = float(ch['step_size'])
            total_values = int(ch['total_values'])
            encoded_values = ch['encoded_values']

            encoded_count = len(encoded_values)

            # per-channel dimensions (what your reader expects)
            f.write(struct.pack('<H', chW))
            f.write(struct.pack('<H', chH))

            f.write(struct.pack('<f', min_val))
            f.write(struct.pack('<f', step_size))
            f.write(struct.pack('<I', total_values))
            f.write(struct.pack('<I', encoded_count))

            data_array = np.asarray(encoded_values, dtype='<u2')  # explicit little-endian u16
            f.write(data_array.tobytes())


        input_size = os.path.getsize(image_path)
        output_size = os.path.getsize(output_filename)

        print('original file size:', input_size, 'new file size:', output_size)
        print('compression ratio:', input_size / output_size)
    return output_filename

#combines the three arrays and applies write_file to produce final result

def encode(path, wavelet_name, L, y_thresh, c_thresh, y_quant, c_quant, chromasub):

    image = Image.open(path).convert('RGB').convert('YCbCr') #converting twice fixes some error god knows why
    image_size = os.path.getsize(path)
    Chroma_array = np.array(image)  # shape (H, W, 3)

    Y = Chroma_array[:, :, 0]
    Cb = Chroma_array[:, :, 1]
    Cr = Chroma_array[:, :, 2]

    if chromasub == 1:   # 4:2:2
        Cb = Cb[:, ::2]
        Cr = Cr[:, ::2]
    elif chromasub == 2: # 4:2:0
        Cb = Cb[::2, ::2]
        Cr = Cr[::2, ::2]


    Y_encoded = encode_array(Y, wavelet_name, L, y_thresh, y_quant)
    Cb_encoded = encode_array(Cb, wavelet_name, L, c_thresh, c_quant)
    Cr_encoded = encode_array(Cr, wavelet_name, L, c_thresh, c_quant)

    return write_file(
        image_path=path,
        image_array_rgb=Chroma_array,
        L=L,
        wavelet_name=wavelet_name,
        y_thresh = y_thresh,
        c_thresh = c_thresh,
        y_quant = y_quant,
        c_quant = c_quant,
        subsamp = chromasub,
        channels=[Y_encoded, Cb_encoded, Cr_encoded],
    )


#----------------------------------- Decoder ------------------------------------ 

#undoes the RLE encoding

def rle_decode(encoded: np.ndarray, sentinel: int = 0xFFFF) -> np.ndarray:
    out = []
    i = 0
    n = int(encoded.size)
    while i < n:
        val = int(encoded[i]); i += 1
        if val != sentinel:
            out.append(val)
        else:
            if i >= n:
                raise ValueError('Malformed RLE stream: sentinel at end with no count.')
            run_len = int(encoded[i]); i += 1
            out.extend([0] * run_len)
    return np.array(out, dtype=np.uint16)

#find structure of array, width height etc

def infer_coeff_shapes(height: int, width: int, wavelet_name: str, L: int, mode: str = 'symmetric'):
    dummy = np.zeros((height, width), dtype=np.float32)
    coeffs = pywt.wavedec2(dummy, wavelet_name, level=L, mode=mode)
    shapes = [coeffs[0].shape] + [tuple(arr.shape for arr in detail) for detail in coeffs[1:]]
    return shapes

#rebuild tree, basically undoes the multilevel part of the multilevel DWT

def build_coeff_tree_from_flat(flat_q: np.ndarray, H: int, W: int, L: int, wavelet: str,
                                min_val: float, step: float, mode: str = 'symmetric'):
    shapes = infer_coeff_shapes(H, W, wavelet, L, mode=mode)

    def dequantise(codes_1d: np.ndarray) -> np.ndarray:
        codes_f = codes_1d.astype(np.float32)
        out = codes_f * step + min_val
        out[codes_f == 0] = 0.0
        return out

    idx = 0

    cA_shape = shapes[0]
    nA = cA_shape[0] * cA_shape[1]
    cA = dequantise(flat_q[idx:idx+nA]).reshape(cA_shape)
    idx += nA

    coeffs = [cA]

    for (sH, sV, sD) in shapes[1:]:
        nH = sH[0] * sH[1]
        nV = sV[0] * sV[1]
        nD = sD[0] * sD[1]

        cH = dequantise(flat_q[idx:idx+nH]).reshape(sH); idx += nH
        cV = dequantise(flat_q[idx:idx+nV]).reshape(sV); idx += nV
        cD = dequantise(flat_q[idx:idx+nD]).reshape(sD); idx += nD

        coeffs.append((cH, cV, cD))

    if idx != flat_q.size:
        raise ValueError(f'Did not consume the whole coefficient stream: used {idx}, total {flat_q.size}')

    return coeffs

def read_ycc_job(data: bytes):
    mv = memoryview(data)
    n = len(mv)
    off = 0

    def require(num_bytes: int, what: str):
        nonlocal off
        if off + num_bytes > n:
            raise ValueError(
                f'Truncated .job while reading {what}: need {num_bytes} bytes at offset {off}, '
                f'but only {n - off} bytes remain (file size {n}).'
            )

    def read_bytes(num_bytes: int, what: str) -> bytes:
        nonlocal off
        require(num_bytes, what)
        b = mv[off:off + num_bytes].tobytes()
        off += num_bytes
        return b

    def read_struct(fmt: str, what: str):
        nonlocal off
        size = struct.calcsize(fmt)
        require(size, what)
        val = struct.unpack_from(fmt, mv, off)
        off += size
        return val[0] if len(val) == 1 else val

    # ---- container header ----
    magic = read_bytes(4, 'magic')
    if magic not in (b'RGB0', b'YCC0'):
        raise ValueError(f'Bad magic {magic!r} (expected b"RGB0" or b"YCC0").')

    subsamp = read_struct('<B', 'subsamp')
    if subsamp not in (0, 1, 2):
        raise ValueError(f'Bad subsamp={subsamp} (expected 0, 1, or 2).')

    width  = read_struct('<H', 'width')
    height = read_struct('<H', 'height')
    L      = read_struct('<B', 'L')

    name_len = read_struct('<B', 'wavelet_name length')
    wavelet_bytes = read_bytes(name_len, 'wavelet_name bytes')
    try:
        wavelet_name = wavelet_bytes.decode('ascii')
    except UnicodeDecodeError as e:
        raise ValueError(
            f'wavelet_name is not ASCII at offset {off - name_len}: {wavelet_bytes!r}'
        ) from e

    # these MUST be read to stay aligned with your writer
    y_quant  = read_struct('<B', 'y_quant')
    c_quant  = read_struct('<B', 'c_quant')
    y_thresh = read_struct('<f', 'y_thresh')
    c_thresh = read_struct('<f', 'c_thresh')

    # ---- channels ----
    channels = []
    for ch_i in range(3):
        chW = read_struct('<H', f'channel[{ch_i}].W')
        chH = read_struct('<H', f'channel[{ch_i}].H')
        min_val   = read_struct('<f', f'channel[{ch_i}].min_val')
        step_size = read_struct('<f', f'channel[{ch_i}].step_size')
        total_count   = read_struct('<I', f'channel[{ch_i}].total_count')
        encoded_count = read_struct('<I', f'channel[{ch_i}].encoded_count')

        # uint16 payload
        byte_count = int(encoded_count) * 2
        require(byte_count, f'channel[{ch_i}].encoded_values[{encoded_count}]')
        enc = np.frombuffer(mv, dtype='<u2', offset=off, count=int(encoded_count)).copy()
        off += byte_count

        channels.append({
            'W': int(chW),
            'H': int(chH),
            'min_val': float(min_val),
            'step_size': float(step_size),
            'total_count': int(total_count),
            'encoded_count': int(encoded_count),
            'encoded': enc,
        })

    # Optional: complain if there are trailing bytes (usually indicates a mismatch somewhere)
    if off != n:
        # You can change this to a warning/print if you want to allow trailing metadata later.
        raise ValueError(f'Extra trailing bytes: parsed {off} of {n} bytes (trailing {n-off}).')

    header = {
        'width': int(width),
        'height': int(height),
        'L': int(L),
        'wavelet_name': wavelet_name,
        'subsamp': int(subsamp),
        'y_quant': int(y_quant),
        'c_quant': int(c_quant),
        'y_thresh': float(y_thresh),
        'c_thresh': float(c_thresh),
    }
    return header, channels

def upsample_chroma(C: np.ndarray, H: int, W: int, subsamp: int) -> np.ndarray:
    if subsamp == 0:  # 444
        return C[:H, :W]
    if subsamp == 1:  # 422
        return np.repeat(C, 2, axis=1)[:, :W]
    if subsamp == 2:  # 420
        return np.repeat(np.repeat(C, 2, axis=0), 2, axis=1)[:H, :W]
    raise ValueError('subsamp must be 0 (444), 1 (422), or 2 (420)')
#IDWT, interpets dictionary / header, saves file and performs analystics

def decode(path, original_path = None):

    with open(path, 'rb') as f:
        data = f.read()
        header, ch_meta = read_ycc_job(data)

    print('Metadata:')
    for k, v in header.items():
        print(f'  {k}: {v}')

    H = int(header['height'])
    W = int(header['width'])
    L = int(header['L'])
    wavelet = header['wavelet_name']

    decoded_channels = []
    for idx_ch, ch in enumerate(ch_meta):
        flat_q = rle_decode(ch['encoded'], sentinel=0xFFFF)
        if flat_q.size != int(ch['total_count']):
            raise ValueError(f'Channel {idx_ch}: decoded length {flat_q.size} != total_count {ch["total_count"]}')

        coeffs_rec = build_coeff_tree_from_flat(
            flat_q, H = ch['H'], W = ch['W'], L = L, wavelet = wavelet,
            min_val=float(ch['min_val']),
            step=float(ch['step_size']),
            mode='symmetric'
        )

        rec = pywt.waverec2(coeffs_rec, wavelet, mode='symmetric')
        rec = rec[:H, :W]
        rec_u8 = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
        decoded_channels.append(rec_u8)

    base = os.path.splitext(path)[0]
    out_path = base + ('_uncompressed.tiff')
    out_path = name_available(out_path)

    Y, Cb, Cr = decoded_channels

    subsamp = header['subsamp']

    # Upsample chroma to full resolution if subsampled
    Cb_full = upsample_chroma(Cb, H, W, subsamp)
    Cr_full = upsample_chroma(Cr, H, W, subsamp)
    Y_full = Y[:H, :W]

    rgb = Image.merge('YCbCr', (
        Image.fromarray(Y_full, mode='L'),
        Image.fromarray(Cb_full, mode='L'),
        Image.fromarray(Cr_full, mode='L'),
    )).convert('RGB')

    rgb.save(out_path, format='TIFF', compression=None)


    #compare compressed .job to original image (not the reconstructed TIFF - that should be similar enough to not matter ideally)

    if original_path is None:
        print('No original_path provided; skipping PSNR/SSIM.')
    else:
        orig_u8 = np.array(Image.open(original_path).convert('L'), dtype=np.uint8)

        # Match shapes preemtivly
        H = min(orig_u8.shape[0], rec_u8.shape[0])
        W = min(orig_u8.shape[1], rec_u8.shape[1])
        orig_u8_c = orig_u8[:H, :W]
        rec_u8_c = rec_u8[:H, :W]

            #find compression metrics
        PSNR = psnr(orig_u8_c, rec_u8_c, data_range=255)
        SSIM = ssim(orig_u8_c, rec_u8_c, data_range=255)
        print(f'PSNR: {PSNR} dB')
        print(f'SSIM: {SSIM}')

        input_size = os.path.getsize(original_path)
        job_size = os.path.getsize(path)
        ratio = input_size / job_size
        
        print('compressed size ', job_size, ' original file size ', input_size)
        print("compression ratio (orig:job):", ratio)
    return PSNR, SSIM, ratio

# --------------------------- terminal argparse stuff ---------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='codec')

    sub = p.add_subparsers(dest='cmd', required=True)

    # encode subcommand
    pe = sub.add_parser('encode', help='Encode an image into a .job file')
    pe.add_argument('path', help='Input image path (e.g. .png, .jpg)',)
    pe.add_argument('-L', type=int, default=3, required=False, help='DWT levels (e.g. 1,2,3...)')
    pe.add_argument('-w', '--wavelet', type=str, default='bior4.4', dest='wavelet_name', required=False, help='Wavelet name (e.g. haar)')
    pe.add_argument('-yt', '--intensity_threshold', type=float, default=0.03, dest='y_thresh', required=False,
                    help='Threshold fraction (e.g. 0.05)')
    pe.add_argument('-ct', '--colour_threshold', type=float, default=0.05, dest='c_thresh', required=False,
                    help='Threshold fraction (e.g. 0.05)')
    pe.add_argument('-yq', '--intensity_quant', dest='y_quant', type=int, default = 8, required=False,
                    help='No. bits for quantisation (e.g. n=8 gives 2**8 = 256 quantisation levels)')
    pe.add_argument('-cq', '--colour_quant', dest='c_quant', type=int, default = 8, required=False,
                    help='No. bits for quantisation (e.g. n=8 gives 2**8 = 256 quantisation levels)')
    pe.add_argument('-c', '-chromasub', type = int, default=2, dest= 'chromasub', required = False, help = 'chroma subsampling reduces colour resolution: 0 = 4:4:4, 1 = 4:2:2, 2 = 4:2:0')

    # decode subcommand
    pd = sub.add_parser('decode', help='Decode a .job file')
    pd.add_argument('path', help='Input .job path')
    pd.add_argument('-o', '--original_path',required= False, help = 'original image path for compression metrics')

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == 'encode':
        encode(
            path=args.path,
            L=args.L,
            wavelet_name=args.wavelet_name,
            y_thresh = args.y_thresh,
            c_thresh = args.c_thresh,
            y_quant=args.y_quant,
            c_quant=args.c_quant,
            chromasub = args.chromasub,
        )

    elif args.cmd == 'decode':
        decode(
            args.path,
            args.original_path
        )

if __name__ == '__main__':
    main()

