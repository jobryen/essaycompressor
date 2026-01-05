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

def encode(path, wavelet_name, L, threshold_fraction, quant_bits):

    image_path = path
    image = Image.open(image_path).convert('L')
    image_array = np.array(image, dtype=np.uint8)

    height, width = image_array.shape
    image_size = os.path.getsize(image_path)
    print(f"Loaded image: {width} x {height} pixels, grayscale,", image_size, "bytes")


    coeffs = pywt.wavedec2(image, wavelet_name, level=L)

    print(f"Decomposed into {len(coeffs)-1} levels of detail coefficients plus approximation.")
    print("Approximation (cA_L) shape:", coeffs[0].shape)
    for level, details in enumerate(coeffs[1:], start=1):
        cH, cV, cD = details
        print(f"Level {L+1-level} detail shapes (cH, cV, cD): {cH.shape}, {cV.shape}, {cD.shape}")


    # find biggest coeff
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
    print(f"Total coefficients: {total_coeffs}, Non-zeros after threshold: {nonzero_coeffs}")

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

    print("Quantization step size:", step_size)
    print("Example quantized coeff range:", np.min(qA), "to", np.max(qA))


    flat_coeffs = []
    # approximation
    flat_coeffs.extend(quantized_coeffs[0].ravel().tolist())
    # Append detail coefficients for each level
    for detail in quantized_coeffs[1:]:
        qH, qV, qD = detail
        flat_coeffs.extend(qH.ravel().tolist())
        flat_coeffs.extend(qV.ravel().tolist())
        flat_coeffs.extend(qD.ravel().tolist())

    total_values = len(flat_coeffs)
    print("Total values to encode (should equal image size):", total_values)

    def rle_encode(values, sentinel=0xFFFF):
        """Basic run-length encoding for a list of integers. 
        Uses `sentinel` followed by count to encode runs of zeros (length >=3)."""
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

    # RLE to the flat coefficient list
    encoded_values = rle_encode(flat_coeffs)
    print(f"RLE encoded length: {len(encoded_values)} (16-bit words)")


    def name_available(path: str, sep: str = "_") -> str:
        folder, filename = os.path.split(path)
        stem, ext = os.path.splitext(filename)

        candidate = path
        i = 1
        while os.path.exists(candidate):
            candidate = os.path.join(folder, f"{stem}{sep}{i}{ext}")
            i += 1
        return candidate

    out_path = os.path.splitext(image_path)[0] + 'encoded.job'
    output_filename = name_available(out_path)

    with open(output_filename, "wb") as f:
        # width height and level
        f.write(struct.pack('<H', image_array.shape[1])) 
        f.write(struct.pack('<H', image_array.shape[0])) 
        f.write(struct.pack('<B', L))
        # Wavelet name
        wavelet_bytes = wavelet_name.encode('ascii')
        f.write(struct.pack('<B', len(wavelet_bytes)))
        f.write(wavelet_bytes)
        # Threshold fraction and quantization parameters
        f.write(struct.pack('<f', threshold_fraction))
        f.write(struct.pack('<f', float(min_val)))
        f.write(struct.pack('<f', float(step_size)))
        # Counts
        total_count = total_values
        encoded_count = len(encoded_values)
        f.write(struct.pack('<I', total_count))
        f.write(struct.pack('<I', encoded_count))
        # Encode list as 16-bit words
        data_array = np.array(encoded_values, dtype=np.uint16)
        f.write(data_array.tobytes())
        return

    
#-------------------------------------------------------- Decoder -------------------------------------------------------- 

def decode(path, original_path):

    def read_job_header_and_payload(path: str):
        with open(path, "rb") as f:
            data = f.read()

        off = 0
        width  = struct.unpack_from("<H", data, off)[0]; off += 2
        height = struct.unpack_from("<H", data, off)[0]; off += 2
        L      = struct.unpack_from("<B", data, off)[0]; off += 1

        name_len = struct.unpack_from("<B", data, off)[0]; off += 1
        wavelet_name = data[off:off+name_len].decode("ascii"); off += name_len

        threshold_fraction = struct.unpack_from("<f", data, off)[0]; off += 4
        min_val = struct.unpack_from("<f", data, off)[0]; off += 4
        step_size = struct.unpack_from("<f", data, off)[0]; off += 4

        total_count = struct.unpack_from("<I", data, off)[0]; off += 4
        encoded_count = struct.unpack_from("<I", data, off)[0]; off += 4

        encoded = np.frombuffer(data, dtype="<u2", offset=off, count=encoded_count).copy()

        header = dict(
            width=width,
            height=height,
            L=L,
            wavelet_name=wavelet_name,
            threshold_fraction=threshold_fraction,
            min_val=min_val,
            step_size=step_size,
            total_count=total_count,
            encoded_count=encoded_count,
        )
        return header, encoded
    
    header, encoded_values = read_job_header_and_payload(path)


    print("Metadata:")
    for k,v in header.items():
        print(f"  {k}: {v}")
    print("Encoded stream uint16 length:", encoded_values.size)

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
                    raise ValueError("Malformed RLE stream: sentinel at end with no count.")
                run_len = int(encoded[i]); i += 1
                out.extend([0] * run_len)
        return np.array(out, dtype=np.uint16)

    flat_q = rle_decode(encoded_values, sentinel=0xFFFF)
    print("Decoded coefficient codes length:", flat_q.size, "(expected", header["total_count"], ")")
    assert flat_q.size == header["total_count"], "Decoded length does not match header total_count."

    def infer_coeff_shapes(height: int, width: int, wavelet_name: str, L: int, mode: str = "symmetric"):
        '''
        Use a dummy zero image to infer the wavedec2 coefficient array shapes for (H,W,wavelet,L).
        Keeps the decoder flexible for wavelet choice and level count.
        '''
        dummy = np.zeros((height, width), dtype=np.float32)
        coeffs = pywt.wavedec2(dummy, wavelet_name, level=L, mode=mode)
        shapes = [coeffs[0].shape] + [tuple(arr.shape for arr in detail) for detail in coeffs[1:]]
        return shapes

    def build_coeff_tree_from_flat(flat_q: np.ndarray, header: dict, mode: str = "symmetric"):

        #Reconstructs the wavedec2 coeff list:

        H = int(header["height"])
        W = int(header["width"])
        L = int(header["L"])
        wavelet = header["wavelet_name"]
        min_val = float(header["min_val"])
        step = float(header["step_size"])

        shapes = infer_coeff_shapes(H, W, wavelet, L, mode=mode)

        def dequantise(codes_1d: np.ndarray) -> np.ndarray:
            codes_f = codes_1d.astype(np.float32)
            out = codes_f * step + min_val
            out[codes_f == 0] = 0.0  # preserve explicit zeros
            return out

        idx = 0

        # Approximation cA_L
        cA_shape = shapes[0]
        nA = cA_shape[0] * cA_shape[1]
        cA_codes = flat_q[idx:idx+nA]; idx += nA
        cA = dequantise(cA_codes).reshape(cA_shape)

        coeffs = [cA]

        # tuples for each level: L down to 1
        for (sH, sV, sD) in shapes[1:]:
            nH = sH[0] * sH[1]
            nV = sV[0] * sV[1]
            nD = sD[0] * sD[1]

            cH = dequantise(flat_q[idx:idx+nH]).reshape(sH); idx += nH
            cV = dequantise(flat_q[idx:idx+nV]).reshape(sV); idx += nV
            cD = dequantise(flat_q[idx:idx+nD]).reshape(sD); idx += nD

            coeffs.append((cH, cV, cD))

        if idx != flat_q.size:
            raise ValueError(f"Did not consume the whole coefficient stream: used {idx}, total {flat_q.size}")

        return coeffs

    coeffs_rec = build_coeff_tree_from_flat(flat_q, header, mode="symmetric")
    print("Rebuilt coeff tree length:", len(coeffs_rec), "(should be L+1 =", header["L"]+1, ")")

    rec = pywt.waverec2(coeffs_rec, header["wavelet_name"], mode="symmetric")

    # ensure correct dimensions (waverec2 can be wrong)
    rec = rec[:header["height"], :header["width"]]

    # Convert to uint8 for display/save
    rec_u8 = np.clip(np.rint(rec), 0, 255).astype(np.uint8)

    print("Reconstructed image shape:", rec_u8.shape, "dtype:", rec_u8.dtype)

    def name_available(path: str, sep: str = "_") -> str:
        folder, filename = os.path.split(path)
        stem, ext = os.path.splitext(filename)

        candidate = path
        i = 1
        while os.path.exists(candidate):
            candidate = os.path.join(folder, f"{stem}{sep}{i}{ext}")
            i += 1
        return candidate

    root, _ = os.path.splitext(path.rsplit("_", 1)[0])

    out_path = root + "_uncompressed.png"
    available_out_path = name_available(out_path)

    Image.fromarray(rec_u8, mode="L").save(available_out_path)
    #Image.fromarray(rec_u8, mode="L").save(available_out_path, compress_level = 0) ideally want to do this so we only have our compression isntead of added lossless compression from png
    print("Saved:", available_out_path)

    if original_path is None:
        print("No original_path provided; skipping PSNR/SSIM.")
    else:
        orig_u8 = np.array(Image.open(original_path).convert("L"), dtype=np.uint8)

        # Match shapes preemtivly
        H = min(orig_u8.shape[0], rec_u8.shape[0])
        W = min(orig_u8.shape[1], rec_u8.shape[1])
        orig_u8_c = orig_u8[:H, :W]
        rec_u8_c = rec_u8[:H, :W]

        print(f"PSNR: {psnr(orig_u8_c, rec_u8_c, data_range=255):.3f} dB")
        print(f"SSIM: {ssim(orig_u8_c, rec_u8_c, data_range=255):.6f}")
            #find compression ratio
        input_size = os.path.getsize(original_path)
        output_size = os.path.getsize(available_out_path)

        print('data lost, ', input_size - output_size, ' of ', input_size)
    return

# ---------------------------------------- terminal argparse stuff --------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="codec")

    sub = p.add_subparsers(dest="cmd", required=True)

    # encode subcommand
    pe = sub.add_parser("encode", help="Encode an image into a .job file")
    pe.add_argument("path", help="Input image path (e.g. .png, .jpg)",)
    pe.add_argument("-L", type=int, default=3, required=False, help="DWT levels (e.g. 1,2,3...)")
    pe.add_argument("-w", "--wavelet", type=str, default="haar", dest="wavelet_name", required=False, help="Wavelet name (e.g. haar)")
    pe.add_argument("-t", "--threshold", type=float, default=0.03, dest="threshold_fraction", required=False,
                    help="Threshold fraction (e.g. 0.05)")
    pe.add_argument("-q", "--quant-levels", dest="quant_bits", type=int, default = 8, required=False,
                    help="No. bits for quantisation (e.g. n=8 gives 2**8 = 256 quantisation levels)")

    # decode subcommand
    pd = sub.add_parser("decode", help="Decode a .job file")
    pd.add_argument("path", help="Input .job path")
    pd.add_argument('original_path', help = 'original image path for compression metrics')

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "encode":
        encode(
            path=args.path,
            L=args.L,
            wavelet_name=args.wavelet_name,
            threshold_fraction=args.threshold_fraction,
            quant_bits=args.quant_bits,
        )

    elif args.cmd == "decode":
        decode(
            args.path,
            args.original_path
        )


if __name__ == "__main__":
    main()

