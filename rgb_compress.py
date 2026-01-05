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

    return quantized_coeffs, min_val, step_size

#compresses long runs of zeros (should replace with entropy or huffman soon)

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

#checks if name if available and if not adds a suffix

def name_available(path: str, sep: str = "_") -> str:
    folder, filename = os.path.split(path)
    stem, ext = os.path.splitext(filename)

    candidate = path
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(folder, f"{stem}{sep}{i}{ext}")
        i += 1
    return candidate

#encodes each individual R G B array same way we do with the greyscale ones. returns data for dict

def encode_array(image_array, wavelet_name, L, threshold_fraction, quant_bits):

    image_array = np.asarray(image_array, dtype=np.float32)
    H, W = image_array.shape
    print(f"Loaded channel: {W} x {H} pixels")

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
        "min_val": float(min_val),
        "step_size": float(step_size),
        "total_values": int(total_values),
        "encoded_values": encoded_values,
    }

#creats dictionary / header then produces the .job file. returns filename (might be useful?)

def write_file(image_path, image_array_rgb, L, wavelet_name, threshold_fraction, quant_bits, channels):
    # filename (make threshold filename-safe)
    out_path = (
        os.path.splitext(image_path)[0]
        + f"_{wavelet_name}_L{L}_t{str(threshold_fraction)}_q{quant_bits}.job"
    )
    output_filename = name_available(out_path)

    H, W = image_array_rgb.shape[:2]

    with open(output_filename, "wb") as f:
        # container header
        f.write(b"RGB0")                       # magic
        f.write(struct.pack("<H", W))
        f.write(struct.pack("<H", H))
        f.write(struct.pack("<B", L))

        wavelet_bytes = wavelet_name.encode("ascii")
        f.write(struct.pack("<B", len(wavelet_bytes)))
        f.write(wavelet_bytes)

        f.write(struct.pack("<B", quant_bits))
        f.write(struct.pack("<f", float(threshold_fraction)))

        # channel blocks: R, G, B
        if len(channels) != 3:
            raise ValueError("Expected 3 channel payloads")

        for ch in channels:
            min_val = float(ch["min_val"])
            step_size = float(ch["step_size"])
            total_values = int(ch["total_values"])
            encoded_values = ch["encoded_values"]
            encoded_count = len(encoded_values)

            f.write(struct.pack("<f", min_val))
            f.write(struct.pack("<f", step_size))
            f.write(struct.pack("<I", total_values))
            f.write(struct.pack("<I", encoded_count))

            data_array = np.array(encoded_values, dtype=np.uint16)
            f.write(data_array.tobytes())

    input_size = os.path.getsize(image_path)
    output_size = os.path.getsize(output_filename)

    print("original file size:", input_size, "new file size:", output_size)
    print("compression ratio:", input_size / output_size)
    return output_filename

#combines the three arrays and applies write_file to produce final result

def encode(path, wavelet_name, L, threshold_fraction, quant_bits):

    image = Image.open(path).convert("RGB")
    image_size = os.path.getsize(path)
    RGB_array = np.array(image)  # shape (H, W, 3)

    R = RGB_array[:, :, 0]
    G = RGB_array[:, :, 1]
    B = RGB_array[:, :, 2]

    R_encoded = encode_array(R, wavelet_name, L, threshold_fraction, quant_bits)
    G_encoded = encode_array(G, wavelet_name, L, threshold_fraction, quant_bits)
    B_encoded = encode_array(B, wavelet_name, L, threshold_fraction, quant_bits)

    return write_file(
        image_path=path,
        image_array_rgb=RGB_array,
        L=L,
        wavelet_name=wavelet_name,
        threshold_fraction=threshold_fraction,
        quant_bits=quant_bits,
        channels=[R_encoded, G_encoded, B_encoded],
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
                raise ValueError("Malformed RLE stream: sentinel at end with no count.")
            run_len = int(encoded[i]); i += 1
            out.extend([0] * run_len)
    return np.array(out, dtype=np.uint16)

#find structure of array, width height etc

def infer_coeff_shapes(height: int, width: int, wavelet_name: str, L: int, mode: str = "symmetric"):
    dummy = np.zeros((height, width), dtype=np.float32)
    coeffs = pywt.wavedec2(dummy, wavelet_name, level=L, mode=mode)
    shapes = [coeffs[0].shape] + [tuple(arr.shape for arr in detail) for detail in coeffs[1:]]
    return shapes

#rebuild tree, basically undoes the multilevel part of the multilevel DWT

def build_coeff_tree_from_flat(flat_q: np.ndarray, H: int, W: int, L: int, wavelet: str,
                                min_val: float, step: float, mode: str = "symmetric"):
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
        raise ValueError(f"Did not consume the whole coefficient stream: used {idx}, total {flat_q.size}")

    return coeffs

def read_greyscale_job(data: bytes):
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
        kind="L",
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
    return header, [encoded]

def read_rgb_job(data: bytes, RGB_MAGIC):
    off = 0
    magic = data[off:off+4]; off += 4
    if magic != RGB_MAGIC:
        raise ValueError("Not an RGB0 file")

    width  = struct.unpack_from("<H", data, off)[0]; off += 2
    height = struct.unpack_from("<H", data, off)[0]; off += 2
    L      = struct.unpack_from("<B", data, off)[0]; off += 1

    name_len = struct.unpack_from("<B", data, off)[0]; off += 1
    wavelet_name = data[off:off+name_len].decode("ascii"); off += name_len

    quant_bits = struct.unpack_from("<B", data, off)[0]; off += 1
    threshold_fraction = struct.unpack_from("<f", data, off)[0]; off += 4

    channels = []
    for _ in range(3):
        min_val = struct.unpack_from("<f", data, off)[0]; off += 4
        step_size = struct.unpack_from("<f", data, off)[0]; off += 4
        total_count = struct.unpack_from("<I", data, off)[0]; off += 4
        encoded_count = struct.unpack_from("<I", data, off)[0]; off += 4

        enc = np.frombuffer(data, dtype="<u2", offset=off, count=encoded_count).copy()
        off += encoded_count * 2

        channels.append(dict(
            min_val=min_val,
            step_size=step_size,
            total_count=total_count,
            encoded_count=encoded_count,
            encoded=enc
        ))

    header = dict(
        kind="RGB",
        width=width,
        height=height,
        L=L,
        wavelet_name=wavelet_name,
        quant_bits=quant_bits,
        threshold_fraction=threshold_fraction,
    )
    return header, channels

#IDWT, interpets dictionary / header, saves file and performs analystics

def decode(path, original_path = None):
    RGB_MAGIC = b"RGB0"

    with open(path, "rb") as f:
        data = f.read()

    if data[:4] == RGB_MAGIC:
        header, ch_meta = read_rgb_job(data, RGB_MAGIC)
    else:
        header, encoded_list = read_greyscale_job(data)
        ch_meta = [dict(
            min_val=header["min_val"],
            step_size=header["step_size"],
            total_count=header["total_count"],
            encoded_count=header["encoded_count"],
            encoded=encoded_list[0]
        )]

    print("Metadata:")
    for k, v in header.items():
        print(f"  {k}: {v}")

    H = int(header["height"])
    W = int(header["width"])
    L = int(header["L"])
    wavelet = header["wavelet_name"]

    decoded_channels = []
    for idx_ch, ch in enumerate(ch_meta):
        flat_q = rle_decode(ch["encoded"], sentinel=0xFFFF)
        if flat_q.size != int(ch["total_count"]):
            raise ValueError(f"Channel {idx_ch}: decoded length {flat_q.size} != total_count {ch['total_count']}")

        coeffs_rec = build_coeff_tree_from_flat(
            flat_q, H, W, L, wavelet,
            min_val=float(ch["min_val"]),
            step=float(ch["step_size"]),
            mode="symmetric"
        )

        rec = pywt.waverec2(coeffs_rec, wavelet, mode="symmetric")
        rec = rec[:H, :W]
        rec_u8 = np.clip(np.rint(rec), 0, 255).astype(np.uint8)
        decoded_channels.append(rec_u8)

    base = os.path.splitext(path)[0]
    out_path = base + ("_rgb_uncompressed.tiff" if header["kind"] == "RGB" else "_uncompressed.tiff")
    out_path = name_available(out_path)

    if header["kind"] == "RGB":
        rgb = np.stack(decoded_channels, axis=-1)  # (H,W,3)
        Image.fromarray(rgb, mode="RGB").save(out_path, format="TIFF", compression=None)
        print("Saved:", out_path)
    else:
        Image.fromarray(decoded_channels[0], mode="L").save(out_path, format="TIFF", compression=None)
        print("Saved:", out_path)

    if original_path is None:
        return out_path

    if header["kind"] == "RGB":
        orig = np.array(Image.open(original_path).convert("RGB"), dtype=np.uint8)
        rec  = np.array(Image.open(out_path).convert("RGB"), dtype=np.uint8)
        h = min(orig.shape[0], rec.shape[0])
        w = min(orig.shape[1], rec.shape[1])
        orig_c = orig[:h, :w, :]
        rec_c  = rec[:h, :w, :]

        print(f"PSNR: {psnr(orig_c, rec_c, data_range=255):.3f} dB")
        print(f"SSIM: {ssim(orig_c, rec_c, data_range=255, channel_axis=-1):.6f}")
    else:
        orig = np.array(Image.open(original_path).convert("L"), dtype=np.uint8)
        rec  = np.array(Image.open(out_path).convert("L"), dtype=np.uint8)
        h = min(orig.shape[0], rec.shape[0])
        w = min(orig.shape[1], rec.shape[1])
        orig_c = orig[:h, :w]
        rec_c  = rec[:h, :w]

        print(f"PSNR: {psnr(orig_c, rec_c, data_range=255):.3f} dB")
        print(f"SSIM: {ssim(orig_c, rec_c, data_range=255):.6f}")

    # Compression size: compare compressed .job to original image (not the reconstructed TIFF)
    input_size = os.path.getsize(original_path)
    job_size = os.path.getsize(path)
    print("original bytes:", input_size, "job bytes:", job_size)
    print("compression ratio (orig:job):", input_size / job_size)

    return out_path

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

