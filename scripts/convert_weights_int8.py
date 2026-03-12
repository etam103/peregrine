#!/usr/bin/env python3
"""Convert f32 Peregrine model weights to int8 per-column symmetric quantized format.

Usage:
    python scripts/convert_weights_int8.py weights/model.bin weights/model_int8.bin

Format per tensor:
    [name_len: u32][name: utf8][dtype: u8][ndim: u32][shape...u32]
    dtype=0 (f32): [f32 data]
    dtype=1 (i8):  [i8 data][f32 scales (per-column)]

Linear weight tensors (2D, name contains ".weight") are quantized to int8.
Bias tensors and 1D tensors are kept as f32.
"""

import struct
import sys
import numpy as np


def read_peregrine_model(path):
    """Read Peregrine binary format: [num_tensors][name+shape+f32_data]..."""
    tensors = []
    with open(path, "rb") as f:
        (num_tensors,) = struct.unpack("<I", f.read(4))
        for _ in range(num_tensors):
            (name_len,) = struct.unpack("<I", f.read(4))
            name = f.read(name_len).decode("utf-8")
            (ndim,) = struct.unpack("<I", f.read(4))
            shape = []
            for _ in range(ndim):
                (s,) = struct.unpack("<I", f.read(4))
                shape.append(s)
            num_elements = 1
            for s in shape:
                num_elements *= s
            data = np.frombuffer(f.read(num_elements * 4), dtype=np.float32).copy()
            data = data.reshape(shape) if shape else data
            tensors.append((name, shape, data))
    return tensors


def write_int8_model(tensors, path):
    """Write mixed f32/i8 format."""
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(tensors)))

        for name, shape, data in tensors:
            name_bytes = name.encode("utf-8")
            f.write(struct.pack("<I", len(name_bytes)))
            f.write(name_bytes)

            is_weight_2d = len(shape) == 2 and ".weight" in name

            if is_weight_2d:
                # Quantize to int8 per-column symmetric
                rows, cols = shape
                col_absmax = np.abs(data).max(axis=0)  # [cols]
                scales = np.where(col_absmax == 0, 1.0, col_absmax / 127.0).astype(np.float32)
                quantized = np.clip(np.round(data / scales[np.newaxis, :]), -127, 127).astype(np.int8)

                # dtype = 1 (i8)
                f.write(struct.pack("B", 1))
                f.write(struct.pack("<I", 2))
                f.write(struct.pack("<I", rows))
                f.write(struct.pack("<I", cols))
                f.write(quantized.tobytes())
                f.write(scales.tobytes())

                # Stats
                dequant = quantized.astype(np.float32) * scales[np.newaxis, :]
                max_err = np.abs(data - dequant).max()
                print(f"  {name}: [{rows}x{cols}] -> int8, max_err={max_err:.6f}")
            else:
                # Keep as f32
                f.write(struct.pack("B", 0))
                f.write(struct.pack("<I", len(shape)))
                for s in shape:
                    f.write(struct.pack("<I", s))
                f.write(data.astype(np.float32).tobytes())
                print(f"  {name}: {shape} -> f32 (kept)")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.bin> <output_int8.bin>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"Loading {input_path}...")
    tensors = read_peregrine_model(input_path)
    print(f"  {len(tensors)} tensors loaded")

    total_f32_bytes = sum(d.nbytes for _, _, d in tensors)

    print(f"\nQuantizing to int8...")
    write_int8_model(tensors, output_path)

    import os
    out_size = os.path.getsize(output_path)
    ratio = out_size / total_f32_bytes
    print(f"\n  Original:   {total_f32_bytes:>12,} bytes")
    print(f"  Quantized:  {out_size:>12,} bytes")
    print(f"  Ratio:      {ratio:.2f}x ({(1-ratio)*100:.1f}% smaller)")
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
