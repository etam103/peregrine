#!/usr/bin/env python3
"""Quantize + Huffman-encode existing .bin weight files.

Reads a Peregrine binary weight file (f32 or i8 quantized), applies int8
per-column quantization (if f32), then Huffman-encodes the i8 values and
writes the result as dtype=3 tensors.

Usage:
    python scripts/convert_huffman.py weights/model.bin weights/model_huffman.bin [--streams 4]
"""

import struct
import sys
from collections import Counter
import heapq


def read_model(path):
    """Read a Peregrine .bin model file. Returns list of (name, dtype, data)."""
    tensors = []
    with open(path, 'rb') as f:
        num_tensors = struct.unpack('<I', f.read(4))[0]
        for _ in range(num_tensors):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            # Check for dtype tag (quantized files have it, vanilla f32 don't)
            # Peek: if next byte could be a dtype tag (0, 1, 2, 3) and followed
            # by ndim=2, treat it as typed. Otherwise treat as vanilla f32.
            pos = f.tell()
            maybe_dtype = struct.unpack('B', f.read(1))[0]

            if maybe_dtype in (1, 2, 3):
                # Typed tensor
                ndim = struct.unpack('<I', f.read(4))[0]
                shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndim)]

                if maybe_dtype == 1:  # i8 quantized
                    rows, cols = shape
                    data_i8 = list(f.read(rows * cols))
                    data_i8 = [x if x < 128 else x - 256 for x in data_i8]
                    scales = struct.unpack(f'<{cols}f', f.read(cols * 4))
                    tensors.append((name, 'i8', rows, cols, data_i8, list(scales)))
                else:
                    raise ValueError(f"Unsupported dtype {maybe_dtype} for conversion")
            else:
                # Vanilla f32 tensor
                f.seek(pos)  # rewind
                ndim = struct.unpack('<I', f.read(4))[0]
                shape = [struct.unpack('<I', f.read(4))[0] for _ in range(ndim)]
                num_elements = 1
                for s in shape:
                    num_elements *= s
                data = list(struct.unpack(f'<{num_elements}f', f.read(num_elements * 4)))

                if ndim == 2:
                    rows, cols = shape
                    tensors.append((name, 'f32', rows, cols, data, None))
                else:
                    # Non-2D tensors: keep as f32, skip Huffman
                    tensors.append((name, 'f32_skip', shape, None, data, None))

    return tensors


def quantize_f32_to_i8(data, rows, cols):
    """Per-column symmetric quantization: i8 + scales."""
    col_absmax = [0.0] * cols
    for m in range(rows):
        for n in range(cols):
            v = abs(data[m * cols + n])
            if v > col_absmax[n]:
                col_absmax[n] = v

    scales = [amax / 127.0 if amax != 0 else 1.0 for amax in col_absmax]

    data_i8 = []
    for m in range(rows):
        for n in range(cols):
            v = data[m * cols + n] / scales[n]
            q = max(-127, min(127, round(v)))
            data_i8.append(q)

    return data_i8, scales


class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq


def build_huffman_table(data_i8):
    """Build canonical Huffman table. Returns bit_lengths[256]."""
    freq = Counter(data_i8)

    if len(freq) <= 1:
        bit_lengths = [0] * 256
        for sym, _ in freq.items():
            bit_lengths[sym + 128] = 1
        return bit_lengths

    # Build tree
    heap = [HuffmanNode(symbol=s, freq=f) for s, f in freq.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        parent = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, parent)

    # Extract bit lengths
    bit_lengths = [0] * 256

    def walk(node, depth):
        if node.symbol is not None:
            bl = min(depth, 12)  # clamp to 12 bits
            bit_lengths[node.symbol + 128] = bl
        else:
            if node.left:
                walk(node.left, depth + 1)
            if node.right:
                walk(node.right, depth + 1)

    walk(heap[0], 0)

    # Handle edge case: root is a leaf
    if heap[0].symbol is not None:
        bit_lengths[heap[0].symbol + 128] = 1

    return bit_lengths


def canonical_codes(bit_lengths):
    """Compute canonical codes from bit lengths."""
    max_len = max(bit_lengths)
    if max_len == 0:
        return [0] * 256

    bl_count = [0] * (max_len + 1)
    for bl in bit_lengths:
        if bl > 0:
            bl_count[bl] += 1

    next_code = [0] * (max_len + 1)
    code = 0
    for bits in range(1, max_len + 1):
        code = (code + bl_count[bits - 1]) << 1
        next_code[bits] = code

    codes = [0] * 256
    for s in range(256):
        bl = bit_lengths[s]
        if bl > 0:
            codes[s] = next_code[bl]
            next_code[bl] += 1

    return codes


def huffman_encode(data_i8, bit_lengths, codes):
    """Encode i8 data to bitstream bytes."""
    current_byte = 0
    bits_in_byte = 0
    result = bytearray()

    for val in data_i8:
        idx = val + 128
        code = codes[idx]
        bl = bit_lengths[idx]

        for bit_pos in range(bl - 1, -1, -1):
            bit = (code >> bit_pos) & 1
            current_byte = (current_byte << 1) | bit
            bits_in_byte += 1
            if bits_in_byte == 8:
                result.append(current_byte)
                current_byte = 0
                bits_in_byte = 0

    if bits_in_byte > 0:
        current_byte <<= (8 - bits_in_byte)
        result.append(current_byte)

    return bytes(result)


def write_huffman_tensor(f, name, rows, cols, data_i8, scales, num_streams):
    """Write a Huffman-compressed tensor."""
    name_bytes = name.encode('utf-8')
    f.write(struct.pack('<I', len(name_bytes)))
    f.write(name_bytes)

    # dtype=3
    f.write(struct.pack('B', 3))

    # shape
    f.write(struct.pack('<I', 2))
    f.write(struct.pack('<I', rows))
    f.write(struct.pack('<I', cols))

    # Build Huffman table
    bit_lengths = build_huffman_table(data_i8)
    codes = canonical_codes(bit_lengths)

    # Codebook
    f.write(bytes(bit_lengths))

    # Split into streams
    total = len(data_i8)
    chunk_size = (total + num_streams - 1) // num_streams

    streams = []
    stream_lengths = []
    for s in range(num_streams):
        start = s * chunk_size
        end = min(start + chunk_size, total)
        if start >= total:
            streams.append(b'')
            stream_lengths.append(0)
        else:
            chunk = data_i8[start:end]
            encoded = huffman_encode(chunk, bit_lengths, codes)
            streams.append(encoded)
            stream_lengths.append(len(chunk))

    f.write(struct.pack('<I', num_streams))

    for sl in stream_lengths:
        f.write(struct.pack('<I', sl))
    for stream in streams:
        f.write(struct.pack('<I', len(stream)))
    for stream in streams:
        f.write(stream)

    # scales
    for s in scales:
        f.write(struct.pack('<f', s))


def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} input.bin output.bin [--streams N]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]
    num_streams = 4

    for i, arg in enumerate(sys.argv):
        if arg == '--streams' and i + 1 < len(sys.argv):
            num_streams = int(sys.argv[i + 1])

    print(f"Reading {input_path}...")
    tensors = read_model(input_path)
    print(f"  Found {len(tensors)} tensors")

    total_orig = 0
    total_compressed = 0

    with open(output_path, 'wb') as f:
        # Count 2D tensors (we'll convert those)
        count_2d = sum(1 for t in tensors if t[1] in ('f32', 'i8'))
        f.write(struct.pack('<I', count_2d))

        for tensor in tensors:
            name = tensor[0]
            dtype = tensor[1]

            if dtype == 'f32_skip':
                print(f"  Skipping non-2D tensor: {name}")
                continue

            if dtype == 'f32':
                rows, cols = tensor[2], tensor[3]
                data = tensor[4]
                data_i8, scales = quantize_f32_to_i8(data, rows, cols)
                orig_bytes = rows * cols * 4
            elif dtype == 'i8':
                rows, cols = tensor[2], tensor[3]
                data_i8 = tensor[4]
                scales = tensor[5]
                orig_bytes = rows * cols + cols * 4
            else:
                continue

            write_huffman_tensor(f, name, rows, cols, data_i8, scales, num_streams)

            # Rough compressed size estimate
            bit_lengths = build_huffman_table(data_i8)
            avg_bits = sum(bit_lengths[v + 128] for v in data_i8) / len(data_i8) if data_i8 else 8
            compressed_bytes = int(len(data_i8) * avg_bits / 8) + 256 + cols * 4
            total_orig += orig_bytes
            total_compressed += compressed_bytes

            ratio = orig_bytes / compressed_bytes if compressed_bytes > 0 else 0
            print(f"  {name}: {rows}x{cols}, {ratio:.2f}x compression")

    print(f"\nTotal: {total_orig / 1048576:.1f} MB → ~{total_compressed / 1048576:.1f} MB "
          f"({total_orig / total_compressed:.2f}x)")
    print(f"Written to {output_path}")


if __name__ == '__main__':
    main()
