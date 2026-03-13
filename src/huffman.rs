//! Huffman compression for int8 quantized weights and KV cache.
//!
//! - `HuffmanTable`: canonical Huffman from i8 symbol frequencies, encode/decode
//! - `HuffmanTensor`: compressed int8 weights (bitstream + codebook + scales)
//! - `HuffmanKVCache`: wraps `StandardKVCache` with hot window + compressed chunks

#[cfg(feature = "metal")]
use crate::metal::{GpuBuffer, GpuContext};
use crate::quant::QuantizedTensor;

/// Maximum number of i8 symbols (256 values: -128..=127).
const NUM_SYMBOLS: usize = 256;

/// A canonical Huffman table for i8 symbols.
///
/// Maps each i8 value (offset by 128 → index 0..255) to a variable-length bit code.
/// The table is canonical: codes are assigned in order of increasing bit length,
/// with lexicographic ordering within the same length.
pub struct HuffmanTable {
    /// Bit length for each symbol (0..255). 0 means symbol not present.
    pub bit_lengths: [u8; NUM_SYMBOLS],
    /// Canonical code for each symbol.
    pub codes: [u32; NUM_SYMBOLS],
    /// Decode LUT: for codes up to MAX_LUT_BITS, maps code → (symbol_i8, bit_length).
    /// Populated lazily for fast decoding.
    pub decode_lut: Vec<(i8, u8)>,
}

/// Max bits for the LUT-based decode fast path.
const MAX_LUT_BITS: u8 = 12;

impl HuffmanTable {
    /// Build a canonical Huffman table from symbol frequencies.
    /// `freqs[i]` = frequency of symbol `(i as i8 - 128)`.
    pub fn from_frequencies(freqs: &[u64; NUM_SYMBOLS]) -> Self {
        // Count number of symbols with non-zero frequency
        let active: Vec<(usize, u64)> = freqs
            .iter()
            .enumerate()
            .filter(|(_, &f)| f > 0)
            .map(|(i, &f)| (i, f))
            .collect();

        if active.is_empty() {
            return HuffmanTable {
                bit_lengths: [0; NUM_SYMBOLS],
                codes: [0; NUM_SYMBOLS],
                decode_lut: Vec::new(),
            };
        }

        if active.len() == 1 {
            // Only one symbol: assign it bit_length=1, code=0
            let mut bit_lengths = [0u8; NUM_SYMBOLS];
            let mut codes = [0u32; NUM_SYMBOLS];
            bit_lengths[active[0].0] = 1;
            codes[active[0].0] = 0;
            let mut table = HuffmanTable {
                bit_lengths,
                codes,
                decode_lut: Vec::new(),
            };
            table.build_decode_lut();
            return table;
        }

        // Build Huffman tree using a min-heap approach (package-merge simplified)
        // Use the standard algorithm: repeatedly merge two lowest-frequency nodes
        let n = active.len();
        let symbols: Vec<usize> = active.iter().map(|&(s, _)| s).collect();
        let freq_vals: Vec<u64> = active.iter().map(|&(_, f)| f).collect();

        // Build bit lengths via Huffman tree simulation
        // Nodes: 0..n-1 are leaves, n..2n-2 are internal
        let mut parent = vec![0usize; 2 * n - 1];
        let mut depth = vec![0u8; 2 * n - 1];

        // Sort leaves by frequency
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| freq_vals[i]);

        // Two-queue merge (optimal O(n log n) Huffman)
        let mut leaf_queue: std::collections::VecDeque<(u64, usize)> = order
            .iter()
            .map(|&i| (freq_vals[i], i))
            .collect();
        let mut internal_queue: std::collections::VecDeque<(u64, usize)> =
            std::collections::VecDeque::new();

        let mut next_node = n;

        for _ in 0..n - 1 {
            // Pick two smallest from either queue
            let pick_smallest =
                |leaf_q: &mut std::collections::VecDeque<(u64, usize)>,
                 int_q: &mut std::collections::VecDeque<(u64, usize)>|
                 -> (u64, usize) {
                    match (leaf_q.front(), int_q.front()) {
                        (Some(&lf), Some(&inf)) => {
                            if lf.0 <= inf.0 {
                                leaf_q.pop_front().unwrap()
                            } else {
                                int_q.pop_front().unwrap()
                            }
                        }
                        (Some(_), None) => leaf_q.pop_front().unwrap(),
                        (None, Some(_)) => int_q.pop_front().unwrap(),
                        (None, None) => unreachable!(),
                    }
                };

            let (f1, n1) = pick_smallest(&mut leaf_queue, &mut internal_queue);
            let (f2, n2) = pick_smallest(&mut leaf_queue, &mut internal_queue);

            parent[n1] = next_node;
            parent[n2] = next_node;
            internal_queue.push_back((f1 + f2, next_node));
            next_node += 1;
        }

        // Compute depths (root is the last node, depth 0)
        let root = 2 * n - 2;
        depth[root] = 0;
        for i in (0..root).rev() {
            depth[i] = depth[parent[i]] + 1;
        }

        // Clamp to MAX_LUT_BITS to avoid excessively long codes
        let max_bits = MAX_LUT_BITS;
        for i in 0..n {
            if depth[i] > max_bits {
                depth[i] = max_bits;
            }
        }

        // Extract bit lengths for actual symbols
        let mut bit_lengths = [0u8; NUM_SYMBOLS];
        for i in 0..n {
            bit_lengths[symbols[i]] = depth[i];
        }

        // Build canonical codes
        let codes = Self::canonical_codes(&bit_lengths);

        let mut table = HuffmanTable {
            bit_lengths,
            codes,
            decode_lut: Vec::new(),
        };
        table.build_decode_lut();
        table
    }

    /// Assign canonical Huffman codes given bit lengths.
    fn canonical_codes(bit_lengths: &[u8; NUM_SYMBOLS]) -> [u32; NUM_SYMBOLS] {
        let mut codes = [0u32; NUM_SYMBOLS];

        // Find max bit length
        let max_len = *bit_lengths.iter().max().unwrap_or(&0) as usize;
        if max_len == 0 {
            return codes;
        }

        // Count symbols per bit length
        let mut bl_count = vec![0u32; max_len + 1];
        for &bl in bit_lengths.iter() {
            if bl > 0 {
                bl_count[bl as usize] += 1;
            }
        }

        // Compute first code for each bit length
        let mut next_code = vec![0u32; max_len + 1];
        let mut code = 0u32;
        for bits in 1..=max_len {
            code = (code + bl_count[bits - 1]) << 1;
            next_code[bits] = code;
        }

        // Assign codes to symbols in order
        for s in 0..NUM_SYMBOLS {
            let bl = bit_lengths[s] as usize;
            if bl > 0 {
                codes[s] = next_code[bl];
                next_code[bl] += 1;
            }
        }

        codes
    }

    /// Build the decode LUT for fast decoding.
    fn build_decode_lut(&mut self) {
        let lut_size = 1usize << MAX_LUT_BITS;
        self.decode_lut = vec![(0i8, 0u8); lut_size];

        for s in 0..NUM_SYMBOLS {
            let bl = self.bit_lengths[s];
            if bl == 0 || bl > MAX_LUT_BITS {
                continue;
            }
            let code = self.codes[s];
            let sym = (s as i16 - 128) as i8;

            // Fill all LUT entries that start with this code
            let prefix_shift = MAX_LUT_BITS - bl;
            let base = (code as usize) << prefix_shift;
            let count = 1usize << prefix_shift;
            for j in 0..count {
                self.decode_lut[base + j] = (sym, bl);
            }
        }
    }

    /// Encode a slice of i8 values into a bitstream. Returns packed bytes.
    pub fn encode(&self, data: &[i8]) -> Vec<u8> {
        if data.is_empty() {
            return Vec::new();
        }

        // Estimate output size: avg ~4 bits per symbol
        let mut bits = Vec::with_capacity(data.len());
        let mut current_byte: u8 = 0;
        let mut bits_in_byte: u8 = 0;

        for &val in data {
            let idx = (val as i16 + 128) as usize;
            let code = self.codes[idx];
            let bl = self.bit_lengths[idx];

            // Write bits MSB first
            for bit_pos in (0..bl).rev() {
                let bit = ((code >> bit_pos) & 1) as u8;
                current_byte = (current_byte << 1) | bit;
                bits_in_byte += 1;
                if bits_in_byte == 8 {
                    bits.push(current_byte);
                    current_byte = 0;
                    bits_in_byte = 0;
                }
            }
        }

        // Flush remaining bits (pad with zeros)
        if bits_in_byte > 0 {
            current_byte <<= 8 - bits_in_byte;
            bits.push(current_byte);
        }

        bits
    }

    /// Decode `num_symbols` values from a bitstream.
    pub fn decode(&self, bitstream: &[u8], num_symbols: usize) -> Vec<i8> {
        if num_symbols == 0 {
            return Vec::new();
        }

        let mut result = Vec::with_capacity(num_symbols);
        let mut bit_pos: usize = 0;
        let total_bits = bitstream.len() * 8;

        for _ in 0..num_symbols {
            // Read MAX_LUT_BITS bits for LUT lookup
            let mut code: u32 = 0;
            let bits_available = total_bits.saturating_sub(bit_pos).min(MAX_LUT_BITS as usize);
            for b in 0..bits_available {
                let byte_idx = (bit_pos + b) / 8;
                let bit_idx = 7 - ((bit_pos + b) % 8);
                code = (code << 1) | ((bitstream[byte_idx] >> bit_idx) as u32 & 1);
            }
            // Pad with zeros if fewer bits available
            if bits_available < MAX_LUT_BITS as usize {
                code <<= MAX_LUT_BITS as usize - bits_available;
            }

            let (sym, bl) = self.decode_lut[code as usize];
            debug_assert!(bl > 0, "invalid code in bitstream");
            result.push(sym);
            bit_pos += bl as usize;
        }

        result
    }

    /// Serialize the codebook to bytes: [bit_lengths: 256 bytes].
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bit_lengths.to_vec()
    }

    /// Deserialize a codebook from bytes.
    pub fn from_bytes(data: &[u8]) -> Self {
        assert!(data.len() >= NUM_SYMBOLS, "codebook too short");
        let mut bit_lengths = [0u8; NUM_SYMBOLS];
        bit_lengths.copy_from_slice(&data[..NUM_SYMBOLS]);
        let codes = Self::canonical_codes(&bit_lengths);
        let mut table = HuffmanTable {
            bit_lengths,
            codes,
            decode_lut: Vec::new(),
        };
        table.build_decode_lut();
        table
    }
}

/// A weight tensor compressed with Huffman coding over int8 quantized values.
///
/// Stores per-column scales (same as `QuantizedTensor`) plus a Huffman-encoded
/// bitstream of the i8 values, split into `num_streams` for parallel decode.
pub struct HuffmanTensor {
    /// Huffman codebook (shared across all streams).
    pub table: HuffmanTable,
    /// Compressed bitstreams, one per stream.
    pub streams: Vec<Vec<u8>>,
    /// Number of i8 symbols per stream.
    pub stream_lengths: Vec<usize>,
    /// Per-column dequantization scales (same as QuantizedTensor).
    pub scales: Vec<f32>,
    /// Matrix dimensions.
    pub rows: usize,
    pub cols: usize,
    /// Number of parallel decode streams.
    pub num_streams: usize,
}

impl HuffmanTensor {
    /// Compress a `QuantizedTensor` into a `HuffmanTensor`.
    ///
    /// Splits the i8 data into `num_streams` chunks (by rows), builds a shared
    /// Huffman table from the full data, and encodes each chunk independently.
    pub fn from_quantized(qt: &QuantizedTensor, num_streams: usize) -> Self {
        let num_streams = num_streams.max(1);
        let total = qt.data_i8.len();

        // Count frequencies across all data
        let mut freqs = [0u64; NUM_SYMBOLS];
        for &v in &qt.data_i8 {
            freqs[(v as i16 + 128) as usize] += 1;
        }

        let table = HuffmanTable::from_frequencies(&freqs);

        // Split data into streams (by contiguous chunks)
        let chunk_size = (total + num_streams - 1) / num_streams;
        let mut streams = Vec::with_capacity(num_streams);
        let mut stream_lengths = Vec::with_capacity(num_streams);

        for s in 0..num_streams {
            let start = s * chunk_size;
            let end = (start + chunk_size).min(total);
            if start >= total {
                streams.push(Vec::new());
                stream_lengths.push(0);
            } else {
                let chunk = &qt.data_i8[start..end];
                streams.push(table.encode(chunk));
                stream_lengths.push(chunk.len());
            }
        }

        HuffmanTensor {
            table,
            streams,
            stream_lengths,
            scales: qt.scales.clone(),
            rows: qt.rows,
            cols: qt.cols,
            num_streams,
        }
    }

    /// Decode back to a `QuantizedTensor`.
    pub fn decode_to_quantized(&self) -> QuantizedTensor {
        let total = self.rows * self.cols;
        let mut data_i8 = Vec::with_capacity(total);

        for (stream, &len) in self.streams.iter().zip(&self.stream_lengths) {
            let decoded = self.table.decode(stream, len);
            data_i8.extend_from_slice(&decoded);
        }

        assert_eq!(data_i8.len(), total);

        QuantizedTensor {
            data_i8,
            scales: self.scales.clone(),
            rows: self.rows,
            cols: self.cols,
            #[cfg(feature = "metal")]
            gpu_data_i8: None,
            #[cfg(feature = "metal")]
            gpu_scales: None,
        }
    }

    /// Compressed size in bytes (bitstreams + codebook + scales).
    pub fn compressed_size(&self) -> usize {
        let bitstream_bytes: usize = self.streams.iter().map(|s| s.len()).sum();
        let codebook_bytes = NUM_SYMBOLS; // bit_lengths
        let scales_bytes = self.cols * 4;
        bitstream_bytes + codebook_bytes + scales_bytes
    }

    /// Uncompressed size in bytes (i8 data + scales).
    pub fn uncompressed_size(&self) -> usize {
        self.rows * self.cols + self.cols * 4
    }

    /// Compression ratio (uncompressed / compressed).
    pub fn compression_ratio(&self) -> f32 {
        self.uncompressed_size() as f32 / self.compressed_size() as f32
    }
}

/// GPU support for HuffmanTensor.
#[cfg(feature = "metal")]
impl HuffmanTensor {
    /// Decode on CPU then upload to GPU for matmul.
    /// Returns i8 and scales buffers ready for `dispatch_matmul_dequant_i8`.
    pub fn to_gpu(&self, gpu: &GpuContext) -> (GpuBuffer<i8>, GpuBuffer<f32>) {
        let qt = self.decode_to_quantized();
        let gpu_i8 = gpu.upload(&qt.data_i8);
        let gpu_scales = gpu.upload(&qt.scales);
        (gpu_i8, gpu_scales)
    }

    /// GPU-accelerated Huffman decode + dequant matmul.
    ///
    /// Decodes Huffman bitstream on CPU, uploads decoded i8 to GPU staging buffer,
    /// then dispatches `dispatch_matmul_dequant_i8` for the actual matmul.
    pub fn matmul_huffman_gpu(
        &self,
        gpu: &GpuContext,
        a_data: &[f32],
        m: usize,
    ) -> GpuBuffer<f32> {
        let k = self.rows;
        let n = self.cols;
        assert_eq!(a_data.len(), m * k);

        let (gpu_i8, gpu_scales) = self.to_gpu(gpu);
        let a_buf = gpu.upload(a_data);
        let out_buf: GpuBuffer<f32> = gpu.alloc(m * n);

        gpu.dispatch_matmul_dequant_i8(
            &a_buf, &gpu_i8, &gpu_scales, &out_buf,
            m as u32, n as u32, k as u32,
        );

        out_buf
    }
}

/// CPU Huffman matmul: decode then use `matmul_quantized`.
pub fn matmul_huffman(
    a_data: &[f32],
    m: usize,
    k: usize,
    ht: &HuffmanTensor,
) -> Vec<f32> {
    assert_eq!(ht.rows, k);
    let qt = ht.decode_to_quantized();
    crate::quant::matmul_quantized(a_data, m, k, &qt)
}

// ---------------------------------------------------------------------------
// HuffmanKVCache: wraps StandardKVCache with transparent compression
// ---------------------------------------------------------------------------

/// A KV cache that transparently compresses older tokens using Huffman coding.
///
/// Maintains a "hot window" of recent f32 tokens in a `StandardKVCache`, and
/// compresses older chunks into Huffman-encoded i8 with per-head scales.
pub struct HuffmanKVCache {
    /// Active (uncompressed) KV cache for recent tokens.
    pub hot: crate::attention::StandardKVCache,
    /// Compressed chunks of older K values.
    pub k_chunks: Vec<CompressedChunk>,
    /// Compressed chunks of older V values.
    pub v_chunks: Vec<CompressedChunk>,
    /// Number of tokens in compressed storage.
    pub compressed_len: usize,
    /// Threshold: compress when hot cache exceeds this many tokens.
    pub compress_threshold: usize,
    /// Size of each compressed chunk (in tokens).
    pub chunk_size: usize,
}

/// A single compressed chunk of KV data for all heads.
pub struct CompressedChunk {
    /// Huffman-encoded i8 data (per head, interleaved).
    pub bitstream: Vec<u8>,
    /// Per-head scales (one scale per head).
    pub scales: Vec<f32>,
    /// Huffman codebook (bit_lengths only — canonical codes are derived).
    pub codebook: Vec<u8>,
    /// Number of tokens in this chunk.
    pub num_tokens: usize,
    /// Number of heads.
    pub num_heads: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl HuffmanKVCache {
    /// Create a new HuffmanKVCache.
    ///
    /// - `num_kv_heads`: number of KV heads
    /// - `head_dim`: dimension per head
    /// - `compress_threshold`: compress when hot cache exceeds this many tokens
    /// - `chunk_size`: number of tokens per compressed chunk
    pub fn new(
        num_kv_heads: usize,
        head_dim: usize,
        compress_threshold: usize,
        chunk_size: usize,
    ) -> Self {
        HuffmanKVCache {
            hot: crate::attention::StandardKVCache::new(num_kv_heads, head_dim),
            k_chunks: Vec::new(),
            v_chunks: Vec::new(),
            compressed_len: 0,
            compress_threshold: compress_threshold.max(chunk_size),
            chunk_size,
        }
    }

    /// Append new K, V data (same interface as `StandardKVCache::append`).
    /// Transparently compresses when the hot cache grows beyond the threshold.
    pub fn append(&mut self, new_k: &[f32], new_v: &[f32], seq_len: usize) {
        self.hot.append(new_k, new_v, seq_len);

        // Check if we should compress
        while self.hot.len >= self.compress_threshold {
            self.compress_oldest_chunk();
        }
    }

    /// Compress the oldest `chunk_size` tokens from the hot cache.
    fn compress_oldest_chunk(&mut self) {
        let chunk_tokens = self.chunk_size.min(self.hot.len);
        if chunk_tokens == 0 {
            return;
        }

        let num_heads = self.hot.num_kv_heads;
        let head_dim = self.hot.head_dim;

        // Extract oldest tokens for K and V
        let k_chunk = self.extract_chunk_data(&self.hot.k, chunk_tokens, num_heads, head_dim);
        let v_chunk = self.extract_chunk_data(&self.hot.v, chunk_tokens, num_heads, head_dim);

        self.k_chunks.push(Self::compress_chunk(&k_chunk, chunk_tokens, num_heads, head_dim));
        self.v_chunks.push(Self::compress_chunk(&v_chunk, chunk_tokens, num_heads, head_dim));
        self.compressed_len += chunk_tokens;

        // Remove oldest tokens from hot cache
        self.remove_oldest_tokens(chunk_tokens);
    }

    /// Extract the first `num_tokens` from the per-head KV layout.
    fn extract_chunk_data(
        &self,
        data: &[f32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Vec<f32> {
        let total_len = self.hot.len;
        let mut result = Vec::with_capacity(num_heads * num_tokens * head_dim);
        for h in 0..num_heads {
            let base = h * total_len * head_dim;
            for t in 0..num_tokens {
                let offset = base + t * head_dim;
                result.extend_from_slice(&data[offset..offset + head_dim]);
            }
        }
        result
    }

    /// Remove the oldest `count` tokens from the hot cache.
    fn remove_oldest_tokens(&mut self, count: usize) {
        let old_len = self.hot.len;
        let new_len = old_len - count;
        let num_heads = self.hot.num_kv_heads;
        let head_dim = self.hot.head_dim;

        let mut new_k = Vec::with_capacity(num_heads * new_len * head_dim);
        let mut new_v = Vec::with_capacity(num_heads * new_len * head_dim);

        for h in 0..num_heads {
            let base = h * old_len * head_dim;
            let start = base + count * head_dim;
            let end = base + old_len * head_dim;
            new_k.extend_from_slice(&self.hot.k[start..end]);
            new_v.extend_from_slice(&self.hot.v[start..end]);
        }

        self.hot.k = new_k;
        self.hot.v = new_v;
        self.hot.len = new_len;
    }

    /// Compress a chunk of f32 KV data into Huffman-encoded i8.
    pub fn compress_chunk(
        data: &[f32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> CompressedChunk {
        let total = num_heads * num_tokens * head_dim;
        assert_eq!(data.len(), total);

        // Quantize per-head: find absmax per head, quantize to i8
        let mut scales = Vec::with_capacity(num_heads);
        let mut all_i8 = Vec::with_capacity(total);

        for h in 0..num_heads {
            let base = h * num_tokens * head_dim;
            let head_data = &data[base..base + num_tokens * head_dim];

            let absmax = head_data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };
            scales.push(scale);

            let inv_scale = 1.0 / scale;
            for &v in head_data {
                all_i8.push((v * inv_scale).round().clamp(-127.0, 127.0) as i8);
            }
        }

        // Build Huffman table and encode
        let mut freqs = [0u64; NUM_SYMBOLS];
        for &v in &all_i8 {
            freqs[(v as i16 + 128) as usize] += 1;
        }
        let table = HuffmanTable::from_frequencies(&freqs);
        let bitstream = table.encode(&all_i8);
        let codebook = table.to_bytes();

        CompressedChunk {
            bitstream,
            scales,
            codebook,
            num_tokens,
            num_heads,
            head_dim,
        }
    }

    /// Decompress a chunk back to f32 data.
    pub fn decompress_chunk(chunk: &CompressedChunk) -> Vec<f32> {
        let total = chunk.num_heads * chunk.num_tokens * chunk.head_dim;
        let table = HuffmanTable::from_bytes(&chunk.codebook);
        let decoded_i8 = table.decode(&chunk.bitstream, total);

        let mut result = Vec::with_capacity(total);
        let head_size = chunk.num_tokens * chunk.head_dim;
        for h in 0..chunk.num_heads {
            let scale = chunk.scales[h];
            let base = h * head_size;
            for i in 0..head_size {
                result.push(decoded_i8[base + i] as f32 * scale);
            }
        }

        result
    }

    /// Get the total number of cached tokens (compressed + hot).
    pub fn len(&self) -> usize {
        self.compressed_len + self.hot.len
    }

    /// Retrieve the full (decompressed) K and V data as contiguous f32 arrays.
    /// Layout: [num_kv_heads, total_len, head_dim]
    pub fn get_kv(&self) -> (Vec<f32>, Vec<f32>) {
        let total_len = self.len();
        let num_heads = self.hot.num_kv_heads;
        let head_dim = self.hot.head_dim;

        let mut k_full = Vec::with_capacity(num_heads * total_len * head_dim);
        let mut v_full = Vec::with_capacity(num_heads * total_len * head_dim);

        // For each head, concatenate compressed chunks then hot data
        for h in 0..num_heads {
            // Compressed chunks
            for chunk in &self.k_chunks {
                let chunk_data = Self::decompress_chunk(chunk);
                let head_size = chunk.num_tokens * head_dim;
                let base = h * head_size;
                k_full.extend_from_slice(&chunk_data[base..base + head_size]);
            }
            // Hot data
            let hot_base = h * self.hot.len * head_dim;
            let hot_end = hot_base + self.hot.len * head_dim;
            if self.hot.len > 0 {
                k_full.extend_from_slice(&self.hot.k[hot_base..hot_end]);
            }

            for chunk in &self.v_chunks {
                let chunk_data = Self::decompress_chunk(chunk);
                let head_size = chunk.num_tokens * head_dim;
                let base = h * head_size;
                v_full.extend_from_slice(&chunk_data[base..base + head_size]);
            }
            if self.hot.len > 0 {
                v_full.extend_from_slice(&self.hot.v[hot_base..hot_end]);
            }
        }

        (k_full, v_full)
    }
}

/// Pack the HuffmanTable decode LUT into a GPU-friendly u32 format.
/// Each entry: lower 8 bits = symbol (as u8, offset by 128), upper 8 bits = bit_length.
impl HuffmanTable {
    pub fn to_gpu_lut(&self) -> Vec<u32> {
        self.decode_lut
            .iter()
            .map(|&(sym, bl)| {
                let sym_u8 = (sym as i16 + 128) as u8;
                (bl as u32) << 8 | sym_u8 as u32
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn random_i8(n: usize, seed: u32) -> Vec<i8> {
        let mut data = Vec::with_capacity(n);
        let mut state = seed;
        for _ in 0..n {
            state = state.wrapping_mul(1103515245).wrapping_add(12345);
            // Bias toward 0 (simulating quantized weights)
            let raw = ((state >> 16) as i32 % 256 - 128) as i8;
            let biased = if (state >> 8) % 4 == 0 { raw } else { (raw / 4) as i8 };
            data.push(biased);
        }
        data
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let data = random_i8(1024, 42);

        let mut freqs = [0u64; NUM_SYMBOLS];
        for &v in &data {
            freqs[(v as i16 + 128) as usize] += 1;
        }

        let table = HuffmanTable::from_frequencies(&freqs);
        let encoded = table.encode(&data);
        let decoded = table.decode(&encoded, data.len());

        assert_eq!(decoded, data, "roundtrip mismatch");
    }

    #[test]
    fn test_empty_input() {
        let freqs = [0u64; NUM_SYMBOLS];
        let table = HuffmanTable::from_frequencies(&freqs);
        let encoded = table.encode(&[]);
        let decoded = table.decode(&encoded, 0);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_all_same_symbol() {
        let data = vec![42i8; 512];

        let mut freqs = [0u64; NUM_SYMBOLS];
        for &v in &data {
            freqs[(v as i16 + 128) as usize] += 1;
        }

        let table = HuffmanTable::from_frequencies(&freqs);
        let encoded = table.encode(&data);
        let decoded = table.decode(&encoded, data.len());

        assert_eq!(decoded, data);
        // With only one symbol and 1-bit code, we expect ~64 bytes
        assert!(encoded.len() <= 65, "encoded {} bytes for 512 same symbols", encoded.len());
    }

    #[test]
    fn test_table_correctness() {
        // Two symbols with known frequencies
        let mut freqs = [0u64; NUM_SYMBOLS];
        freqs[128] = 100; // symbol 0
        freqs[129] = 50;  // symbol 1

        let table = HuffmanTable::from_frequencies(&freqs);

        // More frequent symbol should get shorter or equal code
        assert!(table.bit_lengths[128] <= table.bit_lengths[129]);
        assert!(table.bit_lengths[128] > 0);
        assert!(table.bit_lengths[129] > 0);
    }

    #[test]
    fn test_huffman_tensor_roundtrip() {
        let rows = 64;
        let cols = 32;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 + 3) % 200) as f32 / 200.0 - 0.5)
            .collect();

        let qt = crate::quant::quantize_weights(&data, rows, cols);
        let ht = HuffmanTensor::from_quantized(&qt, 4);
        let qt2 = ht.decode_to_quantized();

        assert_eq!(qt.data_i8, qt2.data_i8);
        assert_eq!(qt.scales, qt2.scales);
        assert_eq!(qt.rows, qt2.rows);
        assert_eq!(qt.cols, qt2.cols);
    }

    #[test]
    fn test_huffman_tensor_compression() {
        let rows = 256;
        let cols = 256;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i * 7 + 3) % 200) as f32 / 200.0 - 0.5)
            .collect();

        let qt = crate::quant::quantize_weights(&data, rows, cols);
        let ht = HuffmanTensor::from_quantized(&qt, 4);

        assert!(
            ht.compression_ratio() > 1.0,
            "expected compression ratio > 1.0, got {}",
            ht.compression_ratio()
        );
    }

    #[test]
    fn test_kv_cache_basic() {
        let num_heads = 2;
        let head_dim = 4;
        let threshold = 8;
        let chunk_size = 4;

        let mut cache = HuffmanKVCache::new(num_heads, head_dim, threshold, chunk_size);

        // Append 4 tokens
        let k = vec![1.0f32; num_heads * 4 * head_dim];
        let v = vec![2.0f32; num_heads * 4 * head_dim];
        cache.append(&k, &v, 4);
        assert_eq!(cache.len(), 4);
        assert_eq!(cache.compressed_len, 0);

        // Append 4 more → triggers compression of first 4
        let k2 = vec![3.0f32; num_heads * 4 * head_dim];
        let v2 = vec![4.0f32; num_heads * 4 * head_dim];
        cache.append(&k2, &v2, 4);
        assert_eq!(cache.len(), 8);
        assert_eq!(cache.compressed_len, 4);

        // Retrieve and verify
        let (k_full, v_full) = cache.get_kv();
        assert_eq!(k_full.len(), num_heads * 8 * head_dim);
        assert_eq!(v_full.len(), num_heads * 8 * head_dim);
    }

    #[test]
    fn test_kv_cache_compress_decompress_roundtrip() {
        let num_heads = 2;
        let head_dim = 8;
        let num_tokens = 16;

        // Create f32 data
        let data: Vec<f32> = (0..num_heads * num_tokens * head_dim)
            .map(|i| ((i * 13 + 7) % 100) as f32 / 50.0 - 1.0)
            .collect();

        let chunk = HuffmanKVCache::compress_chunk(&data, num_tokens, num_heads, head_dim);
        let recovered = HuffmanKVCache::decompress_chunk(&chunk);

        // Check quantization tolerance: per-head, max error <= absmax/127
        for h in 0..num_heads {
            let base = h * num_tokens * head_dim;
            let head_data = &data[base..base + num_tokens * head_dim];
            let absmax = head_data.iter().fold(0.0f32, |acc, &v| acc.max(v.abs()));
            let scale = if absmax == 0.0 { 1.0 } else { absmax / 127.0 };

            for i in 0..num_tokens * head_dim {
                let err = (data[base + i] - recovered[base + i]).abs();
                assert!(
                    err <= scale + 1e-5,
                    "head {h} idx {i}: err {err} > scale {scale}"
                );
            }
        }
    }
}
