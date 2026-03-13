//! Huffman compression parity tests.
//! Run with: cargo test --test huffman_parity --features metal

use peregrine::huffman::{HuffmanTensor, HuffmanKVCache, matmul_huffman};
use peregrine::quant::{quantize_weights, matmul_quantized};

fn random_data(n: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        data.push(f);
    }
    data
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn huffman_vs_quantized_matmul_parity() {
    for &(m, k, n) in &[(8, 32, 16), (1, 64, 1), (32, 128, 64), (16, 256, 32)] {
        let a = random_data(m * k, 42);
        let w = random_data(k * n, 123);

        let qt = quantize_weights(&w, k, n);
        let ht = HuffmanTensor::from_quantized(&qt, 4);

        let quant_result = matmul_quantized(&a, m, k, &qt);
        let huff_result = matmul_huffman(&a, m, k, &ht);

        let err = max_abs_error(&quant_result, &huff_result);
        assert!(
            err < 1e-3,
            "Huffman vs Quantized parity failed for M={m} K={k} N={n}: max_err={err}"
        );
    }
}

#[cfg(feature = "metal")]
#[test]
fn gpu_huffman_vs_cpu_parity() {
    use peregrine::metal;
    metal::init_gpu().expect("Metal GPU init");

    let m = 32;
    let k = 128;
    let n = 64;

    let a = random_data(m * k, 42);
    let w = random_data(k * n, 123);

    let qt = quantize_weights(&w, k, n);
    let ht = HuffmanTensor::from_quantized(&qt, 4);

    // CPU result via Huffman
    let cpu_result = matmul_huffman(&a, m, k, &ht);

    // GPU result via Huffman decode + dequant matmul
    let gpu_result = metal::with_gpu(|gpu| {
        let out_buf = ht.matmul_huffman_gpu(gpu, &a, m);
        gpu.sync();
        out_buf.read()
    }).expect("GPU dispatch");

    // GPU uses f32 path (no activation quantization), allow larger tolerance
    let mut max_diff = 0.0f32;
    for i in 0..m * n {
        let diff = (cpu_result[i] - gpu_result[i]).abs();
        max_diff = max_diff.max(diff);
    }
    eprintln!("  GPU vs CPU Huffman max diff = {max_diff:.6}");

    // Both should be close to f32 reference
    let mut f32_result = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[mi * k + ki] * w[ki * n + ni];
            }
            f32_result[mi * n + ni] = sum;
        }
    }

    for i in 0..m * n {
        let diff = (gpu_result[i] - f32_result[i]).abs();
        let denom = f32_result[i].abs().max(1e-6);
        let rtol = diff / denom;
        assert!(
            rtol < 0.15 || diff < 0.05,
            "GPU Huffman i={i}: got {}, expected {}, rtol={rtol}, diff={diff}",
            gpu_result[i], f32_result[i]
        );
    }
}

#[test]
fn huffman_serialization_roundtrip() {
    use std::io::Cursor;

    let rows = 32;
    let cols = 64;
    let data = random_data(rows * cols, 77);

    let qt = quantize_weights(&data, rows, cols);
    let ht = HuffmanTensor::from_quantized(&qt, 4);

    // Write
    let mut buf = Vec::new();
    peregrine::serial::write_huffman_tensor(&mut buf, "test.huffman", &ht).unwrap();

    // Read back
    let mut cursor = Cursor::new(&buf);
    let (name, ht2) = peregrine::serial::read_huffman_tensor(&mut cursor).unwrap();

    assert_eq!(name, "test.huffman");
    assert_eq!(ht2.rows, rows);
    assert_eq!(ht2.cols, cols);
    assert_eq!(ht2.num_streams, 4);
    assert_eq!(ht2.scales, ht.scales);
    assert_eq!(ht2.stream_lengths, ht.stream_lengths);

    // Verify decode produces same data
    let qt_orig = ht.decode_to_quantized();
    let qt_loaded = ht2.decode_to_quantized();
    assert_eq!(qt_orig.data_i8, qt_loaded.data_i8);
}

#[test]
fn kv_cache_compress_decompress_roundtrip() {
    let num_heads = 4;
    let head_dim = 16;
    let num_tokens = 32;

    let data = random_data(num_heads * num_tokens * head_dim, 99);

    let chunk = HuffmanKVCache::compress_chunk(&data, num_tokens, num_heads, head_dim);
    let recovered = HuffmanKVCache::decompress_chunk(&chunk);

    // Check per-head quantization tolerance
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

#[test]
fn kv_cache_transparent_compression() {
    let num_heads = 2;
    let head_dim = 8;
    let threshold = 16;
    let chunk_size = 8;

    let mut cache = HuffmanKVCache::new(num_heads, head_dim, threshold, chunk_size);

    // Append tokens in batches, should trigger compression transparently
    for batch in 0..4 {
        let k = random_data(num_heads * 8 * head_dim, 100 + batch);
        let v = random_data(num_heads * 8 * head_dim, 200 + batch);
        cache.append(&k, &v, 8);
    }

    assert_eq!(cache.len(), 32);
    assert!(cache.compressed_len > 0, "expected some compressed tokens");

    let (k_full, v_full) = cache.get_kv();
    assert_eq!(k_full.len(), num_heads * 32 * head_dim);
    assert_eq!(v_full.len(), num_heads * 32 * head_dim);
}
