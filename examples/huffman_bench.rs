//! Benchmark: dense f32 vs int8 quantized vs Huffman-compressed matmul.
//!
//! Run with: cargo run --example huffman_bench --release [--features metal]

use peregrine::quant::{quantize_weights, matmul_quantized};
use peregrine::huffman::{HuffmanTensor, matmul_huffman};
use std::time::Instant;

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

fn bench_matmul_f32(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> (f64, Vec<f32>) {
    let start = Instant::now();
    let mut out = vec![0.0f32; m * n];
    for mi in 0..m {
        for ni in 0..n {
            let mut sum = 0.0f32;
            for ki in 0..k {
                sum += a[mi * k + ki] * w[ki * n + ni];
            }
            out[mi * n + ni] = sum;
        }
    }
    (start.elapsed().as_secs_f64(), out)
}

fn main() {
    let sizes: Vec<usize> = vec![256, 512, 1024, 2048, 4096];
    let m = 32; // batch size
    let num_streams = 4;

    println!("{:<8} {:>12} {:>12} {:>12} {:>10} {:>10}",
        "N=K", "f32 (ms)", "i8 (ms)", "huff (ms)", "ratio", "huff/i8");
    println!("{}", "-".repeat(72));

    for &n in &sizes {
        let k = n;
        let a = random_data(m * k, 42);
        let w = random_data(k * n, 123);

        // Dense f32
        let (t_f32, _) = bench_matmul_f32(&a, &w, m, k, n);

        // Int8 quantized
        let qt = quantize_weights(&w, k, n);
        let start = Instant::now();
        let _ = matmul_quantized(&a, m, k, &qt);
        let t_i8 = start.elapsed().as_secs_f64();

        // Huffman compressed
        let ht = HuffmanTensor::from_quantized(&qt, num_streams);
        let start = Instant::now();
        let _ = matmul_huffman(&a, m, k, &ht);
        let t_huff = start.elapsed().as_secs_f64();

        let ratio = ht.compression_ratio();
        let huff_over_i8 = t_huff / t_i8;

        // Sizes
        let dense_bytes = k * n * 4;
        let i8_bytes = k * n + n * 4;
        let huff_bytes = ht.compressed_size();

        println!(
            "{:<8} {:>10.3}ms {:>10.3}ms {:>10.3}ms {:>9.2}x {:>9.2}x",
            n,
            t_f32 * 1000.0,
            t_i8 * 1000.0,
            t_huff * 1000.0,
            ratio,
            huff_over_i8,
        );

        if n == *sizes.last().unwrap() {
            println!("\nSize comparison for {}x{}:", k, n);
            println!("  Dense f32:  {} bytes ({:.1} MB)", dense_bytes, dense_bytes as f64 / 1048576.0);
            println!("  Int8:       {} bytes ({:.1} MB)", i8_bytes, i8_bytes as f64 / 1048576.0);
            println!("  Huffman:    {} bytes ({:.1} MB)", huff_bytes, huff_bytes as f64 / 1048576.0);
            println!("  Huffman compression ratio: {:.2}x vs i8", ratio);
        }
    }

    // GPU benchmark
    #[cfg(feature = "metal")]
    {
        println!("\n--- GPU benchmark ---");
        peregrine::metal::init_gpu().expect("Metal GPU init");

        println!("{:<8} {:>12} {:>12}",
            "N=K", "GPU i8 (ms)", "GPU huff (ms)");
        println!("{}", "-".repeat(36));

        for &n in &sizes {
            let k = n;
            let a = random_data(m * k, 42);
            let w = random_data(k * n, 123);

            let mut qt = quantize_weights(&w, k, n);
            let ht = HuffmanTensor::from_quantized(&qt, num_streams);

            peregrine::metal::with_gpu(|gpu| {
                // Warmup
                qt.to_gpu(gpu);
                let a_buf = gpu.upload(&a);
                let out_buf: peregrine::metal::GpuBuffer<f32> = gpu.alloc(m * n);
                gpu.dispatch_matmul_dequant_i8(
                    &a_buf,
                    qt.gpu_data_i8.as_ref().unwrap(),
                    qt.gpu_scales.as_ref().unwrap(),
                    &out_buf,
                    m as u32, n as u32, k as u32,
                );
                gpu.sync();

                // Timed i8
                let start = Instant::now();
                gpu.dispatch_matmul_dequant_i8(
                    &a_buf,
                    qt.gpu_data_i8.as_ref().unwrap(),
                    qt.gpu_scales.as_ref().unwrap(),
                    &out_buf,
                    m as u32, n as u32, k as u32,
                );
                gpu.sync();
                let t_i8_gpu = start.elapsed().as_secs_f64();

                // Timed huffman (decode on CPU + GPU matmul)
                let start = Instant::now();
                let _ = ht.matmul_huffman_gpu(gpu, &a, m);
                gpu.sync();
                let t_huff_gpu = start.elapsed().as_secs_f64();

                println!(
                    "{:<8} {:>10.3}ms {:>10.3}ms",
                    n,
                    t_i8_gpu * 1000.0,
                    t_huff_gpu * 1000.0,
                );
            }).expect("GPU");
        }
    }
}
