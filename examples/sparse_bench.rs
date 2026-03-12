//! Benchmark: dense vs 2:4 sparse vs int8 quantized matmul.
//! Run with: cargo run --example sparse_bench --release --features metal

use std::time::Instant;

use peregrine::sparse::{prune_to_24, matmul_sparse_24};
use peregrine::quant::{quantize_weights, matmul_quantized};

fn random_data(n: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut state = seed;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data.push((state as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    data
}

fn dense_matmul(a: &[f32], w: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
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
    out
}

fn bench<F: FnMut() -> Vec<f32>>(name: &str, m: usize, n: usize, k: usize, iters: usize, mut f: F) {
    // Warmup
    let _ = f();

    let start = Instant::now();
    for _ in 0..iters {
        let _ = f();
    }
    let elapsed = start.elapsed().as_secs_f64() / iters as f64;

    let flops = 2.0 * m as f64 * n as f64 * k as f64;
    let gflops = flops / elapsed / 1e9;

    // Bandwidth: dense reads A[M,K] + W[K,N], writes C[M,N]
    let bytes_a = (m * k * 4) as f64;
    let bytes_c = (m * n * 4) as f64;

    let bw_label = match name {
        "dense" => {
            let total = bytes_a + (k * n * 4) as f64 + bytes_c;
            format!("{:.1} GB/s", total / elapsed / 1e9)
        }
        "sparse-24" => {
            // values: K/2*N*4 + indices: K/4*N*1
            let w_bytes = (k / 2 * n * 4 + k / 4 * n) as f64;
            let total = bytes_a + w_bytes + bytes_c;
            format!("{:.1} GB/s", total / elapsed / 1e9)
        }
        "int8" => {
            // W: K*N*1 + scales: N*4
            let w_bytes = (k * n + n * 4) as f64;
            let total = bytes_a + w_bytes + bytes_c;
            format!("{:.1} GB/s", total / elapsed / 1e9)
        }
        _ => String::new(),
    };

    println!(
        "  {:<12} {:>8.3} ms  {:>7.2} GFLOPS  {}",
        name, elapsed * 1000.0, gflops, bw_label
    );
}

fn main() {
    let sizes = [256, 512, 1024, 2048, 4096];

    println!("2:4 Structured Sparsity Benchmark");
    println!("==================================");
    println!("M=N=K, comparing dense / sparse-24 / int8 CPU matmul\n");

    for &size in &sizes {
        let m = size;
        let k = size;
        let n = size;
        let iters = if size <= 512 { 20 } else if size <= 1024 { 5 } else { 2 };

        let a = random_data(m * k, 42);
        let w = random_data(k * n, 123);

        let st = prune_to_24(&w, k, n);
        let qt = quantize_weights(&w, k, n);

        println!("[{size}x{size}x{size}] ({iters} iters)");

        bench("dense", m, n, k, iters, || dense_matmul(&a, &w, m, k, n));
        bench("sparse-24", m, n, k, iters, || matmul_sparse_24(&a, m, k, &st));
        bench("int8", m, n, k, iters, || matmul_quantized(&a, m, k, &qt));

        // GPU benchmarks
        #[cfg(feature = "metal")]
        {
            use peregrine::metal::GpuContext;
            use peregrine::sparse::matmul_sparse_24_gpu;
            use peregrine::quant::matmul_quantized_gpu;

            let gpu = GpuContext::new().unwrap();
            let mut st_gpu = prune_to_24(&w, k, n);
            st_gpu.to_gpu(&gpu);
            let mut qt_gpu = quantize_weights(&w, k, n);
            qt_gpu.to_gpu(&gpu);

            // Dense GPU
            {
                let a_buf = gpu.upload(&a);
                let w_buf = gpu.upload(&w);
                let out_buf: peregrine::metal::GpuBuffer<f32> = gpu.alloc(m * n);
                gpu.dispatch_matmul(&a_buf, &w_buf, &out_buf, None, m as u32, n as u32, k as u32, false, false, false);
                gpu.sync();

                let start = Instant::now();
                for _ in 0..iters {
                    let a_buf = gpu.upload(&a);
                    let w_buf = gpu.upload(&w);
                    let out_buf: peregrine::metal::GpuBuffer<f32> = gpu.alloc(m * n);
                    gpu.dispatch_matmul(&a_buf, &w_buf, &out_buf, None, m as u32, n as u32, k as u32, false, false, false);
                    gpu.sync();
                }
                let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                let gflops = 2.0 * m as f64 * n as f64 * k as f64 / elapsed / 1e9;
                println!("  {:<12} {:>8.3} ms  {:>7.2} GFLOPS  (Metal)", "dense-gpu", elapsed * 1000.0, gflops);
            }

            // Sparse-24 GPU
            {
                let _ = matmul_sparse_24_gpu(&gpu, &a, m, k, &st_gpu);
                gpu.sync();

                let start = Instant::now();
                for _ in 0..iters {
                    let _ = matmul_sparse_24_gpu(&gpu, &a, m, k, &st_gpu);
                    gpu.sync();
                }
                let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                let gflops = 2.0 * m as f64 * n as f64 * k as f64 / elapsed / 1e9;
                println!("  {:<12} {:>8.3} ms  {:>7.2} GFLOPS  (Metal)", "sp24-gpu", elapsed * 1000.0, gflops);
            }

            // Int8 GPU
            {
                let _ = matmul_quantized_gpu(&gpu, &a, m, k, &qt_gpu);
                gpu.sync();

                let start = Instant::now();
                for _ in 0..iters {
                    let _ = matmul_quantized_gpu(&gpu, &a, m, k, &qt_gpu);
                    gpu.sync();
                }
                let elapsed = start.elapsed().as_secs_f64() / iters as f64;
                let gflops = 2.0 * m as f64 * n as f64 * k as f64 / elapsed / 1e9;
                println!("  {:<12} {:>8.3} ms  {:>7.2} GFLOPS  (Metal)", "int8-gpu", elapsed * 1000.0, gflops);
            }
        }

        println!();
    }
}
