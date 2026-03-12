//! Benchmark: fused vs unfused GPU kernel dispatches.
//!
//! Run with: cargo run --release --features metal --example pipeline_bench

use peregrine::metal::{self, GpuContext};
use peregrine::tensor::Tensor;
use std::time::Instant;

fn random_data(n: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut state: u32 = seed;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        data.push((state as f32 / u32::MAX as f32) * 2.0 - 1.0);
    }
    data
}

fn bench_fused_vs_unfused_mlp(m: usize, k: usize, n: usize, iters: usize) {
    println!("\n=== MLP forward: [{m}x{k}] @ [{k}x{n}] + bias + gelu + [{m}x{n}] @ [{n}x{k}] + bias ===");

    let a_data = random_data(m * k, 42);
    let w1_data = random_data(k * n, 123);
    let b1_data = random_data(n, 456);
    let w2_data = random_data(n * k, 789);
    let b2_data = random_data(k, 1011);

    // Warm up GPU
    let _gpu = GpuContext::new().unwrap();
    metal::init_gpu();

    // -- Unfused path --
    let a = Tensor::new(a_data.clone(), vec![m, k], false);
    let w1 = Tensor::new(w1_data.clone(), vec![k, n], false);
    let b1 = Tensor::new(b1_data.clone(), vec![1, n], false);
    let w2 = Tensor::new(w2_data.clone(), vec![n, k], false);
    let b2 = Tensor::new(b2_data.clone(), vec![1, k], false);
    a.to_gpu(); w1.to_gpu(); b1.to_gpu(); w2.to_gpu(); b2.to_gpu();

    // Warmup
    for _ in 0..3 {
        let h = a.matmul(&w1).add_bias(&b1).gelu();
        let _out = h.matmul(&w2).add_bias(&b2);
        metal::gpu_sync();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let h = a.matmul(&w1).add_bias(&b1).gelu();
        let _out = h.matmul(&w2).add_bias(&b2);
        metal::gpu_sync();
    }
    let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

    // -- Fused path --
    // Warmup
    for _ in 0..3 {
        let h = a.matmul_bias_gelu(&w1, &b1);
        let _out = h.matmul(&w2).add_bias(&b2);
        metal::gpu_sync();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let h = a.matmul_bias_gelu(&w1, &b1);
        let _out = h.matmul(&w2).add_bias(&b2);
        metal::gpu_sync();
    }
    let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

    let speedup = unfused_us / fused_us;
    println!("  Unfused: {unfused_us:.0} us/iter");
    println!("  Fused:   {fused_us:.0} us/iter");
    println!("  Speedup: {speedup:.2}x");
}

fn bench_double_buffered_matmul(n: usize, iters: usize) {
    println!("\n=== Matmul [{n}x{n}] @ [{n}x{n}]: single-buffered vs double-buffered ===");

    let gpu = GpuContext::new().unwrap();
    let a_data = random_data(n * n, 42);
    let b_data = random_data(n * n, 123);
    let a_buf = gpu.upload(&a_data);
    let b_buf = gpu.upload(&b_data);
    let out_buf: peregrine::metal::GpuBuffer<f32> = gpu.alloc(n * n);

    // Warmup
    for _ in 0..3 {
        gpu.dispatch_matmul_simd(&a_buf, &b_buf, &out_buf, None,
            n as u32, n as u32, n as u32, false, false, false, false);
        gpu.sync();
    }

    let start = Instant::now();
    for _ in 0..iters {
        gpu.dispatch_matmul_simd(&a_buf, &b_buf, &out_buf, None,
            n as u32, n as u32, n as u32, false, false, false, false);
        gpu.sync();
    }
    let single_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Warmup
    for _ in 0..3 {
        gpu.dispatch_matmul_simd_db(&a_buf, &b_buf, &out_buf, None,
            n as u32, n as u32, n as u32, false, false, false, false);
        gpu.sync();
    }

    let start = Instant::now();
    for _ in 0..iters {
        gpu.dispatch_matmul_simd_db(&a_buf, &b_buf, &out_buf, None,
            n as u32, n as u32, n as u32, false, false, false, false);
        gpu.sync();
    }
    let double_us = start.elapsed().as_micros() as f64 / iters as f64;

    let speedup = single_us / double_us;
    println!("  Single-buffered: {single_us:.0} us/iter");
    println!("  Double-buffered: {double_us:.0} us/iter");
    println!("  Speedup: {speedup:.2}x");
}

fn bench_add_layernorm(batch: usize, dim: usize, iters: usize) {
    println!("\n=== Add+LayerNorm [{batch}x{dim}]: unfused vs fused ===");

    metal::init_gpu();
    let x_data = random_data(batch * dim, 42);
    let r_data = random_data(batch * dim, 123);
    let g_data: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 / dim as f32)).collect();
    let b_data = random_data(dim, 456);

    let x = Tensor::new(x_data, vec![batch, dim], false);
    let r = Tensor::new(r_data, vec![batch, dim], false);
    let g = Tensor::new(g_data, vec![dim], false);
    let b = Tensor::new(b_data, vec![dim], false);
    x.to_gpu(); r.to_gpu(); g.to_gpu(); b.to_gpu();

    // Warmup unfused
    for _ in 0..3 {
        let _out = x.add(&r).layer_norm(&g, &b, dim);
        metal::gpu_sync();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _out = x.add(&r).layer_norm(&g, &b, dim);
        metal::gpu_sync();
    }
    let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Warmup fused
    for _ in 0..3 {
        let _out = x.add_layer_norm(&r, &g, &b, dim);
        metal::gpu_sync();
    }

    let start = Instant::now();
    for _ in 0..iters {
        let _out = x.add_layer_norm(&r, &g, &b, dim);
        metal::gpu_sync();
    }
    let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

    let speedup = unfused_us / fused_us;
    println!("  Unfused: {unfused_us:.0} us/iter");
    println!("  Fused:   {fused_us:.0} us/iter");
    println!("  Speedup: {speedup:.2}x");
}

fn main() {
    println!("Pipeline Benchmark: Fused vs Unfused GPU Kernels");
    println!("================================================");

    let iters = 50;

    // MLP forward (typical transformer FFN sizes)
    bench_fused_vs_unfused_mlp(196, 768, 3072, iters);   // ViT-Base FFN
    bench_fused_vs_unfused_mlp(196, 1024, 4096, iters);  // ViT-Large FFN
    bench_fused_vs_unfused_mlp(1024, 768, 3072, iters);  // Large batch FFN

    // Double-buffered matmul
    bench_double_buffered_matmul(1024, iters);
    bench_double_buffered_matmul(2048, iters);
    bench_double_buffered_matmul(4096, iters);

    // Add + LayerNorm
    bench_add_layernorm(196, 768, iters);
    bench_add_layernorm(196, 1024, iters);
    bench_add_layernorm(1024, 768, iters);
}
