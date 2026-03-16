//! PER-39: Map effective systolic array dimensions of AMX and Metal GPU.
//!
//! Sweeps cblas_sgemm (which dispatches to AMX on Apple Silicon) and Metal GPU
//! matmul at various matrix dimensions. The "knee" in the GFLOPS curve reveals
//! the effective tile size of each compute unit.
//!
//! Run (CPU/AMX only):  cargo run --release --example systolic_bench
//! Run (CPU + GPU):     cargo run --release --example systolic_bench --features metal

use peregrine::tensor::Tensor;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WARMUP: usize = 3;
const MIN_ITERS: usize = 5;
const TARGET_SECONDS: f64 = 0.5; // run each size for at least this long

// Square dimensions to sweep
const SQUARE_DIMS: &[usize] = &[
    4, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768,
    1024, 1536, 2048, 3072, 4096,
];

// Non-square: M x K x N patterns to probe rectangular tile shapes
// Format: (M, K, N)
const RECT_DIMS: &[(usize, usize, usize)] = &[
    // Tall-skinny A (M >> K): tests if AMX handles tall tiles efficiently
    (1024, 32, 32),
    (1024, 64, 64),
    (1024, 128, 128),
    // Wide-skinny A (K >> M): column-heavy
    (32, 1024, 32),
    (64, 1024, 64),
    (128, 1024, 128),
    // Large M, small N (decode-like: batch matmul with small output dim)
    (2048, 128, 1),
    (2048, 128, 4),
    (2048, 128, 16),
    (2048, 128, 64),
    (2048, 128, 128),
    // Fixed MN, vary K (inner dimension — probes reduction efficiency)
    (256, 16, 256),
    (256, 32, 256),
    (256, 64, 256),
    (256, 128, 256),
    (256, 256, 256),
    (256, 512, 256),
    (256, 1024, 256),
];

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

fn bench_ns<F: FnMut()>(mut f: F) -> f64 {
    // Warmup
    for _ in 0..WARMUP {
        f();
    }

    // Auto-calibrate iterations: run for at least TARGET_SECONDS
    let mut total_ns: u128 = 0;
    let mut iters: usize = 0;
    let deadline = Instant::now() + std::time::Duration::from_secs_f64(TARGET_SECONDS);

    while iters < MIN_ITERS || Instant::now() < deadline {
        let t0 = Instant::now();
        f();
        total_ns += t0.elapsed().as_nanos();
        iters += 1;
    }

    total_ns as f64 / iters as f64
}

fn gflops(m: usize, n: usize, k: usize, time_ns: f64) -> f64 {
    let flops = 2.0 * m as f64 * n as f64 * k as f64; // 2*M*N*K for matmul
    flops / time_ns // ns -> GFLOPS (10^9 cancels)
}

fn bandwidth_gbps(m: usize, n: usize, k: usize, time_ns: f64) -> f64 {
    // Bytes transferred: read A (M*K*4) + read B (K*N*4) + write C (M*N*4)
    let bytes = ((m * k + k * n + m * n) * 4) as f64;
    bytes / time_ns // ns -> GB/s
}

// ---------------------------------------------------------------------------
// AMX benchmarks (via cblas_sgemm)
// ---------------------------------------------------------------------------

fn bench_amx_square() -> Vec<(usize, f64, f64, f64)> {
    eprintln!("\n=== AMX (cblas_sgemm) — Square Matrices ===");
    eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10}",
        "N", "time_us", "GFLOPS", "GB/s", "util%");
    eprintln!("{}", "-".repeat(52));

    let mut results = Vec::new();
    let mut peak_gflops: f64 = 0.0;

    for &n in SQUARE_DIMS {
        let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.002) % 1.0).collect();
        let mut c_data: Vec<f32> = vec![0.0; n * n];

        let time_ns = bench_ns(|| {
            #[cfg(target_os = "macos")]
            peregrine::tensor::sgemm(
                false, false, n, n, n,
                1.0, &a_data, n,
                &b_data, n,
                0.0, black_box(&mut c_data), n,
            );
            #[cfg(not(target_os = "macos"))]
            {
                // Fallback: naive matmul (won't measure AMX)
                for i in 0..n {
                    for j in 0..n {
                        let mut s = 0.0f32;
                        for p in 0..n { s += a_data[i * n + p] * b_data[p * n + j]; }
                        c_data[i * n + j] = s;
                    }
                }
                black_box(&c_data);
            }
        });

        let gf = gflops(n, n, n, time_ns);
        let bw = bandwidth_gbps(n, n, n, time_ns);
        peak_gflops = peak_gflops.max(gf);
        let util = gf / peak_gflops * 100.0;

        eprintln!("{:>6} {:>10.1} {:>10.1} {:>10.1} {:>9.1}%",
            n, time_ns / 1000.0, gf, bw, util);

        results.push((n, time_ns, gf, bw));
    }

    eprintln!("\nPeak AMX GFLOPS: {:.1}", peak_gflops);
    results
}

fn bench_amx_rect() -> Vec<(usize, usize, usize, f64, f64, f64)> {
    eprintln!("\n=== AMX (cblas_sgemm) — Rectangular Matrices ===");
    eprintln!("{:>6} {:>6} {:>6} {:>10} {:>10} {:>10}",
        "M", "K", "N", "time_us", "GFLOPS", "GB/s");
    eprintln!("{}", "-".repeat(56));

    let mut results = Vec::new();

    for &(m, k, n) in RECT_DIMS {
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.002) % 1.0).collect();
        let mut c_data: Vec<f32> = vec![0.0; m * n];

        let time_ns = bench_ns(|| {
            #[cfg(target_os = "macos")]
            peregrine::tensor::sgemm(
                false, false, m, n, k,
                1.0, &a_data, k,
                &b_data, n,
                0.0, black_box(&mut c_data), n,
            );
            #[cfg(not(target_os = "macos"))]
            {
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0f32;
                        for p in 0..k { s += a_data[i * k + p] * b_data[p * n + j]; }
                        c_data[i * n + j] = s;
                    }
                }
                black_box(&c_data);
            }
        });

        let gf = gflops(m, n, k, time_ns);
        let bw = bandwidth_gbps(m, n, k, time_ns);

        eprintln!("{:>6} {:>6} {:>6} {:>10.1} {:>10.1} {:>10.1}",
            m, k, n, time_ns / 1000.0, gf, bw);

        results.push((m, k, n, time_ns, gf, bw));
    }

    results
}

// ---------------------------------------------------------------------------
// Metal GPU benchmarks
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn bench_gpu_square() -> Vec<(usize, f64, f64, f64)> {
    use peregrine::metal::{init_gpu, gpu_sync};

    init_gpu().expect("Failed to initialize Metal GPU");
    eprintln!("\n=== Metal GPU (matmul_f32 kernel, TILE_SIZE=16) — Square Matrices ===");
    eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10}",
        "N", "time_us", "GFLOPS", "GB/s", "util%");
    eprintln!("{}", "-".repeat(52));

    let mut results = Vec::new();
    let mut peak_gflops: f64 = 0.0;

    for &n in SQUARE_DIMS {
        let a = Tensor::randn(&[n, n], false);
        let b = Tensor::randn(&[n, n], false);
        a.to_gpu();
        b.to_gpu();

        let time_ns = bench_ns(|| {
            let c = a.matmul(&b);
            gpu_sync();
            black_box(c);
        });

        let gf = gflops(n, n, n, time_ns);
        let bw = bandwidth_gbps(n, n, n, time_ns);
        peak_gflops = peak_gflops.max(gf);
        let util = gf / peak_gflops * 100.0;

        eprintln!("{:>6} {:>10.1} {:>10.1} {:>10.1} {:>9.1}%",
            n, time_ns / 1000.0, gf, bw, util);

        results.push((n, time_ns, gf, bw));
    }

    eprintln!("\nPeak Metal GPU GFLOPS: {:.1}", peak_gflops);
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_simd_square() -> Vec<(usize, f64, f64, f64)> {
    use peregrine::metal::{init_gpu, gpu_sync, with_gpu, GpuBuffer};

    init_gpu().unwrap();
    eprintln!("\n=== Metal GPU (simdgroup 8x8 matmul, 64x64 tiles) — Square Matrices ===");
    eprintln!("{:>6} {:>10} {:>10} {:>10} {:>10}",
        "N", "time_us", "GFLOPS", "GB/s", "util%");
    eprintln!("{}", "-".repeat(52));

    let mut results = Vec::new();
    let mut peak_gflops: f64 = 0.0;

    for &n in SQUARE_DIMS {
        let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.001) % 1.0).collect();
        let b_data: Vec<f32> = (0..n * n).map(|i| (i as f32 * 0.002) % 1.0).collect();

        let time_ns = with_gpu(|gpu| {
            let a_buf = GpuBuffer::from_slice(gpu.device(), &a_data);
            let b_buf = GpuBuffer::from_slice(gpu.device(), &b_data);
            let c_buf = GpuBuffer::<f32>::new(gpu.device(), n * n);

            bench_ns(|| {
                gpu.dispatch_matmul_simd(
                    &a_buf, &b_buf, &c_buf, None,
                    n as u32, n as u32, n as u32,
                    false, false, false,
                );
                gpu.sync();
                black_box(());
            })
        }).unwrap();

        let gf = gflops(n, n, n, time_ns);
        let bw = bandwidth_gbps(n, n, n, time_ns);
        peak_gflops = peak_gflops.max(gf);
        let util = gf / peak_gflops * 100.0;

        eprintln!("{:>6} {:>10.1} {:>10.1} {:>10.1} {:>9.1}%",
            n, time_ns / 1000.0, gf, bw, util);

        results.push((n, time_ns, gf, bw));
    }

    eprintln!("\nPeak Metal GPU (simdgroup) GFLOPS: {:.1}", peak_gflops);
    results
}

// ---------------------------------------------------------------------------
// Analysis: find the knee
// ---------------------------------------------------------------------------

fn find_knee(results: &[(usize, f64, f64, f64)]) -> usize {
    // The knee is where GFLOPS first reaches ~80% of peak.
    // This reveals the effective systolic array dimension.
    let peak = results.iter().map(|r| r.2).fold(0.0f64, f64::max);
    let threshold = peak * 0.80;

    for &(n, _, gf, _) in results {
        if gf >= threshold {
            return n;
        }
    }
    results.last().map(|r| r.0).unwrap_or(0)
}

fn analyze_bandwidth_transition(results: &[(usize, f64, f64, f64)]) {
    // At small dims, matmul is bandwidth-bound (low arithmetic intensity).
    // At large dims, it's compute-bound (high arithmetic intensity).
    // The crossover reveals when the systolic array becomes fully utilized.
    eprintln!("\n  Arithmetic Intensity Analysis:");
    eprintln!("  {:>6} {:>12} {:>12}", "N", "ops/byte", "regime");
    eprintln!("  {}", "-".repeat(34));

    for &(n, _, gf, bw) in results {
        // Arithmetic intensity = FLOPS / bytes_transferred
        let flops = 2.0 * (n as f64).powi(3);
        let bytes = ((3 * n * n) * 4) as f64; // A + B + C
        let intensity = flops / bytes;
        let regime = if intensity < 10.0 { "BW-bound" } else { "compute-bound" };
        eprintln!("  {:>6} {:>12.1} {:>12}", n, intensity, regime);
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    eprintln!("PER-39: Systolic Array Dimension Benchmark");
    eprintln!("==========================================");

    // --- AMX ---
    let amx_sq = bench_amx_square();
    let amx_knee = find_knee(&amx_sq);
    eprintln!("\n  >> AMX effective tile size (80% peak): {}x{}", amx_knee, amx_knee);
    analyze_bandwidth_transition(&amx_sq);

    let _amx_rect = bench_amx_rect();

    // --- Metal GPU ---
    #[cfg(feature = "metal")]
    {
        let gpu_sq = bench_gpu_square();
        let gpu_knee = find_knee(&gpu_sq);
        eprintln!("\n  >> Metal GPU (scalar) effective tile size (80% peak): {}x{}", gpu_knee, gpu_knee);

        // Simdgroup matmul benchmark
        let gpu_simd_sq = bench_gpu_simd_square();
        let gpu_simd_knee = find_knee(&gpu_simd_sq);
        eprintln!("\n  >> Metal GPU (simdgroup) effective tile size (80% peak): {}x{}", gpu_simd_knee, gpu_simd_knee);

        // Three-way comparison
        eprintln!("\n=== AMX vs Metal GPU (scalar 16x16) vs Metal GPU (simdgroup 8x8) ===");
        eprintln!("{:>6} {:>12} {:>12} {:>12} {:>10}",
            "N", "AMX GFLOPS", "GPU-scalar", "GPU-simd", "winner");
        eprintln!("{}", "-".repeat(60));

        for i in 0..amx_sq.len().min(gpu_sq.len()).min(gpu_simd_sq.len()) {
            let amx_gf = amx_sq[i].2;
            let gpu_gf = gpu_sq[i].2;
            let simd_gf = gpu_simd_sq[i].2;
            let best = amx_gf.max(gpu_gf).max(simd_gf);
            let winner = if best == amx_gf { "AMX" }
                         else if best == simd_gf { "GPU-simd" }
                         else { "GPU-scalar" };
            eprintln!("{:>6} {:>12.1} {:>12.1} {:>12.1} {:>10}",
                amx_sq[i].0, amx_gf, gpu_gf, simd_gf, winner);
        }

        // Simdgroup speedup over scalar
        eprintln!("\n  Simdgroup speedup over scalar GPU kernel:");
        for i in 0..gpu_sq.len().min(gpu_simd_sq.len()) {
            let scalar_gf = gpu_sq[i].2;
            let simd_gf = gpu_simd_sq[i].2;
            if scalar_gf > 0.01 {
                eprintln!("  N={:>5}: {:.1}x", gpu_sq[i].0, simd_gf / scalar_gf);
            }
        }
    }

    #[cfg(not(feature = "metal"))]
    {
        eprintln!("\n(Skipping Metal GPU benchmarks — rebuild with --features metal)");
    }

    eprintln!("\nDone.");
}
