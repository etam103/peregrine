//! Wall-clock benchmark for Peregrine operations.
//!
//! Standalone binary (harness = false) that outputs JSON to
//! target/bench_compare/peregrine.json with timing stats for each operation.
//!
//! Run: cargo bench --bench wallclock
//! GPU: cargo bench --bench wallclock --features metal

use peregrine::nn;
use peregrine::optim::Adam;
use peregrine::tensor::Tensor;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WARMUP: usize = 5;
const ITERS_FAST: usize = 50;
const ITERS_SLOW: usize = 20; // matmul 512+, training step
const ITERS_VSLOW: usize = 10; // matmul 2048, 10M elements, large training

// ---------------------------------------------------------------------------
// Timing helpers
// ---------------------------------------------------------------------------

fn bench<F: FnMut()>(mut f: F, iters: usize) -> Vec<f64> {
    // warmup
    for _ in 0..WARMUP {
        f();
    }
    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        f();
        let elapsed = t0.elapsed();
        times.push(elapsed.as_nanos() as f64 / 1_000.0); // ns -> us
    }
    times
}

fn median(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 0 {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    } else {
        v[n / 2]
    }
}

fn std_dev(v: &[f64]) -> f64 {
    let n = v.len() as f64;
    let mean = v.iter().sum::<f64>() / n;
    let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt()
}

fn stats(mut times: Vec<f64>) -> serde_json::Value {
    let med = median(&mut times);
    let sd = if times.len() > 1 { std_dev(&times) } else { 0.0 };
    let mn = times.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = times.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    serde_json::json!({
        "median_us": (med * 100.0).round() / 100.0,
        "std_us": (sd * 100.0).round() / 100.0,
        "min_us": (mn * 100.0).round() / 100.0,
        "max_us": (mx * 100.0).round() / 100.0,
        "iters": times.len(),
    })
}

fn make_result(label: &str, times: Vec<f64>) -> serde_json::Value {
    let mut s = stats(times);
    s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!(label));
    s
}

fn size_label(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else {
        format!("{}k", n / 1_000)
    }
}

fn iters_for_elwise(n: usize) -> usize {
    if n >= 10_000_000 { ITERS_VSLOW }
    else if n >= 5_000_000 { ITERS_SLOW }
    else { ITERS_FAST }
}

// ---------------------------------------------------------------------------
// CPU Benchmarks
// ---------------------------------------------------------------------------

fn bench_matmul() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for size in [128, 256, 512, 1024, 2048] {
        let iters = match size {
            2048 => ITERS_VSLOW,
            512 | 1024 => ITERS_SLOW,
            _ => ITERS_FAST,
        };
        let a = Tensor::randn(&[size, size], false);
        let w = Tensor::randn(&[size, size], false);
        let times = bench(|| { black_box(a.matmul(&w)); }, iters);
        results.push(make_result(&format!("matmul_{size}x{size}"), times));
    }
    results
}

fn bench_add() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000, 1_000_000, 5_000_000, 10_000_000] {
        let a = Tensor::randn(&[n], false);
        let b = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.add(&b)); }, iters_for_elwise(n));
        results.push(make_result(&format!("add_{}", size_label(n)), times));
    }
    results
}

fn bench_mul() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000, 1_000_000, 5_000_000, 10_000_000] {
        let a = Tensor::randn(&[n], false);
        let b = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.mul(&b)); }, iters_for_elwise(n));
        results.push(make_result(&format!("mul_{}", size_label(n)), times));
    }
    results
}

fn bench_exp() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000, 1_000_000, 5_000_000, 10_000_000] {
        let a = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.exp()); }, iters_for_elwise(n));
        results.push(make_result(&format!("exp_{}", size_label(n)), times));
    }
    results
}

fn bench_relu() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 1_000_000] {
        let a = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.relu()); }, ITERS_FAST);
        results.push(make_result(&format!("relu_{}", size_label(n)), times));
    }
    results
}

fn bench_softmax() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for seq in [128usize, 512] {
        let x = Tensor::randn(&[8, seq], false);
        let times = bench(|| { black_box(x.softmax(-1)); }, ITERS_FAST);
        results.push(make_result(&format!("softmax_8x{seq}"), times));
    }
    results
}

fn bench_mlp_forward() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Small: batch=64, 784->128->64->10
    {
        let w1 = Tensor::randn(&[784, 128], false);
        let b1 = Tensor::randn(&[1, 128], false);
        let w2 = Tensor::randn(&[128, 64], false);
        let b2 = Tensor::randn(&[1, 64], false);
        let w3 = Tensor::randn(&[64, 10], false);
        let b3 = Tensor::randn(&[1, 10], false);
        let x = Tensor::randn(&[64, 784], false);

        let times = bench(|| {
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let out = h2.matmul(&w3).add_bias(&b3);
            black_box(out);
        }, ITERS_FAST);
        results.push(make_result("mlp_fwd_64x784", times));
    }

    // Large: batch=256, 784->512->256->10
    {
        let w1 = Tensor::randn(&[784, 512], false);
        let b1 = Tensor::randn(&[1, 512], false);
        let w2 = Tensor::randn(&[512, 256], false);
        let b2 = Tensor::randn(&[1, 256], false);
        let w3 = Tensor::randn(&[256, 10], false);
        let b3 = Tensor::randn(&[1, 10], false);
        let x = Tensor::randn(&[256, 784], false);

        let times = bench(|| {
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let out = h2.matmul(&w3).add_bias(&b3);
            black_box(out);
        }, ITERS_SLOW);
        results.push(make_result("mlp_fwd_256x784_wide", times));
    }

    results
}

fn bench_training_step() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Small: batch=64, 784->128->64->10
    {
        let w1 = Tensor::randn(&[784, 128], true);
        let b1 = Tensor::zeros(&[1, 128], true);
        let w2 = Tensor::randn(&[128, 64], true);
        let b2 = Tensor::zeros(&[1, 64], true);
        let w3 = Tensor::randn(&[64, 10], true);
        let b3 = Tensor::zeros(&[1, 10], true);
        let batch = 64usize;
        let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        let times = bench(|| {
            let x = Tensor::randn(&[batch, 784], false);
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss);
        }, ITERS_SLOW);
        results.push(make_result("train_step_64", times));
    }

    // Large: batch=256, 784->256->128->10
    {
        let w1 = Tensor::randn(&[784, 256], true);
        let b1 = Tensor::zeros(&[1, 256], true);
        let w2 = Tensor::randn(&[256, 128], true);
        let b2 = Tensor::zeros(&[1, 128], true);
        let w3 = Tensor::randn(&[128, 10], true);
        let b3 = Tensor::zeros(&[1, 10], true);
        let batch = 256usize;
        let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        let times = bench(|| {
            let x = Tensor::randn(&[batch, 784], false);
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss);
        }, ITERS_VSLOW);
        results.push(make_result("train_step_256_wide", times));
    }

    results
}

// ---------------------------------------------------------------------------
// GPU Benchmarks (Metal)
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn gpu_tensor(shape: &[usize], requires_grad: bool) -> Tensor {
    let t = Tensor::randn(shape, requires_grad);
    t.to_gpu();
    t
}

#[cfg(feature = "metal")]
fn gpu_zeros(shape: &[usize], requires_grad: bool) -> Tensor {
    let t = Tensor::zeros(shape, requires_grad);
    t.to_gpu();
    t
}

#[cfg(feature = "metal")]
fn bench_gpu_matmul() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for size in [128, 256, 512, 1024, 2048] {
        let iters = match size {
            2048 => ITERS_VSLOW,
            512 | 1024 => ITERS_SLOW,
            _ => ITERS_FAST,
        };
        let a = gpu_tensor(&[size, size], false);
        let w = gpu_tensor(&[size, size], false);
        let times = bench(|| { black_box(a.matmul(&w)); }, iters);
        results.push(make_result(&format!("gpu_matmul_{size}x{size}"), times));
    }
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_add() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000, 1_000_000, 5_000_000, 10_000_000] {
        let a = gpu_tensor(&[n], false);
        let b = gpu_tensor(&[n], false);
        let times = bench(|| { black_box(a.add(&b)); }, iters_for_elwise(n));
        results.push(make_result(&format!("gpu_add_{}", size_label(n)), times));
    }
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_mul() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000, 1_000_000, 5_000_000, 10_000_000] {
        let a = gpu_tensor(&[n], false);
        let b = gpu_tensor(&[n], false);
        let times = bench(|| { black_box(a.mul(&b)); }, iters_for_elwise(n));
        results.push(make_result(&format!("gpu_mul_{}", size_label(n)), times));
    }
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_exp() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000, 1_000_000, 5_000_000, 10_000_000] {
        let a = gpu_tensor(&[n], false);
        let times = bench(|| { black_box(a.exp()); }, iters_for_elwise(n));
        results.push(make_result(&format!("gpu_exp_{}", size_label(n)), times));
    }
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_relu() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 1_000_000] {
        let a = gpu_tensor(&[n], false);
        let times = bench(|| { black_box(a.relu()); }, ITERS_FAST);
        results.push(make_result(&format!("gpu_relu_{}", size_label(n)), times));
    }
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_softmax() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for seq in [128usize, 512] {
        let x = gpu_tensor(&[8, seq], false);
        let times = bench(|| { black_box(x.softmax(-1)); }, ITERS_FAST);
        results.push(make_result(&format!("gpu_softmax_8x{seq}"), times));
    }
    results
}

#[cfg(feature = "metal")]
fn bench_gpu_mlp_forward() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Small: batch=64, 784->128->64->10
    {
        let w1 = gpu_tensor(&[784, 128], false);
        let b1 = gpu_tensor(&[1, 128], false);
        let w2 = gpu_tensor(&[128, 64], false);
        let b2 = gpu_tensor(&[1, 64], false);
        let w3 = gpu_tensor(&[64, 10], false);
        let b3 = gpu_tensor(&[1, 10], false);
        let x = gpu_tensor(&[64, 784], false);

        let times = bench(|| {
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let out = h2.matmul(&w3).add_bias(&b3);
            black_box(out);
        }, ITERS_FAST);
        results.push(make_result("gpu_mlp_fwd_64x784", times));
    }

    // Large: batch=256, 784->512->256->10
    {
        let w1 = gpu_tensor(&[784, 512], false);
        let b1 = gpu_tensor(&[1, 512], false);
        let w2 = gpu_tensor(&[512, 256], false);
        let b2 = gpu_tensor(&[1, 256], false);
        let w3 = gpu_tensor(&[256, 10], false);
        let b3 = gpu_tensor(&[1, 10], false);
        let x = gpu_tensor(&[256, 784], false);

        let times = bench(|| {
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let out = h2.matmul(&w3).add_bias(&b3);
            black_box(out);
        }, ITERS_SLOW);
        results.push(make_result("gpu_mlp_fwd_256x784_wide", times));
    }

    results
}

#[cfg(feature = "metal")]
fn bench_gpu_training_step() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Small: batch=64, 784->128->64->10
    {
        let w1 = gpu_tensor(&[784, 128], true);
        let b1 = gpu_zeros(&[1, 128], true);
        let w2 = gpu_tensor(&[128, 64], true);
        let b2 = gpu_zeros(&[1, 64], true);
        let w3 = gpu_tensor(&[64, 10], true);
        let b3 = gpu_zeros(&[1, 10], true);
        let batch = 64usize;
        let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        let times = bench(|| {
            let x = gpu_tensor(&[batch, 784], false);
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss);
        }, ITERS_SLOW);
        results.push(make_result("gpu_train_step_64", times));
    }

    // Large: batch=256, 784->256->128->10
    {
        let w1 = gpu_tensor(&[784, 256], true);
        let b1 = gpu_zeros(&[1, 256], true);
        let w2 = gpu_tensor(&[256, 128], true);
        let b2 = gpu_zeros(&[1, 128], true);
        let w3 = gpu_tensor(&[128, 10], true);
        let b3 = gpu_zeros(&[1, 10], true);
        let batch = 256usize;
        let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        let times = bench(|| {
            let x = gpu_tensor(&[batch, 784], false);
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss);
        }, ITERS_VSLOW);
        results.push(make_result("gpu_train_step_256_wide", times));
    }

    results
}

#[cfg(feature = "metal")]
fn bench_gpu_training_step_fused() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Fused: batch=64, 784->128->64->10 (matmul_bias_relu)
    {
        let w1 = gpu_tensor(&[784, 128], true);
        let b1 = gpu_zeros(&[1, 128], true);
        let w2 = gpu_tensor(&[128, 64], true);
        let b2 = gpu_zeros(&[1, 64], true);
        let w3 = gpu_tensor(&[64, 10], true);
        let b3 = gpu_zeros(&[1, 10], true);
        let batch = 64usize;
        let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        let times = bench(|| {
            let x = gpu_tensor(&[batch, 784], false);
            let h1 = x.matmul_bias_relu(&w1, &b1);
            let h2 = h1.matmul_bias_relu(&w2, &b2);
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss);
        }, ITERS_SLOW);
        results.push(make_result("gpu_train_fused_64", times));
    }

    // Fused: batch=256, 784->256->128->10 (matmul_bias_relu)
    {
        let w1 = gpu_tensor(&[784, 256], true);
        let b1 = gpu_zeros(&[1, 256], true);
        let w2 = gpu_tensor(&[256, 128], true);
        let b2 = gpu_zeros(&[1, 128], true);
        let w3 = gpu_tensor(&[128, 10], true);
        let b3 = gpu_zeros(&[1, 10], true);
        let batch = 256usize;
        let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
        let mut opt = Adam::new(
            vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
            1e-3,
        );
        let times = bench(|| {
            let x = gpu_tensor(&[batch, 784], false);
            let h1 = x.matmul_bias_relu(&w1, &b1);
            let h2 = h1.matmul_bias_relu(&w2, &b2);
            let logits = h2.matmul(&w3).add_bias(&b3);
            let loss = nn::cross_entropy_loss(&logits, &targets);
            loss.backward();
            opt.step();
            opt.zero_grad();
            black_box(loss);
        }, ITERS_VSLOW);
        results.push(make_result("gpu_train_fused_256_wide", times));
    }

    results
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let out_dir = std::path::Path::new("target/bench_compare");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    // CPU benchmarks (always run)
    let cpu_benchmarks: Vec<(&str, fn() -> Vec<serde_json::Value>)> = vec![
        ("bench_matmul", bench_matmul),
        ("bench_add", bench_add),
        ("bench_mul", bench_mul),
        ("bench_exp", bench_exp),
        ("bench_relu", bench_relu),
        ("bench_softmax", bench_softmax),
        ("bench_mlp_forward", bench_mlp_forward),
        ("bench_training_step", bench_training_step),
    ];

    let mut all_results = Vec::new();
    for (name, f) in cpu_benchmarks {
        eprintln!("  CPU: {} ...", name);
        all_results.extend(f());
    }

    // GPU benchmarks (metal feature only)
    #[cfg(feature = "metal")]
    {
        peregrine::metal::init_gpu().expect("Failed to initialize Metal GPU");
        eprintln!("  Metal GPU initialized");

        let gpu_benchmarks: Vec<(&str, fn() -> Vec<serde_json::Value>)> = vec![
            ("bench_gpu_matmul", bench_gpu_matmul),
            ("bench_gpu_add", bench_gpu_add),
            ("bench_gpu_mul", bench_gpu_mul),
            ("bench_gpu_exp", bench_gpu_exp),
            ("bench_gpu_relu", bench_gpu_relu),
            ("bench_gpu_softmax", bench_gpu_softmax),
            ("bench_gpu_mlp_forward", bench_gpu_mlp_forward),
            ("bench_gpu_training_step", bench_gpu_training_step),
            ("bench_gpu_training_fused", bench_gpu_training_step_fused),
        ];

        for (name, f) in gpu_benchmarks {
            eprintln!("  GPU: {} ...", name);
            all_results.extend(f());
        }

        // Print CPU vs GPU comparison table
        eprintln!();
        eprintln!("  CPU vs GPU Comparison (median, microseconds):");
        eprintln!("  {:<30} {:>10} {:>10} {:>10}", "Operation", "CPU (us)", "GPU (us)", "Speedup");
        eprintln!("  {}", "-".repeat(63));

        for result in &all_results {
            let op = result["op"].as_str().unwrap();
            if !op.starts_with("gpu_") { continue; }
            let cpu_op = &op[4..];
            if let Some(cpu_result) = all_results.iter().find(|r| r["op"].as_str().unwrap() == cpu_op) {
                let cpu_med = cpu_result["median_us"].as_f64().unwrap();
                let gpu_med = result["median_us"].as_f64().unwrap();
                let speedup = cpu_med / gpu_med;
                let winner = if speedup > 1.0 { "GPU wins" } else { "CPU wins" };
                eprintln!("  {:<30} {:>10.1} {:>10.1} {:>8.1}x  {}", cpu_op, cpu_med, gpu_med, speedup, winner);
            }
        }
        // Print fused vs unfused comparison
        eprintln!("  Fused vs Unfused GPU Training (median, microseconds):");
        eprintln!("  {:<30} {:>10} {:>10} {:>10}", "Benchmark", "Unfused", "Fused", "Speedup");
        eprintln!("  {}", "-".repeat(63));
        for (unfused, fused) in [
            ("gpu_train_step_64", "gpu_train_fused_64"),
            ("gpu_train_step_256_wide", "gpu_train_fused_256_wide"),
        ] {
            if let (Some(u), Some(f)) = (
                all_results.iter().find(|r| r["op"].as_str().unwrap() == unfused),
                all_results.iter().find(|r| r["op"].as_str().unwrap() == fused),
            ) {
                let u_med = u["median_us"].as_f64().unwrap();
                let f_med = f["median_us"].as_f64().unwrap();
                let speedup = u_med / f_med;
                eprintln!("  {:<30} {:>10.1} {:>10.1} {:>8.1}x", unfused, u_med, f_med, speedup);
            }
        }
        eprintln!();
    }

    let output = serde_json::json!({
        "framework": "peregrine",
        "results": all_results,
    });

    let out_path = out_dir.join("peregrine.json");
    std::fs::write(&out_path, serde_json::to_string_pretty(&output).unwrap())
        .expect("write JSON");
    eprintln!("  Saved {}", out_path.display());
}
