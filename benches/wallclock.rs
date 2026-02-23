//! Wall-clock benchmark for Peregrine operations.
//!
//! Standalone binary (harness = false) that outputs JSON to
//! target/bench_compare/peregrine.json with timing stats for each operation.
//!
//! Run: cargo bench --bench wallclock

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
const ITERS_SLOW: usize = 20; // matmul 512, training step

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

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

fn bench_matmul() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for size in [128, 256, 512] {
        let iters = if size == 512 { ITERS_SLOW } else { ITERS_FAST };
        let a = Tensor::randn(&[size, size], false);
        let w = Tensor::randn(&[size, size], false);
        let times = bench(|| { black_box(a.matmul(&w)); }, iters);
        let mut s = stats(times);
        s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!(format!("matmul_{size}x{size}")));
        results.push(s);
    }
    results
}

fn bench_add() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000] {
        let a = Tensor::randn(&[n], false);
        let b = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.add(&b)); }, ITERS_FAST);
        let label = format!("add_{}k", n / 1000);
        let mut s = stats(times);
        s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!(label));
        results.push(s);
    }
    results
}

fn bench_mul() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000] {
        let a = Tensor::randn(&[n], false);
        let b = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.mul(&b)); }, ITERS_FAST);
        let label = format!("mul_{}k", n / 1000);
        let mut s = stats(times);
        s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!(label));
        results.push(s);
    }
    results
}

fn bench_exp() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for n in [100_000usize, 500_000] {
        let a = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.exp()); }, ITERS_FAST);
        let label = format!("exp_{}k", n / 1000);
        let mut s = stats(times);
        s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!(label));
        results.push(s);
    }
    results
}

fn bench_relu() -> Vec<serde_json::Value> {
    let a = Tensor::randn(&[100_000], false);
    let times = bench(|| { black_box(a.relu()); }, ITERS_FAST);
    let mut s = stats(times);
    s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!("relu_100k"));
    vec![s]
}

fn bench_softmax() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    for seq in [128usize, 512] {
        let x = Tensor::randn(&[8, seq], false);
        let times = bench(|| { black_box(x.softmax(-1)); }, ITERS_FAST);
        let label = format!("softmax_8x{seq}");
        let mut s = stats(times);
        s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!(label));
        results.push(s);
    }
    results
}

fn bench_mlp_forward() -> Vec<serde_json::Value> {
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

    let mut s = stats(times);
    s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!("mlp_fwd_64x784"));
    vec![s]
}

fn bench_training_step() -> Vec<serde_json::Value> {
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

    let mut s = stats(times);
    s.as_object_mut().unwrap().insert("op".to_string(), serde_json::json!("train_step_64"));
    vec![s]
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let out_dir = std::path::Path::new("target/bench_compare");
    std::fs::create_dir_all(out_dir).expect("create output dir");

    let benchmarks: Vec<(&str, fn() -> Vec<serde_json::Value>)> = vec![
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
    for (name, f) in benchmarks {
        eprintln!("  Peregrine: {} ...", name);
        all_results.extend(f());
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
