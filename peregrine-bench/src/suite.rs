//! CPU benchmark suite for Peregrine.
//!
//! Every benchmark function returns `Vec<BenchResult>`.
//! Call `run_all_cpu(quick)` to run the full suite.

use peregrine::nn;
use peregrine::optim::{Adam, RmsProp, Lion, Adafactor};
use peregrine::tensor::Tensor;
use serde::Serialize;
use std::hint::black_box;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

#[derive(Serialize, Clone)]
pub struct BenchResult {
    pub op: String,
    pub category: String,
    pub median_us: f64,
    pub std_us: f64,
    pub min_us: f64,
    pub max_us: f64,
    pub iters: usize,
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const WARMUP: usize = 5;
const ITERS_FAST: usize = 50;
const ITERS_SLOW: usize = 20;  // matmul 512+, training step
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

fn make_result(label: &str, category: &str, times: Vec<f64>) -> BenchResult {
    let mut t = times;
    let med = median(&mut t);
    let sd = if t.len() > 1 { std_dev(&t) } else { 0.0 };
    let mn = t.iter().cloned().fold(f64::INFINITY, f64::min);
    let mx = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    BenchResult {
        op: label.to_string(),
        category: category.to_string(),
        median_us: (med * 100.0).round() / 100.0,
        std_us: (sd * 100.0).round() / 100.0,
        min_us: (mn * 100.0).round() / 100.0,
        max_us: (mx * 100.0).round() / 100.0,
        iters: t.len(),
    }
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

/// Apply quick-mode adjustments: halve iterations, filter out largest sizes.
fn quick_iters(iters: usize) -> usize {
    (iters + 1) / 2
}

// ---------------------------------------------------------------------------
// CPU Benchmarks
// ---------------------------------------------------------------------------

pub fn bench_matmul(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let sizes: Vec<usize> = if quick {
        vec![128, 256, 512, 1024]
    } else {
        vec![128, 256, 512, 1024, 2048]
    };
    for size in sizes {
        let iters = match size {
            2048 => ITERS_VSLOW,
            512 | 1024 => ITERS_SLOW,
            _ => ITERS_FAST,
        };
        let iters = if quick { quick_iters(iters) } else { iters };
        let a = Tensor::randn(&[size, size], false);
        let w = Tensor::randn(&[size, size], false);
        let times = bench(|| { black_box(a.matmul(&w)); }, iters);
        results.push(make_result(&format!("matmul_{size}x{size}"), "matmul", times));
    }
    results
}

pub fn bench_add(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let sizes: Vec<usize> = if quick {
        vec![100_000, 500_000, 1_000_000, 5_000_000]
    } else {
        vec![100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    };
    for n in sizes {
        let iters = if quick { quick_iters(iters_for_elwise(n)) } else { iters_for_elwise(n) };
        let a = Tensor::randn(&[n], false);
        let b = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.add(&b)); }, iters);
        results.push(make_result(&format!("add_{}", size_label(n)), "elementwise", times));
    }
    results
}

pub fn bench_mul(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let sizes: Vec<usize> = if quick {
        vec![100_000, 500_000, 1_000_000, 5_000_000]
    } else {
        vec![100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    };
    for n in sizes {
        let iters = if quick { quick_iters(iters_for_elwise(n)) } else { iters_for_elwise(n) };
        let a = Tensor::randn(&[n], false);
        let b = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.mul(&b)); }, iters);
        results.push(make_result(&format!("mul_{}", size_label(n)), "elementwise", times));
    }
    results
}

pub fn bench_exp(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let sizes: Vec<usize> = if quick {
        vec![100_000, 500_000, 1_000_000, 5_000_000]
    } else {
        vec![100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]
    };
    for n in sizes {
        let iters = if quick { quick_iters(iters_for_elwise(n)) } else { iters_for_elwise(n) };
        let a = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.exp()); }, iters);
        results.push(make_result(&format!("exp_{}", size_label(n)), "elementwise", times));
    }
    results
}

pub fn bench_relu(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let sizes: Vec<usize> = if quick {
        vec![100_000]
    } else {
        vec![100_000, 1_000_000]
    };
    for n in sizes {
        let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
        let a = Tensor::randn(&[n], false);
        let times = bench(|| { black_box(a.relu()); }, iters);
        results.push(make_result(&format!("relu_{}", size_label(n)), "activation", times));
    }
    results
}

pub fn bench_softmax(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    for seq in [128usize, 512] {
        let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
        let x = Tensor::randn(&[8, seq], false);
        let times = bench(|| { black_box(x.softmax(-1)); }, iters);
        results.push(make_result(&format!("softmax_8x{seq}"), "softmax", times));
    }
    results
}

pub fn bench_mlp_forward(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Small: batch=64, 784->128->64->10
    {
        let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
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
        }, iters);
        results.push(make_result("mlp_fwd_64x784", "mlp", times));
    }

    // Large: batch=256, 784->512->256->10
    if !quick {
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
        results.push(make_result("mlp_fwd_256x784_wide", "mlp", times));
    }

    results
}

pub fn bench_training_step(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();

    // Small: batch=64, 784->128->64->10
    {
        let iters = if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW };
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
        }, iters);
        results.push(make_result("train_step_64", "training", times));
    }

    // Large: batch=256, 784->256->128->10
    if !quick {
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
        results.push(make_result("train_step_256_wide", "training", times));
    }

    results
}

// ---------------------------------------------------------------------------
// Unary math ops
// ---------------------------------------------------------------------------

pub fn bench_unary_math(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let n = 100_000usize;
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
    let a = Tensor::randn(&[n], false);
    let a_pos = Tensor::new(
        (0..n).map(|i| (i as f32 + 1.0) * 0.00001).collect(),
        vec![n], false,
    );
    let a_unit = Tensor::new(
        (0..n).map(|i| ((i as f32) / n as f32) * 1.8 - 0.9).collect(),
        vec![n], false,
    );

    let ops: Vec<(&str, Box<dyn Fn()>)> = vec![
        ("reciprocal_100k", Box::new(|| { black_box(a_pos.reciprocal()); })),
        ("square_100k", Box::new(|| { black_box(a.square()); })),
        ("rsqrt_100k", Box::new(|| { black_box(a_pos.rsqrt()); })),
        ("floor_100k", Box::new(|| { black_box(a.floor()); })),
        ("ceil_100k", Box::new(|| { black_box(a.ceil()); })),
        ("round_100k", Box::new(|| { black_box(a.round()); })),
        ("sign_100k", Box::new(|| { black_box(a.sign()); })),
        ("expm1_100k", Box::new(|| { black_box(a.expm1()); })),
        ("log2_100k", Box::new(|| { black_box(a_pos.log2()); })),
        ("log10_100k", Box::new(|| { black_box(a_pos.log10()); })),
        ("log1p_100k", Box::new(|| { black_box(a_pos.log1p()); })),
        ("erf_100k", Box::new(|| { black_box(a.erf()); })),
        ("sinh_100k", Box::new(|| { black_box(a.sinh()); })),
        ("cosh_100k", Box::new(|| { black_box(a.cosh()); })),
        ("arcsin_100k", Box::new(|| { black_box(a_unit.arcsin()); })),
        ("arccos_100k", Box::new(|| { black_box(a_unit.arccos()); })),
        ("arctan_100k", Box::new(|| { black_box(a.arctan()); })),
        ("arcsinh_100k", Box::new(|| { black_box(a.arcsinh()); })),
    ];

    for (name, f) in &ops {
        let times = bench(|| f(), iters);
        results.push(make_result(name, "unary_math", times));
    }
    results
}

// ---------------------------------------------------------------------------
// Binary math ops
// ---------------------------------------------------------------------------

pub fn bench_binary_math(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let n = 100_000usize;
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
    let a = Tensor::randn(&[n], false);
    let b = Tensor::randn(&[n], false);
    let a_pos = Tensor::new(
        (0..n).map(|i| (i as f32 + 1.0) * 0.0001).collect(),
        vec![n], false,
    );
    let b_pos = Tensor::new(
        (0..n).map(|i| (i as f32 + 1.0) * 0.0002).collect(),
        vec![n], false,
    );

    let ops: Vec<(&str, Box<dyn Fn()>)> = vec![
        ("maximum_100k", Box::new(|| { black_box(a.maximum(&b)); })),
        ("minimum_100k", Box::new(|| { black_box(a.minimum(&b)); })),
        ("power_100k", Box::new(|| { black_box(a_pos.power(&b_pos)); })),
        ("arctan2_100k", Box::new(|| { black_box(a.arctan2(&b)); })),
        ("logaddexp_100k", Box::new(|| { black_box(a.logaddexp(&b)); })),
    ];

    for (name, f) in &ops {
        let times = bench(|| f(), iters);
        results.push(make_result(name, "binary_math", times));
    }

    // Clip and where
    let times = bench(|| { black_box(a.clip(-0.5, 0.5)); }, iters);
    results.push(make_result("clip_100k", "binary_math", times));

    let cond = Tensor::new(
        (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect(),
        vec![n], false,
    );
    let times = bench(|| { black_box(Tensor::where_cond(&cond, &a, &b)); }, iters);
    results.push(make_result("where_100k", "binary_math", times));

    // Comparison ops
    let times = bench(|| { black_box(a.greater(&b)); }, iters);
    results.push(make_result("greater_100k", "binary_math", times));

    let times = bench(|| { black_box(a.equal(&b)); }, iters);
    results.push(make_result("equal_100k", "binary_math", times));

    results
}

// ---------------------------------------------------------------------------
// Axis reductions
// ---------------------------------------------------------------------------

pub fn bench_axis_reductions(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };

    // 2D tensor: [256, 512] -- reduce along axis 1
    let x = Tensor::randn(&[256, 512], false);
    let x_pos = Tensor::new(
        (0..256 * 512).map(|i| (i as f32 + 1.0) * 0.0001).collect(),
        vec![256, 512], false,
    );

    let ops: Vec<(&str, Box<dyn Fn()>)> = vec![
        ("sum_axis_256x512", Box::new(|| { black_box(x.sum_axis(1, false)); })),
        ("mean_axis_256x512", Box::new(|| { black_box(x.mean_axis(1, false)); })),
        ("max_axis_256x512", Box::new(|| { black_box(x.max_axis(1, false)); })),
        ("min_axis_256x512", Box::new(|| { black_box(x.min_axis(1, false)); })),
        ("var_256x512", Box::new(|| { black_box(x.var(1, false, 0)); })),
        ("prod_axis_256x512", Box::new(|| { black_box(x_pos.prod_axis(1, false)); })),
        ("logsumexp_256x512", Box::new(|| { black_box(x.logsumexp(1, false)); })),
        ("cumsum_256x512", Box::new(|| { black_box(x.cumsum(1)); })),
        ("argmax_axis_256x512", Box::new(|| { black_box(x.argmax_axis(1)); })),
    ];

    for (name, f) in &ops {
        let times = bench(|| f(), iters);
        results.push(make_result(name, "reduction", times));
    }

    // Larger: [1024, 1024] for sort/topk
    if !quick {
        let iters_slow = ITERS_SLOW;
        let x_large = Tensor::randn(&[1024, 1024], false);
        let times = bench(|| { black_box(x_large.sum_axis(1, false)); }, iters_slow);
        results.push(make_result("sum_axis_1024x1024", "reduction", times));

        let times = bench(|| { black_box(x_large.var(1, false, 0)); }, iters_slow);
        results.push(make_result("var_1024x1024", "reduction", times));
    }

    results
}

// ---------------------------------------------------------------------------
// Shape/indexing ops
// ---------------------------------------------------------------------------

pub fn bench_shape_ops(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };

    let x = Tensor::randn(&[256, 256], false);
    let times = bench(|| { black_box(x.tril(0)); }, iters);
    results.push(make_result("tril_256x256", "shape", times));

    let times = bench(|| { black_box(x.triu(0)); }, iters);
    results.push(make_result("triu_256x256", "shape", times));

    let x_small = Tensor::randn(&[64, 128], false);
    let times = bench(|| { black_box(x_small.repeat(&[2, 3])); }, iters);
    results.push(make_result("repeat_64x128_2x3", "shape", times));

    let times = bench(|| { black_box(x_small.pad(&[(1, 1), (2, 2)], 0.0)); }, iters);
    results.push(make_result("pad_64x128", "shape", times));

    // Stack 8 tensors
    let tensors: Vec<Tensor> = (0..8).map(|_| Tensor::randn(&[64, 128], false)).collect();
    let times = bench(|| { black_box(Tensor::stack(&tensors, 0)); }, iters);
    results.push(make_result("stack_8x64x128", "shape", times));

    // Diagonal
    let x_sq = Tensor::randn(&[512, 512], false);
    let times = bench(|| { black_box(x_sq.diagonal(0)); }, iters);
    results.push(make_result("diagonal_512x512", "shape", times));

    results
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

pub fn bench_activations(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let n = 100_000usize;
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
    let a = Tensor::randn(&[n], false);

    let ops: Vec<(&str, Box<dyn Fn()>)> = vec![
        ("silu_100k", Box::new(|| { black_box(a.silu()); })),
        ("softplus_100k", Box::new(|| { black_box(a.softplus(1.0)); })),
        ("mish_100k", Box::new(|| { black_box(a.mish()); })),
        ("leaky_relu_100k", Box::new(|| { black_box(a.leaky_relu(0.01)); })),
        ("elu_100k", Box::new(|| { black_box(a.elu(1.0)); })),
        ("hard_tanh_100k", Box::new(|| { black_box(a.hard_tanh(-1.0, 1.0)); })),
        ("relu6_100k", Box::new(|| { black_box(a.relu6()); })),
        ("hardswish_100k", Box::new(|| { black_box(a.hardswish()); })),
        ("gelu_100k", Box::new(|| { black_box(a.gelu()); })),
        ("selu_100k", Box::new(|| { black_box(a.selu()); })),
        ("softsign_100k", Box::new(|| { black_box(a.softsign()); })),
    ];

    for (name, f) in &ops {
        let times = bench(|| f(), iters);
        results.push(make_result(name, "activation", times));
    }
    results
}

// ---------------------------------------------------------------------------
// Loss functions
// ---------------------------------------------------------------------------

pub fn bench_losses(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let batch = 64usize;
    let classes = 10usize;
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };

    let pred = Tensor::randn(&[batch, classes], false);
    let target = Tensor::randn(&[batch, classes], false);
    let target_abs = Tensor::new(
        (0..batch * classes).map(|i| ((i as f32) * 0.01).abs()).collect(),
        vec![batch, classes], false,
    );
    let targets_idx: Vec<usize> = (0..batch).map(|i| i % classes).collect();

    let times = bench(|| { black_box(nn::cross_entropy_loss(&pred, &targets_idx)); }, iters);
    results.push(make_result("cross_entropy_64x10", "loss", times));

    let times = bench(|| { black_box(nn::l1_loss(&pred, &target)); }, iters);
    results.push(make_result("l1_loss_64x10", "loss", times));

    let times = bench(|| { black_box(nn::mse_loss(&pred, &target)); }, iters);
    results.push(make_result("mse_loss_64x10", "loss", times));

    let times = bench(|| { black_box(nn::huber_loss(&pred, &target, 1.0)); }, iters);
    results.push(make_result("huber_loss_64x10", "loss", times));

    let times = bench(|| { black_box(nn::smooth_l1_loss(&pred, &target, 1.0)); }, iters);
    results.push(make_result("smooth_l1_loss_64x10", "loss", times));

    let times = bench(|| { black_box(nn::kl_div_loss(&pred, &target_abs)); }, iters);
    results.push(make_result("kl_div_loss_64x10", "loss", times));

    let a_emb = Tensor::randn(&[batch, 64], false);
    let b_emb = Tensor::randn(&[batch, 64], false);
    let times = bench(|| { black_box(nn::cosine_similarity_loss(&a_emb, &b_emb)); }, iters);
    results.push(make_result("cosine_sim_loss_64x64", "loss", times));

    results
}

// ---------------------------------------------------------------------------
// NN layers
// ---------------------------------------------------------------------------

pub fn bench_nn_layers(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters_f = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
    let iters_s = if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW };

    // RMSNorm
    let rmsnorm = nn::RMSNorm::new(512, 1e-5);
    let x = Tensor::randn(&[64, 512], false);
    let times = bench(|| { black_box(rmsnorm.forward(&x)); }, iters_f);
    results.push(make_result("rmsnorm_64x512", "nn_layer", times));

    // Conv1d: 32 channels, kernel 3, stride 1, pad 0
    let conv1d = nn::Conv1d::new(32, 64, 3, 1, 0);
    let x_conv = Tensor::randn(&[1, 32, 128], false);
    let times = bench(|| { black_box(conv1d.forward(&x_conv)); }, iters_f);
    results.push(make_result("conv1d_1x32x128_k3", "nn_layer", times));

    // AvgPool2d
    let pool = nn::AvgPool2d::new(2, 2, 0);
    let x_pool = Tensor::randn(&[1, 16, 32, 32], false);
    let times = bench(|| { black_box(pool.forward(&x_pool)); }, iters_f);
    results.push(make_result("avgpool2d_1x16x32x32", "nn_layer", times));

    // GroupNorm
    let gn = nn::GroupNorm::new(8, 64, 1e-5);
    let x_gn = Tensor::randn(&[4, 64, 16, 16], false);
    let times = bench(|| { black_box(gn.forward(&x_gn)); }, iters_f);
    results.push(make_result("groupnorm_4x64x16x16", "nn_layer", times));

    // RNN: input=128, hidden=256, seq_len=32
    let rnn = nn::RNN::new(128, 256);
    let x_rnn = Tensor::randn(&[32, 128], false);
    let h0 = Tensor::zeros(&[1, 256], false);
    let times = bench(|| { black_box(rnn.forward(&x_rnn, &h0)); }, iters_s);
    results.push(make_result("rnn_seq32_128_256", "nn_layer", times));

    // LSTM: input=128, hidden=256, seq_len=32
    let lstm = nn::LSTM::new(128, 256);
    let c0 = Tensor::zeros(&[1, 256], false);
    let times = bench(|| { black_box(lstm.forward(&x_rnn, &h0, &c0)); }, iters_s);
    results.push(make_result("lstm_seq32_128_256", "nn_layer", times));

    // GRU: input=128, hidden=256, seq_len=32
    let gru = nn::GRU::new(128, 256);
    let times = bench(|| { black_box(gru.forward(&x_rnn, &h0)); }, iters_s);
    results.push(make_result("gru_seq32_128_256", "nn_layer", times));

    results
}

// ---------------------------------------------------------------------------
// Optimizers
// ---------------------------------------------------------------------------

pub fn bench_optimizers(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let batch = 64usize;
    let iters = if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW };
    let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();

    // Helper: create MLP params + optimizer, run one training step
    macro_rules! bench_optimizer {
        ($name:expr, $opt:expr) => {{
            let w1 = Tensor::randn(&[784, 128], true);
            let b1 = Tensor::zeros(&[1, 128], true);
            let w2 = Tensor::randn(&[128, 64], true);
            let b2 = Tensor::zeros(&[1, 64], true);
            let w3 = Tensor::randn(&[64, 10], true);
            let b3 = Tensor::zeros(&[1, 10], true);
            let params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()];
            let mut opt = $opt(params);
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
            }, iters);
            results.push(make_result($name, "optimizer", times));
        }};
    }

    bench_optimizer!("optim_adam_64", |p: Vec<Tensor>| Adam::new(p, 1e-3));
    bench_optimizer!("optim_rmsprop_64", |p: Vec<Tensor>| RmsProp::new(p, 1e-3));
    bench_optimizer!("optim_lion_64", |p: Vec<Tensor>| Lion::new(p, 1e-4));
    bench_optimizer!("optim_adafactor_64", |p: Vec<Tensor>| Adafactor::new(p, 1e-3));

    results
}

// ---------------------------------------------------------------------------
// Random
// ---------------------------------------------------------------------------

pub fn bench_random(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters_f = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };
    let iters_s = if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW };

    let times = bench(|| { black_box(peregrine::random::uniform(&[100_000], 0.0, 1.0, false)); }, iters_f);
    results.push(make_result("rand_uniform_100k", "random", times));

    let times = bench(|| { black_box(peregrine::random::normal(&[100_000], 0.0, 1.0, false)); }, iters_f);
    results.push(make_result("rand_normal_100k", "random", times));

    let times = bench(|| { black_box(peregrine::random::bernoulli(&[100_000], 0.5)); }, iters_f);
    results.push(make_result("rand_bernoulli_100k", "random", times));

    if !quick {
        let times = bench(|| { black_box(peregrine::random::uniform(&[1_000_000], 0.0, 1.0, false)); }, iters_s);
        results.push(make_result("rand_uniform_1M", "random", times));

        let times = bench(|| { black_box(peregrine::random::normal(&[1_000_000], 0.0, 1.0, false)); }, iters_s);
        results.push(make_result("rand_normal_1M", "random", times));
    }

    results
}

// ---------------------------------------------------------------------------
// FFT
// ---------------------------------------------------------------------------

pub fn bench_fft(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };

    let rfft_sizes: Vec<usize> = if quick {
        vec![1024, 4096]
    } else {
        vec![1024, 4096, 16384]
    };
    for n in rfft_sizes {
        let x = Tensor::randn(&[n], false);
        let label_n = if n >= 1000 { format!("{}k", n / 1000) } else { format!("{n}") };
        let times = bench(|| { black_box(peregrine::fft::rfft(&x, None)); }, iters);
        results.push(make_result(&format!("rfft_{label_n}"), "fft", times));
    }

    for n in [1024usize, 4096] {
        let x = Tensor::randn(&[n, 2], false);
        let label_n = if n >= 1000 { format!("{}k", n / 1000) } else { format!("{n}") };
        let times = bench(|| { black_box(peregrine::fft::fft(&x, None)); }, iters);
        results.push(make_result(&format!("fft_{label_n}"), "fft", times));
    }

    results
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

pub fn bench_linalg(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters_f = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };

    // Norm
    let x = Tensor::randn(&[1000], false);
    let times = bench(|| { black_box(peregrine::linalg::norm(&x, Some(2.0))); }, iters_f);
    results.push(make_result("norm_l2_1k", "linalg", times));

    let sizes: Vec<usize> = if quick {
        vec![64, 128]
    } else {
        vec![64, 128, 256]
    };
    for n in sizes {
        let iters = if n >= 256 {
            if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW }
        } else {
            iters_f
        };

        // Make positive-definite matrix for cholesky: A = B^T B + n*I
        let b_data: Vec<f32> = (0..n * n).map(|i| ((i as f32 * 0.7123 + 0.5) % 2.0) - 1.0).collect();
        let b_mat = Tensor::new(b_data, vec![n, n], false);
        let eye = Tensor::eye(n, false);
        let a_pd = b_mat.transpose(0, 1).matmul(&b_mat).add(&eye.scale(n as f32));

        // Solve: Ax = b
        let b_vec = Tensor::randn(&[n, 1], false);
        let times = bench(|| { black_box(peregrine::linalg::solve(&a_pd, &b_vec)); }, iters);
        results.push(make_result(&format!("solve_{n}x{n}"), "linalg", times));

        // Inverse
        let times = bench(|| { black_box(peregrine::linalg::inv(&a_pd)); }, iters);
        results.push(make_result(&format!("inv_{n}x{n}"), "linalg", times));

        // Cholesky
        let times = bench(|| { black_box(peregrine::linalg::cholesky(&a_pd)); }, iters);
        results.push(make_result(&format!("cholesky_{n}x{n}"), "linalg", times));

        // SVD
        let a = Tensor::randn(&[n, n], false);
        let times = bench(|| { black_box(peregrine::linalg::svd(&a)); }, iters);
        results.push(make_result(&format!("svd_{n}x{n}"), "linalg", times));

        // QR
        let times = bench(|| { black_box(peregrine::linalg::qr(&a)); }, iters);
        results.push(make_result(&format!("qr_{n}x{n}"), "linalg", times));

        // Eigendecomposition
        let a_sym = b_mat.transpose(0, 1).matmul(&b_mat);
        let times = bench(|| { black_box(peregrine::linalg::eigh(&a_sym)); }, iters);
        results.push(make_result(&format!("eigh_{n}x{n}"), "linalg", times));

        // Determinant
        let times = bench(|| { black_box(peregrine::linalg::det(&a)); }, iters);
        results.push(make_result(&format!("det_{n}x{n}"), "linalg", times));
    }

    results
}

// ---------------------------------------------------------------------------
// Pipeline ops (fused matmul+bias+gelu, add+layernorm)
// ---------------------------------------------------------------------------

pub fn bench_pipeline_ops(quick: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();
    let iters_s = if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW };
    let iters_f = if quick { quick_iters(ITERS_FAST) } else { ITERS_FAST };

    // matmul_bias_gelu at transformer FFN sizes
    let ffn_sizes: Vec<(usize, usize, usize, &str)> = if quick {
        vec![(196, 768, 3072, "196x768x3072")]
    } else {
        vec![
            (196, 768, 3072, "196x768x3072"),   // ViT-Base FFN
            (196, 1024, 4096, "196x1024x4096"),  // ViT-Large FFN
        ]
    };
    for (m, k, n, label) in &ffn_sizes {
        let x = Tensor::randn(&[*m, *k], false);
        let w = Tensor::randn(&[*k, *n], false);
        let b = Tensor::randn(&[1, *n], false);
        let times = bench(|| {
            let h = x.matmul_bias_gelu(&w, &b);
            black_box(h);
        }, iters_s);
        results.push(make_result(&format!("matmul_bias_gelu_{label}"), "pipeline", times));
    }

    // add + layernorm at transformer sizes
    let ln_sizes: Vec<(usize, usize, &str)> = if quick {
        vec![(196, 768, "196x768")]
    } else {
        vec![
            (196, 768, "196x768"),
            (196, 1024, "196x1024"),
        ]
    };
    for (batch, dim, label) in &ln_sizes {
        let x = Tensor::randn(&[*batch, *dim], false);
        let r = Tensor::randn(&[*batch, *dim], false);
        let g = Tensor::randn(&[*dim], false);
        let b = Tensor::randn(&[*dim], false);
        let times = bench(|| {
            let out = x.add_layer_norm(&r, &g, &b, *dim);
            black_box(out);
        }, iters_f);
        results.push(make_result(&format!("add_layernorm_{label}"), "pipeline", times));
    }

    results
}

// ---------------------------------------------------------------------------
// Quantized matmul (int8)
// ---------------------------------------------------------------------------

pub fn bench_int8_matmul(quick: bool) -> Vec<BenchResult> {
    use peregrine::quant::quantize_weights;
    let mut results = Vec::new();
    let iters = if quick { quick_iters(ITERS_SLOW) } else { ITERS_SLOW };

    let sizes: Vec<(usize, usize, usize, &str)> = if quick {
        vec![(196, 768, 3072, "196x768x3072")]
    } else {
        vec![
            (196, 768, 3072, "196x768x3072"),   // ViT-Base FFN
            (196, 1024, 4096, "196x1024x4096"),  // ViT-Large FFN
        ]
    };

    for (m, k, n, label) in &sizes {
        // f32 matmul baseline
        let a = Tensor::randn(&[*m, *k], false);
        let w_f32 = Tensor::randn(&[*k, *n], false);
        let times_f32 = bench(|| {
            let h = a.matmul(&w_f32);
            black_box(h);
        }, iters);
        results.push(make_result(&format!("matmul_f32_{label}"), "quantized", times_f32));

        // int8 matmul
        let w_data = w_f32.data();
        let qt = quantize_weights(&w_data, *k, *n);
        let times_i8 = bench(|| {
            let h = a.matmul_quantized(&qt);
            black_box(h);
        }, iters);
        results.push(make_result(&format!("matmul_i8_{label}"), "quantized", times_i8));
    }

    results
}

// ---------------------------------------------------------------------------
// Run all CPU benchmarks
// ---------------------------------------------------------------------------

pub fn run_all_cpu(quick: bool) -> Vec<BenchResult> {
    let mut all = Vec::new();

    macro_rules! run_section {
        ($name:expr, $func:expr) => {{
            eprint!("  {}...", $name);
            let r = $func;
            eprintln!(" {} ops", r.len());
            all.extend(r);
        }};
    }

    eprintln!("Running CPU benchmarks{}...", if quick { " (quick)" } else { "" });

    run_section!("matmul", bench_matmul(quick));
    run_section!("add", bench_add(quick));
    run_section!("mul", bench_mul(quick));
    run_section!("exp", bench_exp(quick));
    run_section!("relu", bench_relu(quick));
    run_section!("softmax", bench_softmax(quick));
    run_section!("mlp_forward", bench_mlp_forward(quick));
    run_section!("training_step", bench_training_step(quick));
    run_section!("unary_math", bench_unary_math(quick));
    run_section!("binary_math", bench_binary_math(quick));
    run_section!("axis_reductions", bench_axis_reductions(quick));
    run_section!("shape_ops", bench_shape_ops(quick));
    run_section!("activations", bench_activations(quick));
    run_section!("losses", bench_losses(quick));
    run_section!("nn_layers", bench_nn_layers(quick));
    run_section!("optimizers", bench_optimizers(quick));
    run_section!("random", bench_random(quick));
    run_section!("fft", bench_fft(quick));
    run_section!("linalg", bench_linalg(quick));
    run_section!("pipeline_ops", bench_pipeline_ops(quick));
    run_section!("int8_matmul", bench_int8_matmul(quick));

    eprintln!("Done: {} total benchmarks", all.len());
    all
}
