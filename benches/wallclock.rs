//! Wall-clock benchmark for Peregrine operations.
//!
//! Standalone binary (harness = false) that outputs JSON to
//! target/bench_compare/peregrine.json with timing stats for each operation.
//!
//! Run: cargo bench --bench wallclock
//! GPU: cargo bench --bench wallclock --features metal

use peregrine::nn;
use peregrine::optim::{Adam, RmsProp, Lion, Adafactor};
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
            let h1 = x.matmul_bias_relu(&w1, &b1);
            let h2 = h1.matmul_bias_relu(&w2, &b2);
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
            let h1 = x.matmul_bias_relu(&w1, &b1);
            let h2 = h1.matmul_bias_relu(&w2, &b2);
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
// Phase 1A: Unary math ops
// ---------------------------------------------------------------------------

fn bench_unary_math() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let n = 100_000usize;
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
        let times = bench(|| f(), ITERS_FAST);
        results.push(make_result(name, times));
    }
    results
}

// ---------------------------------------------------------------------------
// Phase 1B: Binary math ops
// ---------------------------------------------------------------------------

fn bench_binary_math() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let n = 100_000usize;
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
        let times = bench(|| f(), ITERS_FAST);
        results.push(make_result(name, times));
    }

    // Clip and where
    let times = bench(|| { black_box(a.clip(-0.5, 0.5)); }, ITERS_FAST);
    results.push(make_result("clip_100k", times));

    let cond = Tensor::new(
        (0..n).map(|i| if i % 2 == 0 { 1.0 } else { 0.0 }).collect(),
        vec![n], false,
    );
    let times = bench(|| { black_box(Tensor::where_cond(&cond, &a, &b)); }, ITERS_FAST);
    results.push(make_result("where_100k", times));

    // Comparison ops
    let times = bench(|| { black_box(a.greater(&b)); }, ITERS_FAST);
    results.push(make_result("greater_100k", times));

    let times = bench(|| { black_box(a.equal(&b)); }, ITERS_FAST);
    results.push(make_result("equal_100k", times));

    results
}

// ---------------------------------------------------------------------------
// Phase 1E: Axis reductions
// ---------------------------------------------------------------------------

fn bench_axis_reductions() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // 2D tensor: [256, 512] — reduce along axis 1
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
        let times = bench(|| f(), ITERS_FAST);
        results.push(make_result(name, times));
    }

    // Larger: [1024, 1024] for sort/topk
    let x_large = Tensor::randn(&[1024, 1024], false);
    let times = bench(|| { black_box(x_large.sum_axis(1, false)); }, ITERS_SLOW);
    results.push(make_result("sum_axis_1024x1024", times));

    let times = bench(|| { black_box(x_large.var(1, false, 0)); }, ITERS_SLOW);
    results.push(make_result("var_1024x1024", times));

    results
}

// ---------------------------------------------------------------------------
// Phase 1F: Shape/indexing ops
// ---------------------------------------------------------------------------

fn bench_shape_ops() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    let x = Tensor::randn(&[256, 256], false);
    let times = bench(|| { black_box(x.tril(0)); }, ITERS_FAST);
    results.push(make_result("tril_256x256", times));

    let times = bench(|| { black_box(x.triu(0)); }, ITERS_FAST);
    results.push(make_result("triu_256x256", times));

    let x_small = Tensor::randn(&[64, 128], false);
    let times = bench(|| { black_box(x_small.repeat(&[2, 3])); }, ITERS_FAST);
    results.push(make_result("repeat_64x128_2x3", times));

    let times = bench(|| { black_box(x_small.pad(&[(1, 1), (2, 2)], 0.0)); }, ITERS_FAST);
    results.push(make_result("pad_64x128", times));

    // Stack 8 tensors
    let tensors: Vec<Tensor> = (0..8).map(|_| Tensor::randn(&[64, 128], false)).collect();
    let times = bench(|| { black_box(Tensor::stack(&tensors, 0)); }, ITERS_FAST);
    results.push(make_result("stack_8x64x128", times));

    // Diagonal
    let x_sq = Tensor::randn(&[512, 512], false);
    let times = bench(|| { black_box(x_sq.diagonal(0)); }, ITERS_FAST);
    results.push(make_result("diagonal_512x512", times));

    results
}

// ---------------------------------------------------------------------------
// Phase 2: Activations
// ---------------------------------------------------------------------------

fn bench_activations() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let n = 100_000usize;
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
        let times = bench(|| f(), ITERS_FAST);
        results.push(make_result(name, times));
    }
    results
}

// ---------------------------------------------------------------------------
// Phase 3: Loss functions
// ---------------------------------------------------------------------------

fn bench_losses() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let batch = 64usize;
    let classes = 10usize;

    let pred = Tensor::randn(&[batch, classes], false);
    let target = Tensor::randn(&[batch, classes], false);
    let target_abs = Tensor::new(
        (0..batch * classes).map(|i| ((i as f32) * 0.01).abs()).collect(),
        vec![batch, classes], false,
    );
    let targets_idx: Vec<usize> = (0..batch).map(|i| i % classes).collect();

    let times = bench(|| { black_box(nn::cross_entropy_loss(&pred, &targets_idx)); }, ITERS_FAST);
    results.push(make_result("cross_entropy_64x10", times));

    let times = bench(|| { black_box(nn::l1_loss(&pred, &target)); }, ITERS_FAST);
    results.push(make_result("l1_loss_64x10", times));

    let times = bench(|| { black_box(nn::mse_loss(&pred, &target)); }, ITERS_FAST);
    results.push(make_result("mse_loss_64x10", times));

    let times = bench(|| { black_box(nn::huber_loss(&pred, &target, 1.0)); }, ITERS_FAST);
    results.push(make_result("huber_loss_64x10", times));

    let times = bench(|| { black_box(nn::smooth_l1_loss(&pred, &target, 1.0)); }, ITERS_FAST);
    results.push(make_result("smooth_l1_loss_64x10", times));

    let times = bench(|| { black_box(nn::kl_div_loss(&pred, &target_abs)); }, ITERS_FAST);
    results.push(make_result("kl_div_loss_64x10", times));

    let a_emb = Tensor::randn(&[batch, 64], false);
    let b_emb = Tensor::randn(&[batch, 64], false);
    let times = bench(|| { black_box(nn::cosine_similarity_loss(&a_emb, &b_emb)); }, ITERS_FAST);
    results.push(make_result("cosine_sim_loss_64x64", times));

    results
}

// ---------------------------------------------------------------------------
// Phase 3B: NN layers
// ---------------------------------------------------------------------------

fn bench_nn_layers() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // RMSNorm
    let rmsnorm = nn::RMSNorm::new(512, 1e-5);
    let x = Tensor::randn(&[64, 512], false);
    let times = bench(|| { black_box(rmsnorm.forward(&x)); }, ITERS_FAST);
    results.push(make_result("rmsnorm_64x512", times));

    // Conv1d: 32 channels, kernel 3, stride 1, pad 0
    let conv1d = nn::Conv1d::new(32, 64, 3, 1, 0);
    let x_conv = Tensor::randn(&[1, 32, 128], false);
    let times = bench(|| { black_box(conv1d.forward(&x_conv)); }, ITERS_FAST);
    results.push(make_result("conv1d_1x32x128_k3", times));

    // AvgPool2d
    let pool = nn::AvgPool2d::new(2, 2, 0);
    let x_pool = Tensor::randn(&[1, 16, 32, 32], false);
    let times = bench(|| { black_box(pool.forward(&x_pool)); }, ITERS_FAST);
    results.push(make_result("avgpool2d_1x16x32x32", times));

    // GroupNorm
    let gn = nn::GroupNorm::new(8, 64, 1e-5);
    let x_gn = Tensor::randn(&[4, 64, 16, 16], false);
    let times = bench(|| { black_box(gn.forward(&x_gn)); }, ITERS_FAST);
    results.push(make_result("groupnorm_4x64x16x16", times));

    // RNN: input=128, hidden=256, seq_len=32
    let rnn = nn::RNN::new(128, 256);
    let x_rnn = Tensor::randn(&[32, 128], false);
    let h0 = Tensor::zeros(&[1, 256], false);
    let times = bench(|| { black_box(rnn.forward(&x_rnn, &h0)); }, ITERS_SLOW);
    results.push(make_result("rnn_seq32_128_256", times));

    // LSTM: input=128, hidden=256, seq_len=32
    let lstm = nn::LSTM::new(128, 256);
    let c0 = Tensor::zeros(&[1, 256], false);
    let times = bench(|| { black_box(lstm.forward(&x_rnn, &h0, &c0)); }, ITERS_SLOW);
    results.push(make_result("lstm_seq32_128_256", times));

    // GRU: input=128, hidden=256, seq_len=32
    let gru = nn::GRU::new(128, 256);
    let times = bench(|| { black_box(gru.forward(&x_rnn, &h0)); }, ITERS_SLOW);
    results.push(make_result("gru_seq32_128_256", times));

    results
}

// ---------------------------------------------------------------------------
// Phase 4: Optimizers
// ---------------------------------------------------------------------------

fn bench_optimizers() -> Vec<serde_json::Value> {
    let mut results = Vec::new();
    let batch = 64usize;
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
            }, ITERS_SLOW);
            results.push(make_result($name, times));
        }};
    }

    bench_optimizer!("optim_adam_64", |p: Vec<Tensor>| Adam::new(p, 1e-3));
    bench_optimizer!("optim_rmsprop_64", |p: Vec<Tensor>| RmsProp::new(p, 1e-3));
    bench_optimizer!("optim_lion_64", |p: Vec<Tensor>| Lion::new(p, 1e-4));
    bench_optimizer!("optim_adafactor_64", |p: Vec<Tensor>| Adafactor::new(p, 1e-3));

    results
}

// ---------------------------------------------------------------------------
// Phase 5: Random
// ---------------------------------------------------------------------------

fn bench_random() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    let times = bench(|| { black_box(peregrine::random::uniform(&[100_000], 0.0, 1.0, false)); }, ITERS_FAST);
    results.push(make_result("rand_uniform_100k", times));

    let times = bench(|| { black_box(peregrine::random::normal(&[100_000], 0.0, 1.0, false)); }, ITERS_FAST);
    results.push(make_result("rand_normal_100k", times));

    let times = bench(|| { black_box(peregrine::random::bernoulli(&[100_000], 0.5)); }, ITERS_FAST);
    results.push(make_result("rand_bernoulli_100k", times));

    let times = bench(|| { black_box(peregrine::random::uniform(&[1_000_000], 0.0, 1.0, false)); }, ITERS_SLOW);
    results.push(make_result("rand_uniform_1M", times));

    let times = bench(|| { black_box(peregrine::random::normal(&[1_000_000], 0.0, 1.0, false)); }, ITERS_SLOW);
    results.push(make_result("rand_normal_1M", times));

    results
}

// ---------------------------------------------------------------------------
// Phase 6: FFT
// ---------------------------------------------------------------------------

fn bench_fft() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    for n in [1024usize, 4096, 16384] {
        let x = Tensor::randn(&[n], false);
        let label_n = if n >= 1000 { format!("{}k", n / 1000) } else { format!("{n}") };
        let times = bench(|| { black_box(peregrine::fft::rfft(&x, None)); }, ITERS_FAST);
        results.push(make_result(&format!("rfft_{label_n}"), times));
    }

    for n in [1024usize, 4096] {
        let x = Tensor::randn(&[n, 2], false);
        let label_n = if n >= 1000 { format!("{}k", n / 1000) } else { format!("{n}") };
        let times = bench(|| { black_box(peregrine::fft::fft(&x, None)); }, ITERS_FAST);
        results.push(make_result(&format!("fft_{label_n}", ), times));
    }

    results
}

// ---------------------------------------------------------------------------
// Phase 7: Linear algebra
// ---------------------------------------------------------------------------

fn bench_linalg() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // Norm
    let x = Tensor::randn(&[1000], false);
    let times = bench(|| { black_box(peregrine::linalg::norm(&x, Some(2.0))); }, ITERS_FAST);
    results.push(make_result("norm_l2_1k", times));

    for n in [64usize, 128, 256] {
        let iters = if n >= 256 { ITERS_SLOW } else { ITERS_FAST };

        // Make positive-definite matrix for cholesky: A = B^T B + n*I
        let b_data: Vec<f32> = (0..n * n).map(|i| ((i as f32 * 0.7123 + 0.5) % 2.0) - 1.0).collect();
        let b_mat = Tensor::new(b_data, vec![n, n], false);
        let eye = Tensor::eye(n, false);
        let a_pd = b_mat.transpose(0, 1).matmul(&b_mat).add(&eye.scale(n as f32));

        // Solve: Ax = b
        let b_vec = Tensor::randn(&[n, 1], false);
        let times = bench(|| { black_box(peregrine::linalg::solve(&a_pd, &b_vec)); }, iters);
        results.push(make_result(&format!("solve_{n}x{n}"), times));

        // Inverse
        let times = bench(|| { black_box(peregrine::linalg::inv(&a_pd)); }, iters);
        results.push(make_result(&format!("inv_{n}x{n}"), times));

        // Cholesky
        let times = bench(|| { black_box(peregrine::linalg::cholesky(&a_pd)); }, iters);
        results.push(make_result(&format!("cholesky_{n}x{n}"), times));

        // SVD
        let a = Tensor::randn(&[n, n], false);
        let times = bench(|| { black_box(peregrine::linalg::svd(&a)); }, iters);
        results.push(make_result(&format!("svd_{n}x{n}"), times));

        // QR
        let times = bench(|| { black_box(peregrine::linalg::qr(&a)); }, iters);
        results.push(make_result(&format!("qr_{n}x{n}"), times));

        // Eigendecomposition
        let a_sym = b_mat.transpose(0, 1).matmul(&b_mat);
        let times = bench(|| { black_box(peregrine::linalg::eigh(&a_sym)); }, iters);
        results.push(make_result(&format!("eigh_{n}x{n}"), times));

        // Determinant
        let times = bench(|| { black_box(peregrine::linalg::det(&a)); }, iters);
        results.push(make_result(&format!("det_{n}x{n}"), times));
    }

    results
}

// ---------------------------------------------------------------------------
// Pipeline ops (fused matmul+bias+gelu, add+layernorm)
// ---------------------------------------------------------------------------

fn bench_pipeline_ops() -> Vec<serde_json::Value> {
    let mut results = Vec::new();

    // matmul_bias_gelu at transformer FFN sizes
    for (m, k, n, label) in [
        (196, 768, 3072, "196x768x3072"),   // ViT-Base FFN
        (196, 1024, 4096, "196x1024x4096"),  // ViT-Large FFN
    ] {
        let x = Tensor::randn(&[m, k], false);
        let w = Tensor::randn(&[k, n], false);
        let b = Tensor::randn(&[1, n], false);
        let times = bench(|| {
            let h = x.matmul_bias_gelu(&w, &b);
            black_box(h);
        }, ITERS_SLOW);
        results.push(make_result(&format!("matmul_bias_gelu_{label}"), times));
    }

    // add + layernorm at transformer sizes
    for (batch, dim, label) in [
        (196, 768, "196x768"),
        (196, 1024, "196x1024"),
    ] {
        let x = Tensor::randn(&[batch, dim], false);
        let r = Tensor::randn(&[batch, dim], false);
        let g = Tensor::randn(&[dim], false);
        let b = Tensor::randn(&[dim], false);
        let times = bench(|| {
            let out = x.add_layer_norm(&r, &g, &b, dim);
            black_box(out);
        }, ITERS_FAST);
        results.push(make_result(&format!("add_layernorm_{label}"), times));
    }

    results
}

fn bench_int8_matmul() -> Vec<serde_json::Value> {
    use peregrine::quant::quantize_weights;
    let mut results = Vec::new();

    for (m, k, n, label) in [
        (196, 768, 3072, "196x768x3072"),   // ViT-Base FFN
        (196, 1024, 4096, "196x1024x4096"),  // ViT-Large FFN
    ] {
        // f32 matmul baseline
        let a = Tensor::randn(&[m, k], false);
        let w_f32 = Tensor::randn(&[k, n], false);
        let times_f32 = bench(|| {
            let h = a.matmul(&w_f32);
            black_box(h);
        }, ITERS_SLOW);
        results.push(make_result(&format!("matmul_f32_{label}"), times_f32));

        // int8 matmul
        let w_data = w_f32.data();
        let qt = quantize_weights(&w_data, k, n);
        let times_i8 = bench(|| {
            let h = a.matmul_quantized(&qt);
            black_box(h);
        }, ITERS_SLOW);
        results.push(make_result(&format!("matmul_i8_{label}"), times_i8));
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
            let h1 = x.matmul_bias_relu(&w1, &b1);
            let h2 = h1.matmul_bias_relu(&w2, &b2);
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
            let h1 = x.matmul_bias_relu(&w1, &b1);
            let h2 = h1.matmul_bias_relu(&w2, &b2);
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
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
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
            let h1 = x.matmul(&w1).add_bias(&b1).relu();
            let h2 = h1.matmul(&w2).add_bias(&b2).relu();
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
// Het decoder benchmark: sequential vs pipelined GPU+CPU overlap
// ---------------------------------------------------------------------------

#[cfg(feature = "metal")]
fn bench_het_decoder() -> Vec<serde_json::Value> {
    use peregrine::tensor::Tensor;
    let mut results = Vec::new();

    // Simulate decoder-like workload: matmul on GPU + matmul on CPU
    let m = 1024;
    let k = 768;
    let n = 768;

    let a_data: Vec<f32> = (0..m*k).map(|i| (i as f32 * 0.001) % 1.0).collect();
    let b_data: Vec<f32> = (0..k*n).map(|i| (i as f32 * 0.002) % 1.0).collect();

    // Sequential: GPU matmul + GPU matmul (current decoder behavior)
    {
        let a1 = Tensor::new(a_data.clone(), vec![m, k], false);
        let b1 = Tensor::new(b_data.clone(), vec![k, n], false);
        let a2 = Tensor::new(a_data.clone(), vec![m, k], false);
        let b2 = Tensor::new(b_data.clone(), vec![k, n], false);
        a1.to_gpu(); b1.to_gpu(); a2.to_gpu(); b2.to_gpu();

        let times = bench(|| {
            let _r1 = a1.matmul(&b1);
            let _r2 = a2.matmul(&b2);
            peregrine::metal::gpu_sync();
        }, ITERS_SLOW);
        results.push(make_result("het_sequential_gpu_gpu", times));
    }

    // Pipelined: GPU matmul + CPU matmul via het_execute
    {
        let a1 = Tensor::new(a_data.clone(), vec![m, k], false);
        let b1 = Tensor::new(b_data.clone(), vec![k, n], false);
        let a2 = Tensor::new(a_data.clone(), vec![m, k], false);
        let b2 = Tensor::new(b_data.clone(), vec![k, n], false);
        a1.to_gpu(); b1.to_gpu();
        // a2, b2 stay on CPU

        let times = bench(|| {
            let (_r1, _r2) = peregrine::metal::het_execute(
                || a1.matmul(&b1),
                || a2.matmul(&b2),
            );
        }, ITERS_SLOW);
        results.push(make_result("het_pipelined_gpu_cpu", times));
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
        // v0.11.0: new feature benchmarks
        ("bench_unary_math", bench_unary_math),
        ("bench_binary_math", bench_binary_math),
        ("bench_axis_reductions", bench_axis_reductions),
        ("bench_shape_ops", bench_shape_ops),
        ("bench_activations", bench_activations),
        ("bench_losses", bench_losses),
        ("bench_nn_layers", bench_nn_layers),
        ("bench_optimizers", bench_optimizers),
        ("bench_random", bench_random),
        ("bench_fft", bench_fft),
        ("bench_linalg", bench_linalg),
        // Pipeline ops (fused matmul+bias+gelu, add+layernorm)
        ("bench_pipeline_ops", bench_pipeline_ops),
        // Int8 quantized matmul
        ("bench_int8_matmul", bench_int8_matmul),
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
            ("bench_het_decoder", bench_het_decoder),
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
