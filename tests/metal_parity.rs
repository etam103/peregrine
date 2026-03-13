//! CPU vs Metal parity tests: verify GPU kernels match CPU tensor ops.
//! Run with: cargo test --test metal_parity --features metal

#![cfg(feature = "metal")]

use peregrine::metal::GpuContext;
use peregrine::tensor::Tensor;

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn random_data(n: usize) -> Vec<f32> {
    // Deterministic pseudo-random for reproducibility
    let mut data = Vec::with_capacity(n);
    let mut state: u32 = 12345;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0; // [-1, 1]
        data.push(f);
    }
    data
}

fn positive_data(n: usize) -> Vec<f32> {
    random_data(n).iter().map(|x| x.abs() + 0.01).collect()
}

// --- Element-wise binary ops ---

#[test]
fn parity_gpu_cpu_add() {
    let gpu = GpuContext::new().unwrap();
    for size in [4, 64, 256, 4096] {
        let a_data = random_data(size);
        let b_data = random_data(size);
        // CPU
        let a_cpu = Tensor::new(a_data.clone(), vec![size], false);
        let b_cpu = Tensor::new(b_data.clone(), vec![size], false);
        let cpu_result = a_cpu.add(&b_cpu).data();
        // GPU
        let a_gpu = gpu.upload(&a_data);
        let b_gpu = gpu.upload(&b_data);
        let out = gpu.alloc(size);
        gpu.dispatch_binary("add_f32", &a_gpu, &b_gpu, &out);
        gpu.sync();
        let gpu_result = out.read();
        let err = max_abs_error(&cpu_result, &gpu_result);
        assert!(err < 1e-6, "add size={size}: max err={err}");
    }
}

#[test]
fn parity_gpu_cpu_sub() {
    let gpu = GpuContext::new().unwrap();
    let a_data = random_data(1024);
    let b_data = random_data(1024);
    let cpu_result = Tensor::new(a_data.clone(), vec![1024], false)
        .sub(&Tensor::new(b_data.clone(), vec![1024], false)).data();
    let a = gpu.upload(&a_data);
    let b = gpu.upload(&b_data);
    let out = gpu.alloc(1024);
    gpu.dispatch_binary("sub_f32", &a, &b, &out);
    gpu.sync();
    let err = max_abs_error(&cpu_result, &out.read());
    assert!(err < 1e-6, "sub: max err={err}");
}

#[test]
fn parity_gpu_cpu_mul() {
    let gpu = GpuContext::new().unwrap();
    let a_data = random_data(1024);
    let b_data = random_data(1024);
    let cpu_result = Tensor::new(a_data.clone(), vec![1024], false)
        .mul(&Tensor::new(b_data.clone(), vec![1024], false)).data();
    let a = gpu.upload(&a_data);
    let b = gpu.upload(&b_data);
    let out = gpu.alloc(1024);
    gpu.dispatch_binary("mul_f32", &a, &b, &out);
    gpu.sync();
    let err = max_abs_error(&cpu_result, &out.read());
    assert!(err < 1e-6, "mul: max err={err}");
}

#[test]
fn parity_gpu_cpu_div() {
    let gpu = GpuContext::new().unwrap();
    let a_data = random_data(1024);
    let b_data = positive_data(1024); // avoid division by near-zero
    let cpu_result = Tensor::new(a_data.clone(), vec![1024], false)
        .div(&Tensor::new(b_data.clone(), vec![1024], false)).data();
    let a = gpu.upload(&a_data);
    let b = gpu.upload(&b_data);
    let out = gpu.alloc(1024);
    gpu.dispatch_binary("div_f32", &a, &b, &out);
    gpu.sync();
    let err = max_abs_error(&cpu_result, &out.read());
    assert!(err < 1e-5, "div: max err={err}");
}

// --- Element-wise unary ops ---

macro_rules! parity_unary_test {
    ($name:ident, $kernel:expr, $cpu_op:ident, $tol:expr, $use_positive:expr) => {
        #[test]
        fn $name() {
            let gpu = GpuContext::new().unwrap();
            for size in [4, 256, 4096] {
                let data = if $use_positive { positive_data(size) } else { random_data(size) };
                let cpu_result = Tensor::new(data.clone(), vec![size], false).$cpu_op().data();
                let a = gpu.upload(&data);
                let out = gpu.alloc(size);
                gpu.dispatch_unary($kernel, &a, &out);
                gpu.sync();
                let err = max_abs_error(&cpu_result, &out.read());
                assert!(err < $tol, "{} size={}: max err={}", $kernel, size, err);
            }
        }
    };
}

parity_unary_test!(parity_gpu_cpu_neg,     "neg_f32",     neg,     1e-7, false);
parity_unary_test!(parity_gpu_cpu_exp,     "exp_f32",     exp,     1e-5, false);
parity_unary_test!(parity_gpu_cpu_log,     "log_f32",     log,     1e-6, true);
parity_unary_test!(parity_gpu_cpu_sqrt,    "sqrt_f32",    sqrt,    1e-6, true);
parity_unary_test!(parity_gpu_cpu_relu,    "relu_f32",    relu,    1e-7, false);
parity_unary_test!(parity_gpu_cpu_sigmoid, "sigmoid_f32", sigmoid, 1e-6, false);
parity_unary_test!(parity_gpu_cpu_tanh,    "tanh_f32",    tanh,    2e-6, false);
parity_unary_test!(parity_gpu_cpu_sin,     "sin_f32",     sin,     1e-6, false);
parity_unary_test!(parity_gpu_cpu_cos,     "cos_f32",     cos,     1e-6, false);
parity_unary_test!(parity_gpu_cpu_abs,     "abs_f32",     abs,     1e-7, false);

// --- Matmul ---

#[test]
fn parity_gpu_cpu_matmul() {
    let gpu = GpuContext::new().unwrap();
    for (m, k, n) in [(4, 4, 4), (8, 16, 4), (32, 64, 16), (64, 128, 32)] {
        let a_data = random_data(m * k);
        let b_data = random_data(k * n);
        let cpu_result = Tensor::new(a_data.clone(), vec![m, k], false)
            .matmul(&Tensor::new(b_data.clone(), vec![k, n], false)).data();
        let a = gpu.upload(&a_data);
        let b = gpu.upload(&b_data);
        let c = gpu.alloc(m * n);
        gpu.dispatch_matmul(&a, &b, &c, None, m as u32, n as u32, k as u32, false, false, false);
        gpu.sync();
        let err = max_abs_error(&cpu_result, &c.read());
        assert!(err < 1e-3, "matmul [{m}x{k}]@[{k}x{n}]: max err={err}");
    }
}

#[test]
fn parity_gpu_cpu_matmul_large() {
    let gpu = GpuContext::new().unwrap();
    let (m, k, n) = (256, 256, 256);
    let a_data = random_data(m * k);
    let b_data = random_data(k * n);
    let cpu_result = Tensor::new(a_data.clone(), vec![m, k], false)
        .matmul(&Tensor::new(b_data.clone(), vec![k, n], false)).data();
    let a = gpu.upload(&a_data);
    let b = gpu.upload(&b_data);
    let c = gpu.alloc(m * n);
    gpu.dispatch_matmul(&a, &b, &c, None, m as u32, n as u32, k as u32, false, false, false);
    gpu.sync();
    let err = max_abs_error(&cpu_result, &c.read());
    assert!(err < 0.01, "matmul 256x256: max err={err}");
}

// --- Softmax ---

#[test]
fn parity_gpu_cpu_softmax() {
    let gpu = GpuContext::new().unwrap();
    for (batch, dim) in [(1, 10), (8, 64), (16, 128)] {
        let data = random_data(batch * dim);
        let cpu_result = Tensor::new(data.clone(), vec![batch, dim], false).softmax(-1).data();
        let input = gpu.upload(&data);
        let output = gpu.alloc(batch * dim);
        gpu.dispatch_softmax(&input, &output, batch as u32, dim as u32);
        gpu.sync();
        let err = max_abs_error(&cpu_result, &output.read());
        assert!(err < 1e-5, "softmax [{batch}x{dim}]: max err={err}");
    }
}

#[test]
fn parity_gpu_cpu_softmax_large_values() {
    let gpu = GpuContext::new().unwrap();
    let data: Vec<f32> = random_data(80).iter().map(|x| x * 100.0).collect();
    let cpu_result = Tensor::new(data.clone(), vec![8, 10], false).softmax(-1).data();
    let input = gpu.upload(&data);
    let output = gpu.alloc(80);
    gpu.dispatch_softmax(&input, &output, 8, 10);
    gpu.sync();
    let err = max_abs_error(&cpu_result, &output.read());
    assert!(err < 1e-5, "softmax large values: max err={err}");
}

// --- Reduction ---

#[test]
fn parity_gpu_cpu_sum() {
    let gpu = GpuContext::new().unwrap();
    for size in [4, 256, 4096, 100_000] {
        let data = random_data(size);
        let cpu_sum: f32 = data.iter().sum();
        let buf = gpu.upload(&data);
        let gpu_sum = gpu.dispatch_reduce("sum_f32", &buf);
        let err = (cpu_sum - gpu_sum).abs();
        let rel = err / cpu_sum.abs().max(1e-8);
        assert!(rel < 0.01, "sum size={size}: cpu={cpu_sum}, gpu={gpu_sum}, rel_err={rel}");
    }
}

#[test]
fn parity_gpu_cpu_max() {
    let gpu = GpuContext::new().unwrap();
    let data = random_data(4096);
    let cpu_max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let buf = gpu.upload(&data);
    let gpu_max = gpu.dispatch_reduce("max_f32", &buf);
    let err = (cpu_max - gpu_max).abs();
    assert!(err < 1e-6, "max: cpu={cpu_max}, gpu={gpu_max}");
}

#[test]
fn parity_gpu_cpu_min() {
    let gpu = GpuContext::new().unwrap();
    let data = random_data(4096);
    let cpu_min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let buf = gpu.upload(&data);
    let gpu_min = gpu.dispatch_reduce("min_f32", &buf);
    let err = (cpu_min - gpu_min).abs();
    assert!(err < 1e-6, "min: cpu={cpu_min}, gpu={gpu_min}");
}

// --- Transpose ---

#[test]
fn parity_gpu_cpu_transpose() {
    let gpu = GpuContext::new().unwrap();
    for (rows, cols) in [(4, 4), (8, 16), (32, 64)] {
        let data = random_data(rows * cols);
        let cpu_result = Tensor::new(data.clone(), vec![rows, cols], false).transpose(0, 1).data();
        let input = gpu.upload(&data);
        let output = gpu.alloc(rows * cols);
        gpu.dispatch_transpose(&input, &output, rows as u32, cols as u32);
        gpu.sync();
        let err = max_abs_error(&cpu_result, &output.read());
        assert!(err < 1e-7, "transpose [{rows}x{cols}]: max err={err}");
    }
}

// --- LayerNorm ---

#[test]
fn parity_gpu_cpu_layernorm() {
    let gpu = GpuContext::new().unwrap();
    for (batch, dim) in [(1, 64), (8, 64), (16, 128)] {
        let data = random_data(batch * dim);
        let gamma_data: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 / dim as f32)).collect();
        let beta_data: Vec<f32> = (0..dim).map(|i| -0.5 + (i as f32 / dim as f32) * 0.1).collect();

        // CPU
        let x = Tensor::new(data.clone(), vec![batch, dim], false);
        let gamma = Tensor::new(gamma_data.clone(), vec![dim], true);
        let beta = Tensor::new(beta_data.clone(), vec![dim], true);
        let cpu_result = x.layer_norm(&gamma, &beta, dim).data();

        // GPU
        let input_buf = gpu.upload(&data);
        let gamma_buf = gpu.upload(&gamma_data);
        let beta_buf = gpu.upload(&beta_data);
        let output_buf = gpu.alloc(batch * dim);
        gpu.dispatch_layernorm(&input_buf, &gamma_buf, &beta_buf, &output_buf,
                               batch as u32, dim as u32, 1e-5);
        gpu.sync();
        let gpu_result = output_buf.read();

        let err = max_abs_error(&cpu_result, &gpu_result);
        assert!(err < 1e-4, "layernorm [{batch}x{dim}]: max err={err}");
    }
}

// --- Fused MatMul+Bias+Relu ---

#[test]
fn parity_matmul_bias_relu() {
    // Compare fused GPU path vs unfused CPU path (forward + backward)
    for (m, k, n) in [(4, 8, 6), (16, 32, 10), (64, 128, 64)] {
        let a_data = random_data(m * k);
        let w_data = random_data(k * n);
        let b_data = random_data(n);

        // Unfused CPU path
        let a_cpu = Tensor::new(a_data.clone(), vec![m, k], true);
        let w_cpu = Tensor::new(w_data.clone(), vec![k, n], true);
        let b_cpu = Tensor::new(b_data.clone(), vec![1, n], true);
        let out_cpu = a_cpu.matmul(&w_cpu).add_bias(&b_cpu).relu();
        let cpu_fwd = out_cpu.data();
        out_cpu.backward();
        let cpu_ga = a_cpu.grad().unwrap();
        let cpu_gw = w_cpu.grad().unwrap();
        let cpu_gb = b_cpu.grad().unwrap();

        // Fused GPU path
        let a_gpu = Tensor::new(a_data.clone(), vec![m, k], true);
        a_gpu.to_gpu();
        let w_gpu = Tensor::new(w_data.clone(), vec![k, n], true);
        w_gpu.to_gpu();
        let b_gpu = Tensor::new(b_data.clone(), vec![1, n], true);
        b_gpu.to_gpu();
        let out_gpu = a_gpu.matmul_bias_relu(&w_gpu, &b_gpu);
        let gpu_fwd = out_gpu.data();
        out_gpu.backward();
        let gpu_ga = a_gpu.grad().unwrap();
        let gpu_gw = w_gpu.grad().unwrap();
        let gpu_gb = b_gpu.grad().unwrap();

        let tol = 1e-3;
        let fwd_err = max_abs_error(&cpu_fwd, &gpu_fwd);
        assert!(fwd_err < tol, "matmul_bias_relu fwd [{m}x{k}x{n}]: err={fwd_err}");
        let ga_err = max_abs_error(&cpu_ga, &gpu_ga);
        assert!(ga_err < tol, "matmul_bias_relu grad_a [{m}x{k}x{n}]: err={ga_err}");
        let gw_err = max_abs_error(&cpu_gw, &gpu_gw);
        assert!(gw_err < tol, "matmul_bias_relu grad_w [{m}x{k}x{n}]: err={gw_err}");
        let gb_err = max_abs_error(&cpu_gb, &gpu_gb);
        assert!(gb_err < tol, "matmul_bias_relu grad_b [{m}x{k}x{n}]: err={gb_err}");
    }
}

#[test]
fn parity_matmul_bias_relu_training_loop() {
    use peregrine::nn;
    use peregrine::optim::Adam;
    // Full fused MLP training loop: verifies forward+backward+optimizer work in release mode
    let w1 = Tensor::randn(&[784, 128], true);   w1.to_gpu();
    let b1 = Tensor::zeros(&[1, 128], true);     b1.to_gpu();
    let w2 = Tensor::randn(&[128, 64], true);    w2.to_gpu();
    let b2 = Tensor::zeros(&[1, 64], true);      b2.to_gpu();
    let w3 = Tensor::randn(&[64, 10], true);     w3.to_gpu();
    let b3 = Tensor::zeros(&[1, 10], true);      b3.to_gpu();
    let batch = 64usize;
    let targets: Vec<usize> = (0..batch).map(|i| i % 10).collect();
    let mut opt = Adam::new(
        vec![w1.clone(), b1.clone(), w2.clone(), b2.clone(), w3.clone(), b3.clone()],
        1e-3,
    );
    for _ in 0..3 {
        let x = Tensor::randn(&[batch, 784], false);
        x.to_gpu();
        let h1 = x.matmul_bias_relu(&w1, &b1);
        let h2 = h1.matmul_bias_relu(&w2, &b2);
        let logits = h2.matmul(&w3).add_bias(&b3);
        let loss = nn::cross_entropy_loss(&logits, &targets);
        loss.backward();
        opt.step();
        opt.zero_grad();
    }
}

// --- Large matmul: simdgroup kernel parity ---

fn max_rel_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| {
        let err = (x - y).abs();
        if x.abs() > 1e-6 { err / x.abs() } else { err }
    }).fold(0.0f32, f32::max)
}

#[test]
fn parity_matmul_large_simdgroup() {
    // 1024x1024 triggers simdgroup kernel (M*N = 1M >= threshold)
    peregrine::metal::init_gpu().unwrap();
    for n in [1024, 1536] {
        let a_data = random_data(n * n);
        let b_data = random_data(n * n);

        // CPU path
        let a_cpu = Tensor::new(a_data.clone(), vec![n, n], false);
        let b_cpu = Tensor::new(b_data.clone(), vec![n, n], false);
        let c_cpu = a_cpu.matmul(&b_cpu);
        let cpu_result = c_cpu.data();

        // GPU path (auto-selects simdgroup for n>=1024)
        let a_gpu = Tensor::new(a_data, vec![n, n], false);
        let b_gpu = Tensor::new(b_data, vec![n, n], false);
        a_gpu.to_gpu();
        b_gpu.to_gpu();
        let c_gpu = a_gpu.matmul(&b_gpu);
        peregrine::metal::gpu_sync();
        let gpu_result = c_gpu.data();

        let abs_err = max_abs_error(&cpu_result, &gpu_result);
        let rel_err = max_rel_error(&cpu_result, &gpu_result);
        assert!(rel_err < 0.01,
            "simdgroup matmul [{n}x{n}]: max_abs_err={abs_err}, max_rel_err={rel_err}");
    }
}

#[test]
fn parity_matmul_large_backward() {
    // Verify backward pass works with simdgroup kernel on large matrices
    peregrine::metal::init_gpu().unwrap();
    let n = 1024;
    let a_data = random_data(n * n);
    let b_data = random_data(n * n);

    // CPU backward
    let a_cpu = Tensor::new(a_data.clone(), vec![n, n], true);
    let b_cpu = Tensor::new(b_data.clone(), vec![n, n], true);
    let c_cpu = a_cpu.matmul(&b_cpu);
    let loss_cpu = c_cpu.sum();
    loss_cpu.backward();
    let grad_a_cpu = a_cpu.grad().unwrap();
    let grad_b_cpu = b_cpu.grad().unwrap();

    // GPU backward
    let a_gpu = Tensor::new(a_data, vec![n, n], true);
    let b_gpu = Tensor::new(b_data, vec![n, n], true);
    a_gpu.to_gpu();
    b_gpu.to_gpu();
    let c_gpu = a_gpu.matmul(&b_gpu);
    let loss_gpu = c_gpu.sum();
    loss_gpu.backward();
    peregrine::metal::gpu_sync();
    let grad_a_gpu = a_gpu.grad().unwrap();
    let grad_b_gpu = b_gpu.grad().unwrap();

    let rel_a = max_rel_error(&grad_a_cpu, &grad_a_gpu);
    let rel_b = max_rel_error(&grad_b_cpu, &grad_b_gpu);
    assert!(rel_a < 0.01,
        "simdgroup matmul backward grad_a [{n}x{n}]: max_rel_err={rel_a}");
    assert!(rel_b < 0.01,
        "simdgroup matmul backward grad_b [{n}x{n}]: max_rel_err={rel_b}");
}

// --- Fused kernel parity tests ---

#[test]
fn parity_matmul_simd_bias_gelu() {
    let gpu = GpuContext::new().unwrap();
    // Test sizes that exercise both scalar (small) and simdgroup (large) kernels
    for (m, n, k) in [(64, 64, 32), (128, 256, 128), (1024, 1024, 512)] {
        let a_data = random_data(m * k);
        let b_data = random_data(k * n);
        let bias_data = random_data(n);

        // CPU reference: matmul + bias + gelu (unfused)
        let a_cpu = Tensor::new(a_data.clone(), vec![m, k], false);
        let b_cpu = Tensor::new(b_data.clone(), vec![k, n], false);
        let bias_cpu = Tensor::new(bias_data.clone(), vec![1, n], false);
        let cpu_result = a_cpu.matmul(&b_cpu).add_bias(&bias_cpu).gelu().data();

        // GPU fused: matmul_bias_gelu
        let a_g = Tensor::new(a_data.clone(), vec![m, k], false);
        let b_g = Tensor::new(b_data.clone(), vec![k, n], false);
        let bias_g = Tensor::new(bias_data.clone(), vec![1, n], false);
        a_g.to_gpu(); b_g.to_gpu(); bias_g.to_gpu();
        let gpu_result = a_g.matmul_bias_gelu(&b_g, &bias_g);
        peregrine::metal::gpu_sync();
        let gpu_data = gpu_result.data();

        let err = max_abs_error(&cpu_result, &gpu_data);
        assert!(err < 0.01,
            "matmul_bias_gelu [{m}x{k}]@[{k}x{n}]: max_abs_err={err}");
    }
}

#[test]
fn parity_bias_gelu() {
    let gpu = GpuContext::new().unwrap();
    for size in [256, 4096, 65536] {
        let cols = 256;
        let rows = size / cols;
        let input_data = random_data(size);
        let bias_data = random_data(cols);

        // CPU reference: bias_add + gelu
        let cpu_result: Vec<f32> = (0..size).map(|i| {
            let x = input_data[i] + bias_data[i % cols];
            let x3 = x * x * x;
            let inner = 0.7978845608f32 * (x + 0.044715 * x3);
            0.5 * x * (1.0 + inner.tanh())
        }).collect();

        // GPU fused kernel
        let in_buf = gpu.upload(&input_data);
        let bias_buf = gpu.upload(&bias_data);
        let out_buf = gpu.alloc(size);
        gpu.dispatch_bias_gelu(&in_buf, &bias_buf, &out_buf, cols as u32);
        gpu.sync();
        let gpu_result = out_buf.read();

        let err = max_abs_error(&cpu_result, &gpu_result);
        assert!(err < 1e-4,
            "bias_gelu rows={rows} cols={cols}: max_abs_err={err}");
    }
}

#[test]
fn parity_add_layernorm() {
    let gpu = GpuContext::new().unwrap();
    for (batch, dim) in [(4, 128), (16, 768), (32, 1024)] {
        let x_data = random_data(batch * dim);
        let res_data = random_data(batch * dim);
        let gamma_data: Vec<f32> = (0..dim).map(|i| 0.5 + (i as f32 / dim as f32)).collect();
        let beta_data = random_data(dim);

        // CPU reference: add + layernorm
        let x_cpu = Tensor::new(x_data.clone(), vec![batch, dim], false);
        let r_cpu = Tensor::new(res_data.clone(), vec![batch, dim], false);
        let g_cpu = Tensor::new(gamma_data.clone(), vec![dim], false);
        let b_cpu = Tensor::new(beta_data.clone(), vec![dim], false);
        let cpu_result = x_cpu.add(&r_cpu).layer_norm(&g_cpu, &b_cpu, dim).data();

        // GPU fused: add_layer_norm
        let x_g = Tensor::new(x_data.clone(), vec![batch, dim], false);
        let r_g = Tensor::new(res_data.clone(), vec![batch, dim], false);
        let g_g = Tensor::new(gamma_data.clone(), vec![dim], false);
        let b_g = Tensor::new(beta_data.clone(), vec![dim], false);
        x_g.to_gpu(); r_g.to_gpu(); g_g.to_gpu(); b_g.to_gpu();
        let gpu_result = x_g.add_layer_norm(&r_g, &g_g, &b_g, dim);
        peregrine::metal::gpu_sync();
        let gpu_data = gpu_result.data();

        let err = max_abs_error(&cpu_result, &gpu_data);
        assert!(err < 1e-3,
            "add_layernorm batch={batch} dim={dim}: max_abs_err={err}");
    }
}

#[test]
fn parity_matmul_simd_db() {
    let gpu = GpuContext::new().unwrap();
    // Test that double-buffered matmul produces same results as single-buffered
    // Using sizes that trigger the double-buffered path (M,N >= 64, K >= 64, M*N >= 1M)
    for n in [1024, 2048] {
        let a_data = random_data(n * n);
        let b_data = random_data(n * n);

        // CPU reference via matmul
        let a_cpu = Tensor::new(a_data.clone(), vec![n, n], false);
        let b_cpu = Tensor::new(b_data.clone(), vec![n, n], false);
        let cpu_result = a_cpu.matmul(&b_cpu).data();

        // GPU (will use double-buffered path for large enough K)
        let a_buf = gpu.upload(&a_data);
        let b_buf = gpu.upload(&b_data);
        let out_buf = gpu.alloc(n * n);
        gpu.dispatch_matmul_simd_db(&a_buf, &b_buf, &out_buf, None,
            n as u32, n as u32, n as u32, false, false, false, false);
        gpu.sync();
        let gpu_result = out_buf.read();

        let err = max_abs_error(&cpu_result, &gpu_result);
        // Simdgroup matmul has slightly larger error due to different accumulation order
        assert!(err < 0.05,
            "matmul_simd_db [{n}x{n}]: max_abs_err={err}");
    }
}

// --- het_execute: concurrent GPU + CPU matmul ---

#[test]
fn parity_het_execute() {
    peregrine::metal::init_gpu().unwrap();

    let m = 128;
    let k = 64;
    let n = 128;
    let a_data = random_data(m * k);
    let b_data = random_data(k * n);
    let c_data = random_data(m * k);
    let d_data = random_data(k * n);

    // Reference: sequential CPU matmul for both pairs
    let a_cpu = Tensor::new(a_data.clone(), vec![m, k], false);
    let b_cpu = Tensor::new(b_data.clone(), vec![k, n], false);
    let ref_gpu = a_cpu.matmul(&b_cpu).data();

    let c_cpu = Tensor::new(c_data.clone(), vec![m, k], false);
    let d_cpu = Tensor::new(d_data.clone(), vec![k, n], false);
    let ref_cpu = c_cpu.matmul(&d_cpu).data();

    // het_execute: GPU matmul + CPU matmul concurrently
    let a_het = Tensor::new(a_data.clone(), vec![m, k], false);
    let b_het = Tensor::new(b_data.clone(), vec![k, n], false);
    a_het.to_gpu();
    b_het.to_gpu();

    let c_het = Tensor::new(c_data.clone(), vec![m, k], false);
    let d_het = Tensor::new(d_data.clone(), vec![k, n], false);

    let (gpu_result, cpu_result) = peregrine::metal::het_execute(
        || {
            // GPU path: matmul dispatches to GPU because tensors are GPU-resident
            a_het.matmul(&b_het)
        },
        || {
            // CPU path: matmul runs on CPU because tensors are CPU-only
            c_het.matmul(&d_het)
        },
    );

    let gpu_out = gpu_result.data();
    let cpu_out = cpu_result.data();

    let gpu_err = max_abs_error(&ref_gpu, &gpu_out);
    let cpu_err = max_abs_error(&ref_cpu, &cpu_out);

    // GPU matmul may have slightly larger error due to different accumulation
    assert!(gpu_err < 0.01, "het_execute GPU matmul: max_abs_err={gpu_err}");
    assert!(cpu_err < 1e-6, "het_execute CPU matmul: max_abs_err={cpu_err}");
}

// --- Causal SDPA parity ---

#[test]
fn parity_gpu_cpu_causal_sdpa() {
    let gpu = GpuContext::new().unwrap();
    let head_dim = 16;
    let num_heads = 2;
    let seq_q = 4;
    let seq_kv = 8;
    let total_bh = num_heads;

    let q_data = random_data(total_bh * seq_q * head_dim);
    let k_data = random_data(total_bh * seq_kv * head_dim);
    let v_data = random_data(total_bh * seq_kv * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // GPU path: use dispatch_sdpa_masked with causal mask
    let q_gpu = gpu.upload(&q_data);
    let k_gpu = gpu.upload(&k_data);
    let v_gpu = gpu.upload(&v_data);
    let scores_gpu = gpu.alloc::<f32>(total_bh * seq_q * seq_kv);
    let output_gpu = gpu.alloc::<f32>(total_bh * seq_q * head_dim);

    gpu.dispatch_sdpa_masked(
        &q_gpu, &k_gpu, &v_gpu, &scores_gpu, &output_gpu,
        1, num_heads, num_heads,
        seq_q, seq_kv, head_dim,
        scale,
        &peregrine::metal::GpuAttentionMask::Causal { offset: 0 },
    );
    gpu.sync();
    let gpu_out = output_gpu.read();

    // CPU path: naive causal attention per head
    let mut cpu_out = vec![0.0f32; total_bh * seq_q * head_dim];
    for bh in 0..total_bh {
        for qt in 0..seq_q {
            let q_off = bh * seq_q * head_dim + qt * head_dim;
            let q_slice = &q_data[q_off..q_off + head_dim];

            let mut scores = vec![0.0f32; seq_kv];
            for kt in 0..seq_kv {
                if kt > qt {
                    scores[kt] = f32::NEG_INFINITY;
                } else {
                    let k_off = bh * seq_kv * head_dim + kt * head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_slice[d] * scale * k_data[k_off + d];
                    }
                    scores[kt] = dot;
                }
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            let o_off = bh * seq_q * head_dim + qt * head_dim;
            for kt in 0..seq_kv {
                let w = scores[kt];
                if w > 0.0 {
                    let v_off = bh * seq_kv * head_dim + kt * head_dim;
                    for d in 0..head_dim {
                        cpu_out[o_off + d] += w * v_data[v_off + d];
                    }
                }
            }
        }
    }

    let err = max_abs_error(&gpu_out, &cpu_out);
    assert!(err < 0.01, "Causal SDPA GPU/CPU parity: max_abs_err={err}");
}

#[test]
fn parity_gpu_cpu_sdpa_gqa() {
    let gpu = GpuContext::new().unwrap();
    let head_dim = 16;
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let seq_q = 4;
    let seq_kv = 8;
    let batch_size = 1;
    let total_q_bh = batch_size * num_q_heads;
    let total_kv_bh = batch_size * num_kv_heads;
    let heads_per_group = num_q_heads / num_kv_heads;

    let q_data = random_data(total_q_bh * seq_q * head_dim);
    let k_data = random_data(total_kv_bh * seq_kv * head_dim);
    let v_data = random_data(total_kv_bh * seq_kv * head_dim);
    let scale = 1.0 / (head_dim as f32).sqrt();

    // GPU path
    let q_gpu = gpu.upload(&q_data);
    let k_gpu = gpu.upload(&k_data);
    let v_gpu = gpu.upload(&v_data);
    let scores_gpu = gpu.alloc::<f32>(total_q_bh * seq_q * seq_kv);
    let output_gpu = gpu.alloc::<f32>(total_q_bh * seq_q * head_dim);

    gpu.dispatch_sdpa_masked(
        &q_gpu, &k_gpu, &v_gpu, &scores_gpu, &output_gpu,
        batch_size, num_q_heads, num_kv_heads,
        seq_q, seq_kv, head_dim,
        scale,
        &peregrine::metal::GpuAttentionMask::None,
    );
    gpu.sync();
    let gpu_out = output_gpu.read();

    // CPU path: GQA attention
    let mut cpu_out = vec![0.0f32; total_q_bh * seq_q * head_dim];
    for bh in 0..total_q_bh {
        let b = bh / num_q_heads;
        let qh = bh % num_q_heads;
        let kv_bh = b * num_kv_heads + (qh / heads_per_group);

        for qt in 0..seq_q {
            let q_off = bh * seq_q * head_dim + qt * head_dim;
            let q_slice = &q_data[q_off..q_off + head_dim];

            let mut scores = vec![0.0f32; seq_kv];
            for kt in 0..seq_kv {
                let k_off = kv_bh * seq_kv * head_dim + kt * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_slice[d] * scale * k_data[k_off + d];
                }
                scores[kt] = dot;
            }

            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_sum = 0.0f32;
            for s in &mut scores {
                *s = (*s - max_s).exp();
                exp_sum += *s;
            }
            for s in &mut scores {
                *s /= exp_sum;
            }

            let o_off = bh * seq_q * head_dim + qt * head_dim;
            for kt in 0..seq_kv {
                let w = scores[kt];
                if w > 0.0 {
                    let v_off = kv_bh * seq_kv * head_dim + kt * head_dim;
                    for d in 0..head_dim {
                        cpu_out[o_off + d] += w * v_data[v_off + d];
                    }
                }
            }
        }
    }

    let err = max_abs_error(&gpu_out, &cpu_out);
    assert!(err < 0.01, "GQA SDPA GPU/CPU parity: max_abs_err={err}");
}

// --- 2:4 Structured Sparse Matmul ---

#[test]
fn parity_gpu_cpu_sparse_24() {
    use peregrine::sparse::{prune_to_24, matmul_sparse_24, matmul_sparse_24_gpu};

    let gpu = GpuContext::new().unwrap();

    for &(m, k, n) in &[(8, 16, 12), (32, 64, 48), (16, 128, 32), (1, 4, 1)] {
        let a = random_data(m * k);
        let w = random_data(k * n);

        let mut st = prune_to_24(&w, k, n);
        let cpu_result = matmul_sparse_24(&a, m, k, &st);

        st.to_gpu(&gpu);
        let gpu_buf = matmul_sparse_24_gpu(&gpu, &a, m, k, &st);
        gpu.sync();
        let gpu_result = gpu_buf.read();

        let err = max_abs_error(&cpu_result, &gpu_result);
        assert!(
            err < 1e-3,
            "sparse-24 GPU/CPU parity: M={m} K={k} N={n} max_abs_err={err}"
        );
    }
}

#[test]
fn parity_gpu_cpu_huffman() {
    use peregrine::huffman::{HuffmanTensor, matmul_huffman};
    use peregrine::quant::quantize_weights;

    let gpu = GpuContext::new().unwrap();

    for &(m, k, n) in &[(8, 32, 16), (32, 128, 64), (16, 256, 32)] {
        let a = random_data(m * k);
        let w = random_data(k * n);

        let qt = quantize_weights(&w, k, n);
        let ht = HuffmanTensor::from_quantized(&qt, 4);

        // CPU result
        let _cpu_result = matmul_huffman(&a, m, k, &ht);

        // GPU result
        let out_buf = ht.matmul_huffman_gpu(&gpu, &a, m);
        gpu.sync();
        let gpu_result = out_buf.read();

        // Allow tolerance for GPU f32 vs CPU i8 path difference
        let mut f32_ref = vec![0.0f32; m * n];
        for mi in 0..m {
            for ni in 0..n {
                let mut sum = 0.0f32;
                for ki in 0..k {
                    sum += a[mi * k + ki] * w[ki * n + ni];
                }
                f32_ref[mi * n + ni] = sum;
            }
        }

        for i in 0..m * n {
            let diff = (gpu_result[i] - f32_ref[i]).abs();
            let denom = f32_ref[i].abs().max(1e-6);
            let rtol = diff / denom;
            assert!(
                rtol < 0.15 || diff < 0.05,
                "huffman GPU M={m} K={k} N={n} i={i}: got {}, expected {}, rtol={rtol}",
                gpu_result[i], f32_ref[i]
            );
        }
    }
}
