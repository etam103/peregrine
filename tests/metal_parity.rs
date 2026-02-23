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
parity_unary_test!(parity_gpu_cpu_tanh,    "tanh_f32",    tanh,    1e-6, false);
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
        gpu.dispatch_matmul(&a, &b, &c, None, m as u32, n as u32, k as u32, false);
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
    gpu.dispatch_matmul(&a, &b, &c, None, m as u32, n as u32, k as u32, false);
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
        let gpu_result = output_buf.read();

        let err = max_abs_error(&cpu_result, &gpu_result);
        assert!(err < 1e-4, "layernorm [{batch}x{dim}]: max err={err}");
    }
}
