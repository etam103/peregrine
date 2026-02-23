//! Metal autograd integration tests: GPU forward+backward parity, training convergence.
//! Run with: cargo test --test metal_autograd --features metal

#![cfg(feature = "metal")]

use peregrine::metal::init_gpu;
use peregrine::tensor::Tensor;
use peregrine::optim::Adam;

fn setup_gpu() {
    init_gpu().expect("Metal GPU required for these tests");
}

fn max_abs_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn seeded_data(n: usize, seed: u32) -> Vec<f32> {
    let mut data = Vec::with_capacity(n);
    let mut state: u32 = seed;
    for _ in 0..n {
        state = state.wrapping_mul(1103515245).wrapping_add(12345);
        let f = (state as f32 / u32::MAX as f32) * 2.0 - 1.0;
        data.push(f);
    }
    data
}

// =========================================================================
// 1. GPU forward-backward parity: verify gradients match CPU
// =========================================================================

/// Binary op parity check: both inputs get gradients.
fn check_binary_grad_parity<F>(name: &str, f: F, tol: f32)
where
    F: Fn(&Tensor, &Tensor) -> Tensor,
{
    setup_gpu();
    let a_data = seeded_data(12, 42);
    let b_data = seeded_data(12, 99);

    // CPU path
    let a_cpu = Tensor::new(a_data.clone(), vec![3, 4], true);
    let b_cpu = Tensor::new(b_data.clone(), vec![3, 4], true);
    let loss_cpu = f(&a_cpu, &b_cpu).sum();
    loss_cpu.backward();
    let ga_cpu = a_cpu.grad().unwrap();
    let gb_cpu = b_cpu.grad().unwrap();

    // GPU path
    let a_gpu = Tensor::new(a_data.clone(), vec![3, 4], true);
    let b_gpu = Tensor::new(b_data.clone(), vec![3, 4], true);
    a_gpu.to_gpu();
    b_gpu.to_gpu();
    let loss_gpu = f(&a_gpu, &b_gpu).sum();
    loss_gpu.backward();
    let ga_gpu = a_gpu.grad().unwrap();
    let gb_gpu = b_gpu.grad().unwrap();

    let err_a = max_abs_error(&ga_cpu, &ga_gpu);
    let err_b = max_abs_error(&gb_cpu, &gb_gpu);
    assert!(err_a < tol, "{name} grad_a: max err={err_a} (tol={tol})");
    assert!(err_b < tol, "{name} grad_b: max err={err_b} (tol={tol})");
}

/// Unary op parity check: only input gets gradient.
fn check_unary_grad_parity<F>(name: &str, f: F, tol: f32)
where
    F: Fn(&Tensor) -> Tensor,
{
    setup_gpu();
    let a_data = seeded_data(12, 42);

    // CPU path
    let a_cpu = Tensor::new(a_data.clone(), vec![3, 4], true);
    let loss_cpu = f(&a_cpu).sum();
    loss_cpu.backward();
    let ga_cpu = a_cpu.grad().unwrap();

    // GPU path
    let a_gpu = Tensor::new(a_data.clone(), vec![3, 4], true);
    a_gpu.to_gpu();
    let loss_gpu = f(&a_gpu).sum();
    loss_gpu.backward();
    let ga_gpu = a_gpu.grad().unwrap();

    let err = max_abs_error(&ga_cpu, &ga_gpu);
    assert!(err < tol, "{name} grad: max err={err} (tol={tol})");
}

#[test]
fn gpu_grad_parity_add() {
    check_binary_grad_parity("add", |a, b| a.add(b), 1e-4);
}

#[test]
fn gpu_grad_parity_mul() {
    check_binary_grad_parity("mul", |a, b| a.mul(b), 1e-4);
}

#[test]
fn gpu_grad_parity_sub() {
    check_binary_grad_parity("sub", |a, b| a.sub(b), 1e-4);
}

#[test]
fn gpu_grad_parity_relu() {
    check_unary_grad_parity("relu", |a| a.relu(), 1e-4);
}

#[test]
fn gpu_grad_parity_sigmoid() {
    check_unary_grad_parity("sigmoid", |a| a.sigmoid(), 1e-4);
}

#[test]
fn gpu_grad_parity_tanh() {
    check_unary_grad_parity("tanh", |a| a.tanh(), 1e-4);
}

#[test]
fn gpu_grad_parity_scale() {
    check_unary_grad_parity("scale", |a| a.scale(2.5), 1e-4);
}

#[test]
fn gpu_grad_parity_neg() {
    check_unary_grad_parity("neg", |a| a.neg(), 1e-4);
}

#[test]
fn gpu_grad_parity_exp() {
    setup_gpu();
    // Use small values to avoid float overflow
    let a_data: Vec<f32> = seeded_data(12, 42).iter().map(|x| x * 0.5).collect();

    let a_cpu = Tensor::new(a_data.clone(), vec![3, 4], true);
    let loss_cpu = a_cpu.exp().sum();
    loss_cpu.backward();
    let ga_cpu = a_cpu.grad().unwrap();

    let a_gpu = Tensor::new(a_data.clone(), vec![3, 4], true);
    a_gpu.to_gpu();
    let loss_gpu = a_gpu.exp().sum();
    loss_gpu.backward();
    let ga_gpu = a_gpu.grad().unwrap();

    let err = max_abs_error(&ga_cpu, &ga_gpu);
    assert!(err < 1e-4, "exp grad: max err={err}");
}

#[test]
fn gpu_grad_parity_matmul() {
    setup_gpu();
    let a_data = seeded_data(8, 42);  // 2x4
    let b_data = seeded_data(12, 99); // 4x3

    // CPU
    let a_cpu = Tensor::new(a_data.clone(), vec![2, 4], true);
    let b_cpu = Tensor::new(b_data.clone(), vec![4, 3], true);
    let loss_cpu = a_cpu.matmul(&b_cpu).sum();
    loss_cpu.backward();
    let ga_cpu = a_cpu.grad().unwrap();
    let gb_cpu = b_cpu.grad().unwrap();

    // GPU
    let a_gpu = Tensor::new(a_data.clone(), vec![2, 4], true);
    let b_gpu = Tensor::new(b_data.clone(), vec![4, 3], true);
    a_gpu.to_gpu();
    b_gpu.to_gpu();
    let loss_gpu = a_gpu.matmul(&b_gpu).sum();
    loss_gpu.backward();
    let ga_gpu = a_gpu.grad().unwrap();
    let gb_gpu = b_gpu.grad().unwrap();

    let err_a = max_abs_error(&ga_cpu, &ga_gpu);
    let err_b = max_abs_error(&gb_cpu, &gb_gpu);
    assert!(err_a < 1e-3, "matmul grad_a: max err={err_a}");
    assert!(err_b < 1e-3, "matmul grad_b: max err={err_b}");
}

#[test]
fn gpu_grad_parity_softmax() {
    setup_gpu();
    let data = seeded_data(15, 42); // 3x5

    // CPU
    let a_cpu = Tensor::new(data.clone(), vec![3, 5], true);
    let loss_cpu = a_cpu.softmax(-1).sum();
    loss_cpu.backward();
    let ga_cpu = a_cpu.grad().unwrap();

    // GPU
    let a_gpu = Tensor::new(data.clone(), vec![3, 5], true);
    a_gpu.to_gpu();
    let loss_gpu = a_gpu.softmax(-1).sum();
    loss_gpu.backward();
    let ga_gpu = a_gpu.grad().unwrap();

    let err = max_abs_error(&ga_cpu, &ga_gpu);
    assert!(err < 1e-4, "softmax grad: max err={err}");
}

// =========================================================================
// 2. GPU MLP training convergence: train small MLP on GPU, verify loss decreases
// =========================================================================

#[test]
fn gpu_mlp_training_converges() {
    setup_gpu();

    // Simple MLP: 4 -> 8 -> 1, learn to approximate sum(x)
    let w1 = Tensor::new(seeded_data(32, 1), vec![4, 8], true);
    let b1 = Tensor::new(vec![0.0; 8], vec![1, 8], true);
    let w2 = Tensor::new(seeded_data(8, 2), vec![8, 1], true);
    let b2 = Tensor::new(vec![0.0; 1], vec![1, 1], true);

    // Move params to GPU
    w1.to_gpu();
    b1.to_gpu();
    w2.to_gpu();
    b2.to_gpu();

    let params = vec![w1.clone(), b1.clone(), w2.clone(), b2.clone()];
    let mut opt = Adam::new(params, 0.01);

    // Training data: x = random, y = sum(x) (simple target)
    let x_data = seeded_data(20, 100); // 5 samples x 4 features
    let y_data: Vec<f32> = (0..5).map(|i| {
        x_data[i*4..(i+1)*4].iter().sum()
    }).collect();

    let mut first_loss = f32::MAX;
    let mut last_loss = f32::MAX;

    for epoch in 0..50 {
        opt.zero_grad();

        let x = Tensor::new(x_data.clone(), vec![5, 4], false);
        let y = Tensor::new(y_data.clone(), vec![5, 1], false);
        x.to_gpu();
        y.to_gpu();

        // Forward: x -> linear1 -> relu -> linear2
        let h = x.matmul(&w1).add(&b1).relu();
        let pred = h.matmul(&w2).add(&b2);

        // MSE loss
        let diff = pred.sub(&y);
        let loss = diff.mul(&diff).mean();

        let loss_val = loss.data()[0];
        if epoch == 0 {
            first_loss = loss_val;
        }
        last_loss = loss_val;

        loss.backward();
        opt.step();
    }

    assert!(
        last_loss < first_loss * 0.5,
        "GPU training should reduce loss: first={first_loss}, last={last_loss}"
    );
}

// =========================================================================
// 3. Mixed CPU/GPU fallback: verify ops gracefully fall back
// =========================================================================

#[test]
fn mixed_cpu_gpu_fallback() {
    setup_gpu();

    // One tensor on GPU, one on CPU — should fall back to CPU path
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], true);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![4], true);
    a.to_gpu();
    // b stays on CPU

    // The add should still work (GPU tensor gets synced to CPU)
    let c = a.add(&b);
    let result = c.data();
    assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

    // Backward should also work
    c.sum().backward();
    let ga = a.grad().unwrap();
    let gb = b.grad().unwrap();
    assert_eq!(ga, vec![1.0, 1.0, 1.0, 1.0]);
    assert_eq!(gb, vec![1.0, 1.0, 1.0, 1.0]);
}

// =========================================================================
// 4. Lazy sync correctness: verify data() returns correct values after GPU ops
// =========================================================================

#[test]
fn lazy_sync_data_returns_correct_values() {
    setup_gpu();

    let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
    let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3], false);
    a.to_gpu();
    b.to_gpu();

    // Perform GPU add
    let c = a.add(&b);
    assert!(c.is_gpu(), "result should be GPU-resident");

    // data() should lazily sync and return correct result
    let result = c.data();
    assert_eq!(result, vec![5.0, 7.0, 9.0]);
}

#[test]
fn lazy_sync_grad_after_gpu_backward() {
    setup_gpu();

    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], true);
    x.to_gpu();

    let y = x.scale(3.0).sum();
    y.backward();

    // grad() should lazily sync GPU grad
    let g = x.grad().unwrap();
    let err = max_abs_error(&g, &[3.0, 3.0, 3.0, 3.0]);
    assert!(err < 1e-6, "scale grad should be 3.0, got {:?}", g);
}

#[test]
fn lazy_sync_chained_ops() {
    setup_gpu();

    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let b = Tensor::new(vec![2.0, 0.0, 0.0, 2.0], vec![2, 2], false);
    a.to_gpu();
    b.to_gpu();

    // Chain: matmul -> relu -> sum
    let c = a.matmul(&b).relu().sum();
    let result = c.data();

    // Manual: A@B = [[2,4],[6,8]], relu = [[2,4],[6,8]], sum = 20
    let expected = 20.0f32;
    assert!(
        (result[0] - expected).abs() < 1e-3,
        "chained GPU ops: expected {expected}, got {}", result[0]
    );
}

#[test]
fn gpu_to_cpu_roundtrip() {
    setup_gpu();

    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let t = Tensor::new(data.clone(), vec![5], false);
    assert!(!t.is_gpu());

    t.to_gpu();
    assert!(t.is_gpu());

    t.to_cpu();
    assert!(!t.is_gpu());
    assert_eq!(t.data(), data);
}
