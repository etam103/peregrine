//! Tests for Phase 2 activation functions.
//!
//! Run: cargo test --test activations

use peregrine::tensor::Tensor;
use peregrine::nn;

// ---- Helper ----

fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tol,
            "mismatch at index {}: {} vs {} (diff={})",
            i, x, y, (x - y).abs()
        );
    }
}

fn numerical_grad(f: impl Fn(&Tensor) -> Tensor, x: &Tensor, eps: f32) -> Vec<f32> {
    let data = x.data();
    let n = data.len();
    let mut grad = vec![0.0f32; n];
    for i in 0..n {
        let mut d_plus = data.clone();
        d_plus[i] += eps;
        let mut d_minus = data.clone();
        d_minus[i] -= eps;
        let t_plus = Tensor::new(d_plus, x.shape(), false);
        let t_minus = Tensor::new(d_minus, x.shape(), false);
        let f_plus = f(&t_plus).data()[0];
        let f_minus = f(&t_minus).data()[0];
        grad[i] = (f_plus - f_minus) / (2.0 * eps);
    }
    grad
}

// ---- Part A: Composable Activations ----

#[test]
fn test_silu() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.silu();
    let data = y.data();
    // silu(x) = x * sigmoid(x)
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = xi * (1.0 / (1.0 + (-xi).exp()));
        assert!((data[i] - expected).abs() < 1e-5, "silu({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_silu_backward() {
    let x = Tensor::new(vec![-1.0, 0.5, 1.5], vec![3], true);
    let y = x.silu().sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.silu().sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

#[test]
fn test_softplus() {
    let x = Tensor::new(vec![-2.0, 0.0, 1.0, 5.0], vec![4], true);
    let y = x.softplus(1.0);
    let data = y.data();
    for (i, &xi) in [-2.0f32, 0.0, 1.0, 5.0].iter().enumerate() {
        let expected = (1.0 + xi.exp()).ln();
        assert!((data[i] - expected).abs() < 1e-4, "softplus({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_softplus_backward() {
    let x = Tensor::new(vec![-1.0, 0.5, 2.0], vec![3], true);
    let y = x.softplus(1.0).sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.softplus(1.0).sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

#[test]
fn test_softplus_beta() {
    let x = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3], false);
    let y = x.softplus(2.0);
    let data = y.data();
    for (i, &xi) in [-1.0f32, 0.0, 1.0].iter().enumerate() {
        let expected = (1.0 + (2.0 * xi).exp()).ln() / 2.0;
        assert!((data[i] - expected).abs() < 1e-4, "softplus(beta=2, {}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_mish() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.mish();
    let data = y.data();
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let softplus = (1.0 + xi.exp()).ln();
        let expected = xi * softplus.tanh();
        assert!((data[i] - expected).abs() < 1e-4, "mish({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_mish_backward() {
    let x = Tensor::new(vec![-1.0, 0.5, 1.5], vec![3], true);
    let y = x.mish().sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.mish().sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

#[test]
fn test_softsign() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5], false);
    let y = x.softsign();
    let data = y.data();
    for (i, &xi) in [-3.0f32, -1.0, 0.0, 1.0, 3.0].iter().enumerate() {
        let expected = xi / (1.0 + xi.abs());
        assert!((data[i] - expected).abs() < 1e-5, "softsign({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_log_sigmoid() {
    let x = Tensor::new(vec![-2.0, 0.0, 1.0, 5.0], vec![4], false);
    let y = x.log_sigmoid();
    let data = y.data();
    for (i, &xi) in [-2.0f32, 0.0, 1.0, 5.0].iter().enumerate() {
        let expected = (1.0 / (1.0 + (-xi).exp())).ln();
        assert!((data[i] - expected).abs() < 1e-4, "log_sigmoid({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_softmin() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
    let y = x.softmin();
    let data = y.data();
    // softmin = softmax(-x)
    let neg_x = x.neg();
    let expected = neg_x.softmax(-1).data();
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_relu6() {
    let x = Tensor::new(vec![-2.0, 0.0, 3.0, 6.0, 10.0], vec![5], true);
    let y = x.relu6();
    let data = y.data();
    let expected = vec![0.0, 0.0, 3.0, 6.0, 6.0];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_relu6_backward() {
    let x = Tensor::new(vec![-1.0, 0.5, 3.0, 7.0], vec![4], true);
    let y = x.relu6().sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.relu6().sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

#[test]
fn test_hardswish() {
    let x = Tensor::new(vec![-4.0, -3.0, 0.0, 3.0, 4.0], vec![5], true);
    let y = x.hardswish();
    let data = y.data();
    // hardswish(x) = x * relu6(x + 3) / 6
    for (i, &xi) in [-4.0f32, -3.0, 0.0, 3.0, 4.0].iter().enumerate() {
        let r6 = (xi + 3.0).max(0.0).min(6.0);
        let expected = xi * r6 / 6.0;
        assert!((data[i] - expected).abs() < 1e-4, "hardswish({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_hardswish_backward() {
    let x = Tensor::new(vec![-2.0, 0.0, 2.0], vec![3], true);
    let y = x.hardswish().sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.hardswish().sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

#[test]
fn test_gelu_fast_approx() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.gelu_fast_approx();
    let data = y.data();
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = (1.0 / (1.0 + (-1.702 * xi).exp())) * xi;
        assert!((data[i] - expected).abs() < 1e-4, "gelu_fast({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_gelu_fast_approx_backward() {
    let x = Tensor::new(vec![-1.0, 0.5, 1.5], vec![3], true);
    let y = x.gelu_fast_approx().sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.gelu_fast_approx().sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

// ---- Part B: Dedicated Kernel Activations ----

#[test]
fn test_leaky_relu() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.leaky_relu(0.01);
    let data = y.data();
    let expected = vec![-0.02, -0.01, 0.0, 1.0, 2.0];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_leaky_relu_backward() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.5, 1.0, 2.0], vec![5], true);
    let y = x.leaky_relu(0.1).sum();
    y.backward();
    let g = x.grad().unwrap();
    // grad should be 1.0 for positive, alpha for negative
    let expected = vec![0.1, 0.1, 1.0, 1.0, 1.0];
    approx_eq(&g, &expected, 1e-5);
}

#[test]
fn test_elu() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.elu(1.0);
    let data = y.data();
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = if xi > 0.0 { xi } else { xi.exp() - 1.0 };
        assert!((data[i] - expected).abs() < 1e-5, "elu({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_elu_backward() {
    let x = Tensor::new(vec![-2.0, -0.5, 0.5, 2.0], vec![4], true);
    let y = x.elu(1.0).sum();
    y.backward();
    let analytical = x.grad().unwrap();
    let numerical = numerical_grad(|t| t.elu(1.0).sum(), &x, 1e-4);
    approx_eq(&analytical, &numerical, 5e-3);
}

#[test]
fn test_elu_alpha() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0], vec![4], false);
    let y = x.elu(2.0);
    let data = y.data();
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0].iter().enumerate() {
        let expected = if xi > 0.0 { xi } else { 2.0 * (xi.exp() - 1.0) };
        assert!((data[i] - expected).abs() < 1e-5, "elu(alpha=2, {}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_hard_tanh() {
    let x = Tensor::new(vec![-3.0, -1.0, 0.0, 0.5, 1.0, 3.0], vec![6], true);
    let y = x.hard_tanh(-1.0, 1.0);
    let data = y.data();
    let expected = vec![-1.0, -1.0, 0.0, 0.5, 1.0, 1.0];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_hard_tanh_backward() {
    let x = Tensor::new(vec![-2.0, -0.5, 0.5, 2.0], vec![4], true);
    let y = x.hard_tanh(-1.0, 1.0).sum();
    y.backward();
    let g = x.grad().unwrap();
    // grad is 1.0 inside the range, 0.0 at boundaries
    let expected = vec![0.0, 1.0, 1.0, 0.0];
    approx_eq(&g, &expected, 1e-5);
}

#[test]
fn test_celu() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.celu(1.0);
    let data = y.data();
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = xi.max(0.0) + (xi.exp() - 1.0).min(0.0);
        assert!((data[i] - expected).abs() < 1e-4, "celu({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_selu() {
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], true);
    let y = x.selu();
    let data = y.data();
    let alpha: f32 = 1.6732632;
    let scale: f32 = 1.0507010;
    for (i, &xi) in [-2.0f32, -1.0, 0.0, 1.0, 2.0].iter().enumerate() {
        let expected = scale * (if xi > 0.0 { xi } else { alpha * (xi.exp() - 1.0) });
        assert!((data[i] - expected).abs() < 1e-4, "selu({}) = {} vs {}", xi, data[i], expected);
    }
}

#[test]
fn test_hard_shrink() {
    let x = Tensor::new(vec![-2.0, -0.3, 0.0, 0.3, 2.0], vec![5], true);
    let y = x.hard_shrink(0.5);
    let data = y.data();
    let expected = vec![-2.0, 0.0, 0.0, 0.0, 2.0];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_hard_shrink_backward() {
    let x = Tensor::new(vec![-2.0, -0.3, 0.3, 2.0], vec![4], true);
    let y = x.hard_shrink(0.5).sum();
    y.backward();
    let g = x.grad().unwrap();
    let expected = vec![1.0, 0.0, 0.0, 1.0];
    approx_eq(&g, &expected, 1e-5);
}

#[test]
fn test_soft_shrink() {
    let x = Tensor::new(vec![-2.0, -0.3, 0.0, 0.3, 2.0], vec![5], true);
    let y = x.soft_shrink(0.5);
    let data = y.data();
    let expected = vec![-1.5, 0.0, 0.0, 0.0, 1.5];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_soft_shrink_backward() {
    let x = Tensor::new(vec![-2.0, -0.3, 0.3, 2.0], vec![4], true);
    let y = x.soft_shrink(0.5).sum();
    y.backward();
    let g = x.grad().unwrap();
    let expected = vec![1.0, 0.0, 0.0, 1.0];
    approx_eq(&g, &expected, 1e-5);
}

#[test]
fn test_step() {
    let x = Tensor::new(vec![-2.0, -0.5, 0.0, 0.5, 2.0], vec![5], false);
    let y = x.step(0.0);
    let data = y.data();
    let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_glu() {
    // Input: [1, 4] -> split along dim -1 into [1, 2] and [1, 2]
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], false);
    let y = x.glu(-1);
    let data = y.data();
    assert_eq!(y.shape(), vec![1, 2]);
    // first_half = [1, 2], second_half = [3, 4]
    // result = first_half * sigmoid(second_half)
    let expected_0 = 1.0 * (1.0 / (1.0 + (-3.0f32).exp()));
    let expected_1 = 2.0 * (1.0 / (1.0 + (-4.0f32).exp()));
    assert!((data[0] - expected_0).abs() < 1e-5, "glu[0] = {} vs {}", data[0], expected_0);
    assert!((data[1] - expected_1).abs() < 1e-5, "glu[1] = {} vs {}", data[1], expected_1);
}

// ---- Part C: PReLU ----

#[test]
fn test_prelu_single_param() {
    let prelu = nn::PReLU::new(1);
    let x = Tensor::new(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], false);
    let y = prelu.forward(&x);
    let data = y.data();
    // weight = 0.25
    let expected = vec![-0.5, -0.25, 0.0, 1.0, 2.0];
    approx_eq(&data, &expected, 1e-5);
}

#[test]
fn test_prelu_multi_param() {
    let prelu = nn::PReLU::new(3);
    // [2, 3] input
    let x = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0], vec![2, 3], false);
    let y = prelu.forward(&x);
    let data = y.data();
    // weight = [0.25, 0.25, 0.25]
    // Row 0: [-0.25, 2.0, -0.75]
    // Row 1: [4.0, -1.25, 6.0]
    let expected = vec![-0.25, 2.0, -0.75, 4.0, -1.25, 6.0];
    approx_eq(&data, &expected, 1e-5);
}

// ---- Helper: ones_like ----

#[test]
fn test_ones_like() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let ones = Tensor::ones_like(&x);
    assert_eq!(ones.shape(), vec![2, 2]);
    assert_eq!(ones.data(), vec![1.0, 1.0, 1.0, 1.0]);
}
