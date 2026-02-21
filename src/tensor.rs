use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

use rayon::prelude::*;

const PAR_THRESHOLD: usize = 10_000;

// --- Apple Accelerate BLAS FFI ---

#[cfg(target_os = "macos")]
const CBLAS_ROW_MAJOR: i32 = 101;
#[cfg(target_os = "macos")]
const CBLAS_NO_TRANS: i32 = 111;
#[cfg(target_os = "macos")]
const CBLAS_TRANS: i32 = 112;

#[cfg(target_os = "macos")]
#[link(name = "Accelerate", kind = "framework")]
extern "C" {
    fn cblas_sgemm(
        order: i32, transA: i32, transB: i32,
        M: i32, N: i32, K: i32,
        alpha: f32, A: *const f32, lda: i32,
        B: *const f32, ldb: i32,
        beta: f32, C: *mut f32, ldc: i32,
    );
}

/// Safe wrapper around cblas_sgemm. Computes C = alpha * op(A) * op(B) + beta * C.
#[cfg(target_os = "macos")]
fn sgemm(
    trans_a: bool, trans_b: bool,
    m: usize, n: usize, k: usize,
    alpha: f32, a: &[f32], lda: usize,
    b: &[f32], ldb: usize,
    beta: f32, c: &mut [f32], ldc: usize,
) {
    unsafe {
        cblas_sgemm(
            CBLAS_ROW_MAJOR,
            if trans_a { CBLAS_TRANS } else { CBLAS_NO_TRANS },
            if trans_b { CBLAS_TRANS } else { CBLAS_NO_TRANS },
            m as i32, n as i32, k as i32,
            alpha, a.as_ptr(), lda as i32,
            b.as_ptr(), ldb as i32,
            beta, c.as_mut_ptr(), ldc as i32,
        );
    }
}

/// The operation that produced a tensor, used for backward pass.
#[derive(Clone)]
pub enum Op {
    None,
    Add(Tensor, Tensor),
    Mul(Tensor, Tensor),
    MatMul(Tensor, Tensor),
    Relu(Tensor),
    Sigmoid(Tensor),
    Sum(Tensor),
    Scale(Tensor, f32),
    AddBias(Tensor, Tensor), // [batch, features] + [1, features] broadcast
    Conv2d(Tensor, Tensor, Tensor), // input, kernel, bias
    MaxPool2d {
        input: Tensor,
        max_indices: Vec<usize>, // flat index into input for each output element
    },
    Flatten {
        input: Tensor,
        original_shape: Vec<usize>, // shape before flatten, for backward reshape
    },
    Select(Tensor, Vec<usize>),       // (input, indices into input's flat data)
    BceLoss(Tensor, Vec<f32>),        // (logits, constant targets)
}

/// Compute flat index for a 4D tensor with shape [N, C, H, W].
fn idx4(n: usize, c: usize, h: usize, w: usize, shape: &[usize]) -> usize {
    n * shape[1] * shape[2] * shape[3]
        + c * shape[2] * shape[3]
        + h * shape[3]
        + w
}

struct TensorInner {
    data: Vec<f32>,
    shape: Vec<usize>,
    grad: Option<Vec<f32>>,
    op: Op,
    requires_grad: bool,
}

#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorInner>>);

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();
        assert_eq!(data.len(), size, "data length must match shape");
        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            shape,
            grad: None,
            op: Op::None,
            requires_grad,
        })))
    }

    pub fn zeros(shape: &[usize], requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![0.0; size], shape.to_vec(), requires_grad)
    }

    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        // Simple LCG PRNG — good enough for a demo.
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = Cell::new(42);
        }
        let size: usize = shape.iter().product();
        let mut data = Vec::with_capacity(size);
        for _ in 0..size {
            // Box-Muller approximation via simple PRNG
            let u1 = SEED.with(|s| {
                let v = s.get().wrapping_mul(6364136223846793005).wrapping_add(1);
                s.set(v);
                (v >> 33) as f32 / (1u64 << 31) as f32
            });
            let u2 = SEED.with(|s| {
                let v = s.get().wrapping_mul(6364136223846793005).wrapping_add(1);
                s.set(v);
                (v >> 33) as f32 / (1u64 << 31) as f32
            });
            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            data.push(z * 0.1); // scale down for stability
        }
        Self::new(data, shape.to_vec(), requires_grad)
    }

    fn from_op(data: Vec<f32>, shape: Vec<usize>, op: Op) -> Self {
        Tensor(Rc::new(RefCell::new(TensorInner {
            data,
            shape,
            grad: None,
            op,
            requires_grad: true,
        })))
    }

    pub fn data(&self) -> Vec<f32> {
        self.0.borrow().data.clone()
    }

    pub fn shape(&self) -> Vec<usize> {
        self.0.borrow().shape.clone()
    }

    pub fn grad(&self) -> Option<Vec<f32>> {
        self.0.borrow().grad.clone()
    }

    pub fn size(&self) -> usize {
        self.0.borrow().data.len()
    }

    // ---- Forward ops ----

    pub fn add(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        assert_eq!(a.shape, b.shape, "shapes must match for add");
        let data: Vec<f32> = if a.data.len() >= PAR_THRESHOLD {
            a.data.par_iter().zip(&b.data).map(|(x, y)| x + y).collect()
        } else {
            a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect()
        };
        Tensor::from_op(data, a.shape.clone(), Op::Add(self.clone(), other.clone()))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        assert_eq!(a.shape, b.shape, "shapes must match for mul");
        let data: Vec<f32> = if a.data.len() >= PAR_THRESHOLD {
            a.data.par_iter().zip(&b.data).map(|(x, y)| x * y).collect()
        } else {
            a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect()
        };
        Tensor::from_op(data, a.shape.clone(), Op::Mul(self.clone(), other.clone()))
    }

    /// 2D matrix multiply: [M, K] x [K, N] -> [M, N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        let (m, k1) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        assert_eq!(k1, k2, "inner dimensions must match for matmul");
        let mut data = vec![0.0f32; m * n];
        #[cfg(target_os = "macos")]
        {
            sgemm(false, false, m, n, k1, 1.0, &a.data, k1, &b.data, n, 0.0, &mut data, n);
        }
        #[cfg(not(target_os = "macos"))]
        {
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for p in 0..k1 {
                        sum += a.data[i * k1 + p] * b.data[p * n + j];
                    }
                    data[i * n + j] = sum;
                }
            }
        }
        Tensor::from_op(data, vec![m, n], Op::MatMul(self.clone(), other.clone()))
    }

    pub fn relu(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.max(0.0)).collect()
        } else {
            inner.data.iter().map(|&x| x.max(0.0)).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Relu(self.clone()))
    }

    pub fn sigmoid(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
        } else {
            inner.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Sigmoid(self.clone()))
    }

    pub fn sum(&self) -> Tensor {
        let inner = self.0.borrow();
        let s: f32 = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().sum()
        } else {
            inner.data.iter().sum()
        };
        Tensor::from_op(vec![s], vec![1], Op::Sum(self.clone()))
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x * s).collect()
        } else {
            inner.data.iter().map(|&x| x * s).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Scale(self.clone(), s))
    }

    /// Broadcast add: self is [batch, features], bias is [1, features]
    pub fn add_bias(&self, bias: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = bias.0.borrow();
        assert_eq!(a.shape.len(), 2);
        assert_eq!(b.shape.len(), 2);
        assert_eq!(b.shape[0], 1);
        assert_eq!(a.shape[1], b.shape[1]);
        let cols = a.shape[1];
        let a_slice: &[f32] = &a.data;
        let b_slice: &[f32] = &b.data;
        let data: Vec<f32> = if a_slice.len() >= PAR_THRESHOLD {
            (0..a_slice.len())
                .into_par_iter()
                .map(|i| a_slice[i] + b_slice[i % cols])
                .collect()
        } else {
            a_slice
                .iter()
                .enumerate()
                .map(|(i, &x)| x + b_slice[i % cols])
                .collect()
        };
        Tensor::from_op(data, a.shape.clone(), Op::AddBias(self.clone(), bias.clone()))
    }

    /// 2D convolution: self=[N,Ci,H,W], kernel=[Co,Ci,kH,kW], bias=[Co]. Stride=1, no padding.
    pub fn conv2d(&self, kernel: &Tensor, bias: &Tensor) -> Tensor {
        let inp = self.0.borrow();
        let ker = kernel.0.borrow();
        let bi = bias.0.borrow();
        assert_eq!(inp.shape.len(), 4);
        assert_eq!(ker.shape.len(), 4);
        let (n, ci, h, w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
        let (co, ci_k, kh, kw) = (ker.shape[0], ker.shape[1], ker.shape[2], ker.shape[3]);
        assert_eq!(ci, ci_k);
        assert_eq!(bi.shape, vec![co]);
        let oh = h - kh + 1;
        let ow = w - kw + 1;
        let out_shape = vec![n, co, oh, ow];

        #[cfg(target_os = "macos")]
        if kh == 1 && kw == 1 {
            let spatial = h * w;
            let mut data = vec![0.0f32; n * co * spatial];
            for b in 0..n {
                sgemm(
                    false, false,
                    co, spatial, ci,
                    1.0,
                    &ker.data, ci,
                    &inp.data[b * ci * spatial..], spatial,
                    0.0,
                    &mut data[b * co * spatial..], spatial,
                );
            }
            let bias_vals: Vec<f32> = bi.data.clone();
            if oh * ow >= PAR_THRESHOLD {
                data.par_chunks_mut(spatial)
                    .enumerate()
                    .for_each(|(idx, chunk)| {
                        let oc = idx % co;
                        let bias_val = bias_vals[oc];
                        for v in chunk.iter_mut() {
                            *v += bias_val;
                        }
                    });
            } else {
                for b in 0..n {
                    for oc in 0..co {
                        let offset = b * co * spatial + oc * spatial;
                        let bias_val = bias_vals[oc];
                        for v in &mut data[offset..offset + spatial] {
                            *v += bias_val;
                        }
                    }
                }
            }
            return Tensor::from_op(
                data,
                out_shape,
                Op::Conv2d(self.clone(), kernel.clone(), bias.clone()),
            );
        }

        let mut data = vec![0.0f32; n * co * oh * ow];
        for b in 0..n {
            for oc in 0..co {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut sum = bi.data[oc];
                        for ic in 0..ci {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    sum += inp.data[idx4(b, ic, i + ki, j + kj, &inp.shape)]
                                        * ker.data[idx4(oc, ic, ki, kj, &ker.shape)];
                                }
                            }
                        }
                        data[idx4(b, oc, i, j, &out_shape)] = sum;
                    }
                }
            }
        }
        Tensor::from_op(
            data,
            out_shape,
            Op::Conv2d(self.clone(), kernel.clone(), bias.clone()),
        )
    }

    /// 2x2 max pooling with stride 2.
    pub fn max_pool2d(&self) -> Tensor {
        let inp = self.0.borrow();
        assert_eq!(inp.shape.len(), 4);
        let (n, c, h, w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
        assert!(h % 2 == 0 && w % 2 == 0, "H and W must be divisible by 2");
        let oh = h / 2;
        let ow = w / 2;
        let out_shape = vec![n, c, oh, ow];
        let out_size = n * c * oh * ow;
        let mut data = vec![0.0f32; out_size];
        let mut max_indices = vec![0usize; out_size];
        for b in 0..n {
            for ch in 0..c {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut best_val = f32::NEG_INFINITY;
                        let mut best_idx = 0;
                        for di in 0..2 {
                            for dj in 0..2 {
                                let fi = idx4(b, ch, i * 2 + di, j * 2 + dj, &inp.shape);
                                if inp.data[fi] > best_val {
                                    best_val = inp.data[fi];
                                    best_idx = fi;
                                }
                            }
                        }
                        let oi = idx4(b, ch, i, j, &out_shape);
                        data[oi] = best_val;
                        max_indices[oi] = best_idx;
                    }
                }
            }
        }
        Tensor::from_op(data, out_shape, Op::MaxPool2d {
            input: self.clone(),
            max_indices,
        })
    }

    /// Flatten [N, C, H, W] -> [N, C*H*W].
    pub fn flatten(&self) -> Tensor {
        let inp = self.0.borrow();
        assert!(inp.shape.len() >= 2, "need at least 2 dims to flatten");
        let n = inp.shape[0];
        let rest: usize = inp.shape[1..].iter().product();
        let original_shape = inp.shape.clone();
        Tensor::from_op(
            inp.data.clone(),
            vec![n, rest],
            Op::Flatten { input: self.clone(), original_shape },
        )
    }

    /// Gather elements by flat index, preserving autograd connectivity.
    /// out[i] = self.data[indices[i]]
    pub fn select(&self, indices: &[usize]) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = indices.iter().map(|&i| inner.data[i]).collect();
        let n = indices.len();
        Tensor::from_op(data, vec![n], Op::Select(self.clone(), indices.to_vec()))
    }

    /// BCE with logits loss, reduced to scalar mean. Targets are constants.
    /// Numerically stable: bce(x,t) = max(x,0) - x*t + ln(1 + exp(-|x|))
    pub fn bce_with_logits_loss(&self, targets: &[f32]) -> Tensor {
        let inner = self.0.borrow();
        let n = inner.data.len();
        assert_eq!(n, targets.len(), "predictions and targets must have same length");
        let loss: f32 = inner.data.iter().zip(targets).map(|(&x, &t)| {
            x.max(0.0) - x * t + (1.0 + (-x.abs()).exp()).ln()
        }).sum::<f32>() / n as f32;
        Tensor::from_op(vec![loss], vec![1], Op::BceLoss(self.clone(), targets.to_vec()))
    }

    // ---- Backward (reverse-mode autodiff) ----

    fn tensor_id(&self) -> usize {
        Rc::as_ptr(&self.0) as usize
    }

    /// Build topological order via post-order DFS.
    fn build_topo(
        &self,
        order: &mut Vec<Tensor>,
        visited: &mut std::collections::HashSet<usize>,
    ) {
        let id = self.tensor_id();
        if !visited.insert(id) {
            return;
        }
        let inputs = {
            let op = self.0.borrow().op.clone();
            match op {
                Op::None => vec![],
                Op::Add(a, b) | Op::Mul(a, b) | Op::MatMul(a, b)
                | Op::AddBias(a, b) => vec![a, b],
                Op::Relu(a) | Op::Sigmoid(a) | Op::Sum(a)
                | Op::Scale(a, _) | Op::Select(a, _) | Op::BceLoss(a, _) => vec![a],
                Op::Conv2d(a, b, c) => vec![a, b, c],
                Op::MaxPool2d { input, .. } | Op::Flatten { input, .. } => vec![input],
            }
        };
        for input in &inputs {
            input.build_topo(order, visited);
        }
        order.push(self.clone());
    }

    pub fn backward(&self) {
        let size = self.size();
        self.0.borrow_mut().grad = Some(vec![1.0; size]);

        // Topological sort: process each node exactly once, from output to inputs.
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited = std::collections::HashSet::new();
        self.build_topo(&mut order, &mut visited);

        for node in order.iter().rev() {
            node.propagate_grad();
        }
    }

    /// Propagate this node's gradient to its op inputs. Non-recursive.
    fn propagate_grad(&self) {
        let (op, grad) = {
            let mut inner = self.0.borrow_mut();
            if matches!(inner.op, Op::None) {
                return; // leaf node — preserve grad for sgd_step
            }
            let grad = match inner.grad.take() {
                Some(g) => g,
                None => return,
            };
            (std::mem::replace(&mut inner.op, Op::None), grad)
        };

        match op {
            Op::None => unreachable!(),
            Op::Add(ref a, ref b) => {
                accumulate_grad(a, &grad);
                accumulate_grad(b, &grad);
            }
            Op::Mul(ref a, ref b) => {
                let a_inner = a.0.borrow();
                let b_inner = b.0.borrow();
                let (grad_a, grad_b) = if grad.len() >= PAR_THRESHOLD {
                    let ga: Vec<f32> = grad.par_iter().zip(b_inner.data.par_iter()).map(|(g, bv)| g * bv).collect();
                    let gb: Vec<f32> = grad.par_iter().zip(a_inner.data.par_iter()).map(|(g, av)| g * av).collect();
                    (ga, gb)
                } else {
                    let ga: Vec<f32> = grad.iter().zip(&b_inner.data).map(|(g, bv)| g * bv).collect();
                    let gb: Vec<f32> = grad.iter().zip(&a_inner.data).map(|(g, av)| g * av).collect();
                    (ga, gb)
                };
                drop(a_inner);
                drop(b_inner);
                accumulate_grad(a, &grad_a);
                accumulate_grad(b, &grad_b);
            }
            Op::MatMul(ref a, ref b) => {
                let a_inner = a.0.borrow();
                let b_inner = b.0.borrow();
                let (m, k) = (a_inner.shape[0], a_inner.shape[1]);
                let n = b_inner.shape[1];

                #[cfg(target_os = "macos")]
                let (grad_a, grad_b) = {
                    let mut ga = vec![0.0f32; m * k];
                    sgemm(false, true, m, k, n, 1.0, &grad, n, &b_inner.data, n, 0.0, &mut ga, k);
                    let mut gb = vec![0.0f32; k * n];
                    sgemm(true, false, k, n, m, 1.0, &a_inner.data, k, &grad, n, 0.0, &mut gb, n);
                    (ga, gb)
                };

                #[cfg(not(target_os = "macos"))]
                let (grad_a, grad_b) = {
                    let mut ga = vec![0.0f32; m * k];
                    for i in 0..m {
                        for j in 0..k {
                            let mut sum = 0.0;
                            for p in 0..n {
                                sum += grad[i * n + p] * b_inner.data[j * n + p];
                            }
                            ga[i * k + j] = sum;
                        }
                    }
                    let mut gb = vec![0.0f32; k * n];
                    for i in 0..k {
                        for j in 0..n {
                            let mut sum = 0.0;
                            for p in 0..m {
                                sum += a_inner.data[p * k + i] * grad[p * n + j];
                            }
                            gb[i * n + j] = sum;
                        }
                    }
                    (ga, gb)
                };

                drop(a_inner);
                drop(b_inner);
                accumulate_grad(a, &grad_a);
                accumulate_grad(b, &grad_b);
            }
            Op::Relu(ref input) => {
                let in_inner = input.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter()
                        .zip(in_inner.data.par_iter())
                        .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 })
                        .collect()
                } else {
                    grad.iter()
                        .zip(&in_inner.data)
                        .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 })
                        .collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Sigmoid(ref input) => {
                let self_inner = self.0.borrow();
                let out_data = &self_inner.data;
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter()
                        .zip(out_data.par_iter())
                        .map(|(g, &s)| g * s * (1.0 - s))
                        .collect()
                } else {
                    grad.iter()
                        .zip(out_data)
                        .map(|(g, &s)| g * s * (1.0 - s))
                        .collect()
                };
                drop(self_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Sum(ref input) => {
                let n = input.size();
                let grad_input = vec![grad[0]; n];
                accumulate_grad(input, &grad_input);
            }
            Op::Scale(ref input, s) => {
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().map(|g| g * s).collect()
                } else {
                    grad.iter().map(|g| g * s).collect()
                };
                accumulate_grad(input, &grad_input);
            }
            Op::AddBias(ref input, ref bias) => {
                let bias_inner = bias.0.borrow();
                let cols = bias_inner.shape[1];
                let rows = input.0.borrow().shape[0];
                drop(bias_inner);
                let mut grad_bias = vec![0.0f32; cols];
                for r in 0..rows {
                    for c in 0..cols {
                        grad_bias[c] += grad[r * cols + c];
                    }
                }
                accumulate_grad(input, &grad);
                accumulate_grad(bias, &grad_bias);
            }
            Op::Conv2d(ref input, ref kernel, ref bias) => {
                let in_inner = input.0.borrow();
                let k_inner = kernel.0.borrow();
                let (n, ci, h, w) = (in_inner.shape[0], in_inner.shape[1], in_inner.shape[2], in_inner.shape[3]);
                let (co, _, kh, kw) = (k_inner.shape[0], k_inner.shape[1], k_inner.shape[2], k_inner.shape[3]);
                let oh = h - kh + 1;
                let ow = w - kw + 1;
                let out_shape = vec![n, co, oh, ow];
                let in_shape = &in_inner.shape;
                let k_shape = &k_inner.shape;

                let mut grad_bias = vec![0.0f32; co];
                for b in 0..n {
                    for oc in 0..co {
                        for i in 0..oh {
                            for j in 0..ow {
                                grad_bias[oc] += grad[idx4(b, oc, i, j, &out_shape)];
                            }
                        }
                    }
                }

                #[cfg(target_os = "macos")]
                let use_blas_1x1 = kh == 1 && kw == 1;
                #[cfg(not(target_os = "macos"))]
                let use_blas_1x1 = false;

                if use_blas_1x1 {
                    #[cfg(target_os = "macos")]
                    {
                        let hw = oh * ow;
                        let mut grad_kernel = vec![0.0f32; co * ci];
                        for b in 0..n {
                            sgemm(
                                false, true, co, ci, hw,
                                1.0, &grad[b * co * hw..], hw,
                                &in_inner.data[b * ci * hw..], hw,
                                1.0, &mut grad_kernel, ci,
                            );
                        }
                        let mut grad_input = vec![0.0f32; n * ci * h * w];
                        for b in 0..n {
                            sgemm(
                                true, false, ci, hw, co,
                                1.0, &k_inner.data, ci,
                                &grad[b * co * hw..], hw,
                                0.0, &mut grad_input[b * ci * hw..], hw,
                            );
                        }
                        drop(in_inner);
                        drop(k_inner);
                        accumulate_grad(input, &grad_input);
                        accumulate_grad(kernel, &grad_kernel);
                        accumulate_grad(bias, &grad_bias);
                    }
                } else {
                    let mut grad_kernel = vec![0.0f32; co * ci * kh * kw];
                    for b in 0..n {
                        for oc in 0..co {
                            for ic in 0..ci {
                                for ki in 0..kh {
                                    for kj in 0..kw {
                                        let mut s = 0.0;
                                        for i in 0..oh {
                                            for j in 0..ow {
                                                s += grad[idx4(b, oc, i, j, &out_shape)]
                                                    * in_inner.data[idx4(b, ic, i + ki, j + kj, in_shape)];
                                            }
                                        }
                                        grad_kernel[idx4(oc, ic, ki, kj, k_shape)] += s;
                                    }
                                }
                            }
                        }
                    }

                    let mut grad_input = vec![0.0f32; n * ci * h * w];
                    for b in 0..n {
                        for ic in 0..ci {
                            for ih in 0..h {
                                for iw in 0..w {
                                    let mut s = 0.0;
                                    for oc in 0..co {
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                let oi = ih as isize - ki as isize;
                                                let oj = iw as isize - kj as isize;
                                                if oi >= 0 && oi < oh as isize
                                                    && oj >= 0 && oj < ow as isize
                                                {
                                                    s += grad[idx4(b, oc, oi as usize, oj as usize, &out_shape)]
                                                        * k_inner.data[idx4(oc, ic, ki, kj, k_shape)];
                                                }
                                            }
                                        }
                                    }
                                    grad_input[idx4(b, ic, ih, iw, in_shape)] += s;
                                }
                            }
                        }
                    }

                    drop(in_inner);
                    drop(k_inner);
                    accumulate_grad(input, &grad_input);
                    accumulate_grad(kernel, &grad_kernel);
                    accumulate_grad(bias, &grad_bias);
                }
            }
            Op::MaxPool2d { ref input, ref max_indices } => {
                let in_size = input.size();
                let mut grad_input = vec![0.0f32; in_size];
                for (out_idx, &in_idx) in max_indices.iter().enumerate() {
                    grad_input[in_idx] += grad[out_idx];
                }
                accumulate_grad(input, &grad_input);
            }
            Op::Flatten { ref input, .. } => {
                accumulate_grad(input, &grad);
            }
            Op::Select(ref input, ref indices) => {
                let in_size = input.size();
                let mut grad_input = vec![0.0f32; in_size];
                for (out_i, &in_i) in indices.iter().enumerate() {
                    grad_input[in_i] += grad[out_i];
                }
                accumulate_grad(input, &grad_input);
            }
            Op::BceLoss(ref logits, ref targets) => {
                let logit_inner = logits.0.borrow();
                let len = logit_inner.data.len() as f32;
                let grad_logits: Vec<f32> = if logit_inner.data.len() >= PAR_THRESHOLD {
                    logit_inner.data.par_iter().zip(targets.par_iter()).map(|(&x, &t)| {
                        let sig = 1.0 / (1.0 + (-x).exp());
                        grad[0] * (sig - t) / len
                    }).collect()
                } else {
                    logit_inner.data.iter().zip(targets).map(|(&x, &t)| {
                        let sig = 1.0 / (1.0 + (-x).exp());
                        grad[0] * (sig - t) / len
                    }).collect()
                };
                drop(logit_inner);
                accumulate_grad(logits, &grad_logits);
            }
        }
    }

    /// Simple SGD update: param -= lr * grad
    pub fn sgd_step(&self, lr: f32) {
        let mut inner = self.0.borrow_mut();
        if let Some(g) = inner.grad.take() {
            for (d, g) in inner.data.iter_mut().zip(&g) {
                *d -= lr * g;
            }
        }
        inner.op = Op::None;
    }
}

fn accumulate_grad(tensor: &Tensor, grad: &[f32]) {
    let mut inner = tensor.0.borrow_mut();
    if !inner.requires_grad {
        return;
    }
    match inner.grad {
        Some(ref mut existing) => {
            for (e, g) in existing.iter_mut().zip(grad) {
                *e += g;
            }
        }
        None => {
            inner.grad = Some(grad.to_vec());
        }
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let inner = self.0.borrow();
        write!(f, "Tensor(shape={:?}, data={:?})", inner.shape, inner.data)
    }
}
