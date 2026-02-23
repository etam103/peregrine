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
    Conv2dReluPool {
        input: Tensor,
        kernel: Tensor,
        bias: Tensor,
        pre_relu_data: Vec<f32>, // conv output before relu, for relu backward
        max_indices: Vec<usize>, // flat index into conv output for pool backward
    },
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
    Reshape {
        input: Tensor,
        original_shape: Vec<usize>,
    },
    Transpose {
        input: Tensor,
        dim0: usize,
        dim1: usize,
    },
    Softmax {
        input: Tensor,
        output_data: Vec<f32>,
        dim: usize,
    },
    LayerNorm {
        input: Tensor,
        gamma: Tensor,
        beta: Tensor,
        normalized: Vec<f32>,
        inv_std: Vec<f32>,
    },
    Gelu(Tensor),
    Sub(Tensor, Tensor),
    Div(Tensor, Tensor),
    Neg(Tensor),
    Exp(Tensor),
    Log(Tensor),
    Sqrt(Tensor),
    Abs(Tensor),
    Pow(Tensor, f32),   // tensor raised to scalar power
    Sin(Tensor),
    Cos(Tensor),
    Tanh(Tensor),
    Mean(Tensor),
    Squeeze {
        input: Tensor,
        original_shape: Vec<usize>,
    },
    Unsqueeze {
        input: Tensor,
        original_shape: Vec<usize>,
    },
    Concat {
        inputs: Vec<Tensor>,
        split_sizes: Vec<usize>,
        dim: usize,
    },
    BatchNorm {
        input: Tensor,
        gamma: Tensor,
        beta: Tensor,
        normalized: Vec<f32>,
        mean: Vec<f32>,
        inv_std: Vec<f32>,
    },
}

/// Compute flat index for a 4D tensor with shape [N, C, H, W].
fn idx4(n: usize, c: usize, h: usize, w: usize, shape: &[usize]) -> usize {
    n * shape[1] * shape[2] * shape[3]
        + c * shape[2] * shape[3]
        + h * shape[3]
        + w
}

// --- Broadcasting helpers ---

/// Compute the broadcast output shape for two input shapes (NumPy rules).
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let ndim = a.len().max(b.len());
    let mut result = vec![0usize; ndim];
    for i in 0..ndim {
        let da = if i < ndim - a.len() { 1 } else { a[i - (ndim - a.len())] };
        let db = if i < ndim - b.len() { 1 } else { b[i - (ndim - b.len())] };
        assert!(
            da == db || da == 1 || db == 1,
            "shapes {:?} and {:?} not broadcastable", a, b
        );
        result[i] = da.max(db);
    }
    result
}

/// Compute virtual strides for a tensor shape within a broadcast output shape.
/// Broadcast dimensions (size 1 in input, >1 in output) get stride 0.
fn broadcast_strides(shape: &[usize], target_ndim: usize) -> Vec<usize> {
    let offset = target_ndim - shape.len();
    let mut strides = vec![0usize; target_ndim];
    let mut stride = 1usize;
    for i in (0..shape.len()).rev() {
        if shape[i] > 1 {
            strides[i + offset] = stride;
        }
        stride *= shape[i];
    }
    strides
}

/// Reduce a gradient from broadcast shape back to the original tensor shape
/// by summing over broadcast dimensions.
fn reduce_grad_for_broadcast(grad: &[f32], grad_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    if grad_shape == target_shape {
        return grad.to_vec();
    }
    let ndim = grad_shape.len();
    let offset = ndim - target_shape.len();

    let mut padded = vec![1usize; ndim];
    for i in 0..target_shape.len() {
        padded[i + offset] = target_shape[i];
    }

    let target_size: usize = target_shape.iter().product();
    let mut result = vec![0.0f32; target_size];

    let mut grad_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        grad_strides[i] = grad_strides[i + 1] * grad_shape[i + 1];
    }

    let mut target_strides_full = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        target_strides_full[i] = target_strides_full[i + 1] * padded[i + 1];
    }

    let total: usize = grad_shape.iter().product();
    for flat in 0..total {
        let mut target_flat = 0;
        let mut rem = flat;
        for d in 0..ndim {
            let coord = rem / grad_strides[d];
            rem %= grad_strides[d];
            let target_coord = if padded[d] == 1 { 0 } else { coord };
            target_flat += target_coord * target_strides_full[d];
        }
        result[target_flat] += grad[flat];
    }
    result
}

/// Apply a binary op element-wise with NumPy broadcasting.
fn broadcast_binary_op(
    a_data: &[f32], a_shape: &[usize],
    b_data: &[f32], b_shape: &[usize],
    op: impl Fn(f32, f32) -> f32,
) -> (Vec<f32>, Vec<usize>) {
    let out_shape = broadcast_shape(a_shape, b_shape);
    let ndim = out_shape.len();
    let a_strides = broadcast_strides(a_shape, ndim);
    let b_strides = broadcast_strides(b_shape, ndim);

    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }

    let total: usize = out_shape.iter().product();
    let mut data = vec![0.0f32; total];
    for flat in 0..total {
        let mut a_idx = 0;
        let mut b_idx = 0;
        let mut rem = flat;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            a_idx += coord * a_strides[d];
            b_idx += coord * b_strides[d];
        }
        data[flat] = op(a_data[a_idx], b_data[b_idx]);
    }
    (data, out_shape)
}

/// Extract element values from a tensor using broadcast strides (for backward pass).
fn broadcast_gather(data: &[f32], shape: &[usize], out_shape: &[usize]) -> Vec<f32> {
    let ndim = out_shape.len();
    let strides = broadcast_strides(shape, ndim);
    let mut out_strides = vec![1usize; ndim];
    for i in (0..ndim.saturating_sub(1)).rev() {
        out_strides[i] = out_strides[i + 1] * out_shape[i + 1];
    }
    let total: usize = out_shape.iter().product();
    let mut result = vec![0.0f32; total];
    for flat in 0..total {
        let mut idx = 0;
        let mut rem = flat;
        for d in 0..ndim {
            let coord = rem / out_strides[d];
            rem %= out_strides[d];
            idx += coord * strides[d];
        }
        result[flat] = data[idx];
    }
    result
}

/// Extract image patches into columns for convolution (same-padding).
/// Input: flat slice of one batch element [Ci, H, W].
/// Output: column matrix [Ci*kH*kW, H*W].
fn im2col(
    input: &[f32], ci: usize, h: usize, w: usize,
    kh: usize, kw: usize,
) -> Vec<f32> {
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let col_rows = ci * kh * kw;
    let col_cols = h * w;
    let mut col = vec![0.0f32; col_rows * col_cols];
    for c in 0..ci {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = c * kh * kw + ki * kw + kj;
                for i in 0..h {
                    let ih = i as isize + ki as isize - pad_h as isize;
                    if ih < 0 || ih >= h as isize { continue; }
                    for j in 0..w {
                        let iw = j as isize + kj as isize - pad_w as isize;
                        if iw < 0 || iw >= w as isize { continue; }
                        col[row * col_cols + i * w + j] =
                            input[c * h * w + ih as usize * w + iw as usize];
                    }
                }
            }
        }
    }
    col
}

/// Scatter column gradients back to input spatial format (adjoint of im2col).
/// grad_col: [Ci*kH*kW, H*W], output: [Ci, H, W].
fn col2im(
    grad_col: &[f32], ci: usize, h: usize, w: usize,
    kh: usize, kw: usize,
) -> Vec<f32> {
    let pad_h = kh / 2;
    let pad_w = kw / 2;
    let col_cols = h * w;
    let mut grad_input = vec![0.0f32; ci * h * w];
    for c in 0..ci {
        for ki in 0..kh {
            for kj in 0..kw {
                let row = c * kh * kw + ki * kw + kj;
                for i in 0..h {
                    let ih = i as isize + ki as isize - pad_h as isize;
                    if ih < 0 || ih >= h as isize { continue; }
                    for j in 0..w {
                        let iw = j as isize + kj as isize - pad_w as isize;
                        if iw < 0 || iw >= w as isize { continue; }
                        grad_input[c * h * w + ih as usize * w + iw as usize]
                            += grad_col[row * col_cols + i * w + j];
                    }
                }
            }
        }
    }
    grad_input
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

    pub fn ones(shape: &[usize], requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![1.0; size], shape.to_vec(), requires_grad)
    }

    pub fn full(shape: &[usize], value: f32, requires_grad: bool) -> Self {
        let size: usize = shape.iter().product();
        Self::new(vec![value; size], shape.to_vec(), requires_grad)
    }

    /// Create a 1D tensor with values [0, 1, ..., n-1].
    pub fn arange(n: usize, requires_grad: bool) -> Self {
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        Self::new(data, vec![n], requires_grad)
    }

    /// Create a 1D tensor with `steps` evenly spaced values from `start` to `end` (inclusive).
    pub fn linspace(start: f32, end: f32, steps: usize, requires_grad: bool) -> Self {
        let data: Vec<f32> = if steps <= 1 {
            vec![start]
        } else {
            (0..steps)
                .map(|i| start + (end - start) * i as f32 / (steps - 1) as f32)
                .collect()
        };
        Self::new(data, vec![steps], requires_grad)
    }

    /// Create an n x n identity matrix.
    pub fn eye(n: usize, requires_grad: bool) -> Self {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Self::new(data, vec![n, n], requires_grad)
    }

    pub fn randn(shape: &[usize], requires_grad: bool) -> Self {
        // Simple LCG PRNG — good enough for a demo.
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = Cell::new(42);
        }
        let size: usize = shape.iter().product();
        // Xavier fan-in initialization: std = sqrt(1 / fan_in)
        let fan_in = match shape.len() {
            2 => shape[1],                              // linear: [out, in]
            4 => shape[1] * shape[2] * shape[3],        // conv: [out, in, kH, kW]
            _ => shape.iter().skip(1).product::<usize>().max(1),
        };
        let init_scale = (1.0 / fan_in as f32).sqrt();
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
            data.push(z * init_scale);
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
        if a.shape == b.shape {
            let data: Vec<f32> = if a.data.len() >= PAR_THRESHOLD {
                a.data.par_iter().zip(&b.data).map(|(x, y)| x + y).collect()
            } else {
                a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect()
            };
            Tensor::from_op(data, a.shape.clone(), Op::Add(self.clone(), other.clone()))
        } else {
            let (data, out_shape) = broadcast_binary_op(&a.data, &a.shape, &b.data, &b.shape, |x, y| x + y);
            Tensor::from_op(data, out_shape, Op::Add(self.clone(), other.clone()))
        }
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        if a.shape == b.shape {
            let data: Vec<f32> = if a.data.len() >= PAR_THRESHOLD {
                a.data.par_iter().zip(&b.data).map(|(x, y)| x * y).collect()
            } else {
                a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect()
            };
            Tensor::from_op(data, a.shape.clone(), Op::Mul(self.clone(), other.clone()))
        } else {
            let (data, out_shape) = broadcast_binary_op(&a.data, &a.shape, &b.data, &b.shape, |x, y| x * y);
            Tensor::from_op(data, out_shape, Op::Mul(self.clone(), other.clone()))
        }
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

    /// 2D convolution with same-padding: self=[N,Ci,H,W], kernel=[Co,Ci,kH,kW], bias=[Co].
    /// Stride=1, pad=kH/2 so output spatial dims equal input spatial dims.
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
        let spatial = h * w;
        let out_shape = vec![n, co, h, w];

        #[cfg(target_os = "macos")]
        if kh == 1 && kw == 1 {
            let mut data = vec![0.0f32; n * co * spatial];
            for b in 0..n {
                sgemm(
                    false, false, co, spatial, ci, 1.0,
                    &ker.data, ci,
                    &inp.data[b * ci * spatial..], spatial,
                    0.0, &mut data[b * co * spatial..], spatial,
                );
            }
            let bias_vals: Vec<f32> = bi.data.clone();
            for b in 0..n {
                for oc in 0..co {
                    let offset = b * co * spatial + oc * spatial;
                    let bv = bias_vals[oc];
                    for v in &mut data[offset..offset + spatial] {
                        *v += bv;
                    }
                }
            }
            return Tensor::from_op(
                data, out_shape,
                Op::Conv2d(self.clone(), kernel.clone(), bias.clone()),
            );
        }

        // General path: im2col + matmul
        let col_rows = ci * kh * kw;
        let mut data = vec![0.0f32; n * co * spatial];
        for b in 0..n {
            let col = im2col(&inp.data[b * ci * h * w..], ci, h, w, kh, kw);
            #[cfg(target_os = "macos")]
            sgemm(
                false, false, co, spatial, col_rows, 1.0,
                &ker.data, col_rows, &col, spatial,
                0.0, &mut data[b * co * spatial..], spatial,
            );
            #[cfg(not(target_os = "macos"))]
            {
                for oc in 0..co {
                    for s in 0..spatial {
                        let mut sum = 0.0;
                        for k in 0..col_rows {
                            sum += ker.data[oc * col_rows + k] * col[k * spatial + s];
                        }
                        data[b * co * spatial + oc * spatial + s] = sum;
                    }
                }
            }
        }
        // Add bias
        let bias_vals: Vec<f32> = bi.data.clone();
        for b in 0..n {
            for oc in 0..co {
                let offset = b * co * spatial + oc * spatial;
                let bv = bias_vals[oc];
                for v in &mut data[offset..offset + spatial] {
                    *v += bv;
                }
            }
        }
        Tensor::from_op(
            data, out_shape,
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
        let plane_out = oh * ow;
        let out_size = n * c * plane_out;
        let mut data = vec![0.0f32; out_size];
        let mut max_indices = vec![0usize; out_size];

        let in_data = &inp.data;
        let in_plane = h * w;

        if out_size >= PAR_THRESHOLD {
            data.par_chunks_mut(plane_out)
                .zip(max_indices.par_chunks_mut(plane_out))
                .enumerate()
                .for_each(|(bc, (out_chunk, idx_chunk))| {
                    let in_base = bc * in_plane;
                    for i in 0..oh {
                        for j in 0..ow {
                            let mut best_val = f32::NEG_INFINITY;
                            let mut best_idx = 0;
                            for di in 0..2 {
                                let fi = in_base + (i * 2 + di) * w + j * 2;
                                if in_data[fi] > best_val {
                                    best_val = in_data[fi];
                                    best_idx = fi;
                                }
                                if in_data[fi + 1] > best_val {
                                    best_val = in_data[fi + 1];
                                    best_idx = fi + 1;
                                }
                            }
                            let oi = i * ow + j;
                            out_chunk[oi] = best_val;
                            idx_chunk[oi] = best_idx;
                        }
                    }
                });
        } else {
            for bc in 0..(n * c) {
                let in_base = bc * in_plane;
                let out_base = bc * plane_out;
                for i in 0..oh {
                    for j in 0..ow {
                        let mut best_val = f32::NEG_INFINITY;
                        let mut best_idx = 0;
                        for di in 0..2 {
                            let fi = in_base + (i * 2 + di) * w + j * 2;
                            if in_data[fi] > best_val {
                                best_val = in_data[fi];
                                best_idx = fi;
                            }
                            if in_data[fi + 1] > best_val {
                                best_val = in_data[fi + 1];
                                best_idx = fi + 1;
                            }
                        }
                        let oi = out_base + i * ow + j;
                        data[oi] = best_val;
                        max_indices[oi] = best_idx;
                    }
                }
            }
        }

        let out_shape = vec![n, c, oh, ow];
        Tensor::from_op(data, out_shape, Op::MaxPool2d {
            input: self.clone(),
            max_indices,
        })
    }

    /// Fused conv2d (same-padding) + ReLU + 2x2 max pool in a single pass.
    pub fn conv2d_relu_pool(&self, kernel: &Tensor, bias: &Tensor) -> Tensor {
        let inp = self.0.borrow();
        let ker = kernel.0.borrow();
        let bi = bias.0.borrow();
        assert_eq!(inp.shape.len(), 4);
        assert_eq!(ker.shape.len(), 4);
        let (n, ci, h, w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
        let (co, ci_k, kh, kw) = (ker.shape[0], ker.shape[1], ker.shape[2], ker.shape[3]);
        assert_eq!(ci, ci_k);
        assert_eq!(bi.shape, vec![co]);
        assert!(h % 2 == 0 && w % 2 == 0, "H and W must be divisible by 2");

        // Same-padding preserves spatial dims
        let spatial = h * w;
        let oh = h / 2;
        let ow = w / 2;
        let plane_out = oh * ow;

        // Step 1: Conv (1x1 BLAS or im2col) + bias
        let mut conv_out = vec![0.0f32; n * co * spatial];
        if kh == 1 && kw == 1 {
            #[cfg(target_os = "macos")]
            for b in 0..n {
                sgemm(
                    false, false, co, spatial, ci, 1.0,
                    &ker.data, ci,
                    &inp.data[b * ci * spatial..], spatial,
                    0.0, &mut conv_out[b * co * spatial..], spatial,
                );
            }
            #[cfg(not(target_os = "macos"))]
            {
                for b in 0..n {
                    for oc in 0..co {
                        for s in 0..spatial {
                            let mut sum = 0.0;
                            for ic in 0..ci {
                                sum += ker.data[oc * ci + ic]
                                    * inp.data[b * ci * spatial + ic * spatial + s];
                            }
                            conv_out[b * co * spatial + oc * spatial + s] = sum;
                        }
                    }
                }
            }
        } else {
            let col_rows = ci * kh * kw;
            for b in 0..n {
                let col = im2col(&inp.data[b * ci * h * w..], ci, h, w, kh, kw);
                #[cfg(target_os = "macos")]
                sgemm(
                    false, false, co, spatial, col_rows, 1.0,
                    &ker.data, col_rows, &col, spatial,
                    0.0, &mut conv_out[b * co * spatial..], spatial,
                );
                #[cfg(not(target_os = "macos"))]
                {
                    for oc in 0..co {
                        for s in 0..spatial {
                            let mut sum = 0.0;
                            for k in 0..col_rows {
                                sum += ker.data[oc * col_rows + k] * col[k * spatial + s];
                            }
                            conv_out[b * co * spatial + oc * spatial + s] = sum;
                        }
                    }
                }
            }
        }

        // Add bias
        let bias_vals = &bi.data;
        for bc in 0..(n * co) {
            let oc = bc % co;
            let bv = bias_vals[oc];
            let offset = bc * spatial;
            for v in &mut conv_out[offset..offset + spatial] {
                *v += bv;
            }
        }

        drop(inp);
        drop(ker);
        drop(bi);

        // Step 2: Fused relu + maxpool
        let out_size = n * co * plane_out;
        let mut data = vec![0.0f32; out_size];
        let mut max_indices = vec![0usize; out_size];

        for bc in 0..(n * co) {
            let in_base = bc * spatial;
            let out_base = bc * plane_out;
            for i in 0..oh {
                for j in 0..ow {
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_idx = 0;
                    for di in 0..2 {
                        let fi = in_base + (i * 2 + di) * w + j * 2;
                        let v0 = conv_out[fi].max(0.0);
                        if v0 > best_val {
                            best_val = v0;
                            best_idx = fi;
                        }
                        let v1 = conv_out[fi + 1].max(0.0);
                        if v1 > best_val {
                            best_val = v1;
                            best_idx = fi + 1;
                        }
                    }
                    let oi = out_base + i * ow + j;
                    data[oi] = best_val;
                    max_indices[oi] = best_idx;
                }
            }
        }

        let out_shape = vec![n, co, oh, ow];
        Tensor::from_op(data, out_shape, Op::Conv2dReluPool {
            input: self.clone(),
            kernel: kernel.clone(),
            bias: bias.clone(),
            pre_relu_data: conv_out,
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

    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let inner = self.0.borrow();
        let new_size: usize = new_shape.iter().product();
        assert_eq!(
            inner.data.len(), new_size,
            "reshape: total elements must match ({} vs {})", inner.data.len(), new_size
        );
        let original_shape = inner.shape.clone();
        Tensor::from_op(
            inner.data.clone(),
            new_shape,
            Op::Reshape { input: self.clone(), original_shape },
        )
    }

    pub fn transpose(&self, dim0: usize, dim1: usize) -> Tensor {
        let inner = self.0.borrow();
        let ndim = inner.shape.len();
        assert!(dim0 < ndim && dim1 < ndim, "transpose dims out of range");
        if dim0 == dim1 {
            return Tensor::from_op(
                inner.data.clone(),
                inner.shape.clone(),
                Op::Transpose { input: self.clone(), dim0, dim1 },
            );
        }

        let mut new_shape = inner.shape.clone();
        new_shape.swap(dim0, dim1);

        let mut strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            strides[i] = strides[i + 1] * inner.shape[i + 1];
        }

        let mut new_strides = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            new_strides[i] = new_strides[i + 1] * new_shape[i + 1];
        }

        let total: usize = inner.data.len();
        let mut data = vec![0.0f32; total];

        for flat in 0..total {
            let mut coords = vec![0usize; ndim];
            let mut rem = flat;
            for d in 0..ndim {
                coords[d] = rem / strides[d];
                rem %= strides[d];
            }
            coords.swap(dim0, dim1);
            let mut dst = 0;
            for d in 0..ndim {
                dst += coords[d] * new_strides[d];
            }
            data[dst] = inner.data[flat];
        }

        Tensor::from_op(
            data,
            new_shape,
            Op::Transpose { input: self.clone(), dim0, dim1 },
        )
    }

    pub fn softmax(&self, dim: isize) -> Tensor {
        let inner = self.0.borrow();
        let ndim = inner.shape.len();
        let dim = if dim < 0 { (ndim as isize + dim) as usize } else { dim as usize };
        assert!(dim < ndim, "softmax dim out of range");

        let outer: usize = inner.shape[..dim].iter().product();
        let dim_size = inner.shape[dim];
        let inner_size: usize = inner.shape[dim + 1..].iter().product();

        let mut data = vec![0.0f32; inner.data.len()];

        for o in 0..outer {
            for i in 0..inner_size {
                let base = o * dim_size * inner_size + i;
                let mut max_val = f32::NEG_INFINITY;
                for d in 0..dim_size {
                    let idx = base + d * inner_size;
                    if inner.data[idx] > max_val {
                        max_val = inner.data[idx];
                    }
                }
                let mut sum_exp = 0.0f32;
                for d in 0..dim_size {
                    let idx = base + d * inner_size;
                    let e = (inner.data[idx] - max_val).exp();
                    data[idx] = e;
                    sum_exp += e;
                }
                for d in 0..dim_size {
                    let idx = base + d * inner_size;
                    data[idx] /= sum_exp;
                }
            }
        }

        let output_data = data.clone();
        Tensor::from_op(
            data,
            inner.shape.clone(),
            Op::Softmax { input: self.clone(), output_data, dim },
        )
    }

    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor, normalized_shape: usize) -> Tensor {
        let inner = self.0.borrow();
        let g = gamma.0.borrow();
        let b = beta.0.borrow();
        assert_eq!(g.data.len(), normalized_shape);
        assert_eq!(b.data.len(), normalized_shape);
        let total = inner.data.len();
        assert!(total % normalized_shape == 0);
        let num_instances = total / normalized_shape;
        let eps = 1e-5f32;

        let mut normalized = vec![0.0f32; total];
        let mut inv_std = vec![0.0f32; num_instances];
        let mut data = vec![0.0f32; total];

        for inst in 0..num_instances {
            let offset = inst * normalized_shape;
            let slice = &inner.data[offset..offset + normalized_shape];

            let mean: f32 = slice.iter().sum::<f32>() / normalized_shape as f32;
            let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>()
                / normalized_shape as f32;
            let istd = 1.0 / (var + eps).sqrt();
            inv_std[inst] = istd;

            for j in 0..normalized_shape {
                let norm_val = (slice[j] - mean) * istd;
                normalized[offset + j] = norm_val;
                data[offset + j] = norm_val * g.data[j] + b.data[j];
            }
        }

        Tensor::from_op(
            data,
            inner.shape.clone(),
            Op::LayerNorm {
                input: self.clone(),
                gamma: gamma.clone(),
                beta: beta.clone(),
                normalized,
                inv_std,
            },
        )
    }

    pub fn gelu(&self) -> Tensor {
        let inner = self.0.borrow();
        let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| {
                let inner_val = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner_val.tanh())
            }).collect()
        } else {
            inner.data.iter().map(|&x| {
                let inner_val = sqrt_2_over_pi * (x + 0.044715 * x * x * x);
                0.5 * x * (1.0 + inner_val.tanh())
            }).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Gelu(self.clone()))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        if a.shape == b.shape {
            let data: Vec<f32> = if a.data.len() >= PAR_THRESHOLD {
                a.data.par_iter().zip(&b.data).map(|(x, y)| x - y).collect()
            } else {
                a.data.iter().zip(&b.data).map(|(x, y)| x - y).collect()
            };
            Tensor::from_op(data, a.shape.clone(), Op::Sub(self.clone(), other.clone()))
        } else {
            let (data, out_shape) = broadcast_binary_op(&a.data, &a.shape, &b.data, &b.shape, |x, y| x - y);
            Tensor::from_op(data, out_shape, Op::Sub(self.clone(), other.clone()))
        }
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        if a.shape == b.shape {
            let data: Vec<f32> = if a.data.len() >= PAR_THRESHOLD {
                a.data.par_iter().zip(&b.data).map(|(x, y)| x / y).collect()
            } else {
                a.data.iter().zip(&b.data).map(|(x, y)| x / y).collect()
            };
            Tensor::from_op(data, a.shape.clone(), Op::Div(self.clone(), other.clone()))
        } else {
            let (data, out_shape) = broadcast_binary_op(&a.data, &a.shape, &b.data, &b.shape, |x, y| x / y);
            Tensor::from_op(data, out_shape, Op::Div(self.clone(), other.clone()))
        }
    }

    pub fn neg(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| -x).collect()
        } else {
            inner.data.iter().map(|&x| -x).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Neg(self.clone()))
    }

    pub fn exp(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.exp()).collect()
        } else {
            inner.data.iter().map(|&x| x.exp()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Exp(self.clone()))
    }

    pub fn log(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.ln()).collect()
        } else {
            inner.data.iter().map(|&x| x.ln()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Log(self.clone()))
    }

    pub fn sqrt(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.sqrt()).collect()
        } else {
            inner.data.iter().map(|&x| x.sqrt()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Sqrt(self.clone()))
    }

    pub fn abs(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.abs()).collect()
        } else {
            inner.data.iter().map(|&x| x.abs()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Abs(self.clone()))
    }

    /// Raise each element to a scalar power: x^p
    pub fn pow(&self, p: f32) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.powf(p)).collect()
        } else {
            inner.data.iter().map(|&x| x.powf(p)).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Pow(self.clone(), p))
    }

    pub fn sin(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.sin()).collect()
        } else {
            inner.data.iter().map(|&x| x.sin()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Sin(self.clone()))
    }

    pub fn cos(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.cos()).collect()
        } else {
            inner.data.iter().map(|&x| x.cos()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Cos(self.clone()))
    }

    pub fn tanh(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = if inner.data.len() >= PAR_THRESHOLD {
            inner.data.par_iter().map(|&x| x.tanh()).collect()
        } else {
            inner.data.iter().map(|&x| x.tanh()).collect()
        };
        Tensor::from_op(data, inner.shape.clone(), Op::Tanh(self.clone()))
    }

    /// Scalar mean of all elements (differentiable).
    pub fn mean(&self) -> Tensor {
        let inner = self.0.borrow();
        let n = inner.data.len();
        let s: f32 = if n >= PAR_THRESHOLD {
            inner.data.par_iter().sum()
        } else {
            inner.data.iter().sum()
        };
        Tensor::from_op(vec![s / n as f32], vec![1], Op::Mean(self.clone()))
    }

    /// Remove dimensions of size 1. If `dim` is None, removes all size-1 dims.
    pub fn squeeze(&self, dim: Option<usize>) -> Tensor {
        let inner = self.0.borrow();
        let original_shape = inner.shape.clone();
        let new_shape: Vec<usize> = match dim {
            Some(d) => {
                assert!(d < original_shape.len(), "squeeze dim out of range");
                assert_eq!(original_shape[d], 1, "can only squeeze dims of size 1");
                original_shape.iter().enumerate()
                    .filter(|&(i, _)| i != d)
                    .map(|(_, &s)| s)
                    .collect()
            }
            None => original_shape.iter().copied().filter(|&s| s != 1).collect(),
        };
        let new_shape = if new_shape.is_empty() { vec![1] } else { new_shape };
        Tensor::from_op(
            inner.data.clone(),
            new_shape,
            Op::Squeeze { input: self.clone(), original_shape },
        )
    }

    /// Insert a dimension of size 1 at position `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Tensor {
        let inner = self.0.borrow();
        let original_shape = inner.shape.clone();
        assert!(dim <= original_shape.len(), "unsqueeze dim out of range");
        let mut new_shape = original_shape.clone();
        new_shape.insert(dim, 1);
        Tensor::from_op(
            inner.data.clone(),
            new_shape,
            Op::Unsqueeze { input: self.clone(), original_shape },
        )
    }

    /// Returns the maximum value as a scalar (not differentiable).
    pub fn max(&self) -> f32 {
        let inner = self.0.borrow();
        inner.data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    }

    /// Returns the minimum value as a scalar (not differentiable).
    pub fn min(&self) -> f32 {
        let inner = self.0.borrow();
        inner.data.iter().copied().fold(f32::INFINITY, f32::min)
    }

    /// Returns the index of the maximum value in the flat data.
    pub fn argmax(&self) -> usize {
        let inner = self.0.borrow();
        inner.data.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Returns the index of the minimum value in the flat data.
    pub fn argmin(&self) -> usize {
        let inner = self.0.borrow();
        inner.data.iter().enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    pub fn concat(tensors: &[&Tensor], dim: usize) -> Tensor {
        assert!(!tensors.is_empty(), "concat requires at least one tensor");
        let first_shape = tensors[0].shape();
        let ndim = first_shape.len();
        assert!(dim < ndim, "concat dim out of range");
        for t in &tensors[1..] {
            let s = t.shape();
            assert_eq!(s.len(), ndim, "concat: all tensors must have same ndim");
            for d in 0..ndim {
                if d != dim {
                    assert_eq!(s[d], first_shape[d], "concat: non-concat dims must match");
                }
            }
        }

        let mut out_shape = first_shape.clone();
        let split_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[dim]).collect();
        out_shape[dim] = split_sizes.iter().sum();

        let outer: usize = out_shape[..dim].iter().product();
        let inner_size: usize = out_shape[dim + 1..].iter().product();
        let total_dim = out_shape[dim];

        let mut data = vec![0.0f32; outer * total_dim * inner_size];

        for o in 0..outer {
            let mut dim_offset = 0;
            for t in tensors {
                let t_inner = t.0.borrow();
                let t_dim = t_inner.shape[dim];
                for d in 0..t_dim {
                    let src_base = o * t_dim * inner_size + d * inner_size;
                    let dst_base = o * total_dim * inner_size + (dim_offset + d) * inner_size;
                    data[dst_base..dst_base + inner_size]
                        .copy_from_slice(&t_inner.data[src_base..src_base + inner_size]);
                }
                dim_offset += t_dim;
            }
        }

        let inputs: Vec<Tensor> = tensors.iter().map(|t| (*t).clone()).collect();
        Tensor::from_op(data, out_shape, Op::Concat { inputs, split_sizes, dim })
    }

    /// Batch normalization: self=[N,C,H,W], gamma=[C], beta=[C].
    pub fn batch_norm(&self, gamma: &Tensor, beta: &Tensor) -> Tensor {
        let inp = self.0.borrow();
        let g = gamma.0.borrow();
        let b = beta.0.borrow();
        assert_eq!(inp.shape.len(), 4);
        let (n, c, h, w) = (inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]);
        let spatial = n * h * w;
        let eps = 1e-5f32;
        assert_eq!(g.data.len(), c);
        assert_eq!(b.data.len(), c);

        let mut mean = vec![0.0f32; c];
        let mut variance = vec![0.0f32; c];
        for ni in 0..n {
            for ci in 0..c {
                for p in 0..(h * w) {
                    mean[ci] += inp.data[ni * c * h * w + ci * h * w + p];
                }
            }
        }
        for ci in 0..c { mean[ci] /= spatial as f32; }

        for ni in 0..n {
            for ci in 0..c {
                for p in 0..(h * w) {
                    let diff = inp.data[ni * c * h * w + ci * h * w + p] - mean[ci];
                    variance[ci] += diff * diff;
                }
            }
        }
        for ci in 0..c { variance[ci] /= spatial as f32; }

        let inv_std: Vec<f32> = variance.iter().map(|v| 1.0 / (v + eps).sqrt()).collect();
        let total = n * c * h * w;
        let mut normalized = vec![0.0f32; total];
        let mut data = vec![0.0f32; total];

        for ni in 0..n {
            for ci in 0..c {
                for p in 0..(h * w) {
                    let idx = ni * c * h * w + ci * h * w + p;
                    let x_hat = (inp.data[idx] - mean[ci]) * inv_std[ci];
                    normalized[idx] = x_hat;
                    data[idx] = x_hat * g.data[ci] + b.data[ci];
                }
            }
        }

        Tensor::from_op(
            data, vec![n, c, h, w],
            Op::BatchNorm {
                input: self.clone(), gamma: gamma.clone(), beta: beta.clone(),
                normalized, mean, inv_std,
            },
        )
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
                | Op::AddBias(a, b) | Op::Sub(a, b) | Op::Div(a, b) => vec![a, b],
                Op::Relu(a) | Op::Sigmoid(a) | Op::Sum(a)
                | Op::Scale(a, _) | Op::Select(a, _) | Op::BceLoss(a, _)
                | Op::Neg(a) | Op::Exp(a) | Op::Log(a) | Op::Sqrt(a)
                | Op::Abs(a) | Op::Pow(a, _) | Op::Sin(a) | Op::Cos(a)
                | Op::Tanh(a) | Op::Mean(a) => vec![a],
                Op::Conv2d(a, b, c) => vec![a, b, c],
                Op::Conv2dReluPool { input, kernel, bias, .. } => vec![input, kernel, bias],
                Op::MaxPool2d { input, .. } | Op::Flatten { input, .. }
                | Op::Squeeze { input, .. } | Op::Unsqueeze { input, .. }
                | Op::Reshape { input, .. } | Op::Transpose { input, .. }
                | Op::Gelu(input) => vec![input],
                Op::Softmax { input, .. } => vec![input],
                Op::LayerNorm { input, gamma, beta, .. }
                | Op::BatchNorm { input, gamma, beta, .. } => vec![input, gamma, beta],
                Op::Concat { inputs, .. } => inputs,
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
                let a_shape = a.shape();
                let b_shape = b.shape();
                if a_shape == b_shape {
                    accumulate_grad(a, &grad);
                    accumulate_grad(b, &grad);
                } else {
                    let out_shape = broadcast_shape(&a_shape, &b_shape);
                    accumulate_grad(a, &reduce_grad_for_broadcast(&grad, &out_shape, &a_shape));
                    accumulate_grad(b, &reduce_grad_for_broadcast(&grad, &out_shape, &b_shape));
                }
            }
            Op::Mul(ref a, ref b) => {
                let a_inner = a.0.borrow();
                let b_inner = b.0.borrow();
                if a_inner.shape == b_inner.shape {
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
                } else {
                    let a_shape = a_inner.shape.clone();
                    let b_shape = b_inner.shape.clone();
                    let out_shape = broadcast_shape(&a_shape, &b_shape);
                    let b_expanded = broadcast_gather(&b_inner.data, &b_shape, &out_shape);
                    let a_expanded = broadcast_gather(&a_inner.data, &a_shape, &out_shape);
                    drop(a_inner);
                    drop(b_inner);
                    let grad_a_full: Vec<f32> = grad.iter().zip(&b_expanded).map(|(g, bv)| g * bv).collect();
                    let grad_b_full: Vec<f32> = grad.iter().zip(&a_expanded).map(|(g, av)| g * av).collect();
                    accumulate_grad(a, &reduce_grad_for_broadcast(&grad_a_full, &out_shape, &a_shape));
                    accumulate_grad(b, &reduce_grad_for_broadcast(&grad_b_full, &out_shape, &b_shape));
                }
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
                let spatial = h * w;
                let col_rows = ci * kh * kw;

                // grad_bias: sum over batch and spatial
                let mut grad_bias = vec![0.0f32; co];
                for b in 0..n {
                    for oc in 0..co {
                        let offset = b * co * spatial + oc * spatial;
                        for s in 0..spatial {
                            grad_bias[oc] += grad[offset + s];
                        }
                    }
                }

                if kh == 1 && kw == 1 {
                    // 1x1 fast path
                    let mut grad_kernel = vec![0.0f32; co * ci];
                    let mut grad_input = vec![0.0f32; n * ci * h * w];
                    #[cfg(target_os = "macos")]
                    for b in 0..n {
                        sgemm(
                            false, true, co, ci, spatial,
                            1.0, &grad[b * co * spatial..], spatial,
                            &in_inner.data[b * ci * spatial..], spatial,
                            1.0, &mut grad_kernel, ci,
                        );
                        sgemm(
                            true, false, ci, spatial, co,
                            1.0, &k_inner.data, ci,
                            &grad[b * co * spatial..], spatial,
                            0.0, &mut grad_input[b * ci * spatial..], spatial,
                        );
                    }
                    #[cfg(not(target_os = "macos"))]
                    for b in 0..n {
                        for oc in 0..co {
                            for ic in 0..ci {
                                let mut s = 0.0;
                                for p in 0..spatial {
                                    s += grad[b * co * spatial + oc * spatial + p]
                                        * in_inner.data[b * ci * spatial + ic * spatial + p];
                                }
                                grad_kernel[oc * ci + ic] += s;
                            }
                        }
                        for ic in 0..ci {
                            for p in 0..spatial {
                                let mut s = 0.0;
                                for oc in 0..co {
                                    s += k_inner.data[oc * ci + ic]
                                        * grad[b * co * spatial + oc * spatial + p];
                                }
                                grad_input[b * ci * spatial + ic * spatial + p] = s;
                            }
                        }
                    }
                    drop(in_inner);
                    drop(k_inner);
                    accumulate_grad(input, &grad_input);
                    accumulate_grad(kernel, &grad_kernel);
                    accumulate_grad(bias, &grad_bias);
                } else {
                    // General im2col-based backward
                    let mut grad_kernel = vec![0.0f32; co * col_rows];
                    let mut grad_input = vec![0.0f32; n * ci * h * w];
                    for b in 0..n {
                        let col = im2col(&in_inner.data[b * ci * h * w..], ci, h, w, kh, kw);
                        // grad_kernel += grad_out[b] @ col^T
                        #[cfg(target_os = "macos")]
                        sgemm(
                            false, true, co, col_rows, spatial,
                            1.0, &grad[b * co * spatial..], spatial,
                            &col, spatial,
                            1.0, &mut grad_kernel, col_rows,
                        );
                        #[cfg(not(target_os = "macos"))]
                        for oc in 0..co {
                            for k in 0..col_rows {
                                let mut s = 0.0;
                                for s_idx in 0..spatial {
                                    s += grad[b * co * spatial + oc * spatial + s_idx]
                                        * col[k * spatial + s_idx];
                                }
                                grad_kernel[oc * col_rows + k] += s;
                            }
                        }
                        // grad_col = kernel^T @ grad_out[b]
                        let mut grad_col = vec![0.0f32; col_rows * spatial];
                        #[cfg(target_os = "macos")]
                        sgemm(
                            true, false, col_rows, spatial, co,
                            1.0, &k_inner.data, col_rows,
                            &grad[b * co * spatial..], spatial,
                            0.0, &mut grad_col, spatial,
                        );
                        #[cfg(not(target_os = "macos"))]
                        for k in 0..col_rows {
                            for s in 0..spatial {
                                let mut sum = 0.0;
                                for oc in 0..co {
                                    sum += k_inner.data[oc * col_rows + k]
                                        * grad[b * co * spatial + oc * spatial + s];
                                }
                                grad_col[k * spatial + s] = sum;
                            }
                        }
                        let gi = col2im(&grad_col, ci, h, w, kh, kw);
                        for i in 0..(ci * h * w) {
                            grad_input[b * ci * h * w + i] += gi[i];
                        }
                    }
                    drop(in_inner);
                    drop(k_inner);
                    accumulate_grad(input, &grad_input);
                    accumulate_grad(kernel, &grad_kernel);
                    accumulate_grad(bias, &grad_bias);
                }
            }
            Op::Conv2dReluPool { ref input, ref kernel, ref bias, ref pre_relu_data, ref max_indices } => {
                let in_inner = input.0.borrow();
                let k_inner = kernel.0.borrow();
                let (n, ci, h, w) = (in_inner.shape[0], in_inner.shape[1], in_inner.shape[2], in_inner.shape[3]);
                let (co, _, kh, kw) = (k_inner.shape[0], k_inner.shape[1], k_inner.shape[2], k_inner.shape[3]);
                let spatial = h * w;
                let col_rows = ci * kh * kw;
                let oh = h / 2;
                let pool_out_size = n * co * oh * (w / 2);
                let conv_size = n * co * spatial;

                // 1. Pool backward
                let mut grad_conv_relu = vec![0.0f32; conv_size];
                for oi in 0..pool_out_size {
                    grad_conv_relu[max_indices[oi]] += grad[oi];
                }
                // 2. ReLU backward
                for i in 0..conv_size {
                    if pre_relu_data[i] <= 0.0 { grad_conv_relu[i] = 0.0; }
                }
                let grad_conv = grad_conv_relu;

                // 3. Bias gradient
                let mut grad_bias = vec![0.0f32; co];
                for b in 0..n {
                    for oc in 0..co {
                        let offset = b * co * spatial + oc * spatial;
                        for s in 0..spatial { grad_bias[oc] += grad_conv[offset + s]; }
                    }
                }

                // 4. Conv backward
                if kh == 1 && kw == 1 {
                    let mut grad_kernel = vec![0.0f32; co * ci];
                    let mut grad_input = vec![0.0f32; n * ci * h * w];
                    #[cfg(target_os = "macos")]
                    for b in 0..n {
                        sgemm(false, true, co, ci, spatial, 1.0,
                            &grad_conv[b * co * spatial..], spatial,
                            &in_inner.data[b * ci * spatial..], spatial,
                            1.0, &mut grad_kernel, ci);
                        sgemm(true, false, ci, spatial, co, 1.0,
                            &k_inner.data, ci,
                            &grad_conv[b * co * spatial..], spatial,
                            0.0, &mut grad_input[b * ci * spatial..], spatial);
                    }
                    #[cfg(not(target_os = "macos"))]
                    for b in 0..n {
                        for oc in 0..co { for ic in 0..ci {
                            let mut s = 0.0;
                            for p in 0..spatial {
                                s += grad_conv[b * co * spatial + oc * spatial + p]
                                    * in_inner.data[b * ci * spatial + ic * spatial + p];
                            }
                            grad_kernel[oc * ci + ic] += s;
                        }}
                        for ic in 0..ci { for p in 0..spatial {
                            let mut s = 0.0;
                            for oc in 0..co {
                                s += k_inner.data[oc * ci + ic]
                                    * grad_conv[b * co * spatial + oc * spatial + p];
                            }
                            grad_input[b * ci * spatial + ic * spatial + p] = s;
                        }}
                    }
                    drop(in_inner); drop(k_inner);
                    accumulate_grad(input, &grad_input);
                    accumulate_grad(kernel, &grad_kernel);
                    accumulate_grad(bias, &grad_bias);
                } else {
                    let mut grad_kernel = vec![0.0f32; co * col_rows];
                    let mut grad_input = vec![0.0f32; n * ci * h * w];
                    for b in 0..n {
                        let col = im2col(&in_inner.data[b * ci * h * w..], ci, h, w, kh, kw);
                        #[cfg(target_os = "macos")]
                        {
                            sgemm(false, true, co, col_rows, spatial, 1.0,
                                &grad_conv[b * co * spatial..], spatial,
                                &col, spatial, 1.0, &mut grad_kernel, col_rows);
                            let mut grad_col = vec![0.0f32; col_rows * spatial];
                            sgemm(true, false, col_rows, spatial, co, 1.0,
                                &k_inner.data, col_rows,
                                &grad_conv[b * co * spatial..], spatial,
                                0.0, &mut grad_col, spatial);
                            let gi = col2im(&grad_col, ci, h, w, kh, kw);
                            for i in 0..(ci * h * w) { grad_input[b * ci * h * w + i] += gi[i]; }
                        }
                        #[cfg(not(target_os = "macos"))]
                        {
                            for oc in 0..co { for k in 0..col_rows {
                                let mut s = 0.0;
                                for s_idx in 0..spatial {
                                    s += grad_conv[b * co * spatial + oc * spatial + s_idx]
                                        * col[k * spatial + s_idx];
                                }
                                grad_kernel[oc * col_rows + k] += s;
                            }}
                            let mut grad_col = vec![0.0f32; col_rows * spatial];
                            for k in 0..col_rows { for s in 0..spatial {
                                let mut sum = 0.0;
                                for oc in 0..co {
                                    sum += k_inner.data[oc * col_rows + k]
                                        * grad_conv[b * co * spatial + oc * spatial + s];
                                }
                                grad_col[k * spatial + s] = sum;
                            }}
                            let gi = col2im(&grad_col, ci, h, w, kh, kw);
                            for i in 0..(ci * h * w) { grad_input[b * ci * h * w + i] += gi[i]; }
                        }
                    }
                    drop(in_inner); drop(k_inner);
                    accumulate_grad(input, &grad_input);
                    accumulate_grad(kernel, &grad_kernel);
                    accumulate_grad(bias, &grad_bias);
                }
            }
            Op::MaxPool2d { ref input, ref max_indices } => {
                let in_inner = input.0.borrow();
                let (h, w) = (in_inner.shape[2], in_inner.shape[3]);
                let in_size = in_inner.data.len();
                let in_plane = h * w;
                let plane_out = (h / 2) * (w / 2);
                drop(in_inner);

                let mut grad_input = vec![0.0f32; in_size];

                if in_size >= PAR_THRESHOLD {
                    grad_input.par_chunks_mut(in_plane)
                        .enumerate()
                        .for_each(|(bc, gi_chunk)| {
                            let out_base = bc * plane_out;
                            for oi in 0..plane_out {
                                let in_idx = max_indices[out_base + oi];
                                gi_chunk[in_idx - bc * in_plane] += grad[out_base + oi];
                            }
                        });
                } else {
                    for (out_idx, &in_idx) in max_indices.iter().enumerate() {
                        grad_input[in_idx] += grad[out_idx];
                    }
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
            Op::Reshape { ref input, .. } => {
                accumulate_grad(input, &grad);
            }
            Op::Transpose { ref input, dim0, dim1 } => {
                if dim0 == dim1 {
                    accumulate_grad(input, &grad);
                } else {
                    let in_shape = input.shape();
                    let ndim = in_shape.len();
                    let mut strides = vec![1usize; ndim];
                    for i in (0..ndim - 1).rev() {
                        strides[i] = strides[i + 1] * in_shape[i + 1];
                    }
                    let mut t_shape = in_shape.clone();
                    t_shape.swap(dim0, dim1);
                    let mut t_strides = vec![1usize; ndim];
                    for i in (0..ndim - 1).rev() {
                        t_strides[i] = t_strides[i + 1] * t_shape[i + 1];
                    }

                    let total = grad.len();
                    let mut grad_input = vec![0.0f32; total];
                    for flat in 0..total {
                        let mut coords = vec![0usize; ndim];
                        let mut rem = flat;
                        for d in 0..ndim {
                            coords[d] = rem / t_strides[d];
                            rem %= t_strides[d];
                        }
                        coords.swap(dim0, dim1);
                        let mut src = 0;
                        for d in 0..ndim {
                            src += coords[d] * strides[d];
                        }
                        grad_input[src] = grad[flat];
                    }
                    accumulate_grad(input, &grad_input);
                }
            }
            Op::Softmax { ref input, ref output_data, dim } => {
                let in_shape = input.shape();
                let outer: usize = in_shape[..dim].iter().product();
                let dim_size = in_shape[dim];
                let inner_size: usize = in_shape[dim + 1..].iter().product();

                let mut grad_input = vec![0.0f32; grad.len()];

                for o in 0..outer {
                    for i in 0..inner_size {
                        let base = o * dim_size * inner_size + i;
                        let mut dot = 0.0f32;
                        for d in 0..dim_size {
                            let idx = base + d * inner_size;
                            dot += grad[idx] * output_data[idx];
                        }
                        for d in 0..dim_size {
                            let idx = base + d * inner_size;
                            grad_input[idx] = output_data[idx] * (grad[idx] - dot);
                        }
                    }
                }
                accumulate_grad(input, &grad_input);
            }
            Op::LayerNorm { ref input, ref gamma, ref beta, ref normalized, ref inv_std } => {
                let g_inner = gamma.0.borrow();
                let normalized_shape = g_inner.data.len();
                let gamma_data = g_inner.data.clone();
                drop(g_inner);

                let total = grad.len();
                let num_instances = total / normalized_shape;
                let mut grad_gamma = vec![0.0f32; normalized_shape];
                let mut grad_beta = vec![0.0f32; normalized_shape];
                let mut grad_input = vec![0.0f32; total];

                for inst in 0..num_instances {
                    let offset = inst * normalized_shape;
                    let istd = inv_std[inst];

                    for j in 0..normalized_shape {
                        let idx = offset + j;
                        grad_gamma[j] += grad[idx] * normalized[idx];
                        grad_beta[j] += grad[idx];
                    }

                    let mut mean_dy = 0.0f32;
                    let mut mean_dy_xhat = 0.0f32;
                    for j in 0..normalized_shape {
                        let idx = offset + j;
                        let dy = grad[idx] * gamma_data[j];
                        mean_dy += dy;
                        mean_dy_xhat += dy * normalized[idx];
                    }
                    mean_dy /= normalized_shape as f32;
                    mean_dy_xhat /= normalized_shape as f32;

                    for j in 0..normalized_shape {
                        let idx = offset + j;
                        let dy = grad[idx] * gamma_data[j];
                        grad_input[idx] = istd * (dy - mean_dy - normalized[idx] * mean_dy_xhat);
                    }
                }

                accumulate_grad(input, &grad_input);
                accumulate_grad(gamma, &grad_gamma);
                accumulate_grad(beta, &grad_beta);
            }
            Op::Sub(ref a, ref b) => {
                let a_shape = a.shape();
                let b_shape = b.shape();
                let neg_grad: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().map(|g| -g).collect()
                } else {
                    grad.iter().map(|g| -g).collect()
                };
                if a_shape == b_shape {
                    accumulate_grad(a, &grad);
                    accumulate_grad(b, &neg_grad);
                } else {
                    let out_shape = broadcast_shape(&a_shape, &b_shape);
                    accumulate_grad(a, &reduce_grad_for_broadcast(&grad, &out_shape, &a_shape));
                    accumulate_grad(b, &reduce_grad_for_broadcast(&neg_grad, &out_shape, &b_shape));
                }
            }
            Op::Div(ref a, ref b) => {
                let a_inner = a.0.borrow();
                let b_inner = b.0.borrow();
                if a_inner.shape == b_inner.shape {
                    let (grad_a, grad_b) = if grad.len() >= PAR_THRESHOLD {
                        let ga: Vec<f32> = grad.par_iter().zip(b_inner.data.par_iter())
                            .map(|(g, bv)| g / bv).collect();
                        let gb: Vec<f32> = grad.par_iter().zip(a_inner.data.par_iter().zip(b_inner.data.par_iter()))
                            .map(|(g, (av, bv))| -g * av / (bv * bv)).collect();
                        (ga, gb)
                    } else {
                        let ga: Vec<f32> = grad.iter().zip(&b_inner.data)
                            .map(|(g, bv)| g / bv).collect();
                        let gb: Vec<f32> = grad.iter().zip(a_inner.data.iter().zip(&b_inner.data))
                            .map(|(g, (av, bv))| -g * av / (bv * bv)).collect();
                        (ga, gb)
                    };
                    drop(a_inner);
                    drop(b_inner);
                    accumulate_grad(a, &grad_a);
                    accumulate_grad(b, &grad_b);
                } else {
                    let a_shape = a_inner.shape.clone();
                    let b_shape = b_inner.shape.clone();
                    let out_shape = broadcast_shape(&a_shape, &b_shape);
                    let a_expanded = broadcast_gather(&a_inner.data, &a_shape, &out_shape);
                    let b_expanded = broadcast_gather(&b_inner.data, &b_shape, &out_shape);
                    drop(a_inner);
                    drop(b_inner);
                    let grad_a_full: Vec<f32> = grad.iter().zip(&b_expanded).map(|(g, bv)| g / bv).collect();
                    let grad_b_full: Vec<f32> = grad.iter().zip(a_expanded.iter().zip(&b_expanded))
                        .map(|(g, (av, bv))| -g * av / (bv * bv)).collect();
                    accumulate_grad(a, &reduce_grad_for_broadcast(&grad_a_full, &out_shape, &a_shape));
                    accumulate_grad(b, &reduce_grad_for_broadcast(&grad_b_full, &out_shape, &b_shape));
                }
            }
            Op::Neg(ref input) => {
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().map(|g| -g).collect()
                } else {
                    grad.iter().map(|g| -g).collect()
                };
                accumulate_grad(input, &grad_input);
            }
            Op::Exp(ref input) => {
                let self_inner = self.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(self_inner.data.par_iter())
                        .map(|(g, &y)| g * y).collect()
                } else {
                    grad.iter().zip(&self_inner.data)
                        .map(|(g, &y)| g * y).collect()
                };
                drop(self_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Log(ref input) => {
                let in_inner = input.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(in_inner.data.par_iter())
                        .map(|(g, &x)| g / x).collect()
                } else {
                    grad.iter().zip(&in_inner.data)
                        .map(|(g, &x)| g / x).collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Sqrt(ref input) => {
                let self_inner = self.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(self_inner.data.par_iter())
                        .map(|(g, &y)| g / (2.0 * y)).collect()
                } else {
                    grad.iter().zip(&self_inner.data)
                        .map(|(g, &y)| g / (2.0 * y)).collect()
                };
                drop(self_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Abs(ref input) => {
                let in_inner = input.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(in_inner.data.par_iter())
                        .map(|(g, &x)| if x >= 0.0 { *g } else { -g }).collect()
                } else {
                    grad.iter().zip(&in_inner.data)
                        .map(|(g, &x)| if x >= 0.0 { *g } else { -g }).collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Pow(ref input, p) => {
                let in_inner = input.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(in_inner.data.par_iter())
                        .map(|(g, &x)| g * p * x.powf(p - 1.0)).collect()
                } else {
                    grad.iter().zip(&in_inner.data)
                        .map(|(g, &x)| g * p * x.powf(p - 1.0)).collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Sin(ref input) => {
                let in_inner = input.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(in_inner.data.par_iter())
                        .map(|(g, &x)| g * x.cos()).collect()
                } else {
                    grad.iter().zip(&in_inner.data)
                        .map(|(g, &x)| g * x.cos()).collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Cos(ref input) => {
                let in_inner = input.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(in_inner.data.par_iter())
                        .map(|(g, &x)| -g * x.sin()).collect()
                } else {
                    grad.iter().zip(&in_inner.data)
                        .map(|(g, &x)| -g * x.sin()).collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Tanh(ref input) => {
                let self_inner = self.0.borrow();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(self_inner.data.par_iter())
                        .map(|(g, &y)| g * (1.0 - y * y)).collect()
                } else {
                    grad.iter().zip(&self_inner.data)
                        .map(|(g, &y)| g * (1.0 - y * y)).collect()
                };
                drop(self_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Mean(ref input) => {
                let n = input.size();
                let grad_input = vec![grad[0] / n as f32; n];
                accumulate_grad(input, &grad_input);
            }
            Op::Squeeze { ref input, .. } | Op::Unsqueeze { ref input, .. } => {
                accumulate_grad(input, &grad);
            }
            Op::Gelu(ref input) => {
                let in_inner = input.0.borrow();
                let sqrt_2_over_pi = (2.0f32 / std::f32::consts::PI).sqrt();
                let grad_input: Vec<f32> = if grad.len() >= PAR_THRESHOLD {
                    grad.par_iter().zip(in_inner.data.par_iter()).map(|(&g, &x)| {
                        let x3 = x * x * x;
                        let inner_val = sqrt_2_over_pi * (x + 0.044715 * x3);
                        let tanh_val = inner_val.tanh();
                        let sech2 = 1.0 - tanh_val * tanh_val;
                        let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
                        g * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
                    }).collect()
                } else {
                    grad.iter().zip(in_inner.data.iter()).map(|(&g, &x)| {
                        let x3 = x * x * x;
                        let inner_val = sqrt_2_over_pi * (x + 0.044715 * x3);
                        let tanh_val = inner_val.tanh();
                        let sech2 = 1.0 - tanh_val * tanh_val;
                        let d_inner = sqrt_2_over_pi * (1.0 + 3.0 * 0.044715 * x * x);
                        g * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
                    }).collect()
                };
                drop(in_inner);
                accumulate_grad(input, &grad_input);
            }
            Op::Concat { ref inputs, ref split_sizes, dim } => {
                let out_shape = self.shape();
                let outer: usize = out_shape[..dim].iter().product();
                let inner_size: usize = out_shape[dim + 1..].iter().product();
                let total_dim = out_shape[dim];

                let mut dim_offset = 0;
                for (t_idx, t) in inputs.iter().enumerate() {
                    let t_dim = split_sizes[t_idx];
                    let t_size = outer * t_dim * inner_size;
                    let mut grad_t = vec![0.0f32; t_size];

                    for o in 0..outer {
                        for d in 0..t_dim {
                            let src_base = o * total_dim * inner_size + (dim_offset + d) * inner_size;
                            let dst_base = o * t_dim * inner_size + d * inner_size;
                            grad_t[dst_base..dst_base + inner_size]
                                .copy_from_slice(&grad[src_base..src_base + inner_size]);
                        }
                    }

                    accumulate_grad(t, &grad_t);
                    dim_offset += t_dim;
                }
            }
            Op::BatchNorm { ref input, ref gamma, ref beta, ref normalized, mean: _, ref inv_std } => {
                let in_shape = input.shape();
                let (n, c, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let spatial = (n * h * w) as f32;
                let gamma_data = gamma.data();

                let mut grad_gamma = vec![0.0f32; c];
                let mut grad_beta = vec![0.0f32; c];
                for ni in 0..n {
                    for ci in 0..c {
                        for p in 0..(h * w) {
                            let idx = ni * c * h * w + ci * h * w + p;
                            grad_gamma[ci] += grad[idx] * normalized[idx];
                            grad_beta[ci] += grad[idx];
                        }
                    }
                }

                let mut grad_input = vec![0.0f32; n * c * h * w];
                for ni in 0..n {
                    for ci in 0..c {
                        let sum_dy_gamma: f32 = (0..(h * w)).map(|p| {
                            grad[ni * c * h * w + ci * h * w + p] * gamma_data[ci]
                        }).sum();
                        let sum_dy_gamma_xhat: f32 = (0..(h * w)).map(|p| {
                            let idx = ni * c * h * w + ci * h * w + p;
                            grad[idx] * gamma_data[ci] * normalized[idx]
                        }).sum();
                        for p in 0..(h * w) {
                            let idx = ni * c * h * w + ci * h * w + p;
                            let dy_gamma = grad[idx] * gamma_data[ci];
                            grad_input[idx] = inv_std[ci] * (dy_gamma
                                - sum_dy_gamma / spatial
                                - normalized[idx] * sum_dy_gamma_xhat / spatial);
                        }
                    }
                }

                accumulate_grad(input, &grad_input);
                accumulate_grad(gamma, &grad_gamma);
                accumulate_grad(beta, &grad_beta);
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
            if existing.len() >= PAR_THRESHOLD {
                let slice: &mut [f32] = existing.as_mut_slice();
                slice.par_iter_mut().zip(grad.par_iter()).for_each(|(e, g)| {
                    *e += g;
                });
            } else {
                for (e, g) in existing.iter_mut().zip(grad) {
                    *e += g;
                }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_3x3_same_padding() {
        let input = Tensor::new(vec![1.0; 1 * 1 * 4 * 4], vec![1, 1, 4, 4], false);
        let kernel = Tensor::new(vec![1.0; 1 * 1 * 3 * 3], vec![1, 1, 3, 3], true);
        let bias = Tensor::zeros(&[1], true);
        let out = input.conv2d(&kernel, &bias);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
        let data = out.data();
        // Center pixel sees full 3x3 patch of ones -> sum = 9
        assert!((data[5] - 9.0).abs() < 1e-5);
        // Corner pixel sees 2x2 patch -> sum = 4
        assert!((data[0] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv2d_1x1_unchanged() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false);
        let kernel = Tensor::new(vec![2.0], vec![1, 1, 1, 1], true);
        let bias = Tensor::new(vec![1.0], vec![1], true);
        let out = input.conv2d(&kernel, &bias);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        assert_eq!(out.data(), vec![3.0, 5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_conv2d_3x3_gradient() {
        let input = Tensor::randn(&[1, 1, 4, 4], true);
        let kernel = Tensor::randn(&[2, 1, 3, 3], true);
        let bias = Tensor::zeros(&[2], true);
        let out = input.conv2d(&kernel, &bias);
        let loss = out.sum();
        loss.backward();
        assert!(kernel.grad().is_some());
        assert!(bias.grad().is_some());
    }

    #[test]
    fn test_conv2d_relu_pool_3x3() {
        let input = Tensor::new(vec![1.0; 1 * 1 * 4 * 4], vec![1, 1, 4, 4], true);
        let kernel = Tensor::new(vec![1.0; 2 * 1 * 3 * 3], vec![2, 1, 3, 3], true);
        let bias = Tensor::zeros(&[2], true);
        let out = input.conv2d_relu_pool(&kernel, &bias);
        // Same-padding preserves spatial, then pool halves: 4->2
        assert_eq!(out.shape(), vec![1, 2, 2, 2]);
        let loss = out.sum();
        loss.backward();
        assert!(kernel.grad().is_some());
    }

    #[test]
    fn test_batch_norm_forward() {
        let input = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 1, 2, 2], true,
        );
        let gamma = Tensor::new(vec![1.0], vec![1], true);
        let beta = Tensor::zeros(&[1], true);
        let out = input.batch_norm(&gamma, &beta);
        assert_eq!(out.shape(), vec![2, 1, 2, 2]);
        let data = out.data();
        // Output should have approximately zero mean per channel
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!(mean.abs() < 1e-5, "batch norm output mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_batch_norm_gradient() {
        let input = Tensor::randn(&[2, 3, 4, 4], true);
        let gamma = Tensor::new(vec![1.0; 3], vec![3], true);
        let beta = Tensor::zeros(&[3], true);
        let out = input.batch_norm(&gamma, &beta);
        let loss = out.sum();
        loss.backward();
        assert!(gamma.grad().is_some());
        assert!(beta.grad().is_some());
    }

    #[test]
    fn test_reshape() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let y = x.reshape(vec![3, 2]);
        assert_eq!(y.shape(), vec![3, 2]);
        assert_eq!(y.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let loss = y.sum();
        loss.backward();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_transpose() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let y = x.transpose(0, 1);
        assert_eq!(y.shape(), vec![3, 2]);
        assert_eq!(y.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        let loss = y.sum();
        loss.backward();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_softmax() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true);
        let y = x.softmax(-1);
        let data = y.data();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1");
        assert!(data[2] > data[1] && data[1] > data[0]);
        let loss = y.sum();
        loss.backward();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_layer_norm() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let gamma = Tensor::new(vec![1.0, 1.0], vec![2], true);
        let beta = Tensor::zeros(&[2], true);
        let y = x.layer_norm(&gamma, &beta, 2);
        assert_eq!(y.shape(), vec![2, 2]);
        let loss = y.sum();
        loss.backward();
        assert!(gamma.grad().is_some());
    }

    #[test]
    fn test_gelu() {
        let x = Tensor::new(vec![-1.0, 0.0, 1.0], vec![3], true);
        let y = x.gelu();
        let data = y.data();
        assert!((data[1]).abs() < 1e-5, "gelu(0) should be ~0");
        assert!(data[2] > 0.0, "gelu(1) should be positive");
        assert!(data[0] < 0.0, "gelu(-1) should be negative");
        let loss = y.sum();
        loss.backward();
        assert!(x.grad().is_some());
    }

    #[test]
    fn test_concat() {
        let a = Tensor::new(vec![1.0, 2.0], vec![1, 2], true);
        let b = Tensor::new(vec![3.0, 4.0, 5.0], vec![1, 3], true);
        let c = Tensor::concat(&[&a, &b], 1);
        assert_eq!(c.shape(), vec![1, 5]);
        assert_eq!(c.data(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let loss = c.sum();
        loss.backward();
        assert!(a.grad().is_some());
        assert!(b.grad().is_some());
    }

    #[test]
    fn test_sub() {
        let a = Tensor::new(vec![3.0, 5.0, 7.0], vec![3], true);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], true);
        let c = a.sub(&b);
        assert_eq!(c.data(), vec![2.0, 3.0, 4.0]);
        let loss = c.sum();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap(), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_div() {
        let a = Tensor::new(vec![6.0, 8.0], vec![2], true);
        let b = Tensor::new(vec![2.0, 4.0], vec![2], true);
        let c = a.div(&b);
        assert_eq!(c.data(), vec![3.0, 2.0]);
        let loss = c.sum();
        loss.backward();
        // grad_a = 1/b = [0.5, 0.25]
        let ga = a.grad().unwrap();
        assert!((ga[0] - 0.5).abs() < 1e-5);
        assert!((ga[1] - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_neg() {
        let a = Tensor::new(vec![1.0, -2.0, 3.0], vec![3], true);
        let b = a.neg();
        assert_eq!(b.data(), vec![-1.0, 2.0, -3.0]);
        let loss = b.sum();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![-1.0, -1.0, -1.0]);
    }

    #[test]
    fn test_exp_log() {
        let a = Tensor::new(vec![0.0, 1.0, 2.0], vec![3], true);
        let b = a.exp();
        let data = b.data();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - std::f32::consts::E).abs() < 1e-4);
        // log(exp(x)) = x
        let c = b.log();
        let roundtrip = c.data();
        for (i, &v) in roundtrip.iter().enumerate() {
            assert!((v - a.data()[i]).abs() < 1e-5, "log(exp(x)) should equal x");
        }
    }

    #[test]
    fn test_sqrt() {
        let a = Tensor::new(vec![4.0, 9.0, 16.0], vec![3], true);
        let b = a.sqrt();
        assert_eq!(b.data(), vec![2.0, 3.0, 4.0]);
        let loss = b.sum();
        loss.backward();
        // grad = 1 / (2*sqrt(x)) = [0.25, 1/6, 0.125]
        let ga = a.grad().unwrap();
        assert!((ga[0] - 0.25).abs() < 1e-5);
        assert!((ga[2] - 0.125).abs() < 1e-5);
    }

    #[test]
    fn test_pow() {
        let a = Tensor::new(vec![2.0, 3.0], vec![2], true);
        let b = a.pow(3.0);
        assert_eq!(b.data(), vec![8.0, 27.0]);
        let loss = b.sum();
        loss.backward();
        // grad = 3 * x^2 = [12, 27]
        let ga = a.grad().unwrap();
        assert!((ga[0] - 12.0).abs() < 1e-4);
        assert!((ga[1] - 27.0).abs() < 1e-4);
    }

    #[test]
    fn test_sin_cos() {
        let a = Tensor::new(vec![0.0, std::f32::consts::FRAC_PI_2], vec![2], true);
        let s = a.sin();
        let sd = s.data();
        assert!((sd[0]).abs() < 1e-5, "sin(0) = 0");
        assert!((sd[1] - 1.0).abs() < 1e-5, "sin(pi/2) = 1");

        let c = a.cos();
        let cd = c.data();
        assert!((cd[0] - 1.0).abs() < 1e-5, "cos(0) = 1");
        assert!((cd[1]).abs() < 1e-5, "cos(pi/2) = 0");
    }

    #[test]
    fn test_tanh() {
        let a = Tensor::new(vec![0.0, 1.0, -1.0], vec![3], true);
        let b = a.tanh();
        let data = b.data();
        assert!((data[0]).abs() < 1e-5, "tanh(0) = 0");
        assert!((data[1] - 1.0_f32.tanh()).abs() < 1e-5);
        let loss = b.sum();
        loss.backward();
        // grad = 1 - tanh^2(x)
        let ga = a.grad().unwrap();
        assert!((ga[0] - 1.0).abs() < 1e-5, "tanh'(0) = 1");
    }

    #[test]
    fn test_abs() {
        let a = Tensor::new(vec![-3.0, 0.0, 5.0], vec![3], true);
        let b = a.abs();
        assert_eq!(b.data(), vec![3.0, 0.0, 5.0]);
        let loss = b.sum();
        loss.backward();
        let ga = a.grad().unwrap();
        assert_eq!(ga, vec![-1.0, 1.0, 1.0]); // sign function (0 maps to +1)
    }

    /// Finite difference gradient check for all unary ops
    #[test]
    fn test_unary_ops_finite_diff() {
        let eps = 1e-3f32;
        let x_val = 1.5f32;

        let ops: Vec<(&str, Box<dyn Fn(&Tensor) -> Tensor>, Box<dyn Fn(f32) -> f32>)> = vec![
            ("exp", Box::new(|t: &Tensor| t.exp()), Box::new(|x: f32| x.exp())),
            ("log", Box::new(|t: &Tensor| t.log()), Box::new(|x: f32| x.ln())),
            ("sqrt", Box::new(|t: &Tensor| t.sqrt()), Box::new(|x: f32| x.sqrt())),
            ("sin", Box::new(|t: &Tensor| t.sin()), Box::new(|x: f32| x.sin())),
            ("cos", Box::new(|t: &Tensor| t.cos()), Box::new(|x: f32| x.cos())),
            ("tanh", Box::new(|t: &Tensor| t.tanh()), Box::new(|x: f32| x.tanh())),
            ("neg", Box::new(|t: &Tensor| t.neg()), Box::new(|x: f32| -x)),
        ];

        for (name, tensor_op, scalar_op) in &ops {
            let x = Tensor::new(vec![x_val], vec![1], true);
            let y = tensor_op(&x);
            let loss = y.sum();
            loss.backward();
            let analytical = x.grad().unwrap()[0];

            let numerical = (scalar_op(x_val + eps) - scalar_op(x_val - eps)) / (2.0 * eps);
            let diff = (analytical - numerical).abs();
            assert!(
                diff < 0.01,
                "{}: analytical={}, numerical={}, diff={}",
                name, analytical, numerical, diff
            );
        }
    }

    #[test]
    fn test_creation_ops() {
        let o = Tensor::ones(&[2, 3], false);
        assert_eq!(o.shape(), vec![2, 3]);
        assert_eq!(o.data(), vec![1.0; 6]);

        let f = Tensor::full(&[2, 2], 7.0, false);
        assert_eq!(f.data(), vec![7.0; 4]);

        let a = Tensor::arange(5, false);
        assert_eq!(a.data(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let l = Tensor::linspace(0.0, 1.0, 5, false);
        assert_eq!(l.data(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);

        let e = Tensor::eye(3, false);
        assert_eq!(e.data(), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_mean() {
        let x = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![2, 2], true);
        let m = x.mean();
        assert_eq!(m.data(), vec![5.0]);
        m.backward();
        // grad = 1/n for each element
        assert_eq!(x.grad().unwrap(), vec![0.25, 0.25, 0.25, 0.25]);
    }

    #[test]
    fn test_squeeze_unsqueeze() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3, 1], true);
        let squeezed = x.squeeze(None);
        assert_eq!(squeezed.shape(), vec![3]);

        let y = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], true);
        let unsqueezed = y.unsqueeze(0);
        assert_eq!(unsqueezed.shape(), vec![1, 3]);
        let unsqueezed2 = y.unsqueeze(1);
        assert_eq!(unsqueezed2.shape(), vec![3, 1]);

        // Backward through squeeze
        let loss = squeezed.sum();
        loss.backward();
        assert_eq!(x.grad().unwrap(), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_max_min_argmax_argmin() {
        let x = Tensor::new(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6], false);
        assert_eq!(x.max(), 9.0);
        assert_eq!(x.min(), 1.0);
        assert_eq!(x.argmax(), 5);
        assert_eq!(x.argmin(), 1);
    }

    #[test]
    fn test_broadcast_add() {
        // [2, 3] + [3] -> [2, 3]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = Tensor::new(vec![10.0, 20.0, 30.0], vec![3], true);
        let c = a.add(&b);
        assert_eq!(c.shape(), vec![2, 3]);
        assert_eq!(c.data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        let loss = c.sum();
        loss.backward();
        // grad_a = 1 for each element (no reduction needed, shapes broadcast naturally)
        assert_eq!(a.grad().unwrap(), vec![1.0; 6]);
        // grad_b = sum over dim 0 = [2.0, 2.0, 2.0]
        assert_eq!(b.grad().unwrap(), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_broadcast_mul() {
        // [2, 3] * [1, 3] -> [2, 3]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = Tensor::new(vec![2.0, 3.0, 4.0], vec![1, 3], true);
        let c = a.mul(&b);
        assert_eq!(c.shape(), vec![2, 3]);
        assert_eq!(c.data(), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
        let loss = c.sum();
        loss.backward();
        // grad_a[i,j] = b[0,j]
        assert_eq!(a.grad().unwrap(), vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);
        // grad_b[0,j] = sum over i of a[i,j]
        assert_eq!(b.grad().unwrap(), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_broadcast_sub_div() {
        // [2, 2] - [2] -> [2, 2]
        let a = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2], true);
        let b = Tensor::new(vec![1.0, 2.0], vec![2], true);
        let c = a.sub(&b);
        assert_eq!(c.data(), vec![9.0, 18.0, 29.0, 38.0]);
        let loss = c.sum();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0; 4]);
        assert_eq!(b.grad().unwrap(), vec![-2.0, -2.0]);
    }

    #[test]
    fn test_broadcast_scalar() {
        // [3] + [1] -> [3]
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], true);
        let b = Tensor::new(vec![10.0], vec![1], true);
        let c = a.add(&b);
        assert_eq!(c.data(), vec![11.0, 12.0, 13.0]);
        let loss = c.sum();
        loss.backward();
        assert_eq!(a.grad().unwrap(), vec![1.0, 1.0, 1.0]);
        assert_eq!(b.grad().unwrap(), vec![3.0]); // sum of grads
    }

    #[test]
    fn test_conv2d_3x3_finite_diff_gradient() {
        let eps = 1e-3;
        let input = Tensor::randn(&[1, 1, 4, 4], false);
        let kernel = Tensor::randn(&[1, 1, 3, 3], true);
        let bias = Tensor::zeros(&[1], true);

        // Forward + backward for analytical gradient
        let out = input.conv2d(&kernel, &bias);
        let loss = out.sum();
        loss.backward();
        let analytical_grad = kernel.grad().unwrap();

        // Numerical gradient via finite differences
        let k_data = kernel.data();
        for idx in 0..k_data.len() {
            let mut k_plus = k_data.clone();
            k_plus[idx] += eps;
            let kp = Tensor::new(k_plus, vec![1, 1, 3, 3], false);
            let bp = Tensor::zeros(&[1], false);
            let loss_plus = input.conv2d(&kp, &bp).sum().data()[0];

            let mut k_minus = k_data.clone();
            k_minus[idx] -= eps;
            let km = Tensor::new(k_minus, vec![1, 1, 3, 3], false);
            let bm = Tensor::zeros(&[1], false);
            let loss_minus = input.conv2d(&km, &bm).sum().data()[0];

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            let diff = (analytical_grad[idx] - numerical).abs();
            assert!(
                diff < 0.05,
                "gradient mismatch at idx {}: analytical={}, numerical={}, diff={}",
                idx, analytical_grad[idx], numerical, diff
            );
        }
    }
}
