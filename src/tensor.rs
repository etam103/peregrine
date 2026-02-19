use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

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
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x + y).collect();
        Tensor::from_op(data, a.shape.clone(), Op::Add(self.clone(), other.clone()))
    }

    pub fn mul(&self, other: &Tensor) -> Tensor {
        let a = self.0.borrow();
        let b = other.0.borrow();
        assert_eq!(a.shape, b.shape, "shapes must match for mul");
        let data: Vec<f32> = a.data.iter().zip(&b.data).map(|(x, y)| x * y).collect();
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
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k1 {
                    sum += a.data[i * k1 + p] * b.data[p * n + j];
                }
                data[i * n + j] = sum;
            }
        }
        Tensor::from_op(data, vec![m, n], Op::MatMul(self.clone(), other.clone()))
    }

    pub fn relu(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = inner.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::from_op(data, inner.shape.clone(), Op::Relu(self.clone()))
    }

    pub fn sigmoid(&self) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = inner.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        Tensor::from_op(data, inner.shape.clone(), Op::Sigmoid(self.clone()))
    }

    pub fn sum(&self) -> Tensor {
        let inner = self.0.borrow();
        let s: f32 = inner.data.iter().sum();
        Tensor::from_op(vec![s], vec![1], Op::Sum(self.clone()))
    }

    pub fn scale(&self, s: f32) -> Tensor {
        let inner = self.0.borrow();
        let data: Vec<f32> = inner.data.iter().map(|&x| x * s).collect();
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
        let data: Vec<f32> = a.data
            .iter()
            .enumerate()
            .map(|(i, &x)| x + b.data[i % cols])
            .collect();
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

    // ---- Backward (reverse-mode autodiff) ----

    pub fn backward(&self) {
        let size = self.size();
        // Seed gradient is 1.0 for the output tensor.
        self.0.borrow_mut().grad = Some(vec![1.0; size]);
        self.backward_inner();
    }

    fn backward_inner(&self) {
        let (op, grad) = {
            let inner = self.0.borrow();
            let grad = match &inner.grad {
                Some(g) => g.clone(),
                None => return,
            };
            (inner.op.clone(), grad)
        };

        match op {
            Op::None => {}
            Op::Add(ref a, ref b) => {
                accumulate_grad(a, &grad);
                accumulate_grad(b, &grad);
                a.backward_inner();
                b.backward_inner();
            }
            Op::Mul(ref a, ref b) => {
                let a_data = a.data();
                let b_data = b.data();
                let grad_a: Vec<f32> = grad.iter().zip(&b_data).map(|(g, b)| g * b).collect();
                let grad_b: Vec<f32> = grad.iter().zip(&a_data).map(|(g, a)| g * a).collect();
                accumulate_grad(a, &grad_a);
                accumulate_grad(b, &grad_b);
                a.backward_inner();
                b.backward_inner();
            }
            Op::MatMul(ref a, ref b) => {
                // C = A @ B  =>  dA = dC @ B^T,  dB = A^T @ dC
                let a_shape = a.shape();
                let b_shape = b.shape();
                let (m, k) = (a_shape[0], a_shape[1]);
                let n = b_shape[1];
                let a_data = a.data();
                let b_data = b.data();

                // dA = grad @ B^T  [M,N] x [N,K] -> [M,K]
                let mut grad_a = vec![0.0f32; m * k];
                for i in 0..m {
                    for j in 0..k {
                        let mut sum = 0.0;
                        for p in 0..n {
                            sum += grad[i * n + p] * b_data[j * n + p];
                        }
                        grad_a[i * k + j] = sum;
                    }
                }

                // dB = A^T @ grad  [K,M] x [M,N] -> [K,N]
                let mut grad_b = vec![0.0f32; k * n];
                for i in 0..k {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for p in 0..m {
                            sum += a_data[p * k + i] * grad[p * n + j];
                        }
                        grad_b[i * n + j] = sum;
                    }
                }

                accumulate_grad(a, &grad_a);
                accumulate_grad(b, &grad_b);
                a.backward_inner();
                b.backward_inner();
            }
            Op::Relu(ref input) => {
                let input_data = input.data();
                let grad_input: Vec<f32> = grad
                    .iter()
                    .zip(&input_data)
                    .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 })
                    .collect();
                accumulate_grad(input, &grad_input);
                input.backward_inner();
            }
            Op::Sigmoid(ref input) => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
                let out_data = self.data();
                let grad_input: Vec<f32> = grad
                    .iter()
                    .zip(&out_data)
                    .map(|(g, &s)| g * s * (1.0 - s))
                    .collect();
                accumulate_grad(input, &grad_input);
                input.backward_inner();
            }
            Op::Sum(ref input) => {
                let n = input.size();
                let grad_input = vec![grad[0]; n];
                accumulate_grad(input, &grad_input);
                input.backward_inner();
            }
            Op::Scale(ref input, s) => {
                let grad_input: Vec<f32> = grad.iter().map(|g| g * s).collect();
                accumulate_grad(input, &grad_input);
                input.backward_inner();
            }
            Op::AddBias(ref input, ref bias) => {
                // grad flows through to input unchanged
                accumulate_grad(input, &grad);
                // bias grad: sum over batch dimension
                let cols = bias.shape()[1];
                let rows = input.shape()[0];
                let mut grad_bias = vec![0.0f32; cols];
                for r in 0..rows {
                    for c in 0..cols {
                        grad_bias[c] += grad[r * cols + c];
                    }
                }
                accumulate_grad(bias, &grad_bias);
                input.backward_inner();
                bias.backward_inner();
            }
            Op::Conv2d(ref input, ref kernel, ref bias) => {
                let in_shape = input.shape();
                let k_shape = kernel.shape();
                let (n, ci, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
                let (co, _, kh, kw) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);
                let oh = h - kh + 1;
                let ow = w - kw + 1;
                let out_shape = vec![n, co, oh, ow];
                let in_data = input.data();
                let k_data = kernel.data();

                // grad_bias[oc] = sum over (n, oh, ow) of grad[n,oc,i,j]
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

                // grad_kernel[oc,ic,ki,kj] = sum over (n,i,j) of
                //   grad[n,oc,i,j] * input[n,ic,i+ki,j+kj]
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
                                                * in_data[idx4(b, ic, i + ki, j + kj, &in_shape)];
                                        }
                                    }
                                    grad_kernel[idx4(oc, ic, ki, kj, &k_shape)] += s;
                                }
                            }
                        }
                    }
                }

                // grad_input[n,ic,ih,iw] = sum over (oc,ki,kj) of
                //   grad[n,oc,ih-ki,iw-kj] * kernel[oc,ic,ki,kj]
                //   where ih-ki and iw-kj are valid output positions
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
                                                    * k_data[idx4(oc, ic, ki, kj, &k_shape)];
                                            }
                                        }
                                    }
                                }
                                grad_input[idx4(b, ic, ih, iw, &in_shape)] += s;
                            }
                        }
                    }
                }

                accumulate_grad(input, &grad_input);
                accumulate_grad(kernel, &grad_kernel);
                accumulate_grad(bias, &grad_bias);
                input.backward_inner();
                kernel.backward_inner();
                bias.backward_inner();
            }
            Op::MaxPool2d { ref input, ref max_indices } => {
                let in_size = input.size();
                let mut grad_input = vec![0.0f32; in_size];
                for (out_idx, &in_idx) in max_indices.iter().enumerate() {
                    grad_input[in_idx] += grad[out_idx];
                }
                accumulate_grad(input, &grad_input);
                input.backward_inner();
            }
            Op::Flatten { ref input, .. } => {
                // Gradient has same flat data, just needs to be accumulated as-is
                // since the underlying data layout is identical.
                accumulate_grad(input, &grad);
                input.backward_inner();
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
