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
