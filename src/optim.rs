use crate::cpu_pool::{pool_get, pool_recycle};
use crate::tensor::Tensor;

// ===========================================================================
// Gradient clipping
// ===========================================================================

/// Clip gradients by global L2 norm. Returns the original norm.
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) -> f32 {
    let mut total_norm_sq = 0.0f32;
    for p in params {
        if let Some(g) = p.grad_data() {
            total_norm_sq += g.iter().map(|x| x * x).sum::<f32>();
        }
    }
    let total_norm = total_norm_sq.sqrt();
    if total_norm > max_norm {
        let scale = max_norm / (total_norm + 1e-6);
        for p in params {
            p.update_data(|data, grad| {
                let _ = (data, grad);
            });
            // Scale gradients in-place
            let mut inner = p.0.borrow_mut();
            if let Some(ref mut g) = inner.grad {
                for v in g.iter_mut() {
                    *v *= scale;
                }
            }
        }
    }
    total_norm
}

/// Clip gradients element-wise to [-value, value].
pub fn clip_grad_value(params: &[Tensor], clip_value: f32) {
    for p in params {
        let mut inner = p.0.borrow_mut();
        if let Some(ref mut g) = inner.grad {
            for v in g.iter_mut() {
                *v = v.clamp(-clip_value, clip_value);
            }
        }
    }
}

// ===========================================================================
// LrSchedule trait
// ===========================================================================

/// Trait for learning rate schedules.
pub trait LrSchedule {
    /// Return the learning rate at the given step (or epoch).
    fn lr_at(&self, step: usize) -> f32;
}

// ===========================================================================
// SGD with momentum
// ===========================================================================

pub struct Sgd {
    params: Vec<Tensor>,
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    velocity: Vec<Vec<f32>>,
}

impl Sgd {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let velocity: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        Sgd { params, lr, momentum: 0.0, weight_decay: 0.0, nesterov: false, velocity }
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let vel = &mut self.velocity[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            for j in 0..data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * data[j];
                }
                if self.momentum != 0.0 {
                    vel[j] = self.momentum * vel[j] + g;
                    if self.nesterov {
                        g = g + self.momentum * vel[j];
                    } else {
                        g = vel[j];
                    }
                }
                data[j] -= self.lr * g;
            }
            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// Adam / AdamW
// ===========================================================================

pub struct Adam {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    decoupled_wd: bool, // true = AdamW, false = L2 regularization
    m: Vec<Vec<f32>>,   // first moment
    v: Vec<Vec<f32>>,   // second moment
    t: u64,             // timestep
    #[cfg(feature = "metal")]
    gpu_m: Vec<Option<crate::metal::GpuBuffer<f32>>>,
    #[cfg(feature = "metal")]
    gpu_v: Vec<Option<crate::metal::GpuBuffer<f32>>>,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        #[allow(unused_variables)]
        let n = params.len();
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        Adam {
            params, lr,
            beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.0, decoupled_wd: false,
            m, v, t: 0,
            #[cfg(feature = "metal")]
            gpu_m: (0..n).map(|_| None).collect(),
            #[cfg(feature = "metal")]
            gpu_v: (0..n).map(|_| None).collect(),
        }
    }

    /// Create an AdamW optimizer (decoupled weight decay).
    pub fn adamw(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        let mut opt = Self::new(params, lr);
        opt.weight_decay = weight_decay;
        opt.decoupled_wd = true;
        opt
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            // --- GPU path: if param has gpu_data + gpu_grad, use fused GPU Adam ---
            #[cfg(feature = "metal")]
            {
                let has_gpu = {
                    let inner = p.0.borrow();
                    inner.gpu_data.is_some() && inner.gpu_grad.is_some()
                };
                if has_gpu {
                    // Lazy-init GPU moment buffers
                    if self.gpu_m[i].is_none() {
                        if let Some((m_buf, v_buf)) = crate::metal::with_gpu(|gpu| {
                            let n = p.size();
                            let m_buf = gpu.alloc(n);
                            let v_buf = gpu.alloc(n);
                            gpu.dispatch_fill(&m_buf, 0.0);
                            gpu.dispatch_fill(&v_buf, 0.0);
                            (m_buf, v_buf)
                        }) {
                            self.gpu_m[i] = Some(m_buf);
                            self.gpu_v[i] = Some(v_buf);
                        }
                    }

                    if let (Some(ref m_buf), Some(ref v_buf)) = (&self.gpu_m[i], &self.gpu_v[i]) {
                        let done = crate::metal::with_gpu(|gpu| {
                            let inner = p.0.borrow();
                            let data_buf = inner.gpu_data.as_ref().unwrap();
                            let grad_buf = inner.gpu_grad.as_ref().unwrap();
                            gpu.dispatch_adam_step(
                                data_buf, grad_buf, m_buf, v_buf,
                                self.lr, self.beta1, self.beta2, self.eps,
                                bc1, bc2,
                                self.weight_decay, self.decoupled_wd,
                            );
                        });
                        if done.is_some() {
                            let mut inner = p.0.borrow_mut();
                            inner.gpu_grad = None;
                            inner.gpu_dirty = true; // GPU data was modified
                            inner.op = crate::tensor::Op::None;
                            continue;
                        }
                    }
                }
            }

            // --- CPU path ---
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            #[cfg(target_arch = "aarch64")]
            {
                crate::simd_kernels::adam_step_f32(
                    data, grad, m, v,
                    self.beta1, self.beta2, self.lr, bc1, bc2, self.eps,
                    self.weight_decay, self.decoupled_wd,
                );
            }

            #[cfg(not(target_arch = "aarch64"))]
            for j in 0..data.len() {
                let mut g = grad[j];

                // L2 regularization (classic Adam)
                if self.weight_decay != 0.0 && !self.decoupled_wd {
                    g += self.weight_decay * data[j];
                }

                // Update biased first & second moment estimates
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;

                // Bias-corrected estimates
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;

                // Decoupled weight decay (AdamW)
                if self.decoupled_wd && self.weight_decay != 0.0 {
                    data[j] -= self.lr * self.weight_decay * data[j];
                }

                data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }

            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// Fused step + zero_grad: applies Adam update and clears gradients in a single pass,
    /// avoiding the extra loop and borrow overhead of separate step() + zero_grad().
    pub fn step_and_zero_grad(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            // --- GPU path ---
            #[cfg(feature = "metal")]
            {
                let has_gpu = {
                    let inner = p.0.borrow();
                    inner.gpu_data.is_some() && inner.gpu_grad.is_some()
                };
                if has_gpu {
                    if self.gpu_m[i].is_none() {
                        if let Some((m_buf, v_buf)) = crate::metal::with_gpu(|gpu| {
                            let n = p.size();
                            let m_buf = gpu.alloc(n);
                            let v_buf = gpu.alloc(n);
                            gpu.dispatch_fill(&m_buf, 0.0);
                            gpu.dispatch_fill(&v_buf, 0.0);
                            (m_buf, v_buf)
                        }) {
                            self.gpu_m[i] = Some(m_buf);
                            self.gpu_v[i] = Some(v_buf);
                        }
                    }

                    if let (Some(ref m_buf), Some(ref v_buf)) = (&self.gpu_m[i], &self.gpu_v[i]) {
                        let done = crate::metal::with_gpu(|gpu| {
                            let inner = p.0.borrow();
                            let data_buf = inner.gpu_data.as_ref().unwrap();
                            let grad_buf = inner.gpu_grad.as_ref().unwrap();
                            gpu.dispatch_adam_step(
                                data_buf, grad_buf, m_buf, v_buf,
                                self.lr, self.beta1, self.beta2, self.eps,
                                bc1, bc2,
                                self.weight_decay, self.decoupled_wd,
                            );
                        });
                        if done.is_some() {
                            let mut inner = p.0.borrow_mut();
                            inner.gpu_grad = None;
                            inner.gpu_dirty = true;
                            inner.op = crate::tensor::Op::None;
                            continue;
                        }
                    }
                }
            }

            // --- CPU path ---
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match inner_ref.grad.take() {
                Some(g) => g,
                None => {
                    inner_ref.op = crate::tensor::Op::None;
                    continue;
                }
            };
            let data = &mut inner_ref.data;

            #[cfg(target_arch = "aarch64")]
            {
                crate::simd_kernels::adam_step_f32(
                    data, &grad, m, v,
                    self.beta1, self.beta2, self.lr, bc1, bc2, self.eps,
                    self.weight_decay, self.decoupled_wd,
                );
            }

            #[cfg(not(target_arch = "aarch64"))]
            for j in 0..data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 && !self.decoupled_wd {
                    g += self.weight_decay * data[j];
                }
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;
                if self.decoupled_wd && self.weight_decay != 0.0 {
                    data[j] -= self.lr * self.weight_decay * data[j];
                }
                data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }

            // Recycle the grad buffer and clear op in one go
            pool_recycle(grad);
            inner_ref.op = crate::tensor::Op::None;
        }
    }
}

// ===========================================================================
// RmsProp
// ===========================================================================

/// RMSProp optimizer.
pub struct RmsProp {
    params: Vec<Tensor>,
    lr: f32,
    alpha: f32,
    eps: f32,
    momentum: f32,
    centered: bool,
    weight_decay: f32,
    v: Vec<Vec<f32>>,        // squared gradient running average
    buf: Vec<Vec<f32>>,      // momentum buffer
    grad_avg: Vec<Vec<f32>>, // for centered variant
}

impl RmsProp {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        let buf: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        let grad_avg: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        RmsProp {
            params, lr, alpha: 0.99, eps: 1e-8, momentum: 0.0, centered: false,
            weight_decay: 0.0, v, buf, grad_avg,
        }
    }

    pub fn alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let v = &mut self.v[i];
            let buf = &mut self.buf[i];
            let grad_avg = &mut self.grad_avg[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            for j in 0..data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * data[j];
                }

                // v = alpha * v + (1 - alpha) * g^2
                v[j] = self.alpha * v[j] + (1.0 - self.alpha) * g * g;

                let avg = if self.centered {
                    // grad_avg = alpha * grad_avg + (1 - alpha) * g
                    grad_avg[j] = self.alpha * grad_avg[j] + (1.0 - self.alpha) * g;
                    (v[j] - grad_avg[j] * grad_avg[j]).sqrt() + self.eps
                } else {
                    v[j].sqrt() + self.eps
                };

                if self.momentum > 0.0 {
                    buf[j] = self.momentum * buf[j] + g / avg;
                    data[j] -= self.lr * buf[j];
                } else {
                    data[j] -= self.lr * g / avg;
                }
            }
            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// Adagrad
// ===========================================================================

/// Adagrad optimizer: adapts learning rate based on historical gradient magnitudes.
pub struct Adagrad {
    params: Vec<Tensor>,
    lr: f32,
    eps: f32,
    weight_decay: f32,
    sum: Vec<Vec<f32>>, // sum of squared gradients
}

impl Adagrad {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let sum: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        Adagrad { params, lr, eps: 1e-10, weight_decay: 0.0, sum }
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let sum = &mut self.sum[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            for j in 0..data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * data[j];
                }
                sum[j] += g * g;
                data[j] -= self.lr * g / (sum[j].sqrt() + self.eps);
            }
            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// Adamax
// ===========================================================================

/// Adamax optimizer: variant of Adam based on infinity norm.
pub struct Adamax {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>,  // 1st moment (exponential average of gradients)
    u: Vec<Vec<f32>>,  // infinity norm (max of weighted abs gradients)
    t: u64,
}

impl Adamax {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        let u: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        Adamax {
            params, lr, beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.0, m, u, t: 0,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            let m = &mut self.m[i];
            let u = &mut self.u[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            for j in 0..data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * data[j];
                }

                // Update biased first moment
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                // Update infinity norm
                u[j] = f32::max(self.beta2 * u[j], g.abs());

                // Bias-corrected first moment
                let m_hat = m[j] / bc1;

                data[j] -= self.lr * m_hat / (u[j] + self.eps);
            }
            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// AdaDelta
// ===========================================================================

/// Adadelta optimizer: adapts learning rate based on running window of gradient updates.
pub struct AdaDelta {
    params: Vec<Tensor>,
    lr: f32,
    rho: f32,
    eps: f32,
    weight_decay: f32,
    v: Vec<Vec<f32>>,     // squared gradient running average
    delta: Vec<Vec<f32>>, // squared update running average
}

impl AdaDelta {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        let delta: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        AdaDelta {
            params, lr, rho: 0.9, eps: 1e-6, weight_decay: 0.0, v, delta,
        }
    }

    pub fn rho(mut self, rho: f32) -> Self {
        self.rho = rho;
        self
    }

    pub fn eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let v = &mut self.v[i];
            let delta = &mut self.delta[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            for j in 0..data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * data[j];
                }

                // Accumulate gradient squared
                v[j] = self.rho * v[j] + (1.0 - self.rho) * g * g;

                // Compute update
                let update = ((delta[j] + self.eps).sqrt() / (v[j] + self.eps).sqrt()) * g;

                // Accumulate update squared
                delta[j] = self.rho * delta[j] + (1.0 - self.rho) * update * update;

                data[j] -= self.lr * update;
            }
            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// Lion
// ===========================================================================

/// Lion optimizer: sign-based update with EMA of gradients.
/// From "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023).
pub struct Lion {
    params: Vec<Tensor>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    weight_decay: f32,
    m: Vec<Vec<f32>>, // EMA of gradients
}

impl Lion {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        Lion {
            params, lr, beta1: 0.9, beta2: 0.99, weight_decay: 0.0, m,
        }
    }

    pub fn betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    pub fn step(&mut self) {
        for (i, p) in self.params.iter().enumerate() {
            let m = &mut self.m[i];
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g,
                None => continue,
            };
            let data = &mut inner_ref.data;

            for j in 0..data.len() {
                let g = grad[j];

                // Weight decay (decoupled)
                if self.weight_decay != 0.0 {
                    data[j] -= self.lr * self.weight_decay * data[j];
                }

                // Update direction: sign(beta1 * m + (1 - beta1) * g)
                let update = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                let sign = if update > 0.0 { 1.0 } else if update < 0.0 { -1.0 } else { 0.0 };

                data[j] -= self.lr * sign;

                // Update EMA
                m[j] = self.beta2 * m[j] + (1.0 - self.beta2) * g;
            }
            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// Adafactor
// ===========================================================================

/// Adafactor optimizer: memory-efficient adaptive optimizer.
/// Uses row and column factorization for 2D parameters.
pub struct Adafactor {
    params: Vec<Tensor>,
    lr: f32,
    eps: (f32, f32),
    clip_threshold: f32,
    beta1: f32,
    decay_rate: f32,
    scale_parameter: bool,
    weight_decay: f32,
    // Per-param state: for 2D params we store row/col factors; for 1D we store full v
    state: Vec<AdafactorState>,
    t: u64,
}

enum AdafactorState {
    Uninitialized,
    /// For 2D+ params: factored representation
    Factored {
        row_factor: Vec<f32>, // [rows]
        col_factor: Vec<f32>, // [cols]
        m: Option<Vec<f32>>,  // first moment (if beta1 > 0)
    },
    /// For 1D params: full second moment
    Full {
        v: Vec<f32>,
        m: Option<Vec<f32>>,
    },
}

impl Adafactor {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let n = params.len();
        Adafactor {
            params, lr,
            eps: (1e-30, 1e-3),
            clip_threshold: 1.0,
            beta1: 0.0,
            decay_rate: 0.8,
            scale_parameter: true,
            weight_decay: 0.0,
            state: (0..n).map(|_| AdafactorState::Uninitialized).collect(),
            t: 0,
        }
    }

    pub fn eps(mut self, eps1: f32, eps2: f32) -> Self {
        self.eps = (eps1, eps2);
        self
    }

    pub fn clip_threshold(mut self, ct: f32) -> Self {
        self.clip_threshold = ct;
        self
    }

    pub fn beta1(mut self, beta1: f32) -> Self {
        self.beta1 = beta1;
        self
    }

    pub fn decay_rate(mut self, dr: f32) -> Self {
        self.decay_rate = dr;
        self
    }

    pub fn scale_parameter(mut self, sp: bool) -> Self {
        self.scale_parameter = sp;
        self
    }

    pub fn weight_decay(mut self, wd: f32) -> Self {
        self.weight_decay = wd;
        self
    }

    pub fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn rho_at(&self, step: u64) -> f32 {
        let rho = 1.0 - (step as f32).powf(-self.decay_rate);
        rho.min(1.0 - self.eps.0)
    }

    pub fn step(&mut self) {
        self.t += 1;
        let rho = self.rho_at(self.t);

        for (i, p) in self.params.iter().enumerate() {
            let mut inner = p.0.borrow_mut();
            let inner_ref = &mut *inner;
            let grad = match &inner_ref.grad {
                Some(g) => g.clone(),
                None => continue,
            };
            let data = &mut inner_ref.data;
            let shape = &inner_ref.shape;
            let use_factored = shape.len() >= 2;

            // Initialize state if needed
            if matches!(self.state[i], AdafactorState::Uninitialized) {
                if use_factored {
                    let rows: usize = shape[..shape.len() - 1].iter().product();
                    let cols = *shape.last().unwrap();
                    self.state[i] = AdafactorState::Factored {
                        row_factor: vec![0.0; rows],
                        col_factor: vec![0.0; cols],
                        m: if self.beta1 > 0.0 { Some(vec![0.0; data.len()]) } else { None },
                    };
                } else {
                    self.state[i] = AdafactorState::Full {
                        v: vec![0.0; data.len()],
                        m: if self.beta1 > 0.0 { Some(vec![0.0; data.len()]) } else { None },
                    };
                }
            }

            // Compute parameter scale
            let param_scale = if self.scale_parameter {
                let rms: f32 = data.iter().map(|x| x * x).sum::<f32>() / data.len() as f32;
                rms.sqrt().max(self.eps.1)
            } else {
                1.0
            };

            // Weight decay
            if self.weight_decay != 0.0 {
                for j in 0..data.len() {
                    data[j] -= self.weight_decay * self.lr * data[j];
                }
            }

            match &mut self.state[i] {
                AdafactorState::Factored { row_factor, col_factor, m } => {
                    let rows = row_factor.len();
                    let cols = col_factor.len();
                    let n = rows * cols;

                    // Precompute grad_sq[i] = grad[i]^2 + eps once
                    let mut grad_sq = pool_get(n);
                    for j in 0..n {
                        grad_sq[j] = grad[j] * grad[j] + self.eps.0;
                    }

                    // Update row factors using precomputed grad_sq
                    let one_minus_rho = 1.0 - rho;
                    let inv_cols = 1.0 / cols as f32;
                    for r in 0..rows {
                        let base = r * cols;
                        let mut row_sum = 0.0f32;
                        for c in 0..cols {
                            row_sum += grad_sq[base + c];
                        }
                        row_factor[r] = rho * row_factor[r] + one_minus_rho * row_sum * inv_cols;
                    }

                    // Update column factors using precomputed grad_sq
                    let inv_rows = 1.0 / rows as f32;
                    for c in 0..cols {
                        let mut col_sum = 0.0f32;
                        for r in 0..rows {
                            col_sum += grad_sq[r * cols + c];
                        }
                        col_factor[c] = rho * col_factor[c] + one_minus_rho * col_sum * inv_rows;
                    }
                    pool_recycle(grad_sq);

                    // Reconstruct second moment estimate: v_hat = row * col / mean(row)
                    let row_mean: f32 = row_factor.iter().sum::<f32>() / rows as f32;
                    let row_mean = row_mean.max(self.eps.0);
                    let inv_row_mean = 1.0 / row_mean;

                    // Compute update using pool buffer
                    let mut update = pool_get(n);
                    for r in 0..rows {
                        let rf = row_factor[r];
                        let base = r * cols;
                        for c in 0..cols {
                            let v_hat = rf * col_factor[c] * inv_row_mean;
                            update[base + c] = grad[base + c] / (v_hat.sqrt() + self.eps.0);
                        }
                    }

                    // Clip update by RMS
                    let update_rms = (update.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
                    if update_rms > self.clip_threshold {
                        let scale = self.clip_threshold / update_rms;
                        for u in update.iter_mut() {
                            *u *= scale;
                        }
                    }

                    // Apply momentum if beta1 > 0
                    let lr_ps = self.lr * param_scale;
                    if let Some(ref mut m_vec) = m {
                        let beta1 = self.beta1;
                        let one_minus_beta1 = 1.0 - beta1;
                        for j in 0..n {
                            m_vec[j] = beta1 * m_vec[j] + one_minus_beta1 * update[j];
                            data[j] -= lr_ps * m_vec[j];
                        }
                    } else {
                        for j in 0..n {
                            data[j] -= lr_ps * update[j];
                        }
                    }
                    pool_recycle(update);
                }
                AdafactorState::Full { v, m } => {
                    let n = data.len();
                    let one_minus_rho = 1.0 - rho;
                    for j in 0..n {
                        v[j] = rho * v[j] + one_minus_rho * (grad[j] * grad[j] + self.eps.0);
                    }

                    let mut update = pool_get(n);
                    for j in 0..n {
                        update[j] = grad[j] / (v[j].sqrt() + self.eps.0);
                    }

                    // Clip
                    let update_rms = (update.iter().map(|x| x * x).sum::<f32>() / n as f32).sqrt();
                    if update_rms > self.clip_threshold {
                        let scale = self.clip_threshold / update_rms;
                        for u in update.iter_mut() {
                            *u *= scale;
                        }
                    }

                    let lr_ps = self.lr * param_scale;
                    if let Some(ref mut m_vec) = m {
                        let beta1 = self.beta1;
                        let one_minus_beta1 = 1.0 - beta1;
                        for j in 0..n {
                            m_vec[j] = beta1 * m_vec[j] + one_minus_beta1 * update[j];
                            data[j] -= lr_ps * m_vec[j];
                        }
                    } else {
                        for j in 0..n {
                            data[j] -= lr_ps * update[j];
                        }
                    }
                    pool_recycle(update);
                }
                AdafactorState::Uninitialized => unreachable!(),
            }

            inner_ref.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// ===========================================================================
// Learning rate schedulers
// ===========================================================================

/// Step decay: multiply LR by gamma every `step_size` epochs.
pub struct StepLr {
    base_lr: f32,
    step_size: usize,
    gamma: f32,
}

impl StepLr {
    pub fn new(base_lr: f32, step_size: usize, gamma: f32) -> Self {
        StepLr { base_lr, step_size, gamma }
    }
}

impl LrSchedule for StepLr {
    fn lr_at(&self, epoch: usize) -> f32 {
        self.base_lr * self.gamma.powi((epoch / self.step_size) as i32)
    }
}

/// Cosine annealing from base_lr to eta_min over T_max epochs.
pub struct CosineAnnealingLr {
    base_lr: f32,
    eta_min: f32,
    t_max: usize,
}

impl CosineAnnealingLr {
    pub fn new(base_lr: f32, t_max: usize) -> Self {
        CosineAnnealingLr { base_lr, eta_min: 0.0, t_max }
    }

    pub fn eta_min(mut self, eta_min: f32) -> Self {
        self.eta_min = eta_min;
        self
    }
}

impl LrSchedule for CosineAnnealingLr {
    fn lr_at(&self, epoch: usize) -> f32 {
        let t = epoch.min(self.t_max) as f32;
        self.eta_min + 0.5 * (self.base_lr - self.eta_min)
            * (1.0 + (std::f32::consts::PI * t / self.t_max as f32).cos())
    }
}

/// Linear warmup for `warmup_steps` then constant at base_lr.
pub struct WarmupLr {
    base_lr: f32,
    warmup_steps: usize,
}

impl WarmupLr {
    pub fn new(base_lr: f32, warmup_steps: usize) -> Self {
        WarmupLr { base_lr, warmup_steps }
    }
}

impl LrSchedule for WarmupLr {
    fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            self.base_lr
        }
    }
}

/// Exponential decay: lr = base_lr * gamma^step
pub struct ExponentialDecayLr {
    pub base_lr: f32,
    pub gamma: f32,
}

impl ExponentialDecayLr {
    pub fn new(base_lr: f32, gamma: f32) -> Self {
        ExponentialDecayLr { base_lr, gamma }
    }
}

impl LrSchedule for ExponentialDecayLr {
    fn lr_at(&self, step: usize) -> f32 {
        self.base_lr * self.gamma.powi(step as i32)
    }
}

/// Linear schedule from start_lr to end_lr over num_steps.
pub struct LinearScheduleLr {
    pub start_lr: f32,
    pub end_lr: f32,
    pub num_steps: usize,
}

impl LinearScheduleLr {
    pub fn new(start_lr: f32, end_lr: f32, num_steps: usize) -> Self {
        LinearScheduleLr { start_lr, end_lr, num_steps }
    }
}

impl LrSchedule for LinearScheduleLr {
    fn lr_at(&self, step: usize) -> f32 {
        let t = step.min(self.num_steps) as f32 / self.num_steps as f32;
        self.start_lr + (self.end_lr - self.start_lr) * t
    }
}

/// Join multiple schedules at given boundaries.
/// Uses schedule\[i\] for steps in \[boundaries\[i-1\], boundaries\[i\]).
pub struct JoinSchedules {
    pub boundaries: Vec<usize>,
    pub schedules: Vec<Box<dyn LrSchedule>>,
}

impl JoinSchedules {
    pub fn new(boundaries: Vec<usize>, schedules: Vec<Box<dyn LrSchedule>>) -> Self {
        assert_eq!(
            boundaries.len() + 1,
            schedules.len(),
            "need exactly one more schedule than boundaries"
        );
        JoinSchedules { boundaries, schedules }
    }
}

impl LrSchedule for JoinSchedules {
    fn lr_at(&self, step: usize) -> f32 {
        for (i, &boundary) in self.boundaries.iter().enumerate() {
            if step < boundary {
                return self.schedules[i].lr_at(step);
            }
        }
        // Past all boundaries: use the last schedule
        self.schedules.last().unwrap().lr_at(step)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_basic() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = Sgd::new(vec![x.clone()], 0.1);

        for _ in 0..50 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(x.data()[0].abs() < 0.01, "SGD should converge near 0, got {}", x.data()[0]);
    }

    #[test]
    fn test_sgd_momentum() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = Sgd::new(vec![x.clone()], 0.01).momentum(0.9);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(x.data()[0].abs() < 0.05, "SGD+momentum should converge, got {}", x.data()[0]);
    }

    #[test]
    fn test_adam_basic() {
        let x = Tensor::new(vec![5.0], vec![1], true);
        let y = Tensor::new(vec![3.0], vec![1], true);
        let mut opt = Adam::new(vec![x.clone(), y.clone()], 0.1);

        for _ in 0..200 {
            let ten = Tensor::new(vec![10.0], vec![1], false);
            let loss = x.pow(2.0).add(&y.pow(2.0).mul(&ten)).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(x.data()[0].abs() < 0.05, "Adam x should converge near 0, got {}", x.data()[0]);
        assert!(y.data()[0].abs() < 0.05, "Adam y should converge near 0, got {}", y.data()[0]);
    }

    #[test]
    fn test_adamw() {
        let x = Tensor::new(vec![5.0], vec![1], true);
        let mut opt = Adam::adamw(vec![x.clone()], 0.1, 0.01);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(x.data()[0].abs() < 0.05, "AdamW should converge, got {}", x.data()[0]);
    }

    #[test]
    fn test_adam_rosenbrock() {
        let x = Tensor::new(vec![-1.0], vec![1], true);
        let y = Tensor::new(vec![-1.0], vec![1], true);
        let mut opt = Adam::new(vec![x.clone(), y.clone()], 0.005).betas(0.9, 0.999);

        for _ in 0..5000 {
            let one = Tensor::new(vec![1.0], vec![1], false);
            let hundred = Tensor::new(vec![100.0], vec![1], false);
            let term1 = one.sub(&x).pow(2.0);
            let term2 = y.sub(&x.pow(2.0)).pow(2.0).mul(&hundred);
            let loss = term1.add(&term2).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        let one = Tensor::new(vec![1.0], vec![1], false);
        let hundred = Tensor::new(vec![100.0], vec![1], false);
        let final_loss = one.sub(&x).pow(2.0).add(&y.sub(&x.pow(2.0)).pow(2.0).mul(&hundred)).sum();
        assert!(
            final_loss.data()[0] < 1.0,
            "Adam on Rosenbrock should reduce loss significantly, got {}", final_loss.data()[0]
        );
    }

    #[test]
    fn test_step_lr() {
        let sched = StepLr::new(0.1, 10, 0.5);
        assert!((sched.lr_at(0) - 0.1).abs() < 1e-6);
        assert!((sched.lr_at(9) - 0.1).abs() < 1e-6);
        assert!((sched.lr_at(10) - 0.05).abs() < 1e-6);
        assert!((sched.lr_at(20) - 0.025).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let sched = CosineAnnealingLr::new(0.1, 100);
        assert!((sched.lr_at(0) - 0.1).abs() < 1e-6);
        assert!((sched.lr_at(100) - 0.0).abs() < 1e-5);
        assert!((sched.lr_at(50) - 0.05).abs() < 0.01);
    }

    #[test]
    fn test_warmup_lr() {
        let sched = WarmupLr::new(0.1, 10);
        assert!((sched.lr_at(0) - 0.01).abs() < 1e-6);
        assert!((sched.lr_at(4) - 0.05).abs() < 1e-6);
        assert!((sched.lr_at(9) - 0.1).abs() < 1e-6);
        assert!((sched.lr_at(100) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], true);
        let loss = x.scale(2.0).sum();
        loss.backward();
        let norm = clip_grad_norm(&[x.clone()], 1.0);
        assert!((norm - (12.0f32).sqrt()).abs() < 1e-4);
        let g = x.grad_data().unwrap();
        let clipped_norm: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 0.01, "clipped norm should be ~1.0, got {}", clipped_norm);
    }

    // --- New optimizer tests ---

    #[test]
    fn test_rmsprop_step() {
        // Minimize f(x) = x^2
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = RmsProp::new(vec![x.clone()], 0.05).alpha(0.9);

        for _ in 0..500 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(
            x.data()[0].abs() < 0.5,
            "RmsProp should converge near 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_rmsprop_centered() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = RmsProp::new(vec![x.clone()], 0.05).alpha(0.9).centered(true);

        for _ in 0..500 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(
            x.data()[0].abs() < 0.5,
            "Centered RmsProp should converge near 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_rmsprop_with_momentum() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = RmsProp::new(vec![x.clone()], 0.01).momentum(0.9);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(
            x.data()[0].abs() < 0.1,
            "RmsProp+momentum should converge near 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_adagrad_step() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = Adagrad::new(vec![x.clone()], 0.5);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(
            x.data()[0].abs() < 0.5,
            "Adagrad should converge near 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_adamax_step() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = Adamax::new(vec![x.clone()], 0.1);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(
            x.data()[0].abs() < 0.1,
            "Adamax should converge near 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_adadelta_step() {
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = AdaDelta::new(vec![x.clone()], 1.0);

        for _ in 0..1000 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        // AdaDelta converges slowly; verify progress toward 0
        assert!(
            x.data()[0].abs() < 2.0,
            "AdaDelta should make progress toward 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_lion_step() {
        // Minimize f(x) = x^2
        // Lion uses sign-based updates, so it oscillates around 0 with step lr.
        // With lr=0.01, it will approach but oscillate within lr of 0.
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = Lion::new(vec![x.clone()], 0.005);

        for _ in 0..1000 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        // Lion oscillates around 0 with amplitude related to lr, check loosely
        assert!(
            x.data()[0].abs() < 0.15,
            "Lion should converge near 0, got {}",
            x.data()[0]
        );
    }

    #[test]
    fn test_lion_sign_update() {
        // Verify that Lion uses sign-based updates
        let x = Tensor::new(vec![2.0, -3.0], vec![2], true);
        let mut opt = Lion::new(vec![x.clone()], 0.1);

        let loss = x.pow(2.0).sum();
        loss.backward();
        // grads = [4.0, -6.0]
        opt.step();

        // With m=0 initially: update = sign(0.9*0 + 0.1*grad) = sign(grad)
        // x[0] -= 0.1 * sign(4.0) = 2.0 - 0.1 = 1.9
        // x[1] -= 0.1 * sign(-6.0) = -3.0 + 0.1 = -2.9
        let d = x.data();
        assert!((d[0] - 1.9).abs() < 1e-5, "Lion x[0]={}", d[0]);
        assert!((d[1] - (-2.9)).abs() < 1e-5, "Lion x[1]={}", d[1]);
    }

    #[test]
    fn test_adafactor_step() {
        // 2D param to exercise factored path
        let x = Tensor::new(vec![4.0, 3.0, 2.0, 1.0], vec![2, 2], true);
        let mut opt = Adafactor::new(vec![x.clone()], 0.1);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        let d = x.data();
        let max_abs = d.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(
            max_abs < 1.0,
            "Adafactor should converge toward 0, got max_abs={}",
            max_abs
        );
    }

    #[test]
    fn test_adafactor_1d() {
        // 1D param to exercise non-factored path
        let x = Tensor::new(vec![4.0], vec![1], true);
        let mut opt = Adafactor::new(vec![x.clone()], 0.1);

        for _ in 0..200 {
            let loss = x.pow(2.0).sum();
            loss.backward();
            opt.step();
            opt.zero_grad();
        }
        assert!(
            x.data()[0].abs() < 1.0,
            "Adafactor 1D should converge toward 0, got {}",
            x.data()[0]
        );
    }

    // --- New scheduler tests ---

    #[test]
    fn test_exponential_decay_lr() {
        let sched = ExponentialDecayLr::new(0.1, 0.9);
        assert!((sched.lr_at(0) - 0.1).abs() < 1e-6);
        assert!((sched.lr_at(1) - 0.09).abs() < 1e-6);
        assert!((sched.lr_at(2) - 0.081).abs() < 1e-5);
    }

    #[test]
    fn test_linear_schedule_lr() {
        let sched = LinearScheduleLr::new(0.1, 0.01, 100);
        assert!((sched.lr_at(0) - 0.1).abs() < 1e-6);
        assert!((sched.lr_at(50) - 0.055).abs() < 1e-5);
        assert!((sched.lr_at(100) - 0.01).abs() < 1e-6);
        // Clamped past num_steps
        assert!((sched.lr_at(200) - 0.01).abs() < 1e-6);
    }

    #[test]
    fn test_join_schedules() {
        let warmup = Box::new(WarmupLr::new(0.1, 10)) as Box<dyn LrSchedule>;
        let cosine = Box::new(CosineAnnealingLr::new(0.1, 90)) as Box<dyn LrSchedule>;
        let sched = JoinSchedules::new(vec![10], vec![warmup, cosine]);

        // During warmup phase (step < 10)
        assert!((sched.lr_at(0) - 0.01).abs() < 1e-6);
        assert!((sched.lr_at(4) - 0.05).abs() < 1e-6);
        // After boundary, uses cosine schedule
        let lr_at_10 = sched.lr_at(10);
        assert!(lr_at_10 > 0.0 && lr_at_10 <= 0.1);
    }

    #[test]
    fn test_lr_schedule_trait() {
        // Verify existing schedulers implement the trait
        let step_sched: Box<dyn LrSchedule> = Box::new(StepLr::new(0.1, 10, 0.5));
        assert!((step_sched.lr_at(0) - 0.1).abs() < 1e-6);

        let warmup: Box<dyn LrSchedule> = Box::new(WarmupLr::new(0.1, 10));
        assert!((warmup.lr_at(0) - 0.01).abs() < 1e-6);

        let cosine: Box<dyn LrSchedule> = Box::new(CosineAnnealingLr::new(0.1, 100));
        assert!((cosine.lr_at(0) - 0.1).abs() < 1e-6);
    }
}
