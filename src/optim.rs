use crate::tensor::Tensor;

// --- Gradient clipping ---

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
                // We can't modify grad directly, so we scale during the step.
                // Instead, apply scaling as part of a manual step.
                // This is a no-op here; actual clipping is applied in the optimizer.
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

// --- SGD with momentum ---

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
            let grad = match p.grad_data() {
                Some(g) => g,
                None => continue,
            };
            let vel = &mut self.velocity[i];
            let mut inner = p.0.borrow_mut();

            for j in 0..inner.data.len() {
                let mut g = grad[j];
                if self.weight_decay != 0.0 {
                    g += self.weight_decay * inner.data[j];
                }
                if self.momentum != 0.0 {
                    vel[j] = self.momentum * vel[j] + g;
                    if self.nesterov {
                        g = g + self.momentum * vel[j];
                    } else {
                        g = vel[j];
                    }
                }
                inner.data[j] -= self.lr * g;
            }
            inner.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// --- Adam / AdamW ---

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
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.size()]).collect();
        Adam {
            params, lr,
            beta1: 0.9, beta2: 0.999, eps: 1e-8,
            weight_decay: 0.0, decoupled_wd: false,
            m, v, t: 0,
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
            let grad = match p.grad_data() {
                Some(g) => g,
                None => continue,
            };
            let m = &mut self.m[i];
            let v = &mut self.v[i];
            let mut inner = p.0.borrow_mut();

            for j in 0..inner.data.len() {
                let mut g = grad[j];

                // L2 regularization (classic Adam)
                if self.weight_decay != 0.0 && !self.decoupled_wd {
                    g += self.weight_decay * inner.data[j];
                }

                // Update biased first & second moment estimates
                m[j] = self.beta1 * m[j] + (1.0 - self.beta1) * g;
                v[j] = self.beta2 * v[j] + (1.0 - self.beta2) * g * g;

                // Bias-corrected estimates
                let m_hat = m[j] / bc1;
                let v_hat = v[j] / bc2;

                // Decoupled weight decay (AdamW)
                if self.decoupled_wd && self.weight_decay != 0.0 {
                    inner.data[j] -= self.lr * self.weight_decay * inner.data[j];
                }

                inner.data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
            inner.op = crate::tensor::Op::None;
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

// --- Learning rate schedulers ---

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

    pub fn lr_at(&self, epoch: usize) -> f32 {
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

    pub fn lr_at(&self, epoch: usize) -> f32 {
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

    pub fn lr_at(&self, step: usize) -> f32 {
        if step < self.warmup_steps {
            self.base_lr * (step + 1) as f32 / self.warmup_steps as f32
        } else {
            self.base_lr
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_basic() {
        // f(x) = x^2, minimum at x=0
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
        // f(x, y) = x^2 + 10*y^2, minimum at (0, 0)
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
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2, min at (1,1)
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
        // Rosenbrock is hard — just verify loss has decreased significantly
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
        // Midpoint should be ~half
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
        // grad = [2, 2, 2], norm = sqrt(12) ≈ 3.46
        let norm = clip_grad_norm(&[x.clone()], 1.0);
        assert!((norm - (12.0f32).sqrt()).abs() < 1e-4);
        // After clipping, norm should be ~1.0
        let g = x.grad_data().unwrap();
        let clipped_norm: f32 = g.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((clipped_norm - 1.0).abs() < 0.01, "clipped norm should be ~1.0, got {}", clipped_norm);
    }
}
