use crate::tensor::Tensor;

// ===========================================================================
// Module trait
// ===========================================================================

/// Trait for neural network modules with forward pass and parameter access.
pub trait Module {
    /// Run the forward pass on input tensor `x`.
    fn forward(&self, x: &Tensor) -> Tensor;

    /// Return references to all learnable parameters.
    fn params(&self) -> Vec<&Tensor>;

    /// Return named parameters with a given prefix.
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        self.params()
            .into_iter()
            .enumerate()
            .map(|(i, p)| (format!("{}.param_{}", prefix, i), p))
            .collect()
    }

    /// Switch module to training mode (default no-op).
    fn train(&mut self) {}

    /// Switch module to evaluation mode (default no-op).
    fn eval(&mut self) {}
}

// ===========================================================================
// Identity
// ===========================================================================

/// Identity module: returns its input unchanged.
pub struct Identity;

impl Module for Identity {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.clone()
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// Sequential
// ===========================================================================

/// Sequential container: chains modules in order.
pub struct Sequential {
    layers: Vec<Box<dyn Module>>,
}

impl Sequential {
    pub fn new(layers: Vec<Box<dyn Module>>) -> Self {
        Sequential { layers }
    }
}

impl Module for Sequential {
    fn forward(&self, x: &Tensor) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }

    fn params(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| l.named_params(&format!("{}.{}", prefix, i)))
            .collect()
    }
}

// ===========================================================================
// Linear (fully connected) layer: y = xW^T + b
// ===========================================================================

pub struct Linear {
    pub weight: Tensor, // [in_features, out_features]
    pub bias: Tensor,   // [1, out_features]
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Linear {
            weight: Tensor::randn(&[in_features, out_features], true),
            bias: Tensor::zeros(&[1, out_features], true),
        }
    }

    /// x: [batch, in_features] -> [batch, out_features]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight).add_bias(&self.bias)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        Linear::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        Linear::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        Linear::named_params(self, prefix)
    }
}

// ===========================================================================
// PReLU: Parametric ReLU with learnable slope for negative inputs
// ===========================================================================

pub struct PReLU {
    pub weight: Tensor, // learnable slope for negative inputs
}

impl PReLU {
    /// Create a PReLU with `num_parameters` learnable slopes, initialized to 0.25.
    pub fn new(num_parameters: usize) -> Self {
        PReLU {
            weight: Tensor::full(&[num_parameters], 0.25, true),
        }
    }

    /// Forward pass: x if x > 0, weight * x otherwise.
    /// If weight has one element, it broadcasts to all channels.
    /// Otherwise weight should match the channel dimension.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Compose as: relu(x) + weight * (-relu(-x))
        // which equals: relu(x) - weight * relu(-x)
        let pos = x.relu();
        let neg = x.neg().relu();
        let w_shape = self.weight.shape();
        if w_shape.len() == 1 && w_shape[0] == 1 {
            // Single parameter: broadcast as scalar
            let w_data = self.weight.data();
            pos.sub(&neg.scale(w_data[0]))
        } else {
            // Multi-parameter: need to match shape for element-wise mul
            // For [batch, channels] input with [channels] weight:
            let x_shape = x.shape();
            if x_shape.len() == 2 && w_shape[0] == x_shape[1] {
                // Reshape weight to [1, channels] for broadcasting
                let w_reshaped = self.weight.reshape(vec![1, w_shape[0]]);
                pos.sub(&neg.mul(&w_reshaped))
            } else {
                // Fallback: element-wise with broadcast
                pos.sub(&neg.mul(&self.weight))
            }
        }
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
        ]
    }
}

impl Module for PReLU {
    fn forward(&self, x: &Tensor) -> Tensor {
        PReLU::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        PReLU::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        PReLU::named_params(self, prefix)
    }
}

// ===========================================================================
// Embedding: lookup table for token embeddings
// ===========================================================================

pub struct Embedding {
    pub weight: Tensor, // [num_embeddings, embedding_dim]
    num_embeddings: usize,
    embedding_dim: usize,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        Embedding {
            weight: Tensor::randn(&[num_embeddings, embedding_dim], true),
            num_embeddings,
            embedding_dim,
        }
    }

    /// indices: flat list of token indices -> [len, embedding_dim]
    pub fn forward(&self, indices: &[usize]) -> Tensor {
        let w = self.weight.data();
        let mut data = Vec::with_capacity(indices.len() * self.embedding_dim);
        for &idx in indices {
            assert!(idx < self.num_embeddings, "embedding index out of range");
            let offset = idx * self.embedding_dim;
            data.extend_from_slice(&w[offset..offset + self.embedding_dim]);
        }
        // Return as a non-grad tensor (embedding backward is handled via select)
        Tensor::new(data, vec![indices.len(), self.embedding_dim], false)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}

impl Module for Embedding {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Interpret x as indices (round to usize)
        let data = x.data();
        let indices: Vec<usize> = data.iter().map(|&v| v as usize).collect();
        Embedding::forward(self, &indices)
    }
    fn params(&self) -> Vec<&Tensor> {
        Embedding::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![(format!("{}.weight", prefix), &self.weight)]
    }
}

// ===========================================================================
// Loss functions
// ===========================================================================

/// CrossEntropyLoss: takes logits [batch, num_classes] and target class indices.
/// Returns scalar loss.
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    let shape = logits.shape();
    assert_eq!(shape.len(), 2);
    let (batch, num_classes) = (shape[0], shape[1]);
    assert_eq!(targets.len(), batch);

    // Log-softmax (numerically stable)
    let log_probs = logits.log_softmax(-1);

    // NLL: -log_prob[i, target[i]] averaged over batch
    let mut indices = Vec::with_capacity(batch);
    for (i, &t) in targets.iter().enumerate() {
        indices.push(i * num_classes + t);
    }
    let selected = log_probs.select(&indices);
    selected.mean().neg()
}

/// Mean squared error: mean((pred - target)^2)
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).pow(2.0).mean()
}

/// Mean absolute error (L1 loss): mean(|pred - target|)
pub fn l1_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target).abs().mean()
}

/// Negative log-likelihood loss. `log_probs` should be [batch, classes] of log-probabilities
/// (e.g., output of log_softmax). `targets` are class indices.
pub fn nll_loss(log_probs: &Tensor, targets: &[usize]) -> Tensor {
    let shape = log_probs.shape();
    assert_eq!(shape.len(), 2);
    let (batch, num_classes) = (shape[0], shape[1]);
    assert_eq!(targets.len(), batch);

    let mut indices = Vec::with_capacity(batch);
    for (i, &t) in targets.iter().enumerate() {
        indices.push(i * num_classes + t);
    }
    let selected = log_probs.select(&indices);
    selected.mean().neg()
}

/// Smooth L1 loss (Huber-like): for each element,
///   0.5 * x^2 / beta   if |x| < beta
///   |x| - 0.5 * beta   otherwise
/// where x = pred - target. Uses tensor ops for autograd.
pub fn smooth_l1_loss(pred: &Tensor, target: &Tensor, beta: f32) -> Tensor {
    assert!(beta > 0.0, "beta must be positive");

    // Fused single-pass for non-grad tensors
    if !pred.requires_grad() && !target.requires_grad() {
        let p = pred.data();
        let t = target.data();
        let n = p.len();
        let half_beta = 0.5 * beta;
        let inv_beta = 0.5 / beta;
        let mut sum = 0.0f64;
        for i in 0..n {
            let d = (p[i] - t[i]).abs();
            sum += if d < beta { (d * d) as f64 * inv_beta as f64 } else { (d - half_beta) as f64 };
        }
        return Tensor::new(vec![(sum / n as f64) as f32], vec![1], false);
    }

    let diff = pred.sub(target);
    let abs_diff = diff.abs();
    let abs_data = abs_diff.data();
    let shape = abs_diff.shape();
    let quadratic = diff.pow(2.0).scale(0.5 / beta);
    let half_beta_t = Tensor::full(&shape, 0.5 * beta, false);
    let linear = abs_diff.sub(&half_beta_t);
    let mask_data: Vec<f32> = abs_data.iter().map(|&v| if v < beta { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::new(mask_data, shape.clone(), false);
    let inv_mask_data: Vec<f32> = abs_data.iter().map(|&v| if v < beta { 0.0 } else { 1.0 }).collect();
    let inv_mask = Tensor::new(inv_mask_data, shape, false);
    let result = quadratic.mul(&mask).add(&linear.mul(&inv_mask));
    result.mean()
}

/// Huber loss with delta threshold:
///   0.5 * x^2              if |x| <= delta
///   delta * (|x| - 0.5*delta)  otherwise
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Tensor {
    assert!(delta > 0.0, "delta must be positive");

    // Fused single-pass path when no gradients are needed — avoids ~8 intermediate tensor allocations
    if !pred.requires_grad() && !target.requires_grad() {
        let pd = pred.data();
        let td = target.data();
        let n = pd.len();
        debug_assert_eq!(n, td.len());
        let half_delta = 0.5 * delta;
        let mut sum = 0.0f64;
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let chunks = n / 4;
            let vdelta = unsafe { vdupq_n_f32(delta) };
            let vhalf = unsafe { vdupq_n_f32(0.5) };
            let vhalf_delta = unsafe { vdupq_n_f32(half_delta) };
            let mut vacc = unsafe { vdupq_n_f32(0.0) };
            unsafe {
                for c in 0..chunks {
                    let off = c * 4;
                    let vp = vld1q_f32(pd.as_ptr().add(off));
                    let vt = vld1q_f32(td.as_ptr().add(off));
                    let diff = vsubq_f32(vp, vt);
                    let abs_diff = vabsq_f32(diff);
                    // quadratic = 0.5 * diff^2
                    let quad = vmulq_f32(vhalf, vmulq_f32(diff, diff));
                    // linear = delta * (|diff| - 0.5 * delta)
                    let lin = vmulq_f32(vdelta, vsubq_f32(abs_diff, vhalf_delta));
                    // mask: |diff| < delta => quad, else => lin
                    let mask = vcltq_f32(abs_diff, vdelta);
                    let result = vbslq_f32(mask, quad, lin);
                    vacc = vaddq_f32(vacc, result);
                }
                sum = vaddvq_f32(vacc) as f64;
            }
            for i in (chunks * 4)..n {
                let diff = pd[i] - td[i];
                let abs_diff = diff.abs();
                if abs_diff < delta {
                    sum += (0.5 * diff * diff) as f64;
                } else {
                    sum += (delta * (abs_diff - half_delta)) as f64;
                }
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..n {
                let diff = pd[i] - td[i];
                let abs_diff = diff.abs();
                if abs_diff < delta {
                    sum += (0.5 * diff * diff) as f64;
                } else {
                    sum += (delta * (abs_diff - half_delta)) as f64;
                }
            }
        }
        let mean = (sum / n as f64) as f32;
        return Tensor::new(vec![mean], vec![1], false);
    }

    // Grad-tracking path: compose from tensor ops for autograd
    let diff = pred.sub(target);
    let abs_diff = diff.abs();
    let abs_data = abs_diff.data();
    let shape = abs_diff.shape();

    let quadratic = diff.pow(2.0).scale(0.5);
    let half_delta = Tensor::full(&shape, 0.5 * delta, false);
    let linear = abs_diff.sub(&half_delta).scale(delta);

    let mask_data: Vec<f32> = abs_data.iter().map(|&v| if v <= delta { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::new(mask_data, shape.clone(), false);
    let inv_mask_data: Vec<f32> = abs_data.iter().map(|&v| if v <= delta { 0.0 } else { 1.0 }).collect();
    let inv_mask = Tensor::new(inv_mask_data, shape, false);

    let result = quadratic.mul(&mask).add(&linear.mul(&inv_mask));
    result.mean()
}

/// KL divergence loss: mean(target * (log(target) - input)).
/// `input` should be log-probabilities, `target` should be probabilities.
/// Following PyTorch convention where `input` is log-space.
pub fn kl_div_loss(input: &Tensor, target: &Tensor) -> Tensor {
    // KL(P || Q) = sum P * (log P - log Q) where input = log Q, target = P
    // = sum target * (log(target) - input)
    // We need to handle target=0 carefully (0*log(0) = 0).
    let target_data = target.data();
    let shape = target.shape();
    // Compute log(target) safely: where target > 0, use log; where 0, use 0.
    let log_target_data: Vec<f32> = target_data
        .iter()
        .map(|&v| if v > 0.0 { v.ln() } else { 0.0 })
        .collect();
    let log_target = Tensor::new(log_target_data, shape, false);
    // target * (log_target - input)
    target.mul(&log_target.sub(input)).mean()
}

/// Cosine similarity loss: 1 - cos_sim(a, b),
/// where cos_sim = dot(a, b) / (||a|| * ||b||).
/// `a` and `b` must have the same shape.
pub fn cosine_similarity_loss(a: &Tensor, b: &Tensor) -> Tensor {
    // Fused single-pass path when no gradients are needed
    if !a.requires_grad() && !b.requires_grad() {
        let ad = a.data();
        let bd = b.data();
        let n = ad.len();
        debug_assert_eq!(n, bd.len());
        let mut dot = 0.0f64;
        let mut sum_a2 = 0.0f64;
        let mut sum_b2 = 0.0f64;
        #[cfg(target_arch = "aarch64")]
        {
            use std::arch::aarch64::*;
            let chunks = n / 4;
            let mut vdot = unsafe { vdupq_n_f32(0.0) };
            let mut va2 = unsafe { vdupq_n_f32(0.0) };
            let mut vb2 = unsafe { vdupq_n_f32(0.0) };
            for i in 0..chunks {
                let off = i * 4;
                unsafe {
                    let va = vld1q_f32(ad.as_ptr().add(off));
                    let vb = vld1q_f32(bd.as_ptr().add(off));
                    vdot = vfmaq_f32(vdot, va, vb);
                    va2 = vfmaq_f32(va2, va, va);
                    vb2 = vfmaq_f32(vb2, vb, vb);
                }
            }
            dot = unsafe { vaddvq_f32(vdot) } as f64;
            sum_a2 = unsafe { vaddvq_f32(va2) } as f64;
            sum_b2 = unsafe { vaddvq_f32(vb2) } as f64;
            for i in (chunks * 4)..n {
                dot += (ad[i] * bd[i]) as f64;
                sum_a2 += (ad[i] * ad[i]) as f64;
                sum_b2 += (bd[i] * bd[i]) as f64;
            }
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            for i in 0..n {
                dot += (ad[i] * bd[i]) as f64;
                sum_a2 += (ad[i] * ad[i]) as f64;
                sum_b2 += (bd[i] * bd[i]) as f64;
            }
        }
        let cos_sim = dot / ((sum_a2.sqrt() * sum_b2.sqrt()) + 1e-8);
        return Tensor::new(vec![1.0 - cos_sim as f32], vec![1], false);
    }

    // Autograd-compatible path
    // dot = sum(a * b), norm_a = sqrt(sum(a^2)), norm_b = sqrt(sum(b^2))
    let dot = a.mul(b).sum();
    let norm_a = a.square().sum().sqrt();
    let norm_b = b.square().sum().sqrt();
    // cos_sim = dot / (norm_a * norm_b + eps)
    let eps = Tensor::new(vec![1e-8], vec![1], false);
    let denom = norm_a.mul(&norm_b).add(&eps);
    let cos_sim = dot.div(&denom);
    let one = Tensor::new(vec![1.0], vec![1], false);
    one.sub(&cos_sim)
}

/// Triplet margin loss: max(0, ||anchor - pos||^2 - ||anchor - neg||^2 + margin)
pub fn triplet_loss(anchor: &Tensor, pos: &Tensor, neg: &Tensor, margin: f32) -> Tensor {
    let d_pos = anchor.sub(pos).square().sum();
    let d_neg = anchor.sub(neg).square().sum();
    let margin_t = Tensor::new(vec![margin], vec![1], false);
    let raw = d_pos.sub(&d_neg).add(&margin_t);
    // max(0, raw) via relu
    raw.relu()
}

/// Hinge loss: mean(max(0, 1 - pred * target))
/// `target` should be +1 or -1.
pub fn hinge_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let one = Tensor::full(&pred.shape(), 1.0, false);
    // 1 - pred * target
    one.sub(&pred.mul(target)).relu().mean()
}

/// Log-cosh loss: mean(log(cosh(pred - target)))
/// Approximation: log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)
/// We use the exact formula via tensor ops: (exp(x) + exp(-x))/2 then log.
pub fn log_cosh_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target);
    // log(cosh(x)) = x + softplus(-2x) - log(2)
    // softplus(y) = log(1 + exp(y))
    // We compose: exp(diff) + exp(-diff) = exp(diff) + exp(neg(diff))
    let exp_pos = diff.exp();
    let exp_neg = diff.neg().exp();
    // cosh = (exp_pos + exp_neg) / 2
    let cosh_val = exp_pos.add(&exp_neg).scale(0.5);
    cosh_val.log().mean()
}

/// Margin ranking loss: mean(max(0, -y * (x1 - x2) + margin))
/// `y` should be +1 or -1.
pub fn margin_ranking_loss(x1: &Tensor, x2: &Tensor, y: &Tensor, margin: f32) -> Tensor {
    let margin_t = Tensor::full(&x1.shape(), margin, false);
    // -y * (x1 - x2) + margin
    let raw = y.neg().mul(&x1.sub(x2)).add(&margin_t);
    raw.relu().mean()
}

/// Gaussian NLL loss: 0.5 * mean(log(var) + (input - target)^2 / var)
/// `var` must be positive.
pub fn gaussian_nll_loss(input: &Tensor, target: &Tensor, var: &Tensor) -> Tensor {
    let diff_sq = input.sub(target).pow(2.0);
    let log_var = var.log();
    // 0.5 * (log(var) + diff^2 / var)
    log_var.add(&diff_sq.div(var)).scale(0.5).mean()
}

// ===========================================================================
// RMSNorm
// ===========================================================================

/// Root Mean Square Layer Normalization.
pub struct RMSNorm {
    pub weight: Tensor,
    pub eps: f32,
    dim: usize,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        RMSNorm {
            weight: Tensor::ones(&[dim], true),
            eps,
            dim,
        }
    }

    /// x: [batch, dim] or any shape where last dim = self.dim.
    /// forward: x * weight / sqrt(mean(x^2) + eps)
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let last = *shape.last().unwrap();
        assert_eq!(last, self.dim, "RMSNorm: last dim must match");

        // Fast fused path for inference (input doesn't need grad)
        if !x.requires_grad() {
            let data = x.data();
            let n = data.len();
            let num_vecs = n / self.dim;
            let w = self.weight.data();
            let mut out = vec![0.0f32; n];
            let dim = self.dim;
            let eps = self.eps;
            let inv_dim = 1.0 / dim as f32;

            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::aarch64::*;
                let chunks4 = dim / 4;
                let rem_start = chunks4 * 4;
                for v in 0..num_vecs {
                    let base = v * dim;
                    let xp = data[base..].as_ptr();
                    let wp = w.as_ptr();
                    let op = out[base..].as_mut_ptr();

                    // Pass 1: compute sum of squares with NEON
                    let mut sum_sq = unsafe { vdupq_n_f32(0.0) };
                    for i in 0..chunks4 {
                        let off = i * 4;
                        unsafe {
                            let vx = vld1q_f32(xp.add(off));
                            sum_sq = vfmaq_f32(sum_sq, vx, vx);
                        }
                    }
                    let mut ss: f32 = unsafe {
                        vaddvq_f32(sum_sq)
                    };
                    for i in rem_start..dim {
                        let val = data[base + i];
                        ss += val * val;
                    }
                    let inv_rms = 1.0 / (ss * inv_dim + eps).sqrt();
                    let v_scale = unsafe { vdupq_n_f32(inv_rms) };

                    // Pass 2: x * inv_rms * weight with NEON
                    for i in 0..chunks4 {
                        let off = i * 4;
                        unsafe {
                            let vx = vld1q_f32(xp.add(off));
                            let vw = vld1q_f32(wp.add(off));
                            let scaled = vmulq_f32(vx, v_scale);
                            vst1q_f32(op.add(off), vmulq_f32(scaled, vw));
                        }
                    }
                    for i in rem_start..dim {
                        out[base + i] = data[base + i] * inv_rms * w[i];
                    }
                }
            }
            #[cfg(not(target_arch = "aarch64"))]
            {
                for v in 0..num_vecs {
                    let base = v * dim;
                    let slice = &data[base..base + dim];
                    let ss: f32 = slice.iter().map(|&val| val * val).sum::<f32>();
                    let inv_rms = 1.0 / (ss * inv_dim + eps).sqrt();
                    for i in 0..dim {
                        out[base + i] = data[base + i] * inv_rms * w[i];
                    }
                }
            }

            return Tensor::new(out, shape, false);
        }

        // Autograd path: compose from tensor ops for gradient tracking
        let data = x.data();
        let n = data.len();
        let num_vecs = n / self.dim;

        // Compute RMS per-vector, build inverse-RMS tensor
        let mut inv_rms_data = Vec::with_capacity(n);
        for v in 0..num_vecs {
            let start = v * self.dim;
            let slice = &data[start..start + self.dim];
            let mean_sq: f32 =
                slice.iter().map(|&val| val * val).sum::<f32>() / self.dim as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            for _ in 0..self.dim {
                inv_rms_data.push(inv_rms);
            }
        }
        let inv_rms_t = Tensor::new(inv_rms_data, shape.clone(), false);

        // Broadcast weight across batch dimension
        let w = self.weight.data();
        let mut weight_data = Vec::with_capacity(n);
        for _ in 0..num_vecs {
            weight_data.extend_from_slice(&w);
        }
        let weight_broad = Tensor::new(weight_data, shape, false);

        // x * inv_rms * weight — uses autograd-aware mul
        x.mul(&inv_rms_t).mul(&weight_broad)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        RMSNorm::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        RMSNorm::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![(format!("{}.weight", prefix), &self.weight)]
    }
}

// ===========================================================================
// Dropout
// ===========================================================================

/// Dropout layer: randomly zeros elements during training.
pub struct Dropout {
    pub p: f32,
    pub training: bool,
}

impl Dropout {
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        Dropout { p, training: true }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }
        // Generate random mask on CPU using simple LCG PRNG
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = Cell::new(123456789);
        }
        let data = x.data();
        let shape = x.shape();
        let n = data.len();
        let scale = 1.0 / (1.0 - self.p);

        let mut mask_data = Vec::with_capacity(n);
        for _ in 0..n {
            let r = SEED.with(|s| {
                let v = s.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s.set(v);
                (v >> 33) as f32 / (1u64 << 31) as f32
            });
            mask_data.push(if r >= self.p { scale } else { 0.0 });
        }

        let mask = Tensor::new(mask_data, shape, false);
        x.mul(&mask)
    }
}

impl Module for Dropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        Dropout::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// RNN (Elman RNN)
// ===========================================================================

/// Simple Elman RNN: h_t = tanh(x_t @ W_ih + h_{t-1} @ W_hh + b_ih + b_hh)
pub struct RNN {
    pub weight_ih: Tensor, // [input_size, hidden_size]
    pub weight_hh: Tensor, // [hidden_size, hidden_size]
    pub bias_ih: Tensor,   // [1, hidden_size]
    pub bias_hh: Tensor,   // [1, hidden_size]
    input_size: usize,
    hidden_size: usize,
}

impl RNN {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        RNN {
            weight_ih: Tensor::randn(&[input_size, hidden_size], true),
            weight_hh: Tensor::randn(&[hidden_size, hidden_size], true),
            bias_ih: Tensor::zeros(&[1, hidden_size], true),
            bias_hh: Tensor::zeros(&[1, hidden_size], true),
            input_size,
            hidden_size,
        }
    }

    /// x: [seq_len, input_size], h0: [1, hidden_size]
    /// Returns (output [seq_len, hidden_size], h_n [1, hidden_size])
    pub fn forward(&self, x: &Tensor, h0: &Tensor) -> (Tensor, Tensor) {
        let shape = x.shape();
        assert_eq!(shape.len(), 2);
        let seq_len = shape[0];
        assert_eq!(shape[1], self.input_size);

        let x_data = x.data();
        let mut h = h0.clone();
        let mut outputs = Vec::with_capacity(seq_len * self.hidden_size);

        for t in 0..seq_len {
            let start = t * self.input_size;
            let x_t_data = x_data[start..start + self.input_size].to_vec();
            let x_t = Tensor::new(x_t_data, vec![1, self.input_size], false);

            // h = tanh(x_t @ W_ih + b_ih + h @ W_hh + b_hh)
            let ih = x_t.matmul(&self.weight_ih).add_bias(&self.bias_ih);
            let hh = h.matmul(&self.weight_hh).add_bias(&self.bias_hh);
            h = ih.add(&hh).tanh();

            outputs.extend(h.data());
        }

        let output = Tensor::new(outputs, vec![seq_len, self.hidden_size], false);
        (output, h)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight_ih, &self.weight_hh, &self.bias_ih, &self.bias_hh]
    }
}

// ===========================================================================
// LSTM
// ===========================================================================

/// Long Short-Term Memory: 4 gates (input, forget, cell, output).
pub struct LSTM {
    pub weight_ih: Tensor, // [input_size, 4 * hidden_size]
    pub weight_hh: Tensor, // [hidden_size, 4 * hidden_size]
    pub bias_ih: Tensor,   // [1, 4 * hidden_size]
    pub bias_hh: Tensor,   // [1, 4 * hidden_size]
    input_size: usize,
    hidden_size: usize,
}

impl LSTM {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        LSTM {
            weight_ih: Tensor::randn(&[input_size, 4 * hidden_size], true),
            weight_hh: Tensor::randn(&[hidden_size, 4 * hidden_size], true),
            bias_ih: Tensor::zeros(&[1, 4 * hidden_size], true),
            bias_hh: Tensor::zeros(&[1, 4 * hidden_size], true),
            input_size,
            hidden_size,
        }
    }

    /// x: [seq_len, input_size], h0: [1, hidden_size], c0: [1, hidden_size]
    /// Returns (output [seq_len, hidden_size], h_n [1, hidden_size], c_n [1, hidden_size])
    pub fn forward(
        &self, x: &Tensor, h0: &Tensor, c0: &Tensor,
    ) -> (Tensor, Tensor, Tensor) {
        let shape = x.shape();
        assert_eq!(shape.len(), 2);
        let seq_len = shape[0];
        assert_eq!(shape[1], self.input_size);

        let mut h = h0.clone();
        let mut c = c0.clone();
        let hs = self.hidden_size;

        // Pre-compute index ranges for gate splitting
        let i_indices: Vec<usize> = (0..hs).collect();
        let f_indices: Vec<usize> = (hs..2 * hs).collect();
        let g_indices: Vec<usize> = (2 * hs..3 * hs).collect();
        let o_indices: Vec<usize> = (3 * hs..4 * hs).collect();

        let mut h_steps: Vec<Tensor> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            // Extract x_t preserving autograd into input
            let x_t = x.index_select(0, &[t]); // [1, input_size]

            // gates = x_t @ W_ih + b_ih + h @ W_hh + b_hh  [1, 4*hidden_size]
            let gates = x_t
                .matmul(&self.weight_ih)
                .add_bias(&self.bias_ih)
                .add(&h.matmul(&self.weight_hh).add_bias(&self.bias_hh));

            // Split into i, f, g, o gates preserving autograd
            let i_gate = gates.index_select(1, &i_indices).sigmoid();
            let f_gate = gates.index_select(1, &f_indices).sigmoid();
            let g_gate = gates.index_select(1, &g_indices).tanh();
            let o_gate = gates.index_select(1, &o_indices).sigmoid();

            // c = f * c + i * g
            c = f_gate.mul(&c).add(&i_gate.mul(&g_gate));
            // h = o * tanh(c)
            h = o_gate.mul(&c.tanh());

            h_steps.push(h.clone());
        }

        // Stack all hidden states preserving autograd
        let output = Tensor::stack(&h_steps, 0).reshape(vec![seq_len, hs]);
        (output, h, c)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight_ih, &self.weight_hh, &self.bias_ih, &self.bias_hh]
    }
}

// ===========================================================================
// GRU
// ===========================================================================

/// Gated Recurrent Unit: 3 gates (reset, update, new).
pub struct GRU {
    pub weight_ih: Tensor, // [input_size, 3 * hidden_size]
    pub weight_hh: Tensor, // [hidden_size, 3 * hidden_size]
    pub bias_ih: Tensor,   // [1, 3 * hidden_size]
    pub bias_hh: Tensor,   // [1, 3 * hidden_size]
    input_size: usize,
    hidden_size: usize,
}

impl GRU {
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        GRU {
            weight_ih: Tensor::randn(&[input_size, 3 * hidden_size], true),
            weight_hh: Tensor::randn(&[hidden_size, 3 * hidden_size], true),
            bias_ih: Tensor::zeros(&[1, 3 * hidden_size], true),
            bias_hh: Tensor::zeros(&[1, 3 * hidden_size], true),
            input_size,
            hidden_size,
        }
    }

    /// x: [seq_len, input_size], h0: [1, hidden_size]
    /// Returns (output [seq_len, hidden_size], h_n [1, hidden_size])
    pub fn forward(&self, x: &Tensor, h0: &Tensor) -> (Tensor, Tensor) {
        let shape = x.shape();
        assert_eq!(shape.len(), 2);
        let seq_len = shape[0];
        assert_eq!(shape[1], self.input_size);

        let x_data = x.data();
        let mut h = h0.clone();
        let mut outputs = Vec::with_capacity(seq_len * self.hidden_size);
        let hs = self.hidden_size;

        for t in 0..seq_len {
            let start = t * self.input_size;
            let x_t_data = x_data[start..start + self.input_size].to_vec();
            let x_t = Tensor::new(x_t_data, vec![1, self.input_size], false);

            // Compute input-hidden part
            let ih = x_t.matmul(&self.weight_ih).add_bias(&self.bias_ih);
            let ih_data = ih.data();

            // Compute hidden-hidden part
            let hh = h.matmul(&self.weight_hh).add_bias(&self.bias_hh);
            let hh_data = hh.data();

            // Split: r, z from ih and hh; apply sigmoid
            let r_data: Vec<f32> = (0..hs).map(|j| ih_data[j] + hh_data[j]).collect();
            let z_data: Vec<f32> = (0..hs).map(|j| ih_data[hs + j] + hh_data[hs + j]).collect();

            let r = Tensor::new(r_data, vec![1, hs], false).sigmoid();
            let z = Tensor::new(z_data, vec![1, hs], false).sigmoid();

            // n = tanh(ih_n + r * hh_n)
            let ih_n: Vec<f32> = ih_data[2 * hs..3 * hs].to_vec();
            let hh_n: Vec<f32> = hh_data[2 * hs..3 * hs].to_vec();
            let ih_n_t = Tensor::new(ih_n, vec![1, hs], false);
            let hh_n_t = Tensor::new(hh_n, vec![1, hs], false);
            let n = ih_n_t.add(&r.mul(&hh_n_t)).tanh();

            // h = (1 - z) * n + z * h
            let one = Tensor::ones(&[1, hs], false);
            h = one.sub(&z).mul(&n).add(&z.mul(&h));

            outputs.extend(h.data());
        }

        let output = Tensor::new(outputs, vec![seq_len, self.hidden_size], false);
        (output, h)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight_ih, &self.weight_hh, &self.bias_ih, &self.bias_hh]
    }
}

// ===========================================================================
// RoPE (Rotary Position Embedding)
// ===========================================================================

/// Rotary Position Embedding: precomputes sin/cos tables and applies to Q/K.
pub struct RoPE {
    cos_cache: Vec<Vec<f32>>, // [max_seq_len][head_dim]
    sin_cache: Vec<Vec<f32>>,
    head_dim: usize,
}

impl RoPE {
    /// Create a RoPE with precomputed tables for up to `max_seq_len` positions.
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32) -> Self {
        assert!(head_dim % 2 == 0, "head_dim must be even for RoPE");
        let half = head_dim / 2;
        let mut cos_cache = Vec::with_capacity(max_seq_len);
        let mut sin_cache = Vec::with_capacity(max_seq_len);

        for pos in 0..max_seq_len {
            let mut cos_row = Vec::with_capacity(head_dim);
            let mut sin_row = Vec::with_capacity(head_dim);
            for i in 0..half {
                let theta = (pos as f32) / base.powf(2.0 * i as f32 / head_dim as f32);
                let c = theta.cos();
                let s = theta.sin();
                cos_row.push(c);
                sin_row.push(s);
            }
            // Duplicate for the second half (same frequencies apply to paired dims)
            let cos_full: Vec<f32> = cos_row.iter().chain(cos_row.iter()).copied().collect();
            let sin_full: Vec<f32> = sin_row.iter().chain(sin_row.iter()).copied().collect();
            cos_cache.push(cos_full);
            sin_cache.push(sin_full);
        }

        RoPE { cos_cache, sin_cache, head_dim }
    }

    /// Apply rotary embedding to a tensor x of shape [seq_len, head_dim].
    /// Returns a new tensor with rotary positions applied.
    pub fn apply(&self, x: &Tensor, offset: usize) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 2);
        let seq_len = shape[0];
        assert_eq!(shape[1], self.head_dim);
        let half = self.head_dim / 2;

        let data = x.data();
        let mut out = Vec::with_capacity(data.len());

        for t in 0..seq_len {
            let pos = offset + t;
            assert!(pos < self.cos_cache.len(), "RoPE: position exceeds max_seq_len");
            let cos = &self.cos_cache[pos];
            let sin = &self.sin_cache[pos];
            let row_start = t * self.head_dim;

            for i in 0..half {
                let x0 = data[row_start + i];
                let x1 = data[row_start + half + i];
                // Rotation: [x0, x1] -> [x0*cos - x1*sin, x0*sin + x1*cos]
                out.push(x0 * cos[i] - x1 * sin[i]);
            }
            for i in 0..half {
                let x0 = data[row_start + i];
                let x1 = data[row_start + half + i];
                out.push(x0 * sin[i] + x1 * cos[i]);
            }
        }

        Tensor::new(out, shape, false)
    }
}

// ===========================================================================
// Conv1d
// ===========================================================================

/// 1D convolution layer.
pub struct Conv1d {
    pub weight: Tensor, // [out_channels, in_channels, kernel_size]
    pub bias: Tensor,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl Conv1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        Conv1d {
            weight: Tensor::randn(&[out_channels, in_channels, kernel_size], true),
            bias: Tensor::zeros(&[out_channels], true),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// x: [batch, in_channels, length] -> [batch, out_channels, out_length]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 3);
        let (batch, in_c, in_len) = (shape[0], shape[1], shape[2]);
        assert_eq!(in_c, self.in_channels);

        let out_len = (in_len + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let x_data = x.data();
        let w_data = self.weight.data();
        let b_data = self.bias.data();

        // im2col + BLAS path: reshape convolution as matrix multiply
        // col matrix: [in_channels * kernel_size, out_len] per batch
        // weight matrix: [out_channels, in_channels * kernel_size]
        // output = weight * col + bias
        let col_rows = in_c * self.kernel_size;
        let mut out_data = vec![0.0f32; batch * self.out_channels * out_len];

        for b in 0..batch {
            // Build im2col matrix for this batch element
            let mut col = vec![0.0f32; col_rows * out_len];
            for ic in 0..in_c {
                for k in 0..self.kernel_size {
                    let row = ic * self.kernel_size + k;
                    for ol in 0..out_len {
                        let il = (ol * self.stride + k) as isize - self.padding as isize;
                        if il >= 0 && (il as usize) < in_len {
                            col[row * out_len + ol] =
                                x_data[b * in_c * in_len + ic * in_len + il as usize];
                        }
                    }
                }
            }

            // Matrix multiply: [out_channels, col_rows] x [col_rows, out_len] -> [out_channels, out_len]
            #[cfg(target_os = "macos")]
            {
                crate::tensor::sgemm(
                    false, false,
                    self.out_channels, out_len, col_rows,
                    1.0, &w_data, col_rows,
                    &col, out_len,
                    0.0, &mut out_data[b * self.out_channels * out_len..], out_len,
                );
            }
            #[cfg(not(target_os = "macos"))]
            {
                for oc in 0..self.out_channels {
                    for ol in 0..out_len {
                        let mut sum = 0.0f32;
                        for k in 0..col_rows {
                            sum += w_data[oc * col_rows + k] * col[k * out_len + ol];
                        }
                        out_data[b * self.out_channels * out_len + oc * out_len + ol] = sum;
                    }
                }
            }
        }

        // Add bias
        for b in 0..batch {
            for oc in 0..self.out_channels {
                let offset = b * self.out_channels * out_len + oc * out_len;
                let bv = b_data[oc];
                for v in &mut out_data[offset..offset + out_len] {
                    *v += bv;
                }
            }
        }

        Tensor::new(
            out_data,
            vec![batch, self.out_channels, out_len],
            false,
        )
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}

// ===========================================================================
// AvgPool2d
// ===========================================================================

/// 2D average pooling layer.
pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        AvgPool2d { kernel_size, stride, padding }
    }

    /// x: [batch, channels, height, width] -> [batch, channels, out_h, out_w]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4);
        let (batch, channels, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let data = x.data();
        let mut out = vec![0.0f32; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0f32;
                        let mut count = 0u32;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let ih = (oh * self.stride + kh) as isize - self.padding as isize;
                                let iw = (ow * self.stride + kw) as isize - self.padding as isize;
                                if ih >= 0
                                    && (ih as usize) < h
                                    && iw >= 0
                                    && (iw as usize) < w
                                {
                                    let idx = b * channels * h * w
                                        + c * h * w
                                        + ih as usize * w
                                        + iw as usize;
                                    sum += data[idx];
                                    count += 1;
                                }
                            }
                        }
                        let out_idx =
                            b * channels * out_h * out_w + c * out_h * out_w + oh * out_w + ow;
                        out[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Tensor::new(out, vec![batch, channels, out_h, out_w], false)
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        AvgPool2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// AdaptiveAvgPool2d
// ===========================================================================

/// Adaptive 2D average pooling: produces a fixed output size regardless of input size.
/// Input: [N, C, H, W] -> Output: [N, C, out_h, out_w].
pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    pub fn new(output_size: (usize, usize)) -> Self {
        AdaptiveAvgPool2d { output_size }
    }

    /// Forward pass: adaptively pool each spatial region.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "AdaptiveAvgPool2d expects [N, C, H, W]");
        let (batch, channels, in_h, in_w) = (shape[0], shape[1], shape[2], shape[3]);
        let (out_h, out_w) = self.output_size;
        let data = x.data();
        let mut out = vec![0.0f32; batch * channels * out_h * out_w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    let start_h = oh * in_h / out_h;
                    let end_h = (oh + 1) * in_h / out_h;
                    for ow in 0..out_w {
                        let start_w = ow * in_w / out_w;
                        let end_w = (ow + 1) * in_w / out_w;
                        let mut sum = 0.0f32;
                        let count = (end_h - start_h) * (end_w - start_w);
                        for ih in start_h..end_h {
                            for iw in start_w..end_w {
                                let idx = b * channels * in_h * in_w
                                    + c * in_h * in_w
                                    + ih * in_w
                                    + iw;
                                sum += data[idx];
                            }
                        }
                        let out_idx = b * channels * out_h * out_w
                            + c * out_h * out_w
                            + oh * out_w
                            + ow;
                        out[out_idx] = if count > 0 { sum / count as f32 } else { 0.0 };
                    }
                }
            }
        }

        Tensor::new(out, vec![batch, channels, out_h, out_w], false)
    }
}

impl Module for AdaptiveAvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        AdaptiveAvgPool2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// AdaptiveAvgPool1d
// ===========================================================================

/// Adaptive 1D average pooling: produces a fixed output length regardless of input length.
/// Input: [N, C, L] -> Output: [N, C, output_size].
pub struct AdaptiveAvgPool1d {
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    pub fn new(output_size: usize) -> Self {
        AdaptiveAvgPool1d { output_size }
    }

    /// Forward pass: reshape to 4D, delegate to AdaptiveAvgPool2d, reshape back.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 3, "AdaptiveAvgPool1d expects [N, C, L]");
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        // Reshape [N, C, L] -> [N, C, L, 1]
        let x4d = x.reshape(vec![batch, channels, length, 1]);
        let pool2d = AdaptiveAvgPool2d::new((self.output_size, 1));
        let out4d = pool2d.forward(&x4d);
        // Reshape [N, C, output_size, 1] -> [N, C, output_size]
        out4d.reshape(vec![batch, channels, self.output_size])
    }
}

impl Module for AdaptiveAvgPool1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        AdaptiveAvgPool1d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// GroupNorm
// ===========================================================================

/// Group Normalization. When num_groups == num_channels, this is InstanceNorm.
pub struct GroupNorm {
    pub weight: Tensor, // [num_channels]
    pub bias: Tensor,   // [num_channels]
    num_groups: usize,
    num_channels: usize,
    eps: f32,
}

impl GroupNorm {
    pub fn new(num_groups: usize, num_channels: usize, eps: f32) -> Self {
        assert!(
            num_channels % num_groups == 0,
            "num_channels must be divisible by num_groups"
        );
        GroupNorm {
            weight: Tensor::ones(&[num_channels], true),
            bias: Tensor::zeros(&[num_channels], true),
            num_groups,
            num_channels,
            eps,
        }
    }

    /// x: [batch, channels, ...spatial_dims...]
    /// Normalizes within each group across (channels_per_group, spatial).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert!(shape.len() >= 2, "GroupNorm requires at least 2 dims");
        let batch = shape[0];
        let channels = shape[1];
        assert_eq!(channels, self.num_channels);
        let spatial: usize = shape[2..].iter().product();
        let cpg = channels / self.num_groups; // channels per group

        // Fused inference path: single-pass mean+var with NEON, no autograd overhead
        if !x.requires_grad() {
            let data = x.data();
            let w = self.weight.data();
            let b = self.bias.data();
            let total = data.len();
            let mut out = crate::cpu_pool::pool_get(total);

            for n in 0..batch {
                for g in 0..self.num_groups {
                    let group_size = cpg * spatial;
                    let inv_group = 1.0 / group_size as f32;

                    // Single-pass mean + variance with NEON
                    let mut sum = 0.0f64;
                    let mut sum_sq = 0.0f64;
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        for c_in_g in 0..cpg {
                            let c = g * cpg + c_in_g;
                            let base = n * channels * spatial + c * spatial;
                            let ptr = data[base..].as_ptr();
                            let chunks4 = spatial / 4;
                            let mut vsum = unsafe { vdupq_n_f32(0.0) };
                            let mut vsum_sq = unsafe { vdupq_n_f32(0.0) };
                            for i in 0..chunks4 {
                                unsafe {
                                    let v = vld1q_f32(ptr.add(i * 4));
                                    vsum = vaddq_f32(vsum, v);
                                    vsum_sq = vfmaq_f32(vsum_sq, v, v);
                                }
                            }
                            sum += unsafe { vaddvq_f32(vsum) } as f64;
                            sum_sq += unsafe { vaddvq_f32(vsum_sq) } as f64;
                            for s in (chunks4 * 4)..spatial {
                                let v = data[base + s] as f64;
                                sum += v;
                                sum_sq += v * v;
                            }
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for c_in_g in 0..cpg {
                            let c = g * cpg + c_in_g;
                            let base = n * channels * spatial + c * spatial;
                            for s in 0..spatial {
                                let v = data[base + s] as f64;
                                sum += v;
                                sum_sq += v * v;
                            }
                        }
                    }
                    let mean = (sum * inv_group as f64) as f32;
                    let var = ((sum_sq * inv_group as f64) - (mean as f64 * mean as f64)) as f32;
                    let inv_std = 1.0 / (var + self.eps).sqrt();

                    // Normalize with fused scale/bias using NEON
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        for c_in_g in 0..cpg {
                            let c = g * cpg + c_in_g;
                            let fused_scale = inv_std * w[c];
                            let fused_bias = b[c] - mean * fused_scale;
                            let base = n * channels * spatial + c * spatial;
                            let sp = data[base..].as_ptr();
                            let dp = out[base..].as_mut_ptr();
                            let chunks4 = spatial / 4;
                            let vs = unsafe { vdupq_n_f32(fused_scale) };
                            let vb = unsafe { vdupq_n_f32(fused_bias) };
                            for i in 0..chunks4 {
                                unsafe {
                                    let v = vld1q_f32(sp.add(i * 4));
                                    let r = vfmaq_f32(vb, v, vs);
                                    vst1q_f32(dp.add(i * 4), r);
                                }
                            }
                            for s in (chunks4 * 4)..spatial {
                                out[base + s] = data[base + s] * fused_scale + fused_bias;
                            }
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for c_in_g in 0..cpg {
                            let c = g * cpg + c_in_g;
                            let fused_scale = inv_std * w[c];
                            let fused_bias = b[c] - mean * fused_scale;
                            let base = n * channels * spatial + c * spatial;
                            for s in 0..spatial {
                                out[base + s] = data[base + s] * fused_scale + fused_bias;
                            }
                        }
                    }
                }
            }
            return Tensor::new(out, shape, false);
        }

        // Autograd-compatible path
        let data = x.data();
        let w = self.weight.data();
        let b = self.bias.data();
        let total = data.len();
        let mut out = crate::cpu_pool::pool_get(total);

        for n in 0..batch {
            for g in 0..self.num_groups {
                let group_size = cpg * spatial;
                let inv_group = 1.0 / group_size as f32;

                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                for c_in_g in 0..cpg {
                    let c = g * cpg + c_in_g;
                    let base = n * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        let v = data[base + s] as f64;
                        sum += v;
                        sum_sq += v * v;
                    }
                }
                let mean = (sum * inv_group as f64) as f32;
                let var = ((sum_sq * inv_group as f64) - (mean as f64 * mean as f64)) as f32;
                let inv_std = 1.0 / (var + self.eps).sqrt();

                for c_in_g in 0..cpg {
                    let c = g * cpg + c_in_g;
                    let fused_scale = inv_std * w[c];
                    let fused_bias = b[c] - mean * fused_scale;
                    let base = n * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        out[base + s] = data[base + s] * fused_scale + fused_bias;
                    }
                }
            }
        }

        Tensor::new(out, shape, false)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }
}

/// InstanceNorm: GroupNorm where num_groups == num_channels.
pub fn instance_norm(num_channels: usize, eps: f32) -> GroupNorm {
    GroupNorm::new(num_channels, num_channels, eps)
}

// ===========================================================================
// MultiHeadAttention
// ===========================================================================

pub struct MultiHeadAttention {
    wq: Tensor,
    wk: Tensor,
    wv: Tensor,
    wo: Tensor,
    bq: Tensor,
    bk: Tensor,
    bv: Tensor,
    bo: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        let head_dim = embed_dim / num_heads;
        MultiHeadAttention {
            wq: Tensor::randn(&[embed_dim, embed_dim], true),
            wk: Tensor::randn(&[embed_dim, embed_dim], true),
            wv: Tensor::randn(&[embed_dim, embed_dim], true),
            wo: Tensor::randn(&[embed_dim, embed_dim], true),
            bq: Tensor::zeros(&[1, embed_dim], true),
            bk: Tensor::zeros(&[1, embed_dim], true),
            bv: Tensor::zeros(&[1, embed_dim], true),
            bo: Tensor::zeros(&[1, embed_dim], true),
            num_heads,
            head_dim,
        }
    }

    /// q, k, v are [batch*seq_len, embed_dim] (2D).
    /// seq_q is the query sequence length, seq_kv is the key/value sequence length.
    /// Returns [batch*seq_q, embed_dim].
    pub fn forward(
        &self, q: &Tensor, k: &Tensor, v: &Tensor,
        batch: usize, seq_q: usize, seq_kv: usize,
    ) -> Tensor {
        let embed_dim = self.num_heads * self.head_dim;

        let q_proj = q.matmul(&self.wq).add_bias(&self.bq);
        let k_proj = k.matmul(&self.wk).add_bias(&self.bk);
        let v_proj = v.matmul(&self.wv).add_bias(&self.bv);

        let q_4d = q_proj.reshape(vec![batch, seq_q, self.num_heads, self.head_dim]);
        let q_4d = q_4d.transpose(1, 2);
        let q_2d = q_4d.reshape(vec![batch * self.num_heads * seq_q, self.head_dim]);

        let k_4d = k_proj.reshape(vec![batch, seq_kv, self.num_heads, self.head_dim]);
        let k_4d = k_4d.transpose(1, 2);
        let k_2d = k_4d.reshape(vec![batch * self.num_heads * seq_kv, self.head_dim]);

        let v_4d = v_proj.reshape(vec![batch, seq_kv, self.num_heads, self.head_dim]);
        let v_4d = v_4d.transpose(1, 2);
        let v_2d = v_4d.reshape(vec![batch * self.num_heads * seq_kv, self.head_dim]);

        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let q_data = q_2d.data();
        let k_data = k_2d.data();
        let v_data = v_2d.data();

        let total_bh = batch * self.num_heads;

        #[cfg(target_os = "macos")]
        let attn_out_data = crate::tensor::multi_head_attention(
            &q_data, &k_data, &v_data,
            total_bh, seq_q, seq_kv, self.head_dim, scale,
        );
        #[cfg(not(target_os = "macos"))]
        let attn_out_data = {
            let mut out = vec![0.0f32; total_bh * seq_q * self.head_dim];
            let mut scores_buf = vec![0.0f32; seq_q * seq_kv];
            for bh in 0..total_bh {
                let q_off = bh * seq_q * self.head_dim;
                let k_off = bh * seq_kv * self.head_dim;
                let v_off = bh * seq_kv * self.head_dim;
                let o_off = bh * seq_q * self.head_dim;

                for i in 0..seq_q {
                    for j in 0..seq_kv {
                        let mut sum = 0.0f32;
                        for p in 0..self.head_dim {
                            sum += q_data[q_off + i * self.head_dim + p]
                                 * k_data[k_off + j * self.head_dim + p];
                        }
                        scores_buf[i * seq_kv + j] = sum * scale;
                    }
                }

                crate::tensor::softmax_rows_inplace(&mut scores_buf, 0, seq_q, seq_kv);

                for i in 0..seq_q {
                    for j in 0..self.head_dim {
                        let mut sum = 0.0f32;
                        for p in 0..seq_kv {
                            sum += scores_buf[i * seq_kv + p]
                                 * v_data[v_off + p * self.head_dim + j];
                        }
                        out[o_off + i * self.head_dim + j] = sum;
                    }
                }
            }
            out
        };

        // Direct transpose [batch, num_heads, seq_q, head_dim] -> [batch*seq_q, embed_dim]
        let mut attn_flat_data = vec![0.0f32; batch * seq_q * embed_dim];
        for b in 0..batch {
            for s in 0..seq_q {
                let dst_row = b * seq_q + s;
                for h in 0..self.num_heads {
                    let src_idx = ((b * self.num_heads + h) * seq_q + s) * self.head_dim;
                    let dst_idx = dst_row * embed_dim + h * self.head_dim;
                    attn_flat_data[dst_idx..dst_idx + self.head_dim]
                        .copy_from_slice(&attn_out_data[src_idx..src_idx + self.head_dim]);
                }
            }
        }
        let attn_flat = Tensor::new(attn_flat_data, vec![batch * seq_q, embed_dim], false);

        attn_flat.matmul(&self.wo).add_bias(&self.bo)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![
            &self.wq, &self.wk, &self.wv, &self.wo,
            &self.bq, &self.bk, &self.bv, &self.bo,
        ]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.wq", prefix), &self.wq),
            (format!("{}.wk", prefix), &self.wk),
            (format!("{}.wv", prefix), &self.wv),
            (format!("{}.wo", prefix), &self.wo),
            (format!("{}.bq", prefix), &self.bq),
            (format!("{}.bk", prefix), &self.bk),
            (format!("{}.bv", prefix), &self.bv),
            (format!("{}.bo", prefix), &self.bo),
        ]
    }
}

// ===========================================================================
// TransformerEncoderLayer
// ===========================================================================

pub struct TransformerEncoderLayer {
    mha: MultiHeadAttention,
    ln1_gamma: Tensor,
    ln1_beta: Tensor,
    ln2_gamma: Tensor,
    ln2_beta: Tensor,
    ffn_w1: Tensor,
    ffn_b1: Tensor,
    ffn_w2: Tensor,
    ffn_b2: Tensor,
    embed_dim: usize,
}

impl TransformerEncoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self::new_with_dim_ff(embed_dim, num_heads, 4 * embed_dim)
    }

    pub fn new_with_dim_ff(embed_dim: usize, num_heads: usize, dim_feedforward: usize) -> Self {
        TransformerEncoderLayer {
            mha: MultiHeadAttention::new(embed_dim, num_heads),
            ln1_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln1_beta: Tensor::zeros(&[embed_dim], true),
            ln2_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln2_beta: Tensor::zeros(&[embed_dim], true),
            ffn_w1: Tensor::randn(&[embed_dim, dim_feedforward], true),
            ffn_b1: Tensor::zeros(&[1, dim_feedforward], true),
            ffn_w2: Tensor::randn(&[dim_feedforward, embed_dim], true),
            ffn_b2: Tensor::zeros(&[1, embed_dim], true),
            embed_dim,
        }
    }

    /// x: [batch*seq, embed_dim]. Returns [batch*seq, embed_dim].
    pub fn forward(&self, x: &Tensor, batch: usize, seq: usize) -> Tensor {
        let normed = x.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.embed_dim);
        let attn_out = self.mha.forward(&normed, &normed, &normed, batch, seq, seq);
        let x = x.add(&attn_out);

        let normed = x.layer_norm(&self.ln2_gamma, &self.ln2_beta, self.embed_dim);
        let h = normed.matmul(&self.ffn_w1).add_bias(&self.ffn_b1).gelu();
        let ffn_out = h.matmul(&self.ffn_w2).add_bias(&self.ffn_b2);
        x.add(&ffn_out)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.mha.params();
        p.extend([
            &self.ln1_gamma, &self.ln1_beta,
            &self.ln2_gamma, &self.ln2_beta,
            &self.ffn_w1, &self.ffn_b1,
            &self.ffn_w2, &self.ffn_b2,
        ]);
        p
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut p = self.mha.named_params(&format!("{}.mha", prefix));
        p.extend([
            (format!("{}.ln1_gamma", prefix), &self.ln1_gamma),
            (format!("{}.ln1_beta", prefix), &self.ln1_beta),
            (format!("{}.ln2_gamma", prefix), &self.ln2_gamma),
            (format!("{}.ln2_beta", prefix), &self.ln2_beta),
            (format!("{}.ffn_w1", prefix), &self.ffn_w1),
            (format!("{}.ffn_b1", prefix), &self.ffn_b1),
            (format!("{}.ffn_w2", prefix), &self.ffn_w2),
            (format!("{}.ffn_b2", prefix), &self.ffn_b2),
        ]);
        p
    }
}

// ===========================================================================
// TransformerDecoderLayer
// ===========================================================================

pub struct TransformerDecoderLayer {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    ln1_gamma: Tensor,
    ln1_beta: Tensor,
    ln2_gamma: Tensor,
    ln2_beta: Tensor,
    ln3_gamma: Tensor,
    ln3_beta: Tensor,
    ffn_w1: Tensor,
    ffn_b1: Tensor,
    ffn_w2: Tensor,
    ffn_b2: Tensor,
    embed_dim: usize,
}

impl TransformerDecoderLayer {
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        Self::new_with_dim_ff(embed_dim, num_heads, 4 * embed_dim)
    }

    pub fn new_with_dim_ff(embed_dim: usize, num_heads: usize, dim_feedforward: usize) -> Self {
        TransformerDecoderLayer {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            cross_attn: MultiHeadAttention::new(embed_dim, num_heads),
            ln1_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln1_beta: Tensor::zeros(&[embed_dim], true),
            ln2_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln2_beta: Tensor::zeros(&[embed_dim], true),
            ln3_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln3_beta: Tensor::zeros(&[embed_dim], true),
            ffn_w1: Tensor::randn(&[embed_dim, dim_feedforward], true),
            ffn_b1: Tensor::zeros(&[1, dim_feedforward], true),
            ffn_w2: Tensor::randn(&[dim_feedforward, embed_dim], true),
            ffn_b2: Tensor::zeros(&[1, embed_dim], true),
            embed_dim,
        }
    }

    /// tgt: [batch*num_queries, embed_dim], memory: [batch*seq_kv, embed_dim].
    /// Returns [batch*num_queries, embed_dim].
    pub fn forward(
        &self, tgt: &Tensor, memory: &Tensor,
        batch: usize, num_queries: usize, seq_kv: usize,
    ) -> Tensor {
        let normed = tgt.layer_norm(&self.ln1_gamma, &self.ln1_beta, self.embed_dim);
        let sa_out = self.self_attn.forward(&normed, &normed, &normed, batch, num_queries, num_queries);
        let x = tgt.add(&sa_out);

        let normed = x.layer_norm(&self.ln2_gamma, &self.ln2_beta, self.embed_dim);
        let ca_out = self.cross_attn.forward(&normed, memory, memory, batch, num_queries, seq_kv);
        let x = x.add(&ca_out);

        let normed = x.layer_norm(&self.ln3_gamma, &self.ln3_beta, self.embed_dim);
        let h = normed.matmul(&self.ffn_w1).add_bias(&self.ffn_b1).gelu();
        let ffn_out = h.matmul(&self.ffn_w2).add_bias(&self.ffn_b2);
        x.add(&ffn_out)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.self_attn.params();
        p.extend(self.cross_attn.params());
        p.extend([
            &self.ln1_gamma, &self.ln1_beta,
            &self.ln2_gamma, &self.ln2_beta,
            &self.ln3_gamma, &self.ln3_beta,
            &self.ffn_w1, &self.ffn_b1,
            &self.ffn_w2, &self.ffn_b2,
        ]);
        p
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut p = self.self_attn.named_params(&format!("{}.self_attn", prefix));
        p.extend(self.cross_attn.named_params(&format!("{}.cross_attn", prefix)));
        p.extend([
            (format!("{}.ln1_gamma", prefix), &self.ln1_gamma),
            (format!("{}.ln1_beta", prefix), &self.ln1_beta),
            (format!("{}.ln2_gamma", prefix), &self.ln2_gamma),
            (format!("{}.ln2_beta", prefix), &self.ln2_beta),
            (format!("{}.ln3_gamma", prefix), &self.ln3_gamma),
            (format!("{}.ln3_beta", prefix), &self.ln3_beta),
            (format!("{}.ffn_w1", prefix), &self.ffn_w1),
            (format!("{}.ffn_b1", prefix), &self.ffn_b1),
            (format!("{}.ffn_w2", prefix), &self.ffn_w2),
            (format!("{}.ffn_b2", prefix), &self.ffn_b2),
        ]);
        p
    }
}

// ===========================================================================
// Conv2d
// ===========================================================================

/// 2D convolution layer with configurable stride and padding.
pub struct Conv2d {
    pub weight: Tensor, // [out_channels, in_channels, kH, kW]
    pub bias: Tensor,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;
        let limit = (6.0 / fan_in as f32).sqrt();
        let weight = crate::random::uniform(
            &[out_channels, in_channels, kh, kw],
            -limit, limit, true,
        );
        Conv2d {
            weight,
            bias: Tensor::zeros(&[out_channels], true),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if self.stride == (1, 1)
            && self.padding == (self.kernel_size.0 / 2, self.kernel_size.1 / 2)
        {
            x.conv2d(&self.weight, &self.bias)
        } else {
            x.conv2d_strided(&self.weight, &self.bias, self.stride, self.padding)
        }
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        Conv2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        Conv2d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        Conv2d::named_params(self, prefix)
    }
}

// ===========================================================================
// ConvTranspose2d
// ===========================================================================

/// 2D transposed convolution (deconvolution) layer.
pub struct ConvTranspose2d {
    pub weight: Tensor, // [in_channels, out_channels, kH, kW]
    pub bias: Tensor,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl ConvTranspose2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        let (kh, kw) = kernel_size;
        let fan_in = in_channels * kh * kw;
        let limit = (6.0 / fan_in as f32).sqrt();
        let weight = crate::random::uniform(
            &[in_channels, out_channels, kh, kw],
            -limit, limit, true,
        );
        ConvTranspose2d {
            weight,
            bias: Tensor::zeros(&[out_channels], true),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.conv_transpose2d(&self.weight, &self.bias, self.stride, self.padding)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for ConvTranspose2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ConvTranspose2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        ConvTranspose2d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        ConvTranspose2d::named_params(self, prefix)
    }
}

// ===========================================================================
// ConvTranspose1d
// ===========================================================================

/// 1D transposed convolution (deconvolution) layer.
pub struct ConvTranspose1d {
    pub weight: Tensor, // [in_channels, out_channels, kernel_size]
    pub bias: Tensor,   // [out_channels]
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl ConvTranspose1d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let fan_in = in_channels * kernel_size;
        let limit = (6.0 / fan_in as f32).sqrt();
        let weight = crate::random::uniform(
            &[in_channels, out_channels, kernel_size],
            -limit, limit, true,
        );
        ConvTranspose1d {
            weight,
            bias: Tensor::zeros(&[out_channels], true),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
        }
    }

    /// x: [batch, in_channels, length] -> [batch, out_channels, out_length]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.conv_transpose1d(&self.weight, &self.bias, self.stride, self.padding)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for ConvTranspose1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ConvTranspose1d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        ConvTranspose1d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        ConvTranspose1d::named_params(self, prefix)
    }
}

// ===========================================================================
// MaxPool2d
// ===========================================================================

/// 2D max pooling layer with configurable kernel, stride, and padding.
pub struct MaxPool2d {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2d {
    pub fn new(kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Self {
        MaxPool2d { kernel_size, stride, padding }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        x.max_pool2d_ext(self.kernel_size, self.stride, self.padding)
    }
    fn params(&self) -> Vec<&Tensor> { vec![] }
}

// ===========================================================================
// MaxPool1d
// ===========================================================================

/// 1D max pooling layer. Internally reshapes to 4D and uses max_pool2d_ext.
pub struct MaxPool1d {
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl MaxPool1d {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        MaxPool1d { kernel_size, stride, padding }
    }
}

impl Module for MaxPool1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [batch, channels, length] -> reshape to [batch, channels, length, 1]
        // apply max_pool2d_ext with kernel (kernel_size, 1), stride (stride, 1), padding (padding, 0)
        // then reshape back
        let shape = x.shape();
        assert_eq!(shape.len(), 3, "MaxPool1d expects [batch, channels, length]");
        let (batch, channels, length) = (shape[0], shape[1], shape[2]);
        let x4d = x.reshape(vec![batch, channels, length, 1]);
        let pooled = x4d.max_pool2d_ext((self.kernel_size, 1), (self.stride, 1), (self.padding, 0));
        let out_shape = pooled.shape();
        let out_len = out_shape[2];
        pooled.reshape(vec![batch, channels, out_len])
    }
    fn params(&self) -> Vec<&Tensor> { vec![] }
}

// ===========================================================================
// LayerNorm
// ===========================================================================

/// Layer Normalization module wrapping the functional `Tensor::layer_norm`.
pub struct LayerNorm {
    pub weight: Tensor,  // gamma [normalized_shape]
    pub bias: Tensor,    // beta [normalized_shape]
    normalized_shape: usize,
    eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_shape: usize) -> Self {
        LayerNorm {
            weight: Tensor::ones(&[normalized_shape], true),
            bias: Tensor::zeros(&[normalized_shape], true),
            normalized_shape,
            eps: 1e-5,
        }
    }

    pub fn with_eps(normalized_shape: usize, eps: f32) -> Self {
        LayerNorm {
            weight: Tensor::ones(&[normalized_shape], true),
            bias: Tensor::zeros(&[normalized_shape], true),
            normalized_shape,
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.layer_norm(&self.weight, &self.bias, self.normalized_shape)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for LayerNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        LayerNorm::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        LayerNorm::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        LayerNorm::named_params(self, prefix)
    }
}

// ===========================================================================
// BatchNorm2d
// ===========================================================================

/// Batch Normalization for 4D input [N, C, H, W].
pub struct BatchNorm2d {
    pub weight: Tensor,       // gamma [num_features]
    pub bias: Tensor,         // beta [num_features]
    pub running_mean: Tensor, // no grad
    pub running_var: Tensor,  // no grad
    num_features: usize,
    momentum: f32,
    eps: f32,
    training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm2d {
            weight: Tensor::ones(&[num_features], true),
            bias: Tensor::zeros(&[num_features], true),
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    pub fn with_params(num_features: usize, momentum: f32, eps: f32) -> Self {
        BatchNorm2d {
            weight: Tensor::ones(&[num_features], true),
            bias: Tensor::zeros(&[num_features], true),
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum,
            eps,
            training: true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "BatchNorm2d requires 4D input [N, C, H, W]");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c, self.num_features);

        if self.training {
            // Training mode: use batch statistics and update running stats
            let result = x.batch_norm(&self.weight, &self.bias);

            // Compute batch mean and variance for running stats update
            let data = x.data();
            let spatial = n * h * w;
            let mut batch_mean = vec![0.0f32; c];
            let mut batch_var = vec![0.0f32; c];
            for ni in 0..n {
                for ci in 0..c {
                    for p in 0..(h * w) {
                        batch_mean[ci] += data[ni * c * h * w + ci * h * w + p];
                    }
                }
            }
            for ci in 0..c {
                batch_mean[ci] /= spatial as f32;
            }
            for ni in 0..n {
                for ci in 0..c {
                    for p in 0..(h * w) {
                        let diff = data[ni * c * h * w + ci * h * w + p] - batch_mean[ci];
                        batch_var[ci] += diff * diff;
                    }
                }
            }
            for ci in 0..c {
                batch_var[ci] /= spatial as f32;
            }

            // Update running stats with exponential moving average
            let rm = self.running_mean.data();
            let rv = self.running_var.data();
            let mom = self.momentum;
            let mut new_rm = vec![0.0f32; c];
            let mut new_rv = vec![0.0f32; c];
            for ci in 0..c {
                new_rm[ci] = (1.0 - mom) * rm[ci] + mom * batch_mean[ci];
                new_rv[ci] = (1.0 - mom) * rv[ci] + mom * batch_var[ci];
            }
            // Overwrite running stats in-place
            {
                let mut inner = self.running_mean.0.borrow_mut();
                inner.data = new_rm;
            }
            {
                let mut inner = self.running_var.0.borrow_mut();
                inner.data = new_rv;
            }

            result
        } else {
            // Eval mode: use running statistics
            let data = x.data();
            let rm = self.running_mean.data();
            let rv = self.running_var.data();
            let wt = self.weight.data();
            let bt = self.bias.data();
            let hw = h * w;
            let total = n * c * hw;
            let mut out = vec![0.0f32; total];

            for ni in 0..n {
                for ci in 0..c {
                    let inv_std = 1.0 / (rv[ci] + self.eps).sqrt();
                    let scale = wt[ci] * inv_std;
                    let shift = bt[ci] - rm[ci] * scale;
                    for p in 0..hw {
                        let idx = ni * c * hw + ci * hw + p;
                        out[idx] = data[idx] * scale + shift;
                    }
                }
            }

            Tensor::new(out, shape, false)
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        BatchNorm2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        BatchNorm2d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        BatchNorm2d::named_params(self, prefix)
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// BatchNorm1d
// ===========================================================================

/// Batch Normalization for 2D input [N, C] or 3D input [N, C, L].
pub struct BatchNorm1d {
    pub weight: Tensor,       // gamma [num_features]
    pub bias: Tensor,         // beta [num_features]
    pub running_mean: Tensor, // no grad
    pub running_var: Tensor,  // no grad
    num_features: usize,
    momentum: f32,
    eps: f32,
    training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        BatchNorm1d {
            weight: Tensor::ones(&[num_features], true),
            bias: Tensor::zeros(&[num_features], true),
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum: 0.1,
            eps: 1e-5,
            training: true,
        }
    }

    pub fn with_params(num_features: usize, momentum: f32, eps: f32) -> Self {
        BatchNorm1d {
            weight: Tensor::ones(&[num_features], true),
            bias: Tensor::zeros(&[num_features], true),
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum,
            eps,
            training: true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert!(
            shape.len() == 2 || shape.len() == 3,
            "BatchNorm1d requires 2D [N, C] or 3D [N, C, L] input"
        );
        let c = shape[1];
        assert_eq!(c, self.num_features);

        if shape.len() == 3 {
            // 3D input [N, C, L]: reshape to [N, C, L, 1] and use 4D batchnorm logic
            let n = shape[0];
            let l = shape[2];
            let x4d = x.reshape(vec![n, c, l, 1]);
            let bn2d = BatchNorm2d {
                weight: self.weight.clone(),
                bias: self.bias.clone(),
                running_mean: self.running_mean.clone(),
                running_var: self.running_var.clone(),
                num_features: self.num_features,
                momentum: self.momentum,
                eps: self.eps,
                training: self.training,
            };
            let out4d = bn2d.forward(&x4d);
            // Copy back updated running stats
            {
                let updated_rm = bn2d.running_mean.data();
                let mut inner = self.running_mean.0.borrow_mut();
                inner.data = updated_rm;
            }
            {
                let updated_rv = bn2d.running_var.data();
                let mut inner = self.running_var.0.borrow_mut();
                inner.data = updated_rv;
            }
            out4d.reshape(vec![n, c, l])
        } else {
            // 2D input [N, C]: treat as [N, C, 1, 1]
            let n = shape[0];
            let x4d = x.reshape(vec![n, c, 1, 1]);
            let bn2d = BatchNorm2d {
                weight: self.weight.clone(),
                bias: self.bias.clone(),
                running_mean: self.running_mean.clone(),
                running_var: self.running_var.clone(),
                num_features: self.num_features,
                momentum: self.momentum,
                eps: self.eps,
                training: self.training,
            };
            let out4d = bn2d.forward(&x4d);
            // Copy back updated running stats
            {
                let updated_rm = bn2d.running_mean.data();
                let mut inner = self.running_mean.0.borrow_mut();
                inner.data = updated_rm;
            }
            {
                let updated_rv = bn2d.running_var.data();
                let mut inner = self.running_var.0.borrow_mut();
                inner.data = updated_rv;
            }
            out4d.reshape(vec![n, c])
        }
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn params(&self) -> Vec<&Tensor> {
        vec![&self.weight, &self.bias]
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        vec![
            (format!("{}.weight", prefix), &self.weight),
            (format!("{}.bias", prefix), &self.bias),
        ]
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        BatchNorm1d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        BatchNorm1d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        BatchNorm1d::named_params(self, prefix)
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// InstanceNorm2d
// ===========================================================================

/// Instance Normalization for 4D input [N, C, H, W].
///
/// Normalizes each instance (per batch element, per channel) independently.
/// In training mode, computes statistics from the current input and updates
/// running statistics with exponential moving average.
/// In eval mode, uses running statistics.
pub struct InstanceNorm2d {
    pub weight: Tensor,       // gamma [num_features]
    pub bias: Tensor,         // beta [num_features]
    pub running_mean: Tensor, // no grad
    pub running_var: Tensor,  // no grad
    num_features: usize,
    momentum: f32,
    eps: f32,
    affine: bool,
    training: bool,
}

impl InstanceNorm2d {
    pub fn new(num_features: usize) -> Self {
        InstanceNorm2d {
            weight: Tensor::ones(&[num_features], true),
            bias: Tensor::zeros(&[num_features], true),
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum: 0.1,
            eps: 1e-5,
            affine: true,
            training: true,
        }
    }

    pub fn with_params(num_features: usize, affine: bool, momentum: f32, eps: f32) -> Self {
        InstanceNorm2d {
            weight: if affine {
                Tensor::ones(&[num_features], true)
            } else {
                Tensor::ones(&[num_features], false)
            },
            bias: if affine {
                Tensor::zeros(&[num_features], true)
            } else {
                Tensor::zeros(&[num_features], false)
            },
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum,
            eps,
            affine,
            training: true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(
            shape.len(),
            4,
            "InstanceNorm2d requires 4D input [N, C, H, W]"
        );
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(c, self.num_features);
        let hw = h * w;

        let data = x.data();
        let wt = self.weight.data();
        let bt = self.bias.data();
        let total = n * c * hw;
        let mut out = vec![0.0f32; total];

        if self.training {
            // Training: normalize each instance (per N, per C) independently
            // and update running stats (averaged across batch for each channel)
            let mut channel_mean_accum = vec![0.0f32; c];
            let mut channel_var_accum = vec![0.0f32; c];

            for ni in 0..n {
                for ci in 0..c {
                    let base = ni * c * hw + ci * hw;
                    // Compute per-instance statistics
                    let mut sum = 0.0f64;
                    let mut sum_sq = 0.0f64;
                    for p in 0..hw {
                        let v = data[base + p] as f64;
                        sum += v;
                        sum_sq += v * v;
                    }
                    let mean = (sum / hw as f64) as f32;
                    let var = ((sum_sq / hw as f64) - (mean as f64 * mean as f64)) as f32;
                    let inv_std = 1.0 / (var + self.eps).sqrt();

                    // Accumulate for running stats
                    channel_mean_accum[ci] += mean;
                    channel_var_accum[ci] += var;

                    // Normalize and apply affine transform
                    let scale = if self.affine { wt[ci] * inv_std } else { inv_std };
                    let shift = if self.affine {
                        bt[ci] - mean * scale
                    } else {
                        -mean * inv_std
                    };
                    for p in 0..hw {
                        out[base + p] = data[base + p] * scale + shift;
                    }
                }
            }

            // Update running stats: average across batch
            let rm = self.running_mean.data();
            let rv = self.running_var.data();
            let mom = self.momentum;
            let inv_n = 1.0 / n as f32;
            let mut new_rm = vec![0.0f32; c];
            let mut new_rv = vec![0.0f32; c];
            for ci in 0..c {
                let batch_mean = channel_mean_accum[ci] * inv_n;
                let batch_var = channel_var_accum[ci] * inv_n;
                new_rm[ci] = (1.0 - mom) * rm[ci] + mom * batch_mean;
                new_rv[ci] = (1.0 - mom) * rv[ci] + mom * batch_var;
            }
            {
                let mut inner = self.running_mean.0.borrow_mut();
                inner.data = new_rm;
            }
            {
                let mut inner = self.running_var.0.borrow_mut();
                inner.data = new_rv;
            }
        } else {
            // Eval mode: use running statistics
            let rm = self.running_mean.data();
            let rv = self.running_var.data();
            for ni in 0..n {
                for ci in 0..c {
                    let inv_std = 1.0 / (rv[ci] + self.eps).sqrt();
                    let scale = if self.affine { wt[ci] * inv_std } else { inv_std };
                    let shift = if self.affine {
                        bt[ci] - rm[ci] * scale
                    } else {
                        -rm[ci] * inv_std
                    };
                    let base = ni * c * hw + ci * hw;
                    for p in 0..hw {
                        out[base + p] = data[base + p] * scale + shift;
                    }
                }
            }
        }

        Tensor::new(out, shape, false)
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn params(&self) -> Vec<&Tensor> {
        if self.affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        if self.affine {
            vec![
                (format!("{}.weight", prefix), &self.weight),
                (format!("{}.bias", prefix), &self.bias),
            ]
        } else {
            vec![]
        }
    }
}

impl Module for InstanceNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        InstanceNorm2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        InstanceNorm2d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        InstanceNorm2d::named_params(self, prefix)
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// InstanceNorm1d
// ===========================================================================

/// Instance Normalization for 3D input [N, C, L].
///
/// Delegates to InstanceNorm2d by reshaping [N, C, L] to [N, C, L, 1].
pub struct InstanceNorm1d {
    pub weight: Tensor,       // gamma [num_features]
    pub bias: Tensor,         // beta [num_features]
    pub running_mean: Tensor, // no grad
    pub running_var: Tensor,  // no grad
    num_features: usize,
    momentum: f32,
    eps: f32,
    affine: bool,
    training: bool,
}

impl InstanceNorm1d {
    pub fn new(num_features: usize) -> Self {
        InstanceNorm1d {
            weight: Tensor::ones(&[num_features], true),
            bias: Tensor::zeros(&[num_features], true),
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum: 0.1,
            eps: 1e-5,
            affine: true,
            training: true,
        }
    }

    pub fn with_params(num_features: usize, affine: bool, momentum: f32, eps: f32) -> Self {
        InstanceNorm1d {
            weight: if affine {
                Tensor::ones(&[num_features], true)
            } else {
                Tensor::ones(&[num_features], false)
            },
            bias: if affine {
                Tensor::zeros(&[num_features], true)
            } else {
                Tensor::zeros(&[num_features], false)
            },
            running_mean: Tensor::zeros(&[num_features], false),
            running_var: Tensor::ones(&[num_features], false),
            num_features,
            momentum,
            eps,
            affine,
            training: true,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(
            shape.len(),
            3,
            "InstanceNorm1d requires 3D input [N, C, L]"
        );
        let (n, c, l) = (shape[0], shape[1], shape[2]);
        assert_eq!(c, self.num_features);

        // Reshape to 4D and delegate to InstanceNorm2d
        let x4d = x.reshape(vec![n, c, l, 1]);
        let in2d = InstanceNorm2d {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
            running_mean: self.running_mean.clone(),
            running_var: self.running_var.clone(),
            num_features: self.num_features,
            momentum: self.momentum,
            eps: self.eps,
            affine: self.affine,
            training: self.training,
        };
        let out4d = in2d.forward(&x4d);

        // Copy back updated running stats
        {
            let updated_rm = in2d.running_mean.data();
            let mut inner = self.running_mean.0.borrow_mut();
            inner.data = updated_rm;
        }
        {
            let updated_rv = in2d.running_var.data();
            let mut inner = self.running_var.0.borrow_mut();
            inner.data = updated_rv;
        }

        out4d.reshape(vec![n, c, l])
    }

    pub fn train(&mut self) {
        self.training = true;
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn params(&self) -> Vec<&Tensor> {
        if self.affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        if self.affine {
            vec![
                (format!("{}.weight", prefix), &self.weight),
                (format!("{}.bias", prefix), &self.bias),
            ]
        } else {
            vec![]
        }
    }
}

impl Module for InstanceNorm1d {
    fn forward(&self, x: &Tensor) -> Tensor {
        InstanceNorm1d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        InstanceNorm1d::params(self)
    }
    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        InstanceNorm1d::named_params(self, prefix)
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// Upsample
// ===========================================================================

/// Interpolation mode for upsampling.
pub enum UpsampleMode {
    Nearest,
    Bilinear,
}

/// Upsampling module supporting nearest-neighbor and bilinear interpolation.
///
/// Use `scale_factor` for nearest-neighbor mode and `size` for bilinear mode
/// (or either for nearest).
pub struct Upsample {
    scale_factor: Option<usize>,
    size: Option<(usize, usize)>,
    mode: UpsampleMode,
}

impl Upsample {
    /// Create a new Upsample module.
    ///
    /// For `Nearest` mode, provide `scale_factor`.
    /// For `Bilinear` mode, provide `size` as `(out_h, out_w)`.
    pub fn new(
        scale_factor: Option<usize>,
        size: Option<(usize, usize)>,
        mode: UpsampleMode,
    ) -> Self {
        match mode {
            UpsampleMode::Nearest => {
                assert!(
                    scale_factor.is_some() || size.is_some(),
                    "Upsample(Nearest): must provide scale_factor or size"
                );
            }
            UpsampleMode::Bilinear => {
                assert!(
                    size.is_some(),
                    "Upsample(Bilinear): must provide size"
                );
            }
        }
        Upsample { scale_factor, size, mode }
    }
}

impl Module for Upsample {
    fn forward(&self, x: &Tensor) -> Tensor {
        match self.mode {
            UpsampleMode::Nearest => {
                if let Some(sf) = self.scale_factor {
                    x.upsample_nearest(sf)
                } else if let Some((oh, ow)) = self.size {
                    // For nearest with explicit size, compute scale from input shape
                    let shape = x.shape();
                    let h = shape[2];
                    let w = shape[3];
                    // Use the min scale factor that covers both dimensions
                    assert!(oh % h == 0 && ow % w == 0,
                        "Upsample(Nearest) with size: output must be exact multiple of input");
                    let sh = oh / h;
                    let sw = ow / w;
                    assert_eq!(sh, sw, "Upsample(Nearest) with size: scale must be uniform");
                    x.upsample_nearest(sh)
                } else {
                    unreachable!()
                }
            }
            UpsampleMode::Bilinear => {
                let (oh, ow) = self.size.expect("Upsample(Bilinear) requires size");
                x.upsample_bilinear(oh, ow)
            }
        }
    }

    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// TransformerEncoder
// ===========================================================================

/// A stack of `TransformerEncoderLayer` modules applied sequentially.
pub struct TransformerEncoder {
    pub layers: Vec<TransformerEncoderLayer>,
    #[allow(dead_code)]
    d_model: usize,
}

impl TransformerEncoder {
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerEncoderLayer::new_with_dim_ff(d_model, nhead, dim_feedforward))
            .collect();
        TransformerEncoder { layers, d_model }
    }

    /// Run the encoder: sequentially applies each encoder layer.
    /// Input x: [batch*seq, d_model]. Returns [batch*seq, d_model].
    pub fn forward(&self, x: &Tensor, batch: usize, seq: usize) -> Tensor {
        let mut out = x.clone();
        for layer in &self.layers {
            out = layer.forward(&out, batch, seq);
        }
        out
    }

    pub fn params(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| l.named_params(&format!("{}.layers.{}", prefix, i)))
            .collect()
    }
}

impl Module for TransformerEncoder {
    fn forward(&self, x: &Tensor) -> Tensor {
        // Infer batch=1, seq=x.shape()[0] / 1 when used via Module trait
        let shape = x.shape();
        let seq = shape[0];
        self.forward(x, 1, seq)
    }

    fn params(&self) -> Vec<&Tensor> {
        TransformerEncoder::params(self)
    }

    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        TransformerEncoder::named_params(self, prefix)
    }
}

// ===========================================================================
// TransformerDecoder
// ===========================================================================

/// A stack of `TransformerDecoderLayer` modules applied sequentially.
pub struct TransformerDecoder {
    pub layers: Vec<TransformerDecoderLayer>,
    #[allow(dead_code)]
    d_model: usize,
}

impl TransformerDecoder {
    pub fn new(d_model: usize, nhead: usize, dim_feedforward: usize, num_layers: usize) -> Self {
        let layers = (0..num_layers)
            .map(|_| TransformerDecoderLayer::new_with_dim_ff(d_model, nhead, dim_feedforward))
            .collect();
        TransformerDecoder { layers, d_model }
    }

    /// Run the decoder: sequentially applies each decoder layer.
    /// tgt: [batch*num_queries, d_model], memory: [batch*seq_kv, d_model].
    /// Returns [batch*num_queries, d_model].
    pub fn forward_with_memory(
        &self, tgt: &Tensor, memory: &Tensor,
        batch: usize, num_queries: usize, seq_kv: usize,
    ) -> Tensor {
        let mut out = tgt.clone();
        for layer in &self.layers {
            out = layer.forward(&out, memory, batch, num_queries, seq_kv);
        }
        out
    }

    pub fn params(&self) -> Vec<&Tensor> {
        self.layers.iter().flat_map(|l| l.params()).collect()
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        self.layers
            .iter()
            .enumerate()
            .flat_map(|(i, l)| l.named_params(&format!("{}.layers.{}", prefix, i)))
            .collect()
    }
}

impl Module for TransformerDecoder {
    fn forward(&self, x: &Tensor) -> Tensor {
        // When used via Module trait with a single input, treat memory as same as input
        let shape = x.shape();
        let seq = shape[0];
        self.forward_with_memory(x, x, 1, seq, seq)
    }

    fn params(&self) -> Vec<&Tensor> {
        TransformerDecoder::params(self)
    }

    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        TransformerDecoder::named_params(self, prefix)
    }
}

// ===========================================================================
// Transformer
// ===========================================================================

/// Full Transformer model with encoder and decoder stacks.
pub struct Transformer {
    pub encoder: TransformerEncoder,
    pub decoder: TransformerDecoder,
    #[allow(dead_code)]
    d_model: usize,
}

impl Transformer {
    pub fn new(
        d_model: usize,
        nhead: usize,
        dim_feedforward: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
    ) -> Self {
        Transformer {
            encoder: TransformerEncoder::new(d_model, nhead, dim_feedforward, num_encoder_layers),
            decoder: TransformerDecoder::new(d_model, nhead, dim_feedforward, num_decoder_layers),
            d_model,
        }
    }

    /// Run the full transformer: encode src, then decode tgt with encoder output as memory.
    /// src: [batch*src_seq, d_model], tgt: [batch*tgt_seq, d_model].
    /// Returns [batch*tgt_seq, d_model].
    pub fn forward(
        &self, src: &Tensor, tgt: &Tensor,
        batch: usize, src_seq: usize, tgt_seq: usize,
    ) -> Tensor {
        let memory = self.encoder.forward(src, batch, src_seq);
        self.decoder.forward_with_memory(tgt, &memory, batch, tgt_seq, src_seq)
    }

    pub fn params(&self) -> Vec<&Tensor> {
        let mut p = self.encoder.params();
        p.extend(self.decoder.params());
        p
    }

    pub fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        let mut p = self.encoder.named_params(&format!("{}.encoder", prefix));
        p.extend(self.decoder.named_params(&format!("{}.decoder", prefix)));
        p
    }
}

impl Module for Transformer {
    fn forward(&self, x: &Tensor) -> Tensor {
        // When used via Module trait, treat input as both src and tgt
        let shape = x.shape();
        let seq = shape[0];
        self.forward(x, x, 1, seq, seq)
    }

    fn params(&self) -> Vec<&Tensor> {
        Transformer::params(self)
    }

    fn named_params(&self, prefix: &str) -> Vec<(String, &Tensor)> {
        Transformer::named_params(self, prefix)
    }
}

// ===========================================================================
// ZeroPad2d
// ===========================================================================

/// Pads 2D input [N,C,H,W] with zeros.
pub struct ZeroPad2d {
    /// (left, right, top, bottom) padding amounts.
    pub padding: (usize, usize, usize, usize),
}

impl ZeroPad2d {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        ZeroPad2d { padding }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "ZeroPad2d: input must be 4D [N,C,H,W]");
        let (left, right, top, bottom) = self.padding;
        x.pad(&[(0, 0), (0, 0), (top, bottom), (left, right)], 0.0)
    }
}

impl Module for ZeroPad2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ZeroPad2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// ConstantPad2d
// ===========================================================================

/// Pads 2D input [N,C,H,W] with a constant value.
pub struct ConstantPad2d {
    /// (left, right, top, bottom) padding amounts.
    pub padding: (usize, usize, usize, usize),
    /// Constant fill value.
    pub value: f32,
}

impl ConstantPad2d {
    pub fn new(padding: (usize, usize, usize, usize), value: f32) -> Self {
        ConstantPad2d { padding, value }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "ConstantPad2d: input must be 4D [N,C,H,W]");
        let (left, right, top, bottom) = self.padding;
        x.pad(&[(0, 0), (0, 0), (top, bottom), (left, right)], self.value)
    }
}

impl Module for ConstantPad2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ConstantPad2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// ReflectionPad2d
// ===========================================================================

/// Pads 2D input [N,C,H,W] by reflecting values at the border.
pub struct ReflectionPad2d {
    /// (left, right, top, bottom) padding amounts.
    pub padding: (usize, usize, usize, usize),
}

impl ReflectionPad2d {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        ReflectionPad2d { padding }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "ReflectionPad2d: input must be 4D [N,C,H,W]");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (left, right, top, bottom) = self.padding;
        assert!(left < w && right < w, "ReflectionPad2d: padding must be less than input width");
        assert!(top < h && bottom < h, "ReflectionPad2d: padding must be less than input height");

        let out_h = h + top + bottom;
        let out_w = w + left + right;
        let data = x.data();
        let mut out = vec![0.0f32; n * c * out_h * out_w];

        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Compute reflected source indices
                        let sh = {
                            let mut y = oh as isize - top as isize;
                            if y < 0 {
                                y = -y;
                            }
                            if y >= h as isize {
                                y = 2 * (h as isize - 1) - y;
                            }
                            y as usize
                        };
                        let sw = {
                            let mut x = ow as isize - left as isize;
                            if x < 0 {
                                x = -x;
                            }
                            if x >= w as isize {
                                x = 2 * (w as isize - 1) - x;
                            }
                            x as usize
                        };
                        let out_idx = ni * c * out_h * out_w + ci * out_h * out_w + oh * out_w + ow;
                        let in_idx = ni * c * h * w + ci * h * w + sh * w + sw;
                        out[out_idx] = data[in_idx];
                    }
                }
            }
        }

        Tensor::new(out, vec![n, c, out_h, out_w], false)
    }
}

impl Module for ReflectionPad2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ReflectionPad2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// ReplicationPad2d
// ===========================================================================

/// Pads 2D input [N,C,H,W] by replicating edge values.
pub struct ReplicationPad2d {
    /// (left, right, top, bottom) padding amounts.
    pub padding: (usize, usize, usize, usize),
}

impl ReplicationPad2d {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        ReplicationPad2d { padding }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "ReplicationPad2d: input must be 4D [N,C,H,W]");
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (left, right, top, bottom) = self.padding;

        let out_h = h + top + bottom;
        let out_w = w + left + right;
        let data = x.data();
        let mut out = vec![0.0f32; n * c * out_h * out_w];

        for ni in 0..n {
            for ci in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        // Clamp source indices to valid range
                        let sh = {
                            let y = oh as isize - top as isize;
                            y.max(0).min(h as isize - 1) as usize
                        };
                        let sw = {
                            let x = ow as isize - left as isize;
                            x.max(0).min(w as isize - 1) as usize
                        };
                        let out_idx = ni * c * out_h * out_w + ci * out_h * out_w + oh * out_w + ow;
                        let in_idx = ni * c * h * w + ci * h * w + sh * w + sw;
                        out[out_idx] = data[in_idx];
                    }
                }
            }
        }

        Tensor::new(out, vec![n, c, out_h, out_w], false)
    }
}

impl Module for ReplicationPad2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        ReplicationPad2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// Dropout2d
// ===========================================================================

/// Dropout2d: zeros entire channels during training.
/// During training, for each (batch, channel), zeros the entire spatial plane
/// with probability p. In eval mode, acts as identity.
pub struct Dropout2d {
    pub p: f32,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        Dropout2d { p, training: true }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }
        let shape = x.shape();
        assert!(shape.len() >= 3, "Dropout2d: input must be at least 3D [N,C,...]");
        let n = shape[0];
        let c = shape[1];
        let spatial: usize = shape[2..].iter().product();
        let scale = 1.0 / (1.0 - self.p);

        // Generate per-channel mask using LCG PRNG
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = Cell::new(987654321);
        }

        let data = x.data();
        let total = data.len();
        let mut mask_data = vec![scale; total];

        for ni in 0..n {
            for ci in 0..c {
                let r = SEED.with(|s| {
                    let v = s.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    s.set(v);
                    (v >> 33) as f32 / (1u64 << 31) as f32
                });
                if r < self.p {
                    // Zero the entire spatial plane for this (batch, channel)
                    let offset = ni * c * spatial + ci * spatial;
                    for s in 0..spatial {
                        mask_data[offset + s] = 0.0;
                    }
                }
            }
        }

        let mask = Tensor::new(mask_data, shape.to_vec(), false);
        x.mul(&mask)
    }
}

impl Module for Dropout2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        Dropout2d::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// AlphaDropout
// ===========================================================================

/// AlphaDropout: dropout variant for SELU networks that preserves
/// self-normalizing properties (mean=0, var=1).
pub struct AlphaDropout {
    pub p: f32,
    pub training: bool,
}

impl AlphaDropout {
    pub fn new(p: f32) -> Self {
        assert!((0.0..1.0).contains(&p), "dropout probability must be in [0, 1)");
        AlphaDropout { p, training: true }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }

        // SELU constants
        let alpha_selu: f64 = -1.7580993408_f64 * 1.0507009873_f64;
        let p64 = self.p as f64;
        let a = (1.0_f64 / ((1.0 - p64) * (1.0 + p64 * alpha_selu * alpha_selu))).sqrt();
        let b = -a * p64 * alpha_selu;

        let a_f32 = a as f32;
        let b_f32 = b as f32;
        let alpha_f32 = alpha_selu as f32;

        // Generate mask using LCG PRNG
        use std::cell::Cell;
        thread_local! {
            static SEED: Cell<u64> = Cell::new(1122334455);
        }

        let data = x.data();
        let shape = x.shape();
        let n = data.len();

        let mut out_data = Vec::with_capacity(n);
        for i in 0..n {
            let r = SEED.with(|s| {
                let v = s.get().wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                s.set(v);
                (v >> 33) as f32 / (1u64 << 31) as f32
            });
            let val = if r >= self.p {
                data[i]
            } else {
                alpha_f32
            };
            out_data.push(a_f32 * val + b_f32);
        }

        Tensor::new(out_data, shape, false)
    }
}

impl Module for AlphaDropout {
    fn forward(&self, x: &Tensor) -> Tensor {
        AlphaDropout::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
}

// ===========================================================================
// PixelShuffle
// ===========================================================================

/// Rearranges [N, C*r^2, H, W] -> [N, C, H*r, W*r] where r is the upscale factor.
/// Used in sub-pixel convolution for super-resolution.
pub struct PixelShuffle {
    pub upscale_factor: usize,
}

impl PixelShuffle {
    pub fn new(upscale_factor: usize) -> Self {
        assert!(upscale_factor > 0, "PixelShuffle: upscale_factor must be > 0");
        PixelShuffle { upscale_factor }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "PixelShuffle: input must be 4D [N,C*r^2,H,W]");
        let (n, c_in, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let r = self.upscale_factor;
        assert_eq!(c_in % (r * r), 0,
            "PixelShuffle: channels ({}) must be divisible by r^2 ({})", c_in, r * r);
        let c = c_in / (r * r);
        let out_h = h * r;
        let out_w = w * r;

        let data = x.data();
        let mut out = vec![0.0f32; n * c * out_h * out_w];

        // Input layout: [N, C*r*r, H, W] -- conceptually [N, C, r, r, H, W]
        // Output layout: [N, C, H*r, W*r]
        for ni in 0..n {
            for ci in 0..c {
                for rh in 0..r {
                    for rw in 0..r {
                        for hi in 0..h {
                            for wi in 0..w {
                                let in_c = ci * r * r + rh * r + rw;
                                let in_idx = ni * c_in * h * w + in_c * h * w + hi * w + wi;
                                let oh = hi * r + rh;
                                let ow = wi * r + rw;
                                let out_idx = ni * c * out_h * out_w + ci * out_h * out_w + oh * out_w + ow;
                                out[out_idx] = data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        Tensor::new(out, vec![n, c, out_h, out_w], false)
    }
}

impl Module for PixelShuffle {
    fn forward(&self, x: &Tensor) -> Tensor {
        PixelShuffle::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// PixelUnshuffle
// ===========================================================================

/// Inverse of PixelShuffle: rearranges [N, C, H*r, W*r] -> [N, C*r^2, H, W].
pub struct PixelUnshuffle {
    pub downscale_factor: usize,
}

impl PixelUnshuffle {
    pub fn new(downscale_factor: usize) -> Self {
        assert!(downscale_factor > 0, "PixelUnshuffle: downscale_factor must be > 0");
        PixelUnshuffle { downscale_factor }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        assert_eq!(shape.len(), 4, "PixelUnshuffle: input must be 4D [N,C,H,W]");
        let (n, c, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
        let r = self.downscale_factor;
        assert_eq!(h_in % r, 0,
            "PixelUnshuffle: height ({}) must be divisible by r ({})", h_in, r);
        assert_eq!(w_in % r, 0,
            "PixelUnshuffle: width ({}) must be divisible by r ({})", w_in, r);
        let h = h_in / r;
        let w = w_in / r;
        let c_out = c * r * r;

        let data = x.data();
        let mut out = vec![0.0f32; n * c_out * h * w];

        // Input layout: [N, C, H*r, W*r]
        // Output layout: [N, C*r*r, H, W]
        for ni in 0..n {
            for ci in 0..c {
                for rh in 0..r {
                    for rw in 0..r {
                        for hi in 0..h {
                            for wi in 0..w {
                                let in_h = hi * r + rh;
                                let in_w = wi * r + rw;
                                let in_idx = ni * c * h_in * w_in + ci * h_in * w_in + in_h * w_in + in_w;
                                let out_c = ci * r * r + rh * r + rw;
                                let out_idx = ni * c_out * h * w + out_c * h * w + hi * w + wi;
                                out[out_idx] = data[in_idx];
                            }
                        }
                    }
                }
            }
        }

        Tensor::new(out, vec![n, c_out, h, w], false)
    }
}

impl Module for PixelUnshuffle {
    fn forward(&self, x: &Tensor) -> Tensor {
        PixelUnshuffle::forward(self, x)
    }
    fn params(&self) -> Vec<&Tensor> {
        vec![]
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l1_loss() {
        let pred = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let target = Tensor::new(vec![1.5, 2.5, 2.5, 3.5], vec![4], false);
        let loss = l1_loss(&pred, &target);
        // |diff| = [0.5, 0.5, 0.5, 0.5], mean = 0.5
        assert!((loss.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_l1_loss_autograd() {
        let pred = Tensor::new(vec![1.0, 3.0], vec![2], true);
        let target = Tensor::new(vec![2.0, 1.0], vec![2], false);
        let loss = l1_loss(&pred, &target);
        loss.backward();
        let grad = pred.grad_data().unwrap();
        // d/dpred mean(|pred-target|) = sign(pred-target)/n
        // sign([-1, 2]) / 2 = [-0.5, 0.5]
        assert!((grad[0] - (-0.5)).abs() < 1e-5);
        assert!((grad[1] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_nll_loss() {
        // log_probs: [2, 3]
        let log_probs = Tensor::new(
            vec![-1.0, -2.0, -0.5, -0.3, -1.5, -2.0],
            vec![2, 3],
            false,
        );
        let targets = vec![2, 0]; // select -0.5 and -0.3
        let loss = nll_loss(&log_probs, &targets);
        // -((-0.5) + (-0.3)) / 2 = 0.4
        assert!((loss.data()[0] - 0.4).abs() < 1e-6);
    }

    #[test]
    fn test_smooth_l1_loss() {
        let pred = Tensor::new(vec![0.0, 0.0], vec![2], false);
        let target = Tensor::new(vec![0.5, 2.0], vec![2], false);
        let loss = smooth_l1_loss(&pred, &target, 1.0);
        // |0.5| < 1.0: 0.5*0.25/1.0 = 0.125
        // |2.0| >= 1.0: 2.0 - 0.5 = 1.5
        // mean = (0.125 + 1.5) / 2 = 0.8125
        assert!((loss.data()[0] - 0.8125).abs() < 1e-5);
    }

    #[test]
    fn test_huber_loss() {
        let pred = Tensor::new(vec![0.0, 0.0], vec![2], false);
        let target = Tensor::new(vec![0.5, 2.0], vec![2], false);
        let loss = huber_loss(&pred, &target, 1.0);
        // |0.5| <= 1.0: 0.5 * 0.25 = 0.125
        // |2.0| > 1.0: 1.0 * (2.0 - 0.5) = 1.5
        // mean = (0.125 + 1.5) / 2 = 0.8125
        assert!((loss.data()[0] - 0.8125).abs() < 1e-5);
    }

    #[test]
    fn test_huber_loss_autograd() {
        let pred = Tensor::new(vec![0.5, 3.0], vec![2], true);
        let target = Tensor::new(vec![0.0, 0.0], vec![2], false);
        let loss = huber_loss(&pred, &target, 1.0);
        loss.backward();
        let grad = pred.grad_data().unwrap();
        // |0.5| <= 1: grad = 0.5 * 2 * 0.5 / 2 = 0.5 * 0.5 = 0.25
        // |3.0| > 1: grad = delta * sign(diff) / 2 = 1.0 * 1.0 / 2 = 0.5
        assert!((grad[0] - 0.25).abs() < 1e-4, "huber grad[0]={}", grad[0]);
        assert!((grad[1] - 0.5).abs() < 1e-4, "huber grad[1]={}", grad[1]);
    }

    #[test]
    fn test_cosine_similarity_loss() {
        // Identical vectors => cos_sim = 1 => loss = 0
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
        let b = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
        let loss = cosine_similarity_loss(&a, &b);
        assert!(loss.data()[0].abs() < 1e-5);

        // Orthogonal vectors => cos_sim = 0 => loss = 1
        let a = Tensor::new(vec![1.0, 0.0], vec![2], false);
        let b = Tensor::new(vec![0.0, 1.0], vec![2], false);
        let loss = cosine_similarity_loss(&a, &b);
        assert!((loss.data()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_triplet_loss() {
        let anchor = Tensor::new(vec![0.0, 0.0], vec![2], false);
        let pos = Tensor::new(vec![1.0, 0.0], vec![2], false);
        let neg = Tensor::new(vec![3.0, 0.0], vec![2], false);
        // d_pos = 1, d_neg = 9, loss = max(0, 1 - 9 + 1) = max(0, -7) = 0
        let loss = triplet_loss(&anchor, &pos, &neg, 1.0);
        assert!(loss.data()[0].abs() < 1e-6);

        // neg is closer than pos
        let neg2 = Tensor::new(vec![0.5, 0.0], vec![2], false);
        // d_pos = 1, d_neg = 0.25, loss = max(0, 1 - 0.25 + 1) = 1.75
        let loss2 = triplet_loss(&anchor, &pos, &neg2, 1.0);
        assert!((loss2.data()[0] - 1.75).abs() < 1e-5);
    }

    #[test]
    fn test_hinge_loss() {
        let pred = Tensor::new(vec![0.5, -1.0, 2.0], vec![3], false);
        let target = Tensor::new(vec![1.0, -1.0, 1.0], vec![3], false);
        let loss = hinge_loss(&pred, &target);
        // 1 - pred*target = [0.5, 0.0, -1.0]
        // relu = [0.5, 0.0, 0.0]
        // mean = 0.5/3
        assert!((loss.data()[0] - 0.5 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_log_cosh_loss() {
        let pred = Tensor::new(vec![0.0], vec![1], false);
        let target = Tensor::new(vec![0.0], vec![1], false);
        let loss = log_cosh_loss(&pred, &target);
        // log(cosh(0)) = log(1) = 0
        assert!(loss.data()[0].abs() < 1e-6);
    }

    #[test]
    fn test_gaussian_nll_loss() {
        let input = Tensor::new(vec![1.0, 2.0], vec![2], false);
        let target = Tensor::new(vec![1.0, 2.0], vec![2], false);
        let var = Tensor::new(vec![1.0, 1.0], vec![2], false);
        let loss = gaussian_nll_loss(&input, &target, &var);
        // 0.5 * mean(log(1) + 0) = 0
        assert!(loss.data()[0].abs() < 1e-6);
    }

    #[test]
    fn test_rmsnorm_forward() {
        let dim = 4;
        let norm = RMSNorm::new(dim, 1e-5);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], false);
        let out = norm.forward(&x);
        let out_data = out.data();

        // Manual: mean(x^2) = (1+4+9+16)/4 = 7.5
        // rms = sqrt(7.5 + 1e-5) ~ 2.7386
        // normalized: [1/2.7386, 2/2.7386, 3/2.7386, 4/2.7386]
        let rms = (7.5f32 + 1e-5).sqrt();
        for i in 0..4 {
            let expected = (i as f32 + 1.0) / rms;
            assert!(
                (out_data[i] - expected).abs() < 1e-4,
                "RMSNorm out[{}]={}, expected={}",
                i,
                out_data[i],
                expected
            );
        }
    }

    #[test]
    fn test_rmsnorm_batch() {
        let norm = RMSNorm::new(2, 1e-5);
        let x = Tensor::new(vec![3.0, 4.0, 1.0, 0.0], vec![2, 2], false);
        let out = norm.forward(&x);
        let out_data = out.data();

        // Row 0: [3, 4], mean(x^2)=(9+16)/2=12.5, rms=sqrt(12.5)~3.5355
        let rms0 = 12.5f32.sqrt();
        assert!((out_data[0] - 3.0 / rms0).abs() < 1e-4);
        assert!((out_data[1] - 4.0 / rms0).abs() < 1e-4);

        // Row 1: [1, 0], mean(x^2)=(1+0)/2=0.5, rms=sqrt(0.5)~0.7071
        let rms1 = 0.5f32.sqrt();
        assert!((out_data[2] - 1.0 / rms1).abs() < 1e-4);
        assert!(out_data[3].abs() < 1e-4);
    }

    #[test]
    fn test_sequential_forward() {
        let layers: Vec<Box<dyn Module>> = vec![
            Box::new(Linear::new(4, 8)),
            Box::new(Linear::new(8, 2)),
        ];
        let model = Sequential::new(layers);
        let x = Tensor::new(vec![1.0; 4], vec![1, 4], false);
        let out = model.forward(&x);
        assert_eq!(out.shape(), vec![1, 2]);

        // Check params count: linear1(4*8+8=40) + linear2(8*2+2=18) = 58 values
        let total_params: usize = model.params().iter().map(|p| p.size()).sum();
        assert_eq!(total_params, 4 * 8 + 8 + 8 * 2 + 2);
    }

    #[test]
    fn test_identity() {
        let id = Identity;
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3], false);
        let out = id.forward(&x);
        assert_eq!(out.data(), x.data());
        assert!(id.params().is_empty());
    }

    #[test]
    fn test_dropout_eval_mode() {
        let mut drop = Dropout::new(0.5);
        drop.eval();
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let out = drop.forward(&x);
        assert_eq!(out.data(), x.data());
    }

    #[test]
    fn test_dropout_training_mode() {
        let drop = Dropout::new(0.5);
        let x = Tensor::new(vec![1.0; 1000], vec![1000], false);
        let out = drop.forward(&x);
        let out_data = out.data();
        // Some should be zero, some should be 2.0 (scaled by 1/(1-0.5))
        let zeros = out_data.iter().filter(|&&v| v == 0.0).count();
        let twos = out_data.iter().filter(|&&v| (v - 2.0).abs() < 1e-5).count();
        assert!(zeros > 100, "should have some zeros, got {}", zeros);
        assert!(twos > 100, "should have some scaled values, got {}", twos);
        assert_eq!(zeros + twos, 1000);
    }

    #[test]
    fn test_rnn_forward() {
        let rnn = RNN::new(3, 4);
        let x = Tensor::new(vec![0.1; 6], vec![2, 3], false); // seq_len=2
        let h0 = Tensor::zeros(&[1, 4], false);
        let (output, h_n) = rnn.forward(&x, &h0);
        assert_eq!(output.shape(), vec![2, 4]);
        assert_eq!(h_n.shape(), vec![1, 4]);
        // last hidden state should match last row of output
        let out_data = output.data();
        let hn_data = h_n.data();
        for i in 0..4 {
            assert!(
                (out_data[4 + i] - hn_data[i]).abs() < 1e-6,
                "h_n should match last output step"
            );
        }
    }

    #[test]
    fn test_lstm_forward() {
        let lstm = LSTM::new(3, 4);
        let x = Tensor::new(vec![0.1; 6], vec![2, 3], false);
        let h0 = Tensor::zeros(&[1, 4], false);
        let c0 = Tensor::zeros(&[1, 4], false);
        let (output, h_n, c_n) = lstm.forward(&x, &h0, &c0);
        assert_eq!(output.shape(), vec![2, 4]);
        assert_eq!(h_n.shape(), vec![1, 4]);
        assert_eq!(c_n.shape(), vec![1, 4]);
    }

    #[test]
    fn test_gru_forward() {
        let gru = GRU::new(3, 4);
        let x = Tensor::new(vec![0.1; 6], vec![2, 3], false);
        let h0 = Tensor::zeros(&[1, 4], false);
        let (output, h_n) = gru.forward(&x, &h0);
        assert_eq!(output.shape(), vec![2, 4]);
        assert_eq!(h_n.shape(), vec![1, 4]);
    }

    #[test]
    fn test_rope() {
        let rope = RoPE::new(4, 128, 10000.0);
        let x = Tensor::new(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], vec![2, 4], false);
        let out = rope.apply(&x, 0);
        assert_eq!(out.shape(), vec![2, 4]);
        // At position 0, cos=1, sin=0 so output should equal input
        let out_data = out.data();
        assert!((out_data[0] - 1.0).abs() < 1e-5);
        assert!((out_data[1] - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_conv1d_forward() {
        let _conv = Conv1d::new(1, 1, 3, 1, 0);
        // Set weights to all 1s and bias to 0
        let w = Tensor::new(vec![1.0, 1.0, 1.0], vec![1, 1, 3], true);
        let b = Tensor::zeros(&[1], true);
        let conv = Conv1d {
            weight: w,
            bias: b,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: 0,
        };
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1, 1, 5], false);
        let out = conv.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 3]);
        let out_data = out.data();
        assert!((out_data[0] - 6.0).abs() < 1e-5); // 1+2+3
        assert!((out_data[1] - 9.0).abs() < 1e-5); // 2+3+4
        assert!((out_data[2] - 12.0).abs() < 1e-5); // 3+4+5
    }

    #[test]
    fn test_avgpool2d_forward() {
        let pool = AvgPool2d::new(2, 2, 0);
        // [1, 1, 4, 4] input
        let x = Tensor::new(
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
                13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 1, 4, 4],
            false,
        );
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        let out_data = out.data();
        assert!((out_data[0] - 3.5).abs() < 1e-5); // avg(1,2,5,6)
        assert!((out_data[1] - 5.5).abs() < 1e-5); // avg(3,4,7,8)
        assert!((out_data[2] - 11.5).abs() < 1e-5); // avg(9,10,13,14)
        assert!((out_data[3] - 13.5).abs() < 1e-5); // avg(11,12,15,16)
    }

    #[test]
    fn test_groupnorm_forward() {
        // 1 batch, 4 channels, 1x1 spatial, 2 groups
        let gn = GroupNorm::new(2, 4, 1e-5);
        let x = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0],
            vec![1, 4, 1, 1],
            false,
        );
        let out = gn.forward(&x);
        let out_data = out.data();
        // Group 0: channels [0,1] = [1, 3], mean=2, var=1
        // normalized: [-1, 1] (approx)
        assert!((out_data[0] - (-1.0)).abs() < 0.01);
        assert!((out_data[1] - 1.0).abs() < 0.01);
        // Group 1: channels [2,3] = [5, 7], mean=6, var=1
        assert!((out_data[2] - (-1.0)).abs() < 0.01);
        assert!((out_data[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_module_trait_linear() {
        let linear = Linear::new(3, 2);
        let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], false);
        let out: Tensor = Module::forward(&linear, &x);
        assert_eq!(out.shape(), vec![1, 2]);
        assert_eq!(Module::params(&linear).len(), 2);
    }

    #[test]
    fn test_kl_div_loss() {
        // P = Q => KL = 0
        let p = Tensor::new(vec![0.25, 0.25, 0.25, 0.25], vec![4], false);
        // log_q = log(p) for uniform distribution
        let log_q = p.log();
        let loss = kl_div_loss(&log_q, &p);
        // KL(P||Q) when P=Q is 0
        assert!(loss.data()[0].abs() < 1e-5, "KL divergence of identical distributions should be 0, got {}", loss.data()[0]);
    }

    #[test]
    fn test_margin_ranking_loss() {
        let x1 = Tensor::new(vec![1.0, 2.0], vec![2], false);
        let x2 = Tensor::new(vec![2.0, 1.0], vec![2], false);
        let y = Tensor::new(vec![1.0, 1.0], vec![2], false);
        let loss = margin_ranking_loss(&x1, &x2, &y, 0.0);
        // -1*(1-2) + 0 = 1, relu = 1
        // -1*(2-1) + 0 = -1, relu = 0
        // mean = 0.5
        assert!((loss.data()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_maxpool2d_module() {
        let pool = MaxPool2d::new((2, 2), (2, 2), (0, 0));
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            vec![1, 1, 4, 4], false
        );
        let out = Module::forward(&pool, &x);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        let d = out.data();
        assert!((d[0] - 6.0).abs() < 1e-5);  // max(1,2,5,6)
        assert!((d[1] - 8.0).abs() < 1e-5);  // max(3,4,7,8)
        assert!((d[2] - 14.0).abs() < 1e-5); // max(9,10,13,14)
        assert!((d[3] - 16.0).abs() < 1e-5); // max(11,12,15,16)
    }

    #[test]
    fn test_maxpool2d_with_padding() {
        let pool = MaxPool2d::new((3, 3), (1, 1), (1, 1));
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3], false
        );
        let out = Module::forward(&pool, &x);
        assert_eq!(out.shape(), vec![1, 1, 3, 3]);
        // Top-left output: max of padded 3x3 region around (0,0)
        // = max(0,0,0, 0,1,2, 0,4,5) = 5.0
        let d = out.data();
        assert!((d[0] - 5.0).abs() < 1e-5);
        // Center output: max of 3x3 region around (1,1) = max(1..9) = 9.0
        assert!((d[4] - 9.0).abs() < 1e-5);
    }

    #[test]
    fn test_maxpool1d_module() {
        let pool = MaxPool1d::new(2, 2, 0);
        let x = Tensor::new(
            vec![1.0, 3.0, 2.0, 4.0, 5.0, 1.0],
            vec![1, 1, 6], false
        );
        let out = Module::forward(&pool, &x);
        assert_eq!(out.shape(), vec![1, 1, 3]);
        let d = out.data();
        assert!((d[0] - 3.0).abs() < 1e-5); // max(1, 3)
        assert!((d[1] - 4.0).abs() < 1e-5); // max(2, 4)
        assert!((d[2] - 5.0).abs() < 1e-5); // max(5, 1)
    }

    #[test]
    fn test_maxpool2d_ext_backward() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            vec![1, 1, 4, 4], true
        );
        let out = x.max_pool2d_ext((2, 2), (2, 2), (0, 0));
        let loss = out.sum();
        loss.backward();
        let g = x.grad().unwrap();
        // Grad should be 1.0 at max positions (6,8,14,16 -> indices 5,7,13,15) and 0 elsewhere
        assert!((g[5] - 1.0).abs() < 1e-5);
        assert!((g[7] - 1.0).abs() < 1e-5);
        assert!((g[13] - 1.0).abs() < 1e-5);
        assert!((g[15] - 1.0).abs() < 1e-5);
        assert!((g[0]).abs() < 1e-5);
        assert!((g[1]).abs() < 1e-5);
    }

    #[test]
    fn test_conv2d_module_shape() {
        crate::random::seed(42);
        let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (1, 1));
        let x = Tensor::rand(&[2, 3, 8, 8], false);
        let out = conv.forward(&x);
        assert_eq!(out.shape(), vec![2, 16, 8, 8]);
    }

    #[test]
    fn test_conv2d_module_stride() {
        crate::random::seed(42);
        let conv = Conv2d::new(1, 1, (3, 3), (2, 2), (1, 1));
        let x = Tensor::rand(&[1, 1, 8, 8], false);
        let out = conv.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
    }

    #[test]
    fn test_conv2d_module_params() {
        crate::random::seed(42);
        let conv = Conv2d::new(3, 16, (3, 3), (1, 1), (1, 1));
        let params = conv.params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), vec![16, 3, 3, 3]); // weight
        assert_eq!(params[1].shape(), vec![16]);            // bias
    }

    #[test]
    fn test_layer_norm_module() {
        let ln = LayerNorm::new(4);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4], false);
        let out = ln.forward(&x);
        let d = out.data();
        // mean=2.5, var=1.25, std~=1.118
        // normalized: (-1.5, -0.5, 0.5, 1.5) / 1.118 ~= (-1.342, -0.447, 0.447, 1.342)
        // With weight=1, bias=0, output == normalized
        assert!(d.len() == 4);
        // Values should be zero-mean
        let sum: f32 = d.iter().sum();
        assert!(sum.abs() < 1e-4, "LayerNorm output should be zero-mean, got sum={}", sum);
        // Check symmetry
        assert!((d[0] + d[3]).abs() < 1e-4);
        assert!((d[1] + d[2]).abs() < 1e-4);
    }

    #[test]
    fn test_layer_norm_module_trait() {
        let ln = LayerNorm::new(3);
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], false);
        let out = Module::forward(&ln, &x);
        assert_eq!(out.shape(), vec![2, 3]);
        assert_eq!(Module::params(&ln).len(), 2);
        let named = ln.named_params("ln");
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "ln.weight");
        assert_eq!(named[1].0, "ln.bias");
    }

    #[test]
    fn test_batchnorm2d_training() {
        let bn = BatchNorm2d::new(2);
        // [N=2, C=2, H=1, W=1]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2, 1, 1],
            false,
        );
        let out = bn.forward(&x);
        let d = out.data();
        assert_eq!(d.len(), 4);
        // Channel 0: values [1, 3], mean=2, var=1 -> normalized [-1, 1]
        assert!((d[0] - (-1.0)).abs() < 0.01, "got {}", d[0]);
        assert!((d[2] - 1.0).abs() < 0.01, "got {}", d[2]);
        // Channel 1: values [2, 4], mean=3, var=1 -> normalized [-1, 1]
        assert!((d[1] - (-1.0)).abs() < 0.01, "got {}", d[1]);
        assert!((d[3] - 1.0).abs() < 0.01, "got {}", d[3]);
    }

    #[test]
    fn test_batchnorm2d_running_stats() {
        let bn = BatchNorm2d::new(1);
        let x = Tensor::new(
            vec![2.0, 4.0],
            vec![2, 1, 1, 1],
            false,
        );
        let _ = bn.forward(&x);
        // batch mean = 3.0, batch var = 1.0
        // running_mean = 0.9 * 0.0 + 0.1 * 3.0 = 0.3
        // running_var  = 0.9 * 1.0 + 0.1 * 1.0 = 1.0
        let rm = bn.running_mean.data();
        let rv = bn.running_var.data();
        assert!((rm[0] - 0.3).abs() < 1e-5, "running_mean should be 0.3, got {}", rm[0]);
        assert!((rv[0] - 1.0).abs() < 1e-5, "running_var should be 1.0, got {}", rv[0]);
    }

    #[test]
    fn test_batchnorm2d_eval_mode() {
        let mut bn = BatchNorm2d::new(1);
        // Set known running stats
        {
            let mut inner = bn.running_mean.0.borrow_mut();
            inner.data = vec![2.0];
        }
        {
            let mut inner = bn.running_var.0.borrow_mut();
            inner.data = vec![4.0];
        }
        bn.eval();
        // x=4.0, running_mean=2.0, running_var=4.0, eps=1e-5
        // result = (4 - 2) / sqrt(4 + 1e-5) * 1 + 0 ~= 1.0
        let x = Tensor::new(vec![4.0], vec![1, 1, 1, 1], false);
        let out = bn.forward(&x);
        let d = out.data();
        assert!((d[0] - 1.0).abs() < 0.01, "eval mode output should be ~1.0, got {}", d[0]);
    }

    #[test]
    fn test_batchnorm2d_module_trait() {
        let mut bn = BatchNorm2d::new(3);
        let x = Tensor::new(vec![0.0; 2 * 3 * 2 * 2], vec![2, 3, 2, 2], false);
        let out = Module::forward(&bn, &x);
        assert_eq!(out.shape(), vec![2, 3, 2, 2]);
        assert_eq!(Module::params(&bn).len(), 2);
        Module::train(&mut bn);
        assert!(bn.training);
        Module::eval(&mut bn);
        assert!(!bn.training);
    }

    #[test]
    fn test_batchnorm1d_2d_input() {
        let bn = BatchNorm1d::new(2);
        // [N=2, C=2]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![2, 2],
            false,
        );
        let out = bn.forward(&x);
        let d = out.data();
        assert_eq!(out.shape(), vec![2, 2]);
        // Channel 0: values [1, 3], mean=2, var=1 -> normalized [-1, 1]
        assert!((d[0] - (-1.0)).abs() < 0.01, "got {}", d[0]);
        assert!((d[2] - 1.0).abs() < 0.01, "got {}", d[2]);
    }

    #[test]
    fn test_batchnorm1d_3d_input() {
        let bn = BatchNorm1d::new(1);
        // [N=2, C=1, L=2]
        let x = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0],
            vec![2, 1, 2],
            false,
        );
        let out = bn.forward(&x);
        assert_eq!(out.shape(), vec![2, 1, 2]);
        let d = out.data();
        // Channel 0: all values [1, 3, 5, 7], mean=4, var=5
        // normalized: (-3, -1, 1, 3) / sqrt(5)
        let sum: f32 = d.iter().sum();
        assert!(sum.abs() < 0.01, "output should be zero-mean, got sum={}", sum);
    }

    #[test]
    fn test_batchnorm1d_module_trait() {
        let mut bn = BatchNorm1d::new(4);
        let x = Tensor::new(vec![0.0; 2 * 4], vec![2, 4], false);
        let out = Module::forward(&bn, &x);
        assert_eq!(out.shape(), vec![2, 4]);
        assert_eq!(Module::params(&bn).len(), 2);
        Module::eval(&mut bn);
        assert!(!bn.training);
        Module::train(&mut bn);
        assert!(bn.training);
    }

    // ===== Upsample tests =====

    #[test]
    fn test_upsample_nearest_forward() {
        // [1, 1, 2, 2] -> scale 2 -> [1, 1, 4, 4]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
            false,
        );
        let up = Upsample::new(Some(2), None, UpsampleMode::Nearest);
        let out = up.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
        let d = out.data();
        // Each input pixel is replicated into a 2x2 block
        let expected = vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(d, expected);
    }

    #[test]
    fn test_upsample_nearest_backward() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
            true,
        );
        let out = x.upsample_nearest(2);
        let loss = out.sum();
        loss.backward();
        let g = x.grad().unwrap();
        // Each input pixel contributes to 4 output pixels, each with grad 1.0
        assert_eq!(g, vec![4.0, 4.0, 4.0, 4.0]);
    }

    #[test]
    fn test_upsample_bilinear_forward() {
        // [1, 1, 2, 2] -> upsample to 4x4
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
            false,
        );
        let up = Upsample::new(None, Some((4, 4)), UpsampleMode::Bilinear);
        let out = up.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
        // Verify corners are close to original values
        let d = out.data();
        // Top-left region should be close to 1.0
        assert!((d[0] - 1.0).abs() < 0.5, "top-left should be near 1.0, got {}", d[0]);
    }

    #[test]
    fn test_upsample_bilinear_backward() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
            true,
        );
        let out = x.upsample_bilinear(4, 4);
        let loss = out.sum();
        loss.backward();
        let g = x.grad().unwrap();
        // All output gradients are 1.0; bilinear weights sum to 1 per output pixel
        // Total grad should sum to 16 (4x4 output pixels)
        let total: f32 = g.iter().sum();
        assert!((total - 16.0).abs() < 0.01, "total grad should be 16.0, got {}", total);
    }

    #[test]
    fn test_upsample_nearest_scale1() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0],
            vec![1, 1, 2, 2],
            false,
        );
        let out = x.upsample_nearest(1);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        assert_eq!(out.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_upsample_module_no_params() {
        let up = Upsample::new(Some(2), None, UpsampleMode::Nearest);
        assert_eq!(up.params().len(), 0);
    }

    // ===== TransformerEncoder tests =====

    #[test]
    fn test_transformer_encoder_forward() {
        let enc = TransformerEncoder::new(8, 2, 16, 2);
        // batch=1, seq=3, d_model=8
        let x = Tensor::new(vec![0.1; 3 * 8], vec![3, 8], false);
        let out = enc.forward(&x, 1, 3);
        assert_eq!(out.shape(), vec![3, 8]);
    }

    #[test]
    fn test_transformer_encoder_params() {
        let enc = TransformerEncoder::new(8, 2, 16, 2);
        let p = enc.params();
        // Each encoder layer has: MHA (4 weight + 4 bias = 8) + LN1(2) + LN2(2) + FFN(4) = 16
        assert_eq!(p.len(), 2 * 16, "expected 32 params for 2 encoder layers");
    }

    #[test]
    fn test_transformer_encoder_named_params() {
        let enc = TransformerEncoder::new(8, 2, 16, 2);
        let np = enc.named_params("encoder");
        assert!(np.len() > 0);
        // Check that layer indexing is present
        let names: Vec<&str> = np.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("layers.0")));
        assert!(names.iter().any(|n| n.contains("layers.1")));
    }

    #[test]
    fn test_transformer_encoder_module_trait() {
        let enc = TransformerEncoder::new(8, 2, 16, 1);
        // Module::forward with batch=1 inference
        let x = Tensor::new(vec![0.1; 2 * 8], vec![2, 8], false);
        let out = Module::forward(&enc, &x);
        assert_eq!(out.shape(), vec![2, 8]);
    }

    // ===== TransformerDecoder tests =====

    #[test]
    fn test_transformer_decoder_forward() {
        let dec = TransformerDecoder::new(8, 2, 16, 2);
        // batch=1, num_queries=3, seq_kv=4
        let tgt = Tensor::new(vec![0.1; 3 * 8], vec![3, 8], false);
        let memory = Tensor::new(vec![0.1; 4 * 8], vec![4, 8], false);
        let out = dec.forward_with_memory(&tgt, &memory, 1, 3, 4);
        assert_eq!(out.shape(), vec![3, 8]);
    }

    #[test]
    fn test_transformer_decoder_params() {
        let dec = TransformerDecoder::new(8, 2, 16, 2);
        let p = dec.params();
        // Each decoder layer: self_attn(8) + cross_attn(8) + LN1(2) + LN2(2) + LN3(2) + FFN(4) = 26
        assert_eq!(p.len(), 2 * 26, "expected 52 params for 2 decoder layers");
    }

    #[test]
    fn test_transformer_decoder_named_params() {
        let dec = TransformerDecoder::new(8, 2, 16, 2);
        let np = dec.named_params("decoder");
        assert!(np.len() > 0);
        let names: Vec<&str> = np.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("layers.0")));
        assert!(names.iter().any(|n| n.contains("layers.1")));
    }

    // ===== Transformer tests =====

    #[test]
    fn test_transformer_forward() {
        let t = Transformer::new(8, 2, 16, 2, 2);
        let src = Tensor::new(vec![0.1; 4 * 8], vec![4, 8], false);
        let tgt = Tensor::new(vec![0.1; 3 * 8], vec![3, 8], false);
        let out = t.forward(&src, &tgt, 1, 4, 3);
        assert_eq!(out.shape(), vec![3, 8]);
    }

    #[test]
    fn test_transformer_params() {
        let t = Transformer::new(8, 2, 16, 2, 2);
        let p = t.params();
        // encoder: 2 * 16 = 32, decoder: 2 * 26 = 52
        assert_eq!(p.len(), 32 + 52, "expected 84 total params");
    }

    #[test]
    fn test_transformer_named_params() {
        let t = Transformer::new(8, 2, 16, 1, 1);
        let np = t.named_params("transformer");
        let names: Vec<&str> = np.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.contains("encoder")));
        assert!(names.iter().any(|n| n.contains("decoder")));
    }

    #[test]
    fn test_transformer_module_trait() {
        let t = Transformer::new(8, 2, 16, 1, 1);
        let x = Tensor::new(vec![0.1; 2 * 8], vec![2, 8], false);
        let out = Module::forward(&t, &x);
        assert_eq!(out.shape(), vec![2, 8]);
    }

    #[test]
    fn test_conv_transpose2d_module_shape() {
        crate::random::seed(42);
        let ct = ConvTranspose2d::new(3, 16, (3, 3), (1, 1), (0, 0));
        let x = Tensor::rand(&[2, 3, 4, 4], false);
        let out = ct.forward(&x);
        // out_h = (4-1)*1 - 0 + 3 = 6, out_w = same
        assert_eq!(out.shape(), vec![2, 16, 6, 6]);
    }

    #[test]
    fn test_conv_transpose2d_module_stride2() {
        crate::random::seed(42);
        let ct = ConvTranspose2d::new(1, 1, (3, 3), (2, 2), (1, 1));
        let x = Tensor::rand(&[1, 1, 4, 4], false);
        let out = ct.forward(&x);
        // out_h = (4-1)*2 - 2 + 3 = 7, out_w = same
        assert_eq!(out.shape(), vec![1, 1, 7, 7]);
    }

    #[test]
    fn test_conv_transpose2d_module_params() {
        crate::random::seed(42);
        let ct = ConvTranspose2d::new(3, 16, (3, 3), (1, 1), (0, 0));
        let params = ct.params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), vec![3, 16, 3, 3]); // weight [in_ch, out_ch, kH, kW]
        assert_eq!(params[1].shape(), vec![16]);            // bias
    }

    #[test]
    fn test_conv_transpose2d_module_trait() {
        crate::random::seed(42);
        let ct = ConvTranspose2d::new(2, 4, (3, 3), (1, 1), (0, 0));
        let x = Tensor::rand(&[1, 2, 3, 3], false);
        let out = Module::forward(&ct, &x);
        assert_eq!(out.shape(), vec![1, 4, 5, 5]);
        assert_eq!(Module::params(&ct).len(), 2);
        let named = ct.named_params("ct");
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "ct.weight");
        assert_eq!(named[1].0, "ct.bias");
    }

    #[test]
    fn test_conv_transpose1d_module_shape() {
        crate::random::seed(42);
        let ct = ConvTranspose1d::new(2, 4, 3, 1, 0);
        let x = Tensor::rand(&[1, 2, 5], false);
        let out = ct.forward(&x);
        // out_l = (5-1)*1 + 3 = 7
        assert_eq!(out.shape(), vec![1, 4, 7]);
    }

    #[test]
    fn test_conv_transpose1d_module_stride2() {
        crate::random::seed(42);
        let ct = ConvTranspose1d::new(1, 1, 3, 2, 1);
        let x = Tensor::rand(&[1, 1, 4], false);
        let out = ct.forward(&x);
        // out_l = (4-1)*2 - 2 + 3 = 7
        assert_eq!(out.shape(), vec![1, 1, 7]);
    }

    #[test]
    fn test_conv_transpose1d_module_params() {
        let ct = ConvTranspose1d::new(3, 8, 5, 1, 0);
        let params = ct.params();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), vec![3, 8, 5]); // weight [in_ch, out_ch, K]
        assert_eq!(params[1].shape(), vec![8]);         // bias
    }

    #[test]
    fn test_conv_transpose1d_module_trait() {
        crate::random::seed(42);
        let ct = ConvTranspose1d::new(2, 3, 3, 1, 0);
        let x = Tensor::rand(&[1, 2, 4], false);
        let out = Module::forward(&ct, &x);
        // out_l = (4-1)*1 + 3 = 6
        assert_eq!(out.shape(), vec![1, 3, 6]);
        assert_eq!(Module::params(&ct).len(), 2);
        let named = ct.named_params("ct1d");
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "ct1d.weight");
        assert_eq!(named[1].0, "ct1d.bias");
    }

    // -----------------------------------------------------------------------
    // AdaptiveAvgPool2d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_avg_pool2d_identity() {
        // Input 4x4 -> output 4x4 should be identity
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = Tensor::new(data.clone(), vec![1, 1, 4, 4], false);
        let pool = AdaptiveAvgPool2d::new((4, 4));
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
        let od = out.data();
        for i in 0..16 {
            assert!((od[i] - data[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adaptive_avg_pool2d_downsample() {
        // Input 4x4 -> output 2x2
        // Each 2x2 block gets averaged
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = Tensor::new(data, vec![1, 1, 4, 4], false);
        let pool = AdaptiveAvgPool2d::new((2, 2));
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        let od = out.data();
        // Top-left 2x2: (1+2+5+6)/4 = 3.5
        assert!((od[0] - 3.5).abs() < 1e-5);
        // Top-right 2x2: (3+4+7+8)/4 = 5.5
        assert!((od[1] - 5.5).abs() < 1e-5);
        // Bottom-left 2x2: (9+10+13+14)/4 = 11.5
        assert!((od[2] - 11.5).abs() < 1e-5);
        // Bottom-right 2x2: (11+12+15+16)/4 = 13.5
        assert!((od[3] - 13.5).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avg_pool2d_to_1x1() {
        // Global average pooling: output (1, 1)
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = Tensor::new(data, vec![1, 1, 4, 4], false);
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 1, 1]);
        // Mean of 1..=16 = 8.5
        assert!((out.data()[0] - 8.5).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avg_pool2d_multi_batch_channel() {
        // 2 batches, 2 channels, 4x4 -> 2x2
        let mut data = Vec::new();
        for _ in 0..4 {
            data.extend((1..=16).map(|x| x as f32));
        }
        let x = Tensor::new(data, vec![2, 2, 4, 4], false);
        let pool = AdaptiveAvgPool2d::new((2, 2));
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![2, 2, 2, 2]);
    }

    #[test]
    fn test_adaptive_avg_pool2d_module_trait() {
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let x = Tensor::new(data, vec![1, 1, 4, 4], false);
        let pool = AdaptiveAvgPool2d::new((2, 2));
        let out = Module::forward(&pool, &x);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        assert_eq!(Module::params(&pool).len(), 0);
    }

    // -----------------------------------------------------------------------
    // AdaptiveAvgPool1d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_avg_pool1d_downsample() {
        // Input length 6 -> output length 3
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 6], false);
        let pool = AdaptiveAvgPool1d::new(3);
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 3]);
        let od = out.data();
        // Each pair gets averaged: (1+2)/2=1.5, (3+4)/2=3.5, (5+6)/2=5.5
        assert!((od[0] - 1.5).abs() < 1e-5);
        assert!((od[1] - 3.5).abs() < 1e-5);
        assert!((od[2] - 5.5).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avg_pool1d_to_1() {
        // Global average pooling 1D
        let x = Tensor::new(vec![2.0, 4.0, 6.0, 8.0], vec![1, 1, 4], false);
        let pool = AdaptiveAvgPool1d::new(1);
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 1]);
        // Mean = 5.0
        assert!((out.data()[0] - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_adaptive_avg_pool1d_identity() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4], false);
        let pool = AdaptiveAvgPool1d::new(4);
        let out = pool.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 4]);
        let od = out.data();
        for i in 0..4 {
            assert!((od[i] - (i as f32 + 1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_adaptive_avg_pool1d_module_trait() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 1, 6], false);
        let pool = AdaptiveAvgPool1d::new(2);
        let out = Module::forward(&pool, &x);
        assert_eq!(out.shape(), vec![1, 1, 2]);
        assert_eq!(Module::params(&pool).len(), 0);
        let od = out.data();
        // (1+2+3)/3=2.0, (4+5+6)/3=5.0
        assert!((od[0] - 2.0).abs() < 1e-5);
        assert!((od[1] - 5.0).abs() < 1e-5);
    }

    // ===== Phase 5A: Padding Layers =====

    #[test]
    fn test_zero_pad2d_shape() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
            false,
        );
        let pad = ZeroPad2d::new((1, 1, 1, 1));
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 5, 5]);
    }

    #[test]
    fn test_zero_pad2d_values() {
        // Input [1,1,2,2] = [[1,2],[3,4]], pad left=1,right=0,top=0,bottom=1
        // Output [1,1,3,3] = [[0,1,2],[0,3,4],[0,0,0]]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false);
        let pad = ZeroPad2d::new((1, 0, 0, 1));
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 3, 3]);
        let d = out.data();
        assert_eq!(d, vec![0.0, 1.0, 2.0, 0.0, 3.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_zero_pad2d_no_params() {
        let pad = ZeroPad2d::new((1, 1, 1, 1));
        assert_eq!(pad.params().len(), 0);
    }

    #[test]
    fn test_zero_pad2d_module_trait() {
        let x = Tensor::new(vec![1.0; 4], vec![1, 1, 2, 2], false);
        let pad = ZeroPad2d::new((1, 1, 1, 1));
        let out = Module::forward(&pad, &x);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
        assert_eq!(Module::params(&pad).len(), 0);
    }

    #[test]
    fn test_constant_pad2d_shape() {
        let x = Tensor::new(vec![1.0; 12], vec![1, 1, 3, 4], false);
        let pad = ConstantPad2d::new((2, 2, 1, 1), 5.0);
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 5, 8]);
    }

    #[test]
    fn test_constant_pad2d_values() {
        // Input [1,1,2,2] = [[1,2],[3,4]], pad left=1,right=0,top=0,bottom=1 with value 7.0
        // Output [1,1,3,3] = [[7,1,2],[7,3,4],[7,7,7]]
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false);
        let pad = ConstantPad2d::new((1, 0, 0, 1), 7.0);
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 3, 3]);
        let d = out.data();
        assert_eq!(d, vec![7.0, 1.0, 2.0, 7.0, 3.0, 4.0, 7.0, 7.0, 7.0]);
    }

    #[test]
    fn test_constant_pad2d_no_params() {
        let pad = ConstantPad2d::new((1, 1, 1, 1), 0.0);
        assert_eq!(pad.params().len(), 0);
    }

    #[test]
    fn test_reflection_pad2d_shape() {
        let x = Tensor::new(vec![1.0; 12], vec![1, 1, 3, 4], false);
        let pad = ReflectionPad2d::new((1, 1, 1, 1));
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 5, 6]);
    }

    #[test]
    fn test_reflection_pad2d_values() {
        // Input [1,1,3,3]:
        // [[1,2,3],[4,5,6],[7,8,9]]
        // pad (1,1,1,1) reflect -> [1,1,5,5]:
        // [[5,4,5,6,5],[2,1,2,3,2],[5,4,5,6,5],[8,7,8,9,8],[5,4,5,6,5]]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
            false,
        );
        let pad = ReflectionPad2d::new((1, 1, 1, 1));
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 5, 5]);
        let d = out.data();
        let expected = vec![
            5.0, 4.0, 5.0, 6.0, 5.0,
            2.0, 1.0, 2.0, 3.0, 2.0,
            5.0, 4.0, 5.0, 6.0, 5.0,
            8.0, 7.0, 8.0, 9.0, 8.0,
            5.0, 4.0, 5.0, 6.0, 5.0,
        ];
        for (a, b) in d.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {} expected {}", a, b);
        }
    }

    #[test]
    fn test_reflection_pad2d_no_params() {
        let pad = ReflectionPad2d::new((1, 1, 1, 1));
        assert_eq!(pad.params().len(), 0);
    }

    #[test]
    fn test_reflection_pad2d_module_trait() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
            false,
        );
        let pad = ReflectionPad2d::new((1, 1, 1, 1));
        let out = Module::forward(&pad, &x);
        assert_eq!(out.shape(), vec![1, 1, 5, 5]);
    }

    #[test]
    fn test_replication_pad2d_shape() {
        let x = Tensor::new(vec![1.0; 12], vec![1, 1, 3, 4], false);
        let pad = ReplicationPad2d::new((2, 3, 1, 1));
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 5, 9]);
    }

    #[test]
    fn test_replication_pad2d_values() {
        // Input [1,1,2,3] = [[1,2,3],[4,5,6]]
        // pad (1,1,1,1) replicate -> [1,1,4,5]:
        // [[1,1,2,3,3],[1,1,2,3,3],[4,4,5,6,6],[4,4,5,6,6]]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 2, 3],
            false,
        );
        let pad = ReplicationPad2d::new((1, 1, 1, 1));
        let out = pad.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 4, 5]);
        let d = out.data();
        let expected = vec![
            1.0, 1.0, 2.0, 3.0, 3.0,
            1.0, 1.0, 2.0, 3.0, 3.0,
            4.0, 4.0, 5.0, 6.0, 6.0,
            4.0, 4.0, 5.0, 6.0, 6.0,
        ];
        for (a, b) in d.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "got {} expected {}", a, b);
        }
    }

    #[test]
    fn test_replication_pad2d_no_params() {
        let pad = ReplicationPad2d::new((1, 1, 1, 1));
        assert_eq!(pad.params().len(), 0);
    }

    #[test]
    fn test_replication_pad2d_module_trait() {
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1, 1, 2, 3],
            false,
        );
        let pad = ReplicationPad2d::new((1, 1, 1, 1));
        let out = Module::forward(&pad, &x);
        assert_eq!(out.shape(), vec![1, 1, 4, 5]);
    }

    // ===== Phase 5B: Dropout2d / AlphaDropout =====

    #[test]
    fn test_dropout2d_eval_identity() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 2, 2, 2], false);
        let mut d2d = Dropout2d::new(0.5);
        d2d.eval();
        let out = d2d.forward(&x);
        assert_eq!(out.data(), x.data());
    }

    #[test]
    fn test_dropout2d_training_shape() {
        let x = Tensor::new(vec![1.0; 16], vec![1, 2, 2, 4], false);
        let d2d = Dropout2d::new(0.5);
        let out = d2d.forward(&x);
        assert_eq!(out.shape(), vec![1, 2, 2, 4]);
    }

    #[test]
    fn test_dropout2d_channel_wise() {
        // Entire channels should be either all zero or all scaled
        let x = Tensor::new(vec![1.0; 24], vec![1, 3, 2, 4], false);
        let d2d = Dropout2d::new(0.5);
        let out = d2d.forward(&x);
        let d = out.data();
        let spatial = 2 * 4;
        let scale = 1.0 / (1.0 - 0.5);
        for ci in 0..3 {
            let offset = ci * spatial;
            let first = d[offset];
            for s in 0..spatial {
                assert!(
                    (d[offset + s] - first).abs() < 1e-6,
                    "Channel {} not uniform: d[{}]={} vs d[{}]={}",
                    ci, offset, first, offset + s, d[offset + s]
                );
            }
            assert!(
                first.abs() < 1e-6 || (first - scale).abs() < 1e-6,
                "Channel {} value {} not 0 or {}", ci, first, scale
            );
        }
    }

    #[test]
    fn test_dropout2d_no_params() {
        let d2d = Dropout2d::new(0.3);
        assert_eq!(d2d.params().len(), 0);
    }

    #[test]
    fn test_dropout2d_train_eval() {
        let mut d2d = Dropout2d::new(0.5);
        assert!(d2d.training);
        d2d.eval();
        assert!(!d2d.training);
        d2d.train();
        assert!(d2d.training);
    }

    #[test]
    fn test_dropout2d_p_zero() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], false);
        let d2d = Dropout2d::new(0.0);
        let out = d2d.forward(&x);
        assert_eq!(out.data(), x.data());
    }

    #[test]
    fn test_dropout2d_module_trait() {
        let x = Tensor::new(vec![1.0; 8], vec![1, 2, 2, 2], false);
        let mut d2d = Dropout2d::new(0.5);
        let _out = Module::forward(&d2d, &x);
        assert_eq!(Module::params(&d2d).len(), 0);
        Module::eval(&mut d2d);
        assert!(!d2d.training);
        Module::train(&mut d2d);
        assert!(d2d.training);
    }

    #[test]
    fn test_alpha_dropout_eval_identity() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let mut ad = AlphaDropout::new(0.5);
        ad.eval();
        let out = ad.forward(&x);
        assert_eq!(out.data(), x.data());
    }

    #[test]
    fn test_alpha_dropout_training_shape() {
        let x = Tensor::new(vec![1.0; 10], vec![10], false);
        let ad = AlphaDropout::new(0.3);
        let out = ad.forward(&x);
        assert_eq!(out.shape(), vec![10]);
        assert_eq!(out.data().len(), 10);
    }

    #[test]
    fn test_alpha_dropout_no_zero_values() {
        // AlphaDropout uses alpha (not 0) for dropped values
        let x = Tensor::new(vec![1.0; 100], vec![100], false);
        let ad = AlphaDropout::new(0.5);
        let out = ad.forward(&x);
        let d = out.data();
        for &v in &d {
            assert!(v.is_finite(), "AlphaDropout produced non-finite value: {}", v);
        }
    }

    #[test]
    fn test_alpha_dropout_p_zero() {
        let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4], false);
        let ad = AlphaDropout::new(0.0);
        let out = ad.forward(&x);
        assert_eq!(out.data(), x.data());
    }

    #[test]
    fn test_alpha_dropout_train_eval() {
        let mut ad = AlphaDropout::new(0.5);
        assert!(ad.training);
        ad.eval();
        assert!(!ad.training);
        ad.train();
        assert!(ad.training);
    }

    #[test]
    fn test_alpha_dropout_no_params() {
        let ad = AlphaDropout::new(0.3);
        assert_eq!(ad.params().len(), 0);
    }

    #[test]
    fn test_alpha_dropout_module_trait() {
        let x = Tensor::new(vec![1.0; 8], vec![8], false);
        let mut ad = AlphaDropout::new(0.5);
        let _out = Module::forward(&ad, &x);
        assert_eq!(Module::params(&ad).len(), 0);
        Module::eval(&mut ad);
        assert!(!ad.training);
        Module::train(&mut ad);
        assert!(ad.training);
    }

    // ===== Phase 5D: PixelShuffle / PixelUnshuffle =====

    #[test]
    fn test_pixel_shuffle_shape() {
        let x = Tensor::new(vec![1.0; 48], vec![1, 8, 2, 3], false);
        let ps = PixelShuffle::new(2);
        let out = ps.forward(&x);
        assert_eq!(out.shape(), vec![1, 2, 4, 6]);
    }

    #[test]
    fn test_pixel_shuffle_values() {
        // Input [1,4,1,1] with r=2 -> [1,1,2,2]
        let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 4, 1, 1], false);
        let ps = PixelShuffle::new(2);
        let out = ps.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 2, 2]);
        let d = out.data();
        assert_eq!(d, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_pixel_shuffle_r3() {
        let x = Tensor::new(vec![1.0; 36], vec![1, 9, 2, 2], false);
        let ps = PixelShuffle::new(3);
        let out = ps.forward(&x);
        assert_eq!(out.shape(), vec![1, 1, 6, 6]);
    }

    #[test]
    fn test_pixel_shuffle_no_params() {
        let ps = PixelShuffle::new(2);
        assert_eq!(ps.params().len(), 0);
    }

    #[test]
    fn test_pixel_shuffle_module_trait() {
        let x = Tensor::new(vec![1.0; 16], vec![1, 4, 2, 2], false);
        let ps = PixelShuffle::new(2);
        let out = Module::forward(&ps, &x);
        assert_eq!(out.shape(), vec![1, 1, 4, 4]);
        assert_eq!(Module::params(&ps).len(), 0);
    }

    #[test]
    fn test_pixel_unshuffle_shape() {
        let x = Tensor::new(vec![1.0; 48], vec![1, 2, 4, 6], false);
        let pu = PixelUnshuffle::new(2);
        let out = pu.forward(&x);
        assert_eq!(out.shape(), vec![1, 8, 2, 3]);
    }

    #[test]
    fn test_pixel_unshuffle_values() {
        // Input [1,1,2,2] = [[10,20],[30,40]] with r=2 -> [1,4,1,1]
        let x = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![1, 1, 2, 2], false);
        let pu = PixelUnshuffle::new(2);
        let out = pu.forward(&x);
        assert_eq!(out.shape(), vec![1, 4, 1, 1]);
        let d = out.data();
        assert_eq!(d, vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_pixel_unshuffle_r3() {
        let x = Tensor::new(vec![1.0; 36], vec![1, 1, 6, 6], false);
        let pu = PixelUnshuffle::new(3);
        let out = pu.forward(&x);
        assert_eq!(out.shape(), vec![1, 9, 2, 2]);
    }

    #[test]
    fn test_pixel_unshuffle_no_params() {
        let pu = PixelUnshuffle::new(2);
        assert_eq!(pu.params().len(), 0);
    }

    #[test]
    fn test_pixel_unshuffle_module_trait() {
        let x = Tensor::new(vec![1.0; 16], vec![1, 1, 4, 4], false);
        let pu = PixelUnshuffle::new(2);
        let out = Module::forward(&pu, &x);
        assert_eq!(out.shape(), vec![1, 4, 2, 2]);
        assert_eq!(Module::params(&pu).len(), 0);
    }

    #[test]
    fn test_pixel_shuffle_unshuffle_roundtrip() {
        let x = Tensor::new(
            (0..48).map(|i| i as f32).collect(),
            vec![1, 4, 3, 4],
            false,
        );
        let ps = PixelShuffle::new(2);
        let pu = PixelUnshuffle::new(2);
        let shuffled = ps.forward(&x);
        assert_eq!(shuffled.shape(), vec![1, 1, 6, 8]);
        let recovered = pu.forward(&shuffled);
        assert_eq!(recovered.shape(), vec![1, 4, 3, 4]);
        let orig = x.data();
        let rec = recovered.data();
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_pixel_unshuffle_shuffle_roundtrip() {
        let x = Tensor::new(
            (0..48).map(|i| i as f32).collect(),
            vec![1, 1, 6, 8],
            false,
        );
        let pu = PixelUnshuffle::new(2);
        let ps = PixelShuffle::new(2);
        let unshuffled = pu.forward(&x);
        assert_eq!(unshuffled.shape(), vec![1, 4, 3, 4]);
        let recovered = ps.forward(&unshuffled);
        assert_eq!(recovered.shape(), vec![1, 1, 6, 8]);
        let orig = x.data();
        let rec = recovered.data();
        for (a, b) in orig.iter().zip(rec.iter()) {
            assert!((a - b).abs() < 1e-6, "roundtrip mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_pixel_shuffle_batch() {
        // Batch size > 1: [2,4,1,1] with r=2 -> [2,1,2,2]
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 4, 1, 1],
            false,
        );
        let ps = PixelShuffle::new(2);
        let out = ps.forward(&x);
        assert_eq!(out.shape(), vec![2, 1, 2, 2]);
        let d = out.data();
        assert_eq!(&d[0..4], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&d[4..8], &[5.0, 6.0, 7.0, 8.0]);
    }

    // ===== Phase 6B: InstanceNorm2d / InstanceNorm1d =====

    #[test]
    fn test_instancenorm2d_training_basic() {
        let inn = InstanceNorm2d::new(2);
        // [N=1, C=2, H=2, W=2]
        // Channel 0: [1, 2, 3, 4], mean=2.5, var=1.25
        // Channel 1: [5, 6, 7, 8], mean=6.5, var=1.25
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 2, 2],
            false,
        );
        let out = inn.forward(&x);
        let d = out.data();
        assert_eq!(d.len(), 8);
        // Each channel should be normalized to zero mean
        let ch0_mean: f32 = d[0..4].iter().sum::<f32>() / 4.0;
        let ch1_mean: f32 = d[4..8].iter().sum::<f32>() / 4.0;
        assert!(ch0_mean.abs() < 1e-4, "ch0 mean should be ~0, got {}", ch0_mean);
        assert!(ch1_mean.abs() < 1e-4, "ch1 mean should be ~0, got {}", ch1_mean);
    }

    #[test]
    fn test_instancenorm2d_per_instance() {
        let inn = InstanceNorm2d::new(1);
        // [N=2, C=1, H=1, W=2]
        // Instance 0: [10, 20], mean=15, var=25
        // Instance 1: [100, 200], mean=150, var=2500
        let x = Tensor::new(
            vec![10.0, 20.0, 100.0, 200.0],
            vec![2, 1, 1, 2],
            false,
        );
        let out = inn.forward(&x);
        let d = out.data();
        // Each instance should have zero mean
        let inst0_mean = (d[0] + d[1]) / 2.0;
        let inst1_mean = (d[2] + d[3]) / 2.0;
        assert!(inst0_mean.abs() < 1e-4, "instance 0 mean should be ~0, got {}", inst0_mean);
        assert!(inst1_mean.abs() < 1e-4, "instance 1 mean should be ~0, got {}", inst1_mean);
        // The normalized values should be the same for both instances
        // because [10,20] and [100,200] have the same structure
        assert!((d[0] - d[2]).abs() < 1e-4, "normalized values should match");
        assert!((d[1] - d[3]).abs() < 1e-4, "normalized values should match");
    }

    #[test]
    fn test_instancenorm2d_running_stats() {
        let inn = InstanceNorm2d::new(1);
        // [N=2, C=1, H=1, W=2]
        let x = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0],
            vec![2, 1, 1, 2],
            false,
        );
        let _ = inn.forward(&x);
        // Instance 0: mean=2, var=1
        // Instance 1: mean=6, var=1
        // Average across batch: mean=4, var=1
        // running_mean = 0.9 * 0.0 + 0.1 * 4.0 = 0.4
        // running_var  = 0.9 * 1.0 + 0.1 * 1.0 = 1.0
        let rm = inn.running_mean.data();
        let rv = inn.running_var.data();
        assert!(
            (rm[0] - 0.4).abs() < 1e-5,
            "running_mean should be ~0.4, got {}",
            rm[0]
        );
        assert!(
            (rv[0] - 1.0).abs() < 1e-5,
            "running_var should be ~1.0, got {}",
            rv[0]
        );
    }

    #[test]
    fn test_instancenorm2d_eval_mode() {
        let mut inn = InstanceNorm2d::new(1);
        // Set known running stats
        {
            let mut inner = inn.running_mean.0.borrow_mut();
            inner.data = vec![5.0];
        }
        {
            let mut inner = inn.running_var.0.borrow_mut();
            inner.data = vec![4.0];
        }
        inn.eval();

        let x = Tensor::new(
            vec![5.0, 7.0, 3.0, 9.0],
            vec![2, 1, 1, 2],
            false,
        );
        let out = inn.forward(&x);
        let d = out.data();
        // Using running stats: inv_std = 1/sqrt(4 + 1e-5) ~= 0.5
        // x=5: (5-5)*0.5 = 0
        // x=7: (7-5)*0.5 = 1
        // x=3: (3-5)*0.5 = -1
        // x=9: (9-5)*0.5 = 2
        assert!((d[0] - 0.0).abs() < 0.01, "got {}", d[0]);
        assert!((d[1] - 1.0).abs() < 0.01, "got {}", d[1]);
        assert!((d[2] - (-1.0)).abs() < 0.01, "got {}", d[2]);
        assert!((d[3] - 2.0).abs() < 0.01, "got {}", d[3]);
    }

    #[test]
    fn test_instancenorm2d_module_trait() {
        let mut inn = InstanceNorm2d::new(3);
        let x = Tensor::new(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], false);
        let _out = Module::forward(&inn, &x);
        assert_eq!(Module::params(&inn).len(), 2);
        let named = inn.named_params("in2d");
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "in2d.weight");
        assert_eq!(named[1].0, "in2d.bias");
        Module::eval(&mut inn);
        assert!(!inn.training);
        Module::train(&mut inn);
        assert!(inn.training);
    }

    #[test]
    fn test_instancenorm1d_training_basic() {
        let inn = InstanceNorm1d::new(2);
        // [N=1, C=2, L=4]
        // Channel 0: [1, 2, 3, 4], mean=2.5, var=1.25
        // Channel 1: [5, 6, 7, 8], mean=6.5, var=1.25
        let x = Tensor::new(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![1, 2, 4],
            false,
        );
        let out = inn.forward(&x);
        let d = out.data();
        assert_eq!(d.len(), 8);
        assert_eq!(out.shape(), vec![1, 2, 4]);
        // Each channel should have zero mean after normalization
        let ch0_mean: f32 = d[0..4].iter().sum::<f32>() / 4.0;
        let ch1_mean: f32 = d[4..8].iter().sum::<f32>() / 4.0;
        assert!(ch0_mean.abs() < 1e-4, "ch0 mean should be ~0, got {}", ch0_mean);
        assert!(ch1_mean.abs() < 1e-4, "ch1 mean should be ~0, got {}", ch1_mean);
    }

    #[test]
    fn test_instancenorm1d_running_stats() {
        let inn = InstanceNorm1d::new(1);
        // [N=2, C=1, L=2]
        let x = Tensor::new(
            vec![1.0, 3.0, 5.0, 7.0],
            vec![2, 1, 2],
            false,
        );
        let _ = inn.forward(&x);
        let rm = inn.running_mean.data();
        let rv = inn.running_var.data();
        // Same as InstanceNorm2d test:
        // Instance 0: mean=2, var=1; Instance 1: mean=6, var=1
        // Avg mean=4, avg var=1
        assert!(
            (rm[0] - 0.4).abs() < 1e-5,
            "running_mean should be ~0.4, got {}",
            rm[0]
        );
        assert!(
            (rv[0] - 1.0).abs() < 1e-5,
            "running_var should be ~1.0, got {}",
            rv[0]
        );
    }

    #[test]
    fn test_instancenorm1d_module_trait() {
        let mut inn = InstanceNorm1d::new(3);
        let x = Tensor::new(vec![1.0; 3 * 8], vec![1, 3, 8], false);
        let _out = Module::forward(&inn, &x);
        assert_eq!(Module::params(&inn).len(), 2);
        let named = inn.named_params("in1d");
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "in1d.weight");
        assert_eq!(named[1].0, "in1d.bias");
        Module::eval(&mut inn);
        assert!(!inn.training);
        Module::train(&mut inn);
        assert!(inn.training);
    }
}
