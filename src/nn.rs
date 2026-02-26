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
    let diff = pred.sub(target);
    let abs_diff = diff.abs();
    // We compose using: if |d| < beta then 0.5*d^2/beta else |d| - 0.5*beta
    // Build mask on CPU, apply via tensor ops
    let abs_data = abs_diff.data();
    let shape = abs_diff.shape();

    // Quadratic region: 0.5 * diff^2 / beta
    let quadratic = diff.pow(2.0).scale(0.5 / beta);
    // Linear region: |diff| - 0.5 * beta
    let half_beta = Tensor::full(&shape, 0.5 * beta, false);
    let linear = abs_diff.sub(&half_beta);

    // Build mask tensor: 1.0 where |diff| < beta, 0.0 otherwise
    let mask_data: Vec<f32> = abs_data.iter().map(|&v| if v < beta { 1.0 } else { 0.0 }).collect();
    let mask = Tensor::new(mask_data, shape.clone(), false);
    let inv_mask_data: Vec<f32> = abs_data.iter().map(|&v| if v < beta { 0.0 } else { 1.0 }).collect();
    let inv_mask = Tensor::new(inv_mask_data, shape, false);

    // result = mask * quadratic + inv_mask * linear
    let result = quadratic.mul(&mask).add(&linear.mul(&inv_mask));
    result.mean()
}

/// Huber loss with delta threshold:
///   0.5 * x^2              if |x| <= delta
///   delta * (|x| - 0.5*delta)  otherwise
pub fn huber_loss(pred: &Tensor, target: &Tensor, delta: f32) -> Tensor {
    assert!(delta > 0.0, "delta must be positive");
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
    // dot = sum(a * b), norm_a = sqrt(sum(a^2)), norm_b = sqrt(sum(b^2))
    let dot = a.mul(b).sum();
    let norm_a = a.pow(2.0).sum().sqrt();
    let norm_b = b.pow(2.0).sum().sqrt();
    // cos_sim = dot / (norm_a * norm_b + eps)
    let eps = Tensor::new(vec![1e-8], vec![1], false);
    let denom = norm_a.mul(&norm_b).add(&eps);
    let cos_sim = dot.div(&denom);
    let one = Tensor::new(vec![1.0], vec![1], false);
    one.sub(&cos_sim)
}

/// Triplet margin loss: max(0, ||anchor - pos||^2 - ||anchor - neg||^2 + margin)
pub fn triplet_loss(anchor: &Tensor, pos: &Tensor, neg: &Tensor, margin: f32) -> Tensor {
    let d_pos = anchor.sub(pos).pow(2.0).sum();
    let d_neg = anchor.sub(neg).pow(2.0).sum();
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

        let x_data = x.data();
        let mut h = h0.clone();
        let mut c = c0.clone();
        let mut outputs = Vec::with_capacity(seq_len * self.hidden_size);
        let hs = self.hidden_size;

        for t in 0..seq_len {
            let start = t * self.input_size;
            let x_t_data = x_data[start..start + self.input_size].to_vec();
            let x_t = Tensor::new(x_t_data, vec![1, self.input_size], false);

            // gates = x_t @ W_ih + b_ih + h @ W_hh + b_hh  [1, 4*hidden_size]
            let gates = x_t
                .matmul(&self.weight_ih)
                .add_bias(&self.bias_ih)
                .add(&h.matmul(&self.weight_hh).add_bias(&self.bias_hh));

            let gates_data = gates.data();

            // Split into i, f, g, o gates
            let i_data: Vec<f32> = gates_data[0..hs].to_vec();
            let f_data: Vec<f32> = gates_data[hs..2 * hs].to_vec();
            let g_data: Vec<f32> = gates_data[2 * hs..3 * hs].to_vec();
            let o_data: Vec<f32> = gates_data[3 * hs..4 * hs].to_vec();

            let i_gate = Tensor::new(i_data, vec![1, hs], false).sigmoid();
            let f_gate = Tensor::new(f_data, vec![1, hs], false).sigmoid();
            let g_gate = Tensor::new(g_data, vec![1, hs], false).tanh();
            let o_gate = Tensor::new(o_data, vec![1, hs], false).sigmoid();

            // c = f * c + i * g
            c = f_gate.mul(&c).add(&i_gate.mul(&g_gate));
            // h = o * tanh(c)
            h = o_gate.mul(&c.tanh());

            outputs.extend(h.data());
        }

        let output = Tensor::new(outputs, vec![seq_len, self.hidden_size], false);
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

        let data = x.data();
        let w = self.weight.data();
        let b = self.bias.data();
        let mut out = vec![0.0f32; data.len()];

        for n in 0..batch {
            for g in 0..self.num_groups {
                let group_size = cpg * spatial;
                // Compute mean & variance of this group
                let mut sum = 0.0f32;
                for c_in_g in 0..cpg {
                    let c = g * cpg + c_in_g;
                    let base = n * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        sum += data[base + s];
                    }
                }
                let mean = sum / group_size as f32;
                let mut var = 0.0f32;
                for c_in_g in 0..cpg {
                    let c = g * cpg + c_in_g;
                    let base = n * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        let d = data[base + s] - mean;
                        var += d * d;
                    }
                }
                var /= group_size as f32;
                let inv_std = 1.0 / (var + self.eps).sqrt();

                // Normalize and apply affine
                for c_in_g in 0..cpg {
                    let c = g * cpg + c_in_g;
                    let base = n * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        out[base + s] =
                            (data[base + s] - mean) * inv_std * w[c] + b[c];
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
        let mut attn_out_data: Vec<f32> = Vec::with_capacity(batch * self.num_heads * seq_q * self.head_dim);
        let q_data = q_2d.data();
        let k_data = k_2d.data();
        let v_data = v_2d.data();

        for bh in 0..(batch * self.num_heads) {
            let q_offset = bh * seq_q * self.head_dim;
            let k_offset = bh * seq_kv * self.head_dim;
            let v_offset = bh * seq_kv * self.head_dim;

            let q_slice = q_data[q_offset..q_offset + seq_q * self.head_dim].to_vec();
            let k_slice = k_data[k_offset..k_offset + seq_kv * self.head_dim].to_vec();
            let v_slice = v_data[v_offset..v_offset + seq_kv * self.head_dim].to_vec();

            let q_t = Tensor::new(q_slice, vec![seq_q, self.head_dim], false);
            let k_t = Tensor::new(k_slice, vec![seq_kv, self.head_dim], false);
            let v_t = Tensor::new(v_slice, vec![seq_kv, self.head_dim], false);

            let k_transposed = k_t.transpose(0, 1);
            let scores = q_t.matmul(&k_transposed).scale(scale);
            let attn_weights = scores.softmax(-1);

            let context = attn_weights.matmul(&v_t);
            attn_out_data.extend(context.data());
        }

        let attn_out = Tensor::new(
            attn_out_data,
            vec![batch, self.num_heads, seq_q, self.head_dim],
            false,
        );
        let attn_out = attn_out.transpose(1, 2);
        let attn_flat = attn_out.reshape(vec![batch * seq_q, embed_dim]);

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
        let hidden_dim = 4 * embed_dim;
        TransformerEncoderLayer {
            mha: MultiHeadAttention::new(embed_dim, num_heads),
            ln1_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln1_beta: Tensor::zeros(&[embed_dim], true),
            ln2_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln2_beta: Tensor::zeros(&[embed_dim], true),
            ffn_w1: Tensor::randn(&[embed_dim, hidden_dim], true),
            ffn_b1: Tensor::zeros(&[1, hidden_dim], true),
            ffn_w2: Tensor::randn(&[hidden_dim, embed_dim], true),
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
        let hidden_dim = 4 * embed_dim;
        TransformerDecoderLayer {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            cross_attn: MultiHeadAttention::new(embed_dim, num_heads),
            ln1_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln1_beta: Tensor::zeros(&[embed_dim], true),
            ln2_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln2_beta: Tensor::zeros(&[embed_dim], true),
            ln3_gamma: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], true),
            ln3_beta: Tensor::zeros(&[embed_dim], true),
            ffn_w1: Tensor::randn(&[embed_dim, hidden_dim], true),
            ffn_b1: Tensor::zeros(&[1, hidden_dim], true),
            ffn_w2: Tensor::randn(&[hidden_dim, embed_dim], true),
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
}
