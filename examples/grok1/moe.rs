use peregrine::tensor::Tensor;
use std::collections::HashMap;

/// SwiGLU FFN block (one expert).
/// gate = gelu(x @ linear_gate)
/// value = x @ linear_v
/// output = (gate * value) @ linear_out
pub struct DenseBlock {
    pub linear_gate: Tensor, // [model_dim, ffn_dim]
    pub linear_v: Tensor,    // [model_dim, ffn_dim]
    pub linear_out: Tensor,  // [ffn_dim, model_dim]
    _model_dim: usize,
    _ffn_dim: usize,
}

impl DenseBlock {
    pub fn new(model_dim: usize, ffn_dim: usize) -> Self {
        DenseBlock {
            linear_gate: Tensor::zeros(&[model_dim, ffn_dim], false),
            linear_v: Tensor::zeros(&[model_dim, ffn_dim], false),
            linear_out: Tensor::zeros(&[ffn_dim, model_dim], false),
            _model_dim: model_dim,
            _ffn_dim: ffn_dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // gate = gelu(x @ linear_gate)
        let gate = x.matmul(&self.linear_gate).gelu();
        // value = x @ linear_v
        let value = x.matmul(&self.linear_v);
        // output = (gate * value) @ linear_out
        gate.mul(&value).matmul(&self.linear_out)
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        if let Some((shape, data)) = params.get(&format!("{}.linear_gate", prefix)) {
            self.linear_gate = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.linear_v", prefix)) {
            self.linear_v = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.linear_out", prefix)) {
            self.linear_out = Tensor::new(data.clone(), shape.clone(), false);
        }
    }
}

/// Mixture-of-Experts layer with top-k routing.
pub struct MoELayer {
    pub router: Tensor, // [model_dim, num_experts]
    pub experts: Vec<DenseBlock>,
    num_experts: usize,
    top_k: usize,
}

impl MoELayer {
    pub fn new(model_dim: usize, ffn_dim: usize, num_experts: usize, top_k: usize) -> Self {
        let experts = (0..num_experts)
            .map(|_| DenseBlock::new(model_dim, ffn_dim))
            .collect();
        MoELayer {
            router: Tensor::zeros(&[model_dim, num_experts], false),
            experts,
            num_experts,
            top_k,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];
        let model_dim = shape[1];

        // Router: logits = x @ router_weight -> [seq_len, num_experts]
        let logits = x.matmul(&self.router);
        let probs = logits.softmax(1);
        let probs_data = probs.data();

        let x_data = x.data();
        let mut output_data = vec![0.0f32; seq_len * model_dim];

        for t in 0..seq_len {
            // Get top-k experts for this token
            let prob_offset = t * self.num_experts;
            let mut expert_probs: Vec<(usize, f32)> = (0..self.num_experts)
                .map(|e| (e, probs_data[prob_offset + e]))
                .collect();
            expert_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let token_input = Tensor::new(
                x_data[t * model_dim..(t + 1) * model_dim].to_vec(),
                vec![1, model_dim],
                false,
            );

            let out_offset = t * model_dim;
            for k in 0..self.top_k {
                let (expert_idx, prob) = expert_probs[k];
                let expert_out = self.experts[expert_idx].forward(&token_input);
                let expert_data = expert_out.data();
                for d in 0..model_dim {
                    output_data[out_offset + d] += prob * expert_data[d];
                }
            }
        }

        Tensor::new(output_data, vec![seq_len, model_dim], false)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        if let Some((shape, data)) = params.get(&format!("{}.router", prefix)) {
            self.router = Tensor::new(data.clone(), shape.clone(), false);
        }
        for j in 0..self.num_experts {
            let expert_prefix = format!("{}.experts.{}", prefix, j);
            self.experts[j].load_weights(params, &expert_prefix);
        }
    }
}
