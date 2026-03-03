use peregrine::tensor::Tensor;
use std::collections::HashMap;

/// SwiGLU MLP: output = w2(silu(w1(x)) * w3(x))
pub struct MLP {
    pub w1: Tensor, // [dim, inter_dim]
    pub w2: Tensor, // [inter_dim, dim]
    pub w3: Tensor, // [dim, inter_dim]
}

impl MLP {
    pub fn new(dim: usize, inter_dim: usize) -> Self {
        MLP {
            w1: Tensor::zeros(&[dim, inter_dim], false),
            w2: Tensor::zeros(&[inter_dim, dim], false),
            w3: Tensor::zeros(&[dim, inter_dim], false),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = x.matmul(&self.w1).silu();
        let h3 = x.matmul(&self.w3);
        h1.mul(&h3).matmul(&self.w2)
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        if let Some((s, d)) = params.get(&format!("{}.w1", prefix)) {
            self.w1 = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((s, d)) = params.get(&format!("{}.w2", prefix)) {
            self.w2 = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((s, d)) = params.get(&format!("{}.w3", prefix)) {
            self.w3 = Tensor::new(d.clone(), s.clone(), false);
        }
    }
}

/// Single expert (same as MLP but uses non-parallel Linear).
pub struct Expert {
    pub w1: Tensor, // [dim, inter_dim]
    pub w2: Tensor, // [inter_dim, dim]
    pub w3: Tensor, // [dim, inter_dim]
}

impl Expert {
    pub fn new(dim: usize, inter_dim: usize) -> Self {
        Expert {
            w1: Tensor::zeros(&[dim, inter_dim], false),
            w2: Tensor::zeros(&[inter_dim, dim], false),
            w3: Tensor::zeros(&[dim, inter_dim], false),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let h1 = x.matmul(&self.w1).silu();
        let h3 = x.matmul(&self.w3);
        h1.mul(&h3).matmul(&self.w2)
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        if let Some((s, d)) = params.get(&format!("{}.w1", prefix)) {
            self.w1 = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((s, d)) = params.get(&format!("{}.w2", prefix)) {
            self.w2 = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((s, d)) = params.get(&format!("{}.w3", prefix)) {
            self.w3 = Tensor::new(d.clone(), s.clone(), false);
        }
    }
}

/// Router/Gate with sigmoid scoring and group-limited top-k selection.
pub struct Gate {
    pub weight: Tensor,        // [n_routed_experts, dim]
    pub bias: Option<Vec<f32>>, // [n_routed_experts] (only for 671B)
    n_routed_experts: usize,
    topk: usize,
    n_groups: usize,
    topk_groups: usize,
    route_scale: f32,
    use_sigmoid: bool,
}

impl Gate {
    pub fn new(
        dim: usize,
        n_routed_experts: usize,
        topk: usize,
        n_groups: usize,
        topk_groups: usize,
        route_scale: f32,
        use_sigmoid: bool,
        has_bias: bool,
    ) -> Self {
        Gate {
            weight: Tensor::zeros(&[n_routed_experts, dim], false),
            bias: if has_bias { Some(vec![0.0; n_routed_experts]) } else { None },
            n_routed_experts,
            topk,
            n_groups,
            topk_groups,
            route_scale,
            use_sigmoid,
        }
    }

    /// Returns (weights, indices) for each token.
    /// weights: Vec<Vec<(expert_idx, weight)>> per token
    pub fn forward(&self, x: &Tensor) -> Vec<Vec<(usize, f32)>> {
        let shape = x.shape();
        let n_tokens = shape[0];
        let dim = shape[1];
        let x_data = x.data();
        let w_data = self.weight.data();

        let mut result = Vec::with_capacity(n_tokens);

        for t in 0..n_tokens {
            // Compute scores = x @ weight^T -> [n_routed_experts]
            let mut scores = vec![0.0f32; self.n_routed_experts];
            for e in 0..self.n_routed_experts {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += x_data[t * dim + d] * w_data[e * dim + d];
                }
                scores[e] = if self.use_sigmoid {
                    1.0 / (1.0 + (-dot).exp())
                } else {
                    dot // softmax applied later
                };
            }

            // For softmax scoring
            if !self.use_sigmoid {
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = scores.iter().map(|&s| (s - max_s).exp()).sum();
                for s in &mut scores {
                    *s = (*s - max_s).exp() / sum;
                }
            }

            let original_scores = scores.clone();

            // Add bias if present
            if let Some(ref bias) = self.bias {
                for e in 0..self.n_routed_experts {
                    scores[e] += bias[e];
                }
            }

            // Group-limited routing
            if self.n_groups > 1 {
                let experts_per_group = self.n_routed_experts / self.n_groups;
                let mut group_scores = vec![0.0f32; self.n_groups];

                for g in 0..self.n_groups {
                    let start = g * experts_per_group;
                    let end = start + experts_per_group;
                    let mut group_vals: Vec<f32> = scores[start..end].to_vec();
                    group_vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    // Sum of top-2 if bias present, else max
                    group_scores[g] = if self.bias.is_some() {
                        group_vals.iter().take(2).sum()
                    } else {
                        group_vals[0]
                    };
                }

                // Select top-k groups
                let mut group_indices: Vec<usize> = (0..self.n_groups).collect();
                group_indices.sort_by(|&a, &b| group_scores[b].partial_cmp(&group_scores[a]).unwrap());
                let selected_groups: Vec<usize> = group_indices[..self.topk_groups].to_vec();

                // Zero out non-selected groups
                for g in 0..self.n_groups {
                    if !selected_groups.contains(&g) {
                        let start = g * experts_per_group;
                        let end = start + experts_per_group;
                        for e in start..end {
                            scores[e] = 0.0;
                        }
                    }
                }
            }

            // Select top-k experts by biased scores
            let mut expert_indices: Vec<usize> = (0..self.n_routed_experts).collect();
            expert_indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap());
            let selected: Vec<usize> = expert_indices[..self.topk].to_vec();

            // Use original scores (pre-bias) for the weights
            let mut weights: Vec<(usize, f32)> = selected
                .iter()
                .map(|&e| (e, original_scores[e]))
                .collect();

            // Normalize weights for sigmoid scoring
            if self.use_sigmoid {
                let sum: f32 = weights.iter().map(|(_, w)| w).sum();
                if sum > 0.0 {
                    for w in &mut weights {
                        w.1 /= sum;
                    }
                }
            }

            // Apply route_scale
            for w in &mut weights {
                w.1 *= self.route_scale;
            }

            result.push(weights);
        }

        result
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        if let Some((s, d)) = params.get(&format!("{}.gate", prefix)) {
            self.weight = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((_s, d)) = params.get(&format!("{}.bias", prefix)) {
            self.bias = Some(d.clone());
        }
    }
}

/// Mixture-of-Experts layer with shared expert.
pub struct MoE {
    pub gate: Gate,
    pub experts: Vec<Expert>,
    pub shared_experts: MLP,
    dim: usize,
}

impl MoE {
    pub fn new(
        dim: usize,
        moe_inter_dim: usize,
        n_routed_experts: usize,
        n_shared_experts: usize,
        n_activated_experts: usize,
        n_expert_groups: usize,
        n_limited_groups: usize,
        route_scale: f32,
        use_sigmoid: bool,
        has_bias: bool,
    ) -> Self {
        let experts = (0..n_routed_experts)
            .map(|_| Expert::new(dim, moe_inter_dim))
            .collect();
        MoE {
            gate: Gate::new(
                dim,
                n_routed_experts,
                n_activated_experts,
                n_expert_groups,
                n_limited_groups,
                route_scale,
                use_sigmoid,
                has_bias,
            ),
            experts,
            shared_experts: MLP::new(dim, n_shared_experts * moe_inter_dim),
            dim,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let n_tokens = shape[0];
        let x_data = x.data();

        // Route tokens to experts
        let routing = self.gate.forward(x);

        // Compute routed expert outputs
        let mut y_data = vec![0.0f32; n_tokens * self.dim];
        for t in 0..n_tokens {
            let token = Tensor::new(
                x_data[t * self.dim..(t + 1) * self.dim].to_vec(),
                vec![1, self.dim],
                false,
            );
            for &(expert_idx, weight) in &routing[t] {
                let expert_out = self.experts[expert_idx].forward(&token);
                let out_data = expert_out.data();
                let off = t * self.dim;
                for d in 0..self.dim {
                    y_data[off + d] += weight * out_data[d];
                }
            }
        }

        let y = Tensor::new(y_data, vec![n_tokens, self.dim], false);

        // Add shared expert output
        let z = self.shared_experts.forward(x);
        y.add(&z)
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        self.gate.load_weights(params, prefix);
        for (i, expert) in self.experts.iter_mut().enumerate() {
            expert.load_weights(params, &format!("{}.experts.{}", prefix, i));
        }
        self.shared_experts.load_weights(params, &format!("{}.shared_experts", prefix));
    }
}
