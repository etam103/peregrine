use peregrine::nn::RMSNorm;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::{KVCache, MLA};
use crate::moe::{MLP, MoE};

/// FFN variant: either dense MLP (first n_dense_layers) or MoE.
pub enum FFN {
    Dense(MLP),
    Moe(MoE),
}

impl FFN {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            FFN::Dense(mlp) => mlp.forward(x),
            FFN::Moe(moe) => moe.forward(x),
        }
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        match self {
            FFN::Dense(mlp) => mlp.load_weights(params, prefix),
            FFN::Moe(moe) => moe.load_weights(params, prefix),
        }
    }
}

/// DeepSeek Block: pre-norm attention + pre-norm FFN/MoE with residuals.
///   x = x + attn(attn_norm(x))
///   x = x + ffn(ffn_norm(x))
pub struct Block {
    pub attn: MLA,
    pub ffn: FFN,
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
}

impl Block {
    pub fn new(
        layer_id: usize,
        dim: usize,
        n_heads: usize,
        inter_dim: usize,
        moe_inter_dim: usize,
        n_dense_layers: usize,
        n_routed_experts: usize,
        n_shared_experts: usize,
        n_activated_experts: usize,
        n_expert_groups: usize,
        n_limited_groups: usize,
        route_scale: f32,
        score_func_sigmoid: bool,
        has_bias: bool,
        q_lora_rank: usize,
        kv_lora_rank: usize,
        qk_nope_head_dim: usize,
        qk_rope_head_dim: usize,
        v_head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
        rope_factor: f32,
        original_seq_len: usize,
        beta_fast: usize,
        beta_slow: usize,
        mscale: f32,
    ) -> Self {
        let ffn = if layer_id < n_dense_layers {
            FFN::Dense(MLP::new(dim, inter_dim))
        } else {
            FFN::Moe(MoE::new(
                dim,
                moe_inter_dim,
                n_routed_experts,
                n_shared_experts,
                n_activated_experts,
                n_expert_groups,
                n_limited_groups,
                route_scale,
                score_func_sigmoid,
                has_bias,
            ))
        };

        Block {
            attn: MLA::new(
                dim,
                n_heads,
                q_lora_rank,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                max_seq_len,
                rope_theta,
                rope_factor,
                original_seq_len,
                beta_fast,
                beta_slow,
                mscale,
            ),
            ffn,
            attn_norm: RMSNorm::new(dim, 1e-6),
            ffn_norm: RMSNorm::new(dim, 1e-6),
        }
    }

    pub fn forward(&self, x: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let attn_in = self.attn_norm.forward(x);
        let attn_out = self.attn.forward(&attn_in, kv_cache);
        let h = x.add(&attn_out);

        let ffn_in = self.ffn_norm.forward(&h);
        let ffn_out = self.ffn.forward(&ffn_in);
        h.add(&ffn_out)
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        if let Some((_s, d)) = params.get(&format!("{}.attn_norm.weight", prefix)) {
            self.attn_norm.weight = Tensor::new(d.clone(), vec![d.len()], false);
        }
        if let Some((_s, d)) = params.get(&format!("{}.ffn_norm.weight", prefix)) {
            self.ffn_norm.weight = Tensor::new(d.clone(), vec![d.len()], false);
        }
        self.attn.load_weights(params, &format!("{}.attn", prefix));
        self.ffn.load_weights(params, &format!("{}.ffn", prefix));
    }
}
