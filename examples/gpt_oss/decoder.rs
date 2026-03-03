use peregrine::nn::RMSNorm;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::{AttentionBlock, KVCache};
use crate::moe::MoELayer;

/// Transformer block: pre-norm attention + pre-norm MoE, each with residual.
///   h = x + attn(attn_norm(x))
///   h = h + moe(ffn_norm(h))
pub struct TransformerBlock {
    pub attn: AttentionBlock,
    pub moe: MoELayer,
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
}

impl TransformerBlock {
    pub fn new(
        layer_id: usize,
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        intermediate: usize,
        num_experts: usize,
        top_k: usize,
        max_seq_len: usize,
        rope_theta: f32,
        rope_factor: f32,
        initial_ctx: usize,
        beta_fast: usize,
        beta_slow: usize,
        sliding_window: usize,
        swiglu_alpha: f32,
        swiglu_clamp: f32,
    ) -> Self {
        let use_sliding_window = layer_id % 2 == 0;

        TransformerBlock {
            attn: AttentionBlock::new(
                model_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                rope_theta,
                rope_factor,
                initial_ctx,
                beta_fast,
                beta_slow,
                use_sliding_window,
                sliding_window,
            ),
            moe: MoELayer::new(
                model_dim,
                intermediate,
                num_experts,
                top_k,
                swiglu_alpha,
                swiglu_clamp,
            ),
            attn_norm: RMSNorm::new(model_dim, 1e-5),
            ffn_norm: RMSNorm::new(model_dim, 1e-5),
        }
    }

    /// h: [seq_len, model_dim]
    pub fn forward(&self, h: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        // Attention block
        let attn_in = self.attn_norm.forward(h);
        let attn_out = self.attn.forward(&attn_in, kv_cache);
        let h = h.add(&attn_out);

        // MoE block
        let moe_in = self.ffn_norm.forward(&h);
        let moe_out = self.moe.forward(&moe_in);
        h.add(&moe_out)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        if let Some((_s, data)) = params.get(&format!("{}.attn_norm.weight", prefix)) {
            self.attn_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }
        if let Some((_s, data)) = params.get(&format!("{}.ffn_norm.weight", prefix)) {
            self.ffn_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }

        self.attn
            .load_weights(params, &format!("{}.attn", prefix));
        self.moe.load_weights(params, &format!("{}.moe", prefix));
    }
}
