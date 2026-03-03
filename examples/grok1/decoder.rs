use peregrine::nn::RMSNorm;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::{GroupedQueryAttention, KVCache};
use crate::moe::MoELayer;

/// One decoder layer: pre-norm attention + pre-norm MoE, each with post-norm and residual.
/// Matches model.py DecoderLayer (lines 1048-1096):
///   attn_in = rms_norm_1(h)
///   attn_out = attention(attn_in)
///   attn_out = rms_norm_2(attn_out)
///   h = h + attn_out
///   moe_in = rms_norm_3(h)
///   moe_out = moe(moe_in)
///   moe_out = rms_norm_4(moe_out)
///   h = h + moe_out
pub struct DecoderLayer {
    pub pre_attn_norm: RMSNorm,
    pub post_attn_norm: RMSNorm,
    pub attention: GroupedQueryAttention,
    pub pre_moe_norm: RMSNorm,
    pub post_moe_norm: RMSNorm,
    pub moe: MoELayer,
}

impl DecoderLayer {
    pub fn new(
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        num_experts: usize,
        top_k: usize,
        max_seq_len: usize,
        rope_base: f32,
        attn_output_mult: f32,
        logit_cap: f32,
    ) -> Self {
        DecoderLayer {
            pre_attn_norm: RMSNorm::new(model_dim, 1e-5),
            post_attn_norm: RMSNorm::new(model_dim, 1e-5),
            attention: GroupedQueryAttention::new(
                model_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                attn_output_mult,
                logit_cap,
                max_seq_len,
                rope_base,
            ),
            pre_moe_norm: RMSNorm::new(model_dim, 1e-5),
            post_moe_norm: RMSNorm::new(model_dim, 1e-5),
            moe: MoELayer::new(model_dim, ffn_dim, num_experts, top_k),
        }
    }

    /// h: [seq_len, model_dim]
    pub fn forward(&self, h: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        // Attention block
        let attn_in = self.pre_attn_norm.forward(h);
        let attn_out = self.attention.forward(&attn_in, kv_cache);
        let attn_out = self.post_attn_norm.forward(&attn_out);
        let h = h.add(&attn_out);

        // MoE block
        let moe_in = self.pre_moe_norm.forward(&h);
        let moe_out = self.moe.forward(&moe_in);
        let moe_out = self.post_moe_norm.forward(&moe_out);
        h.add(&moe_out)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        // RMSNorm weights
        if let Some((_s, data)) = params.get(&format!("{}.pre_attn_norm.weight", prefix)) {
            self.pre_attn_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }
        if let Some((_s, data)) = params.get(&format!("{}.post_attn_norm.weight", prefix)) {
            self.post_attn_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }
        if let Some((_s, data)) = params.get(&format!("{}.pre_moe_norm.weight", prefix)) {
            self.pre_moe_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }
        if let Some((_s, data)) = params.get(&format!("{}.post_moe_norm.weight", prefix)) {
            self.post_moe_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }

        // Attention weights
        self.attention
            .load_weights(params, &format!("{}.attention", prefix));

        // MoE weights
        self.moe.load_weights(params, &format!("{}.moe", prefix));
    }
}
