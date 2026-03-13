use peregrine::nn::RMSNorm;
use peregrine::tensor::Tensor;

use crate::attention::{KVCache, LlamaAttention};

/// One Llama decoder layer: pre-norm GQA attention + pre-norm SwiGLU FFN with residuals.
///
///   h' = h + attention(rms_norm(h))
///   out = h' + swiglu_ffn(rms_norm(h'))
pub struct LlamaBlock {
    pub attn_norm: RMSNorm,
    pub ffn_norm: RMSNorm,
    pub attention: LlamaAttention,
    // SwiGLU FFN: down(silu(gate(x)) * up(x))
    pub gate_proj: Tensor, // [model_dim, ffn_dim]
    pub up_proj: Tensor,   // [model_dim, ffn_dim]
    pub down_proj: Tensor, // [ffn_dim, model_dim]
}

impl LlamaBlock {
    pub fn new(
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        ffn_dim: usize,
        max_seq_len: usize,
        rope_base: f32,
        rms_eps: f32,
    ) -> Self {
        LlamaBlock {
            attn_norm: RMSNorm::new(model_dim, rms_eps),
            ffn_norm: RMSNorm::new(model_dim, rms_eps),
            attention: LlamaAttention::new(
                model_dim,
                num_q_heads,
                num_kv_heads,
                head_dim,
                max_seq_len,
                rope_base,
            ),
            gate_proj: Tensor::zeros(&[model_dim, ffn_dim], false),
            up_proj: Tensor::zeros(&[model_dim, ffn_dim], false),
            down_proj: Tensor::zeros(&[ffn_dim, model_dim], false),
        }
    }

    /// h: [seq_len, model_dim] → [seq_len, model_dim]
    pub fn forward(&self, h: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        // Attention block with residual
        let normed = self.attn_norm.forward(h);
        let attn_out = self.attention.forward(&normed, kv_cache);
        let h = h.add(&attn_out);

        // SwiGLU FFN with residual
        let normed = self.ffn_norm.forward(&h);
        let gate = normed.matmul(&self.gate_proj); // [seq_len, ffn_dim]
        let up = normed.matmul(&self.up_proj);     // [seq_len, ffn_dim]

        // SiLU(gate) * up
        let gate_data = gate.data();
        let up_data = up.data();
        let ffn_act: Vec<f32> = gate_data
            .iter()
            .zip(up_data.iter())
            .map(|(&g, &u)| {
                let silu = g / (1.0 + (-g).exp()); // silu(x) = x * sigmoid(x)
                silu * u
            })
            .collect();
        let ffn_hidden = Tensor::new(ffn_act, gate.shape(), false);

        let ffn_out = ffn_hidden.matmul(&self.down_proj); // [seq_len, model_dim]
        h.add(&ffn_out)
    }
}
