use crate::attention::{
    gqa_attention_cpu, AttentionMask, PostScoreTransform,
};
use crate::nn::RoPE;
use crate::tensor::Tensor;

pub use crate::attention::StandardKVCache as KVCache;

/// Grouped Query Attention with RoPE for Llama.
pub struct LlamaAttention {
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub rope: RoPE,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LlamaAttention {
    pub fn new(
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_base: f32,
    ) -> Self {
        LlamaAttention {
            q_proj: Tensor::zeros(&[model_dim, num_q_heads * head_dim], false),
            k_proj: Tensor::zeros(&[model_dim, num_kv_heads * head_dim], false),
            v_proj: Tensor::zeros(&[model_dim, num_kv_heads * head_dim], false),
            o_proj: Tensor::zeros(&[num_q_heads * head_dim, model_dim], false),
            rope: RoPE::new(head_dim, max_seq_len, rope_base),
            num_q_heads,
            num_kv_heads,
            head_dim,
        }
    }

    pub fn forward(&self, x: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];
        let hd = self.head_dim;
        let nqh = self.num_q_heads;
        let nkvh = self.num_kv_heads;

        let q_all = x.matmul(&self.q_proj);
        let k_all = x.matmul(&self.k_proj);
        let v_all = x.matmul(&self.v_proj);

        let q_data = q_all.data();
        let k_data = k_all.data();
        let v_data = v_all.data();

        let offset = kv_cache.len;

        let mut q_rope = vec![0.0f32; nqh * seq_len * hd];
        let mut k_rope = vec![0.0f32; nkvh * seq_len * hd];

        for h in 0..nqh {
            let mut head_data = Vec::with_capacity(seq_len * hd);
            for t in 0..seq_len {
                let start = t * nqh * hd + h * hd;
                head_data.extend_from_slice(&q_data[start..start + hd]);
            }
            let head_tensor = Tensor::new(head_data, vec![seq_len, hd], false);
            let rotated = self.rope.apply(&head_tensor, offset);
            let rot_data = rotated.data();
            for t in 0..seq_len {
                let dst = h * seq_len * hd + t * hd;
                let src = t * hd;
                q_rope[dst..dst + hd].copy_from_slice(&rot_data[src..src + hd]);
            }
        }

        for h in 0..nkvh {
            let mut head_data = Vec::with_capacity(seq_len * hd);
            for t in 0..seq_len {
                let start = t * nkvh * hd + h * hd;
                head_data.extend_from_slice(&k_data[start..start + hd]);
            }
            let head_tensor = Tensor::new(head_data, vec![seq_len, hd], false);
            let rotated = self.rope.apply(&head_tensor, offset);
            let rot_data = rotated.data();
            for t in 0..seq_len {
                let dst = h * seq_len * hd + t * hd;
                let src = t * hd;
                k_rope[dst..dst + hd].copy_from_slice(&rot_data[src..src + hd]);
            }
        }

        let mut v_arranged = vec![0.0f32; nkvh * seq_len * hd];
        for t in 0..seq_len {
            for h in 0..nkvh {
                let src = t * nkvh * hd + h * hd;
                let dst = h * seq_len * hd + t * hd;
                v_arranged[dst..dst + hd].copy_from_slice(&v_data[src..src + hd]);
            }
        }

        kv_cache.append(&k_rope, &v_arranged, seq_len);

        let mut output = vec![0.0f32; seq_len * nqh * hd];
        let mask = AttentionMask::Causal { offset };
        let scale = 1.0 / (hd as f32).sqrt();
        let transform = PostScoreTransform::None;

        gqa_attention_cpu(
            &q_rope,
            kv_cache,
            kv_cache,
            nqh,
            nkvh,
            seq_len,
            hd,
            scale,
            &mask,
            &transform,
            &mut output,
        );

        let attn_out = Tensor::new(output, vec![seq_len, nqh * hd], false);
        attn_out.matmul(&self.o_proj)
    }
}
