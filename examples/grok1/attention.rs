use peregrine::attention::{
    gqa_attention_cpu, AttentionMask, PostScoreTransform,
};
use peregrine::nn::RoPE;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

// Re-export StandardKVCache as KVCache for backward compat within this example
pub use peregrine::attention::StandardKVCache as KVCache;

/// Grouped Query Attention with RoPE, logit capping, and causal masking.
/// Delegates core attention math to peregrine::attention::gqa_attention_cpu.
pub struct GroupedQueryAttention {
    pub q_proj: Tensor, // [model_dim, num_q_heads * head_dim]
    pub k_proj: Tensor, // [model_dim, num_kv_heads * head_dim]
    pub v_proj: Tensor, // [model_dim, num_kv_heads * head_dim]
    pub o_proj: Tensor, // [num_q_heads * head_dim, model_dim]
    pub rope: RoPE,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attn_output_mult: f32,
    logit_cap: f32,
}

impl GroupedQueryAttention {
    pub fn new(
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        attn_output_mult: f32,
        logit_cap: f32,
        max_seq_len: usize,
        rope_base: f32,
    ) -> Self {
        GroupedQueryAttention {
            q_proj: Tensor::zeros(&[model_dim, num_q_heads * head_dim], false),
            k_proj: Tensor::zeros(&[model_dim, num_kv_heads * head_dim], false),
            v_proj: Tensor::zeros(&[model_dim, num_kv_heads * head_dim], false),
            o_proj: Tensor::zeros(&[num_q_heads * head_dim, model_dim], false),
            rope: RoPE::new(head_dim, max_seq_len, rope_base),
            num_q_heads,
            num_kv_heads,
            head_dim,
            attn_output_mult,
            logit_cap,
        }
    }

    /// x: [seq_len, model_dim], returns [seq_len, model_dim]
    pub fn forward(&self, x: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];
        let hd = self.head_dim;
        let nqh = self.num_q_heads;
        let nkvh = self.num_kv_heads;

        // Project Q, K, V
        let q_all = x.matmul(&self.q_proj); // [seq_len, nqh * hd]
        let k_all = x.matmul(&self.k_proj); // [seq_len, nkvh * hd]
        let v_all = x.matmul(&self.v_proj); // [seq_len, nkvh * hd]

        let q_data = q_all.data();
        let k_data = k_all.data();
        let v_data = v_all.data();

        let offset = kv_cache.len;

        // Apply RoPE per head: reshape to [seq_len, hd] per head, apply, collect
        let mut q_rope = vec![0.0f32; nqh * seq_len * hd];
        let mut k_rope = vec![0.0f32; nkvh * seq_len * hd];

        // Q heads
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

        // K heads
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

        // Rearrange V into [nkvh, seq_len, hd]
        let mut v_arranged = vec![0.0f32; nkvh * seq_len * hd];
        for t in 0..seq_len {
            for h in 0..nkvh {
                let src = t * nkvh * hd + h * hd;
                let dst = h * seq_len * hd + t * hd;
                v_arranged[dst..dst + hd].copy_from_slice(&v_data[src..src + hd]);
            }
        }

        // Append to KV cache
        kv_cache.append(&k_rope, &v_arranged, seq_len);

        // Delegate attention to core GQA
        let mut output = vec![0.0f32; seq_len * nqh * hd];
        let mask = AttentionMask::Causal { offset };
        let transform = PostScoreTransform::LogitCap { cap: self.logit_cap };

        gqa_attention_cpu(
            &q_rope,
            kv_cache,
            kv_cache,
            nqh,
            nkvh,
            seq_len,
            hd,
            self.attn_output_mult,
            &mask,
            &transform,
            &mut output,
        );

        // output is [seq_len, nqh * hd], project to [seq_len, model_dim]
        let attn_out = Tensor::new(output, vec![seq_len, nqh * hd], false);
        attn_out.matmul(&self.o_proj)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        if let Some((shape, data)) = params.get(&format!("{}.q_proj", prefix)) {
            self.q_proj = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.k_proj", prefix)) {
            self.k_proj = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.v_proj", prefix)) {
            self.v_proj = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.o_proj", prefix)) {
            self.o_proj = Tensor::new(data.clone(), shape.clone(), false);
        }
    }
}
