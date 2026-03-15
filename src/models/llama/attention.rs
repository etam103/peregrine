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
    qkv_fused: Option<Tensor>,
    q_dim: usize,
    kv_dim: usize,
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
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        LlamaAttention {
            q_proj: Tensor::zeros(&[model_dim, q_dim], false),
            k_proj: Tensor::zeros(&[model_dim, kv_dim], false),
            v_proj: Tensor::zeros(&[model_dim, kv_dim], false),
            o_proj: Tensor::zeros(&[q_dim, model_dim], false),
            rope: RoPE::new(head_dim, max_seq_len, rope_base),
            num_q_heads,
            num_kv_heads,
            head_dim,
            qkv_fused: None,
            q_dim,
            kv_dim,
        }
    }

    /// Fuse Q, K, V projections into a single weight matrix.
    pub fn fuse_qkv(&mut self) {
        let q_data = self.q_proj.data();
        let k_data = self.k_proj.data();
        let v_data = self.v_proj.data();
        let model_dim = self.q_proj.shape()[0];
        let q_dim = self.q_dim;
        let kv_dim = self.kv_dim;
        let total = q_dim + 2 * kv_dim;

        let mut fused = vec![0.0f32; model_dim * total];
        for r in 0..model_dim {
            let dst_off = r * total;
            let q_off = r * q_dim;
            let k_off = r * kv_dim;
            let v_off = r * kv_dim;
            fused[dst_off..dst_off + q_dim].copy_from_slice(&q_data[q_off..q_off + q_dim]);
            fused[dst_off + q_dim..dst_off + q_dim + kv_dim].copy_from_slice(&k_data[k_off..k_off + kv_dim]);
            fused[dst_off + q_dim + kv_dim..dst_off + total].copy_from_slice(&v_data[v_off..v_off + kv_dim]);
        }
        self.qkv_fused = Some(Tensor::new(fused, vec![model_dim, total], false));
    }

    pub fn forward(&self, x: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];
        let hd = self.head_dim;
        let nqh = self.num_q_heads;
        let nkvh = self.num_kv_heads;
        let q_dim = self.q_dim;
        let kv_dim = self.kv_dim;

        let (q_data, k_data, v_data);
        if let Some(ref fused) = self.qkv_fused {
            let qkv = x.matmul(fused);
            let qkv_data = qkv.data();
            let total = q_dim + 2 * kv_dim;

            let mut q = vec![0.0f32; seq_len * q_dim];
            let mut k = vec![0.0f32; seq_len * kv_dim];
            let mut v = vec![0.0f32; seq_len * kv_dim];
            for s in 0..seq_len {
                let src = s * total;
                q[s * q_dim..(s + 1) * q_dim].copy_from_slice(&qkv_data[src..src + q_dim]);
                k[s * kv_dim..(s + 1) * kv_dim].copy_from_slice(&qkv_data[src + q_dim..src + q_dim + kv_dim]);
                v[s * kv_dim..(s + 1) * kv_dim].copy_from_slice(&qkv_data[src + q_dim + kv_dim..src + total]);
            }
            q_data = q;
            k_data = k;
            v_data = v;
        } else {
            let q_all = x.matmul(&self.q_proj);
            let k_all = x.matmul(&self.k_proj);
            let v_all = x.matmul(&self.v_proj);
            q_data = q_all.data().to_vec();
            k_data = k_all.data().to_vec();
            v_data = v_all.data().to_vec();
        }

        let offset = kv_cache.len;

        let mut q_rope = vec![0.0f32; nqh * seq_len * hd];
        let mut k_rope = vec![0.0f32; nkvh * seq_len * hd];

        if seq_len == 1 {
            let pos = offset;
            for h in 0..nqh {
                let src_start = h * hd;
                let dst_start = h * hd;
                self.rope.apply_one(
                    &q_data[src_start..src_start + hd],
                    &mut q_rope[dst_start..dst_start + hd],
                    pos,
                );
            }
            for h in 0..nkvh {
                let src_start = h * hd;
                let dst_start = h * hd;
                self.rope.apply_one(
                    &k_data[src_start..src_start + hd],
                    &mut k_rope[dst_start..dst_start + hd],
                    pos,
                );
            }
        } else {
            for h in 0..nqh {
                for t in 0..seq_len {
                    let src_start = t * nqh * hd + h * hd;
                    let dst_start = h * seq_len * hd + t * hd;
                    self.rope.apply_one(
                        &q_data[src_start..src_start + hd],
                        &mut q_rope[dst_start..dst_start + hd],
                        offset + t,
                    );
                }
            }
            for h in 0..nkvh {
                for t in 0..seq_len {
                    let src_start = t * nkvh * hd + h * hd;
                    let dst_start = h * seq_len * hd + t * hd;
                    self.rope.apply_one(
                        &k_data[src_start..src_start + hd],
                        &mut k_rope[dst_start..dst_start + hd],
                        offset + t,
                    );
                }
            }
        }

        let mut v_arranged = vec![0.0f32; nkvh * seq_len * hd];
        if seq_len == 1 {
            v_arranged.copy_from_slice(&v_data);
        } else {
            for t in 0..seq_len {
                for h in 0..nkvh {
                    let src = t * nkvh * hd + h * hd;
                    let dst = h * seq_len * hd + t * hd;
                    v_arranged[dst..dst + hd].copy_from_slice(&v_data[src..src + hd]);
                }
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
