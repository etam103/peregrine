use peregrine::nn::RoPE;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

/// KV cache for autoregressive generation.
pub struct KVCache {
    /// Stored K values: [num_kv_heads, cached_len, head_dim]
    pub k: Vec<f32>,
    /// Stored V values: [num_kv_heads, cached_len, head_dim]
    pub v: Vec<f32>,
    pub len: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl KVCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        KVCache {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            num_kv_heads,
            head_dim,
        }
    }

    /// Append new K, V of shape [num_kv_heads, seq_len, head_dim].
    pub fn append(&mut self, new_k: &[f32], new_v: &[f32], seq_len: usize) {
        if self.len == 0 {
            self.k = new_k.to_vec();
            self.v = new_v.to_vec();
            self.len = seq_len;
        } else {
            // Insert new entries into each head's slice
            let old_len = self.len;
            let new_len = old_len + seq_len;
            let hd = self.head_dim;

            let mut new_k_buf = Vec::with_capacity(self.num_kv_heads * new_len * hd);
            let mut new_v_buf = Vec::with_capacity(self.num_kv_heads * new_len * hd);

            for h in 0..self.num_kv_heads {
                let old_offset = h * old_len * hd;
                let append_offset = h * seq_len * hd;
                new_k_buf.extend_from_slice(&self.k[old_offset..old_offset + old_len * hd]);
                new_k_buf.extend_from_slice(&new_k[append_offset..append_offset + seq_len * hd]);

                new_v_buf.extend_from_slice(&self.v[old_offset..old_offset + old_len * hd]);
                new_v_buf.extend_from_slice(&new_v[append_offset..append_offset + seq_len * hd]);
            }

            self.k = new_k_buf;
            self.v = new_v_buf;
            self.len = new_len;
        }
    }
}

/// Grouped Query Attention with RoPE, logit capping, and causal masking.
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
        let heads_per_group = nqh / nkvh;

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
        let total_len = kv_cache.len;

        // Compute attention: Q @ K^T with logit capping and causal mask
        let cap = self.logit_cap;
        let scale = self.attn_output_mult;

        let mut output = vec![0.0f32; seq_len * nqh * hd];

        for qh in 0..nqh {
            let kvh = qh / heads_per_group;

            for qt in 0..seq_len {
                let q_off = qh * seq_len * hd + qt * hd;
                let q_slice = &q_rope[q_off..q_off + hd];

                // Compute scores against all cached K
                let mut scores = Vec::with_capacity(total_len);
                let k_base = kvh * total_len * hd;
                for kt in 0..total_len {
                    // Causal mask: only attend to positions <= current query position
                    let query_pos = offset + qt;
                    if kt > query_pos {
                        scores.push(f32::NEG_INFINITY);
                    } else {
                        let k_off = k_base + kt * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q_slice[d] * kv_cache.k[k_off + d];
                        }
                        // Scale and cap: cap * tanh(score * scale / cap)
                        let scaled = dot * scale;
                        let capped = cap * (scaled / cap).tanh();
                        scores.push(capped);
                    }
                }

                // Softmax
                let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_score).exp();
                    exp_sum += *s;
                }
                if exp_sum > 0.0 {
                    for s in &mut scores {
                        *s /= exp_sum;
                    }
                }

                // Weighted sum of V
                let v_base = kvh * total_len * hd;
                let out_off = qt * nqh * hd + qh * hd;
                for kt in 0..total_len {
                    let w = scores[kt];
                    if w > 0.0 {
                        let v_off = v_base + kt * hd;
                        for d in 0..hd {
                            output[out_off + d] += w * kv_cache.v[v_off + d];
                        }
                    }
                }
            }
        }

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
