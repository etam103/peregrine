use peregrine::tensor::Tensor;
use std::collections::HashMap;

// Re-export StandardKVCache as KVCache for backward compat within this example
pub use peregrine::attention::StandardKVCache as KVCache;

/// Precompute YaRN-extended RoPE frequencies with half-split rotation.
/// Returns cos/sin tables of shape [max_seq_len, half_dim].
pub fn precompute_yarn_freqs(
    head_dim: usize,
    max_seq_len: usize,
    rope_theta: f32,
    rope_factor: f32,
    initial_ctx: usize,
    beta_fast: usize,
    beta_slow: usize,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;

    // Base frequencies
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / head_dim as f32))
        .collect();

    // YaRN NTK-by-parts interpolation
    let scaled_ctx = (initial_ctx as f32) * rope_factor;
    if scaled_ctx > initial_ctx as f32 {
        let find_correction_dim = |num_rotations: f32| -> f32 {
            head_dim as f32
                * (initial_ctx as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * rope_theta.ln())
        };
        let low = find_correction_dim(beta_fast as f32).floor().max(0.0) as usize;
        let high = find_correction_dim(beta_slow as f32)
            .ceil()
            .min(half as f32 - 1.0) as usize;

        // Linear ramp
        let mut smooth = vec![0.0f32; half];
        for i in 0..half {
            let ramp = if low == high {
                1.0
            } else {
                ((i as f32 - low as f32) / (high as f32 - low as f32)).clamp(0.0, 1.0)
            };
            smooth[i] = 1.0 - ramp;
        }

        for i in 0..half {
            freqs[i] = freqs[i] / rope_factor * (1.0 - smooth[i]) + freqs[i] * smooth[i];
        }
    }

    // Concentration scaling
    let concentration = 0.1 * (rope_factor).ln() + 1.0;

    // Build cos/sin tables: [max_seq_len, half]
    let mut cos_table = vec![0.0f32; max_seq_len * half];
    let mut sin_table = vec![0.0f32; max_seq_len * half];
    for t in 0..max_seq_len {
        for i in 0..half {
            let angle = t as f32 * freqs[i];
            cos_table[t * half + i] = angle.cos() * concentration;
            sin_table[t * half + i] = angle.sin() * concentration;
        }
    }

    (cos_table, sin_table)
}

/// Apply half-split rotary embedding.
/// x: [seq_len, head_dim], positions start at `offset`.
/// Half-split: x1 = x[0..d/2], x2 = x[d/2..d]
///   o1 = x1*cos - x2*sin, o2 = x2*cos + x1*sin
fn apply_rotary_emb_half_split(
    x: &[f32],
    seq_len: usize,
    head_dim: usize,
    offset: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut out = vec![0.0f32; seq_len * head_dim];
    for t in 0..seq_len {
        let pos = offset + t;
        for i in 0..half {
            let x1 = x[t * head_dim + i];
            let x2 = x[t * head_dim + i + half];
            let cos_val = cos_table[pos * half + i];
            let sin_val = sin_table[pos * half + i];
            out[t * head_dim + i] = x1 * cos_val - x2 * sin_val;
            out[t * head_dim + i + half] = x2 * cos_val + x1 * sin_val;
        }
    }
    out
}

/// GQA attention with YaRN RoPE, sliding window, and learned attention sinks.
/// Delegates core attention math to peregrine::attention::gqa_attention_cpu,
/// with sink logits handled as a wrapper.
pub struct AttentionBlock {
    pub qkv_weight: Tensor, // [model_dim, (num_q_heads + 2*num_kv_heads) * head_dim]
    pub qkv_bias: Tensor,   // [(num_q_heads + 2*num_kv_heads) * head_dim]
    pub o_weight: Tensor,    // [num_q_heads * head_dim, model_dim]
    pub o_bias: Tensor,      // [model_dim]
    pub sinks: Vec<f32>,     // [num_q_heads] — learned per-head sink logit

    cos_table: Vec<f32>,
    sin_table: Vec<f32>,

    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_sliding_window: bool,
    sliding_window: usize,
}

impl AttentionBlock {
    pub fn new(
        model_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f32,
        rope_factor: f32,
        initial_ctx: usize,
        beta_fast: usize,
        beta_slow: usize,
        use_sliding_window: bool,
        sliding_window: usize,
    ) -> Self {
        let total_heads = num_q_heads + 2 * num_kv_heads;
        let qkv_dim = total_heads * head_dim;

        let (cos_table, sin_table) = precompute_yarn_freqs(
            head_dim,
            max_seq_len,
            rope_theta,
            rope_factor,
            initial_ctx,
            beta_fast,
            beta_slow,
        );

        AttentionBlock {
            qkv_weight: Tensor::zeros(&[model_dim, qkv_dim], false),
            qkv_bias: Tensor::zeros(&[qkv_dim], false),
            o_weight: Tensor::zeros(&[num_q_heads * head_dim, model_dim], false),
            o_bias: Tensor::zeros(&[model_dim], false),
            sinks: vec![0.0; num_q_heads],
            cos_table,
            sin_table,
            num_q_heads,
            num_kv_heads,
            head_dim,
            use_sliding_window,
            sliding_window,
        }
    }

    /// x: [seq_len, model_dim], returns [seq_len, model_dim]
    pub fn forward(&self, x: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];
        let model_dim = shape[1];
        let hd = self.head_dim;
        let nqh = self.num_q_heads;
        let nkvh = self.num_kv_heads;
        let heads_per_group = nqh / nkvh;

        // QKV projection: x @ qkv_weight + bias
        let qkv = x.matmul(&self.qkv_weight);
        let qkv_data = qkv.data();
        let bias_data = self.qkv_bias.data();

        let q_dim = nqh * hd;
        let k_dim = nkvh * hd;
        let total_qkv_dim = (nqh + 2 * nkvh) * hd;

        // Split into Q, K, V and add bias
        let offset = kv_cache.len;

        // Apply RoPE per Q head
        let mut q_rope = vec![0.0f32; nqh * seq_len * hd];
        for h in 0..nqh {
            let mut head_data = Vec::with_capacity(seq_len * hd);
            for t in 0..seq_len {
                let base = t * total_qkv_dim;
                let start = base + h * hd;
                for d in 0..hd {
                    head_data.push(qkv_data[start + d] + bias_data[h * hd + d]);
                }
            }
            let rotated = apply_rotary_emb_half_split(
                &head_data,
                seq_len,
                hd,
                offset,
                &self.cos_table,
                &self.sin_table,
            );
            for t in 0..seq_len {
                let dst = h * seq_len * hd + t * hd;
                let src = t * hd;
                q_rope[dst..dst + hd].copy_from_slice(&rotated[src..src + hd]);
            }
        }

        // Apply RoPE per K head
        let mut k_rope = vec![0.0f32; nkvh * seq_len * hd];
        for h in 0..nkvh {
            let mut head_data = Vec::with_capacity(seq_len * hd);
            for t in 0..seq_len {
                let base = t * total_qkv_dim + q_dim;
                let start = base + h * hd;
                for d in 0..hd {
                    head_data.push(qkv_data[start + d] + bias_data[q_dim + h * hd + d]);
                }
            }
            let rotated = apply_rotary_emb_half_split(
                &head_data,
                seq_len,
                hd,
                offset,
                &self.cos_table,
                &self.sin_table,
            );
            for t in 0..seq_len {
                let dst = h * seq_len * hd + t * hd;
                let src = t * hd;
                k_rope[dst..dst + hd].copy_from_slice(&rotated[src..src + hd]);
            }
        }

        // Rearrange V into [nkvh, seq_len, hd] with bias
        let mut v_arranged = vec![0.0f32; nkvh * seq_len * hd];
        for t in 0..seq_len {
            for h in 0..nkvh {
                let base = t * total_qkv_dim + q_dim + k_dim;
                let src = base + h * hd;
                let dst = h * seq_len * hd + t * hd;
                for d in 0..hd {
                    v_arranged[dst + d] =
                        qkv_data[src + d] + bias_data[q_dim + k_dim + h * hd + d];
                }
            }
        }

        // Append to KV cache
        kv_cache.append(&k_rope, &v_arranged, seq_len);
        let total_len = kv_cache.len;

        // Attention — sink tokens require custom handling since they're virtual
        // (not in KV cache). Use core GQA for the KV-cache-based attention,
        // then add sink score externally.
        let scale = 1.0 / (hd as f32).sqrt();

        let mut output = vec![0.0f32; seq_len * nqh * hd];

        // Sink logits are virtual (not in KV cache), so we compute
        // attention manually with sinks inline rather than using core GQA.
        for qh in 0..nqh {
            let kvh = qh / heads_per_group;
            let sink_val = self.sinks[qh];

            for qt in 0..seq_len {
                let q_off = qh * seq_len * hd + qt * hd;
                let q_slice = &q_rope[q_off..q_off + hd];
                let query_pos = offset + qt;

                // Compute scores against all cached K + 1 sink element
                let mut scores = Vec::with_capacity(total_len + 1);
                let k_base = kvh * total_len * hd;
                for kt in 0..total_len {
                    // Causal mask
                    if kt > query_pos {
                        scores.push(f32::NEG_INFINITY);
                    } else if self.use_sliding_window && query_pos - kt > self.sliding_window {
                        scores.push(f32::NEG_INFINITY);
                    } else {
                        let k_off = k_base + kt * hd;
                        let mut dot = 0.0f32;
                        for d in 0..hd {
                            dot += q_slice[d] * kv_cache.k[k_off + d];
                        }
                        scores.push(dot * scale);
                    }
                }
                // Append sink score
                scores.push(sink_val);

                // Softmax over [total_len + 1] elements
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

                // Weighted sum of V using first total_len weights (discard sink weight)
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

        // Output projection: [seq_len, nqh * hd] @ o_weight + o_bias
        let attn_out = Tensor::new(output, vec![seq_len, nqh * hd], false);
        let projected = attn_out.matmul(&self.o_weight);
        let proj_data = projected.data();
        let o_bias_data = self.o_bias.data();

        let mut result = vec![0.0f32; seq_len * model_dim];
        for i in 0..seq_len * model_dim {
            result[i] = proj_data[i] + o_bias_data[i % model_dim];
        }

        Tensor::new(result, vec![seq_len, model_dim], false)
    }

    pub fn load_weights(
        &mut self,
        params: &HashMap<String, (Vec<usize>, Vec<f32>)>,
        prefix: &str,
    ) {
        if let Some((shape, data)) = params.get(&format!("{}.qkv_weight", prefix)) {
            self.qkv_weight = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.qkv_bias", prefix)) {
            self.qkv_bias = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.o_weight", prefix)) {
            self.o_weight = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((shape, data)) = params.get(&format!("{}.o_bias", prefix)) {
            self.o_bias = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((_s, data)) = params.get(&format!("{}.sinks", prefix)) {
            self.sinks = data.clone();
        }
    }
}
