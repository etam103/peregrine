use peregrine::tensor::Tensor;
use std::collections::HashMap;

/// Compressed KV cache for MLA (absorb mode).
/// Stores only the compressed latent (kv_lora_rank) + RoPE key (qk_rope_head_dim) per token.
pub struct KVCache {
    /// Compressed KV latent: [cached_len, kv_lora_rank]
    pub kv: Vec<f32>,
    /// RoPE key component: [cached_len, qk_rope_head_dim]
    pub pe: Vec<f32>,
    pub len: usize,
    kv_lora_rank: usize,
    qk_rope_head_dim: usize,
}

impl KVCache {
    pub fn new(kv_lora_rank: usize, qk_rope_head_dim: usize) -> Self {
        KVCache {
            kv: Vec::new(),
            pe: Vec::new(),
            len: 0,
            kv_lora_rank,
            qk_rope_head_dim,
        }
    }

    pub fn append(&mut self, new_kv: &[f32], new_pe: &[f32], seq_len: usize) {
        self.kv.extend_from_slice(&new_kv[..seq_len * self.kv_lora_rank]);
        self.pe.extend_from_slice(&new_pe[..seq_len * self.qk_rope_head_dim]);
        self.len += seq_len;
    }
}

/// Precompute YaRN-extended RoPE frequencies.
/// Returns cos/sin tables of shape [max_seq_len, qk_rope_head_dim].
pub fn precompute_yarn_freqs(
    qk_rope_head_dim: usize,
    max_seq_len: usize,
    rope_theta: f32,
    rope_factor: f32,
    original_seq_len: usize,
    beta_fast: usize,
    beta_slow: usize,
) -> (Vec<f32>, Vec<f32>) {
    let dim = qk_rope_head_dim;
    let half = dim / 2;

    // Base frequencies
    let mut freqs: Vec<f32> = (0..half)
        .map(|i| 1.0 / rope_theta.powf(2.0 * i as f32 / dim as f32))
        .collect();

    // YaRN correction if sequence length exceeds original
    if max_seq_len > original_seq_len {
        let find_correction_dim = |num_rotations: f32| -> f32 {
            dim as f32
                * (original_seq_len as f32 / (num_rotations * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * (rope_theta).ln())
        };
        let low = find_correction_dim(beta_fast as f32).floor().max(0.0) as usize;
        let high = find_correction_dim(beta_slow as f32).ceil().min(half as f32 - 1.0) as usize;

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

    // Build cos/sin tables: [max_seq_len, dim] where each row has [cos(f0*t), cos(f1*t), ..., sin(f0*t), sin(f1*t), ...]
    let mut cos_table = vec![0.0f32; max_seq_len * dim];
    let mut sin_table = vec![0.0f32; max_seq_len * dim];
    for t in 0..max_seq_len {
        for i in 0..half {
            let angle = t as f32 * freqs[i];
            let c = angle.cos();
            let s = angle.sin();
            // Complex rotation: pairs at [2*i, 2*i+1]
            cos_table[t * dim + 2 * i] = c;
            cos_table[t * dim + 2 * i + 1] = c;
            sin_table[t * dim + 2 * i] = s;
            sin_table[t * dim + 2 * i + 1] = s;
        }
    }

    (cos_table, sin_table)
}

/// Apply rotary embedding using complex multiplication.
/// x: [seq_len, rope_dim], positions start at `offset`.
/// Uses interleaved pairs: (x[2i], x[2i+1]) treated as complex number.
fn apply_rotary_emb(
    x: &[f32],
    seq_len: usize,
    rope_dim: usize,
    offset: usize,
    cos_table: &[f32],
    sin_table: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0f32; seq_len * rope_dim];
    let half = rope_dim / 2;
    for t in 0..seq_len {
        let pos = offset + t;
        for i in 0..half {
            let x_re = x[t * rope_dim + 2 * i];
            let x_im = x[t * rope_dim + 2 * i + 1];
            let cos_val = cos_table[pos * rope_dim + 2 * i];
            let sin_val = sin_table[pos * rope_dim + 2 * i];
            out[t * rope_dim + 2 * i] = x_re * cos_val - x_im * sin_val;
            out[t * rope_dim + 2 * i + 1] = x_re * sin_val + x_im * cos_val;
        }
    }
    out
}

/// RMSNorm applied inline to a slice (no learnable weight, just normalization).
fn rms_norm_inline(data: &[f32], dim: usize, weight: &[f32]) -> Vec<f32> {
    let n = data.len() / dim;
    let mut out = vec![0.0f32; data.len()];
    for i in 0..n {
        let row = &data[i * dim..(i + 1) * dim];
        let mut sum_sq = 0.0f32;
        for &v in row {
            sum_sq += v * v;
        }
        let rms = (sum_sq / dim as f32 + 1e-6).sqrt();
        for j in 0..dim {
            out[i * dim + j] = row[j] / rms * weight[j];
        }
    }
    out
}

/// Multi-head Latent Attention (MLA) with compressed KV cache.
/// Uses "absorb" mode: folds W_kv_b into query-side computation.
pub struct MLA {
    // Query path (with LoRA when q_lora_rank > 0)
    pub wq: Option<Tensor>,       // [dim, n_heads * qk_head_dim] (no LoRA)
    pub wq_a: Option<Tensor>,     // [dim, q_lora_rank]
    pub q_norm_weight: Vec<f32>,   // [q_lora_rank]
    pub wq_b: Option<Tensor>,     // [q_lora_rank, n_heads * qk_head_dim]

    // KV path
    pub wkv_a: Tensor,            // [dim, kv_lora_rank + qk_rope_head_dim]
    pub kv_norm_weight: Vec<f32>,  // [kv_lora_rank]
    pub wkv_b: Tensor,            // [kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)]

    // Output
    pub wo: Tensor,               // [n_heads * v_head_dim, dim]

    // RoPE tables
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,

    // Dimensions
    n_heads: usize,
    q_lora_rank: usize,
    kv_lora_rank: usize,
    qk_nope_head_dim: usize,
    qk_rope_head_dim: usize,
    v_head_dim: usize,
    softmax_scale: f32,
}

impl MLA {
    pub fn new(
        dim: usize,
        n_heads: usize,
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
        let qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;

        let (cos_table, sin_table) = precompute_yarn_freqs(
            qk_rope_head_dim,
            max_seq_len,
            rope_theta,
            rope_factor,
            original_seq_len,
            beta_fast,
            beta_slow,
        );

        let mut softmax_scale = (qk_head_dim as f32).powf(-0.5);
        if max_seq_len > original_seq_len {
            let ms = 0.1 * mscale * (rope_factor).ln() + 1.0;
            softmax_scale *= ms * ms;
        }

        let (wq, wq_a, wq_b) = if q_lora_rank == 0 {
            (
                Some(Tensor::zeros(&[dim, n_heads * qk_head_dim], false)),
                None,
                None,
            )
        } else {
            (
                None,
                Some(Tensor::zeros(&[dim, q_lora_rank], false)),
                Some(Tensor::zeros(&[q_lora_rank, n_heads * qk_head_dim], false)),
            )
        };

        MLA {
            wq,
            wq_a,
            q_norm_weight: vec![1.0; q_lora_rank],
            wq_b,
            wkv_a: Tensor::zeros(&[dim, kv_lora_rank + qk_rope_head_dim], false),
            kv_norm_weight: vec![1.0; kv_lora_rank],
            wkv_b: Tensor::zeros(
                &[kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)],
                false,
            ),
            wo: Tensor::zeros(&[n_heads * v_head_dim, dim], false),
            cos_table,
            sin_table,
            n_heads,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
            softmax_scale,
        }
    }

    /// x: [seq_len, dim], returns [seq_len, dim]
    pub fn forward(&self, x: &Tensor, kv_cache: &mut KVCache) -> Tensor {
        let shape = x.shape();
        let seq_len = shape[0];
        let offset = kv_cache.len;
        let qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim;

        // === Query path ===
        let q_flat = if self.q_lora_rank == 0 {
            // Direct: x @ wq -> [seq_len, n_heads * qk_head_dim]
            let q = x.matmul(self.wq.as_ref().unwrap());
            q.data().to_vec()
        } else {
            // LoRA: x @ wq_a -> norm -> @ wq_b
            let qa = x.matmul(self.wq_a.as_ref().unwrap()); // [seq_len, q_lora_rank]
            let qa_norm = rms_norm_inline(&qa.data(), self.q_lora_rank, &self.q_norm_weight);
            let qa_t = Tensor::new(qa_norm, vec![seq_len, self.q_lora_rank], false);
            let q = qa_t.matmul(self.wq_b.as_ref().unwrap());
            q.data().to_vec()
        };

        // Split Q into q_nope [seq_len, n_heads, qk_nope_head_dim] and q_pe [seq_len, n_heads, qk_rope_head_dim]
        // Then apply RoPE to q_pe
        let mut q_nope = vec![0.0f32; seq_len * self.n_heads * self.qk_nope_head_dim];
        let mut q_pe_flat = vec![0.0f32; seq_len * self.n_heads * self.qk_rope_head_dim];

        for t in 0..seq_len {
            for h in 0..self.n_heads {
                let src = t * self.n_heads * qk_head_dim + h * qk_head_dim;
                let nope_dst = t * self.n_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim;
                let pe_dst = t * self.n_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim;
                q_nope[nope_dst..nope_dst + self.qk_nope_head_dim]
                    .copy_from_slice(&q_flat[src..src + self.qk_nope_head_dim]);
                q_pe_flat[pe_dst..pe_dst + self.qk_rope_head_dim]
                    .copy_from_slice(&q_flat[src + self.qk_nope_head_dim..src + qk_head_dim]);
            }
        }

        // Apply RoPE to each head's q_pe
        let mut q_pe_rotated = vec![0.0f32; seq_len * self.n_heads * self.qk_rope_head_dim];
        for h in 0..self.n_heads {
            let mut head_pe = vec![0.0f32; seq_len * self.qk_rope_head_dim];
            for t in 0..seq_len {
                let src = t * self.n_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim;
                let dst = t * self.qk_rope_head_dim;
                head_pe[dst..dst + self.qk_rope_head_dim]
                    .copy_from_slice(&q_pe_flat[src..src + self.qk_rope_head_dim]);
            }
            let rotated = apply_rotary_emb(
                &head_pe,
                seq_len,
                self.qk_rope_head_dim,
                offset,
                &self.cos_table,
                &self.sin_table,
            );
            for t in 0..seq_len {
                let src = t * self.qk_rope_head_dim;
                let dst = t * self.n_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim;
                q_pe_rotated[dst..dst + self.qk_rope_head_dim]
                    .copy_from_slice(&rotated[src..src + self.qk_rope_head_dim]);
            }
        }

        // === KV path ===
        // x @ wkv_a -> [seq_len, kv_lora_rank + qk_rope_head_dim]
        let kva = x.matmul(&self.wkv_a);
        let kva_data = kva.data();

        // Split into kv_compressed [seq_len, kv_lora_rank] and k_pe [seq_len, qk_rope_head_dim]
        let mut kv_compressed = vec![0.0f32; seq_len * self.kv_lora_rank];
        let mut k_pe_raw = vec![0.0f32; seq_len * self.qk_rope_head_dim];
        for t in 0..seq_len {
            let src = t * (self.kv_lora_rank + self.qk_rope_head_dim);
            kv_compressed[t * self.kv_lora_rank..(t + 1) * self.kv_lora_rank]
                .copy_from_slice(&kva_data[src..src + self.kv_lora_rank]);
            k_pe_raw[t * self.qk_rope_head_dim..(t + 1) * self.qk_rope_head_dim]
                .copy_from_slice(&kva_data[src + self.kv_lora_rank..src + self.kv_lora_rank + self.qk_rope_head_dim]);
        }

        // Apply RoPE to k_pe (shared across heads)
        let k_pe_rotated = apply_rotary_emb(
            &k_pe_raw,
            seq_len,
            self.qk_rope_head_dim,
            offset,
            &self.cos_table,
            &self.sin_table,
        );

        // RMSNorm on kv_compressed
        let kv_normed = rms_norm_inline(&kv_compressed, self.kv_lora_rank, &self.kv_norm_weight);

        // === Absorb mode: fold W_kv_b into query side ===
        // W_kv_b: [kv_lora_rank, n_heads * (qk_nope_head_dim + v_head_dim)]
        // Reshape as [n_heads, (qk_nope_head_dim + v_head_dim), kv_lora_rank]
        let wkv_b_data = self.wkv_b.data();
        let nope_v_dim = self.qk_nope_head_dim + self.v_head_dim;

        // q_nope_absorbed = q_nope @ W_kv_b[:, :nope]^T per head
        // -> [seq_len, n_heads, kv_lora_rank]
        let mut q_nope_absorbed = vec![0.0f32; seq_len * self.n_heads * self.kv_lora_rank];
        for h in 0..self.n_heads {
            // W_kv_b for this head's nope part: rows [kv_lora_rank], cols mapped from head h
            // wkv_b layout: [kv_lora_rank, n_heads * nope_v_dim]
            // For head h, nope cols: h * nope_v_dim .. h * nope_v_dim + qk_nope_head_dim
            for t in 0..seq_len {
                let q_off = t * self.n_heads * self.qk_nope_head_dim + h * self.qk_nope_head_dim;
                let out_off = t * self.n_heads * self.kv_lora_rank + h * self.kv_lora_rank;
                for c in 0..self.kv_lora_rank {
                    let mut dot = 0.0f32;
                    let w_row_base = c * self.n_heads * nope_v_dim + h * nope_v_dim;
                    for d in 0..self.qk_nope_head_dim {
                        dot += q_nope[q_off + d] * wkv_b_data[w_row_base + d];
                    }
                    q_nope_absorbed[out_off + c] = dot;
                }
            }
        }

        // Append to KV cache (kv_normed and k_pe_rotated)
        kv_cache.append(&kv_normed, &k_pe_rotated, seq_len);
        let total_len = kv_cache.len;

        // === Attention scores ===
        // scores = q_nope_absorbed @ kv_cache^T + q_pe @ pe_cache^T
        // Per head, per query token:
        //   score[t, h, kt] = sum_c(q_nope_absorbed[t,h,c] * kv_cache[kt,c])
        //                   + sum_r(q_pe[t,h,r] * pe_cache[kt,r])
        let mut output = vec![0.0f32; seq_len * self.n_heads * self.v_head_dim];

        for h in 0..self.n_heads {
            for qt in 0..seq_len {
                let query_pos = offset + qt;

                // Compute scores
                let mut scores = Vec::with_capacity(total_len);
                let qna_off = qt * self.n_heads * self.kv_lora_rank + h * self.kv_lora_rank;
                let qpe_off = qt * self.n_heads * self.qk_rope_head_dim + h * self.qk_rope_head_dim;

                for kt in 0..total_len {
                    if kt > query_pos {
                        scores.push(f32::NEG_INFINITY);
                    } else {
                        // Nope score: q_nope_absorbed @ kv_cache
                        let kv_off = kt * self.kv_lora_rank;
                        let mut dot = 0.0f32;
                        for c in 0..self.kv_lora_rank {
                            dot += q_nope_absorbed[qna_off + c] * kv_cache.kv[kv_off + c];
                        }
                        // PE score: q_pe @ pe_cache
                        let pe_off = kt * self.qk_rope_head_dim;
                        for r in 0..self.qk_rope_head_dim {
                            dot += q_pe_rotated[qpe_off + r] * kv_cache.pe[pe_off + r];
                        }
                        scores.push(dot * self.softmax_scale);
                    }
                }

                // Softmax
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                }
                if exp_sum > 0.0 {
                    for s in &mut scores {
                        *s /= exp_sum;
                    }
                }

                // Weighted sum: attn @ kv_cache -> [kv_lora_rank], then @ W_kv_b_v -> [v_head_dim]
                // First accumulate in compressed space
                let mut compressed_sum = vec![0.0f32; self.kv_lora_rank];
                for kt in 0..total_len {
                    let w = scores[kt];
                    if w > 0.0 {
                        let kv_off = kt * self.kv_lora_rank;
                        for c in 0..self.kv_lora_rank {
                            compressed_sum[c] += w * kv_cache.kv[kv_off + c];
                        }
                    }
                }

                // Project back: compressed_sum @ W_kv_b_v^T for this head
                // W_kv_b layout: [kv_lora_rank, n_heads * nope_v_dim]
                // V part for head h: col offset h * nope_v_dim + qk_nope_head_dim
                let out_off = qt * self.n_heads * self.v_head_dim + h * self.v_head_dim;
                for d in 0..self.v_head_dim {
                    let mut val = 0.0f32;
                    let w_col = h * nope_v_dim + self.qk_nope_head_dim + d;
                    for c in 0..self.kv_lora_rank {
                        val += compressed_sum[c] * wkv_b_data[c * self.n_heads * nope_v_dim + w_col];
                    }
                    output[out_off + d] = val;
                }
            }
        }

        // Reshape to [seq_len, n_heads * v_head_dim] and project
        let attn_out = Tensor::new(output, vec![seq_len, self.n_heads * self.v_head_dim], false);
        attn_out.matmul(&self.wo)
    }

    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        if let Some((s, d)) = params.get(&format!("{}.wq", prefix)) {
            self.wq = Some(Tensor::new(d.clone(), s.clone(), false));
        }
        if let Some((s, d)) = params.get(&format!("{}.wq_a", prefix)) {
            self.wq_a = Some(Tensor::new(d.clone(), s.clone(), false));
        }
        if let Some((_s, d)) = params.get(&format!("{}.q_norm", prefix)) {
            self.q_norm_weight = d.clone();
        }
        if let Some((s, d)) = params.get(&format!("{}.wq_b", prefix)) {
            self.wq_b = Some(Tensor::new(d.clone(), s.clone(), false));
        }
        if let Some((s, d)) = params.get(&format!("{}.wkv_a", prefix)) {
            self.wkv_a = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((_s, d)) = params.get(&format!("{}.kv_norm", prefix)) {
            self.kv_norm_weight = d.clone();
        }
        if let Some((s, d)) = params.get(&format!("{}.wkv_b", prefix)) {
            self.wkv_b = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((s, d)) = params.get(&format!("{}.wo", prefix)) {
            self.wo = Tensor::new(d.clone(), s.clone(), false);
        }
    }
}
