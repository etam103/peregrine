/// Core GQA attention with KV cache, masking, and BLAS-accelerated score computation.
///
/// Extracted from identical patterns in grok1 and gpt_oss examples.

/// KV cache for autoregressive generation.
/// Layout: k/v are [num_kv_heads, cached_len, head_dim] contiguous.
pub struct StandardKVCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl StandardKVCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        StandardKVCache {
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

    /// Rollback cache to `new_len` tokens (for speculative decoding rejection).
    /// Panics if `new_len > self.len`.
    pub fn rollback_to(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "rollback_to: new_len {} > current len {}", new_len, self.len);
        if new_len == self.len {
            return;
        }
        let hd = self.head_dim;
        let old_len = self.len;

        let mut new_k = Vec::with_capacity(self.num_kv_heads * new_len * hd);
        let mut new_v = Vec::with_capacity(self.num_kv_heads * new_len * hd);

        for h in 0..self.num_kv_heads {
            let old_offset = h * old_len * hd;
            new_k.extend_from_slice(&self.k[old_offset..old_offset + new_len * hd]);
            new_v.extend_from_slice(&self.v[old_offset..old_offset + new_len * hd]);
        }

        self.k = new_k;
        self.v = new_v;
        self.len = new_len;
    }
}

/// Attention masking strategies.
#[derive(Clone, Debug)]
pub enum AttentionMask {
    /// No masking (full attention).
    None,
    /// Causal mask: only attend to positions <= query position.
    Causal { offset: usize },
    /// Causal + sliding window: attend within window + sink tokens at start.
    CausalSlidingWindow {
        offset: usize,
        window: usize,
        sink_tokens: usize,
    },
    /// Local + global sparse attention pattern.
    LocalGlobal {
        offset: usize,
        local_window: usize,
        global_positions: Vec<usize>,
    },
}

/// Post-score transforms applied before softmax.
#[derive(Clone, Debug)]
pub enum PostScoreTransform {
    /// No transform.
    None,
    /// Grok-1 style logit capping: cap * tanh(score / cap).
    LogitCap { cap: f32 },
}

/// BLAS-accelerated Grouped Query Attention on CPU.
///
/// q: [num_q_heads, seq_q, head_dim] — query heads (already RoPE'd)
/// k_cache, v_cache: StandardKVCache containing all KV up to (and including) current step
/// num_q_heads, num_kv_heads: GQA grouping (num_q_heads must be divisible by num_kv_heads)
/// head_dim: dimension per head
/// scale: typically 1/sqrt(head_dim)
/// mask: AttentionMask
/// transform: PostScoreTransform
/// output: [seq_q, num_q_heads * head_dim] row-major
pub fn gqa_attention_cpu(
    q: &[f32],
    k_cache: &StandardKVCache,
    v_cache: &StandardKVCache,
    num_q_heads: usize,
    num_kv_heads: usize,
    seq_q: usize,
    head_dim: usize,
    scale: f32,
    mask: &AttentionMask,
    transform: &PostScoreTransform,
    output: &mut [f32],
) {
    let total_len = k_cache.len;
    let heads_per_group = num_q_heads / num_kv_heads;

    debug_assert_eq!(q.len(), num_q_heads * seq_q * head_dim);
    debug_assert_eq!(output.len(), seq_q * num_q_heads * head_dim);

    for qh in 0..num_q_heads {
        let kvh = qh / heads_per_group;

        for qt in 0..seq_q {
            let q_off = qh * seq_q * head_dim + qt * head_dim;
            let q_slice = &q[q_off..q_off + head_dim];

            // Compute scores against all cached K positions
            let mut scores = vec![0.0f32; total_len];
            let k_base = kvh * total_len * head_dim;

            for kt in 0..total_len {
                if is_masked(mask, qt, kt, total_len) {
                    scores[kt] = f32::NEG_INFINITY;
                } else {
                    let k_off = k_base + kt * head_dim;
                    let mut dot = 0.0f32;
                    // Use chunks for better autovectorization
                    let k_slice = &k_cache.k[k_off..k_off + head_dim];
                    for d in 0..head_dim {
                        dot += q_slice[d] * k_slice[d];
                    }
                    let scaled = dot * scale;
                    scores[kt] = apply_transform(transform, scaled);
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
                let inv_sum = 1.0 / exp_sum;
                for s in &mut scores {
                    *s *= inv_sum;
                }
            }

            // Weighted sum of V
            let v_base = kvh * total_len * head_dim;
            let out_off = qt * num_q_heads * head_dim + qh * head_dim;
            for kt in 0..total_len {
                let w = scores[kt];
                if w > 0.0 {
                    let v_off = v_base + kt * head_dim;
                    for d in 0..head_dim {
                        output[out_off + d] += w * v_cache.v[v_off + d];
                    }
                }
            }
        }
    }
}

/// Check if position `kt` should be masked for query at position `qt`.
#[inline]
fn is_masked(mask: &AttentionMask, qt: usize, kt: usize, _total_len: usize) -> bool {
    match mask {
        AttentionMask::None => false,
        AttentionMask::Causal { offset } => {
            let query_pos = offset + qt;
            kt > query_pos
        }
        AttentionMask::CausalSlidingWindow { offset, window, sink_tokens } => {
            let query_pos = offset + qt;
            if kt > query_pos {
                return true; // causal
            }
            if kt < *sink_tokens {
                return false; // sink tokens always visible
            }
            // sliding window: mask if distance exceeds window
            query_pos.saturating_sub(kt) > *window
        }
        AttentionMask::LocalGlobal { offset, local_window, global_positions } => {
            let query_pos = offset + qt;
            if kt > query_pos {
                return true; // causal
            }
            // Check if in local window
            if query_pos.saturating_sub(kt) <= *local_window {
                return false;
            }
            // Check if in global positions
            if global_positions.contains(&kt) {
                return false;
            }
            true
        }
    }
}

/// Apply post-score transform.
#[inline]
fn apply_transform(transform: &PostScoreTransform, score: f32) -> f32 {
    match transform {
        PostScoreTransform::None => score,
        PostScoreTransform::LogitCap { cap } => cap * (score / cap).tanh(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_cache_append_and_rollback() {
        let mut cache = StandardKVCache::new(2, 4);
        // Append 3 tokens: [2, 3, 4] layout
        let k = vec![1.0; 2 * 3 * 4]; // 2 heads, 3 tokens, 4 dim
        let v = vec![2.0; 2 * 3 * 4];
        cache.append(&k, &v, 3);
        assert_eq!(cache.len, 3);
        assert_eq!(cache.k.len(), 2 * 3 * 4);

        // Append 1 more token
        let k2 = vec![3.0; 2 * 1 * 4];
        let v2 = vec![4.0; 2 * 1 * 4];
        cache.append(&k2, &v2, 1);
        assert_eq!(cache.len, 4);
        assert_eq!(cache.k.len(), 2 * 4 * 4);

        // Rollback to 3
        cache.rollback_to(3);
        assert_eq!(cache.len, 3);
        assert_eq!(cache.k.len(), 2 * 3 * 4);

        // Rollback to same length is no-op
        cache.rollback_to(3);
        assert_eq!(cache.len, 3);
    }

    #[test]
    #[should_panic(expected = "rollback_to")]
    fn test_rollback_panics_on_invalid() {
        let mut cache = StandardKVCache::new(1, 4);
        let k = vec![1.0; 1 * 2 * 4];
        let v = vec![1.0; 1 * 2 * 4];
        cache.append(&k, &v, 2);
        cache.rollback_to(3); // should panic
    }
}
