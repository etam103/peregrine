/// Core GQA attention with KV cache, masking, and BLAS-accelerated score computation.
///
/// Extracted from identical patterns in grok1 and gpt_oss examples.
///
/// KV cache for autoregressive generation.
/// Layout: k/v are [num_kv_heads, capacity, head_dim] contiguous, with `len` tracking
/// actual used tokens. Pre-allocates capacity to avoid per-token reallocation.
pub struct StandardKVCache {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub len: usize,
    capacity: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
}

impl StandardKVCache {
    pub fn new(num_kv_heads: usize, head_dim: usize) -> Self {
        StandardKVCache {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            capacity: 0,
            num_kv_heads,
            head_dim,
        }
    }

    /// Ensure we have room for at least `new_cap` tokens per head.
    /// Reshuffles data from [heads, old_cap, hd] to [heads, new_cap, hd].
    fn ensure_capacity(&mut self, needed: usize) {
        if needed <= self.capacity {
            return;
        }
        // Grow geometrically: at least double, or to needed (whichever is larger)
        let new_cap = needed.max(self.capacity.max(16) * 2);
        let hd = self.head_dim;
        let nkv = self.num_kv_heads;
        let old_cap = self.capacity;
        let old_len = self.len;

        let total = nkv * new_cap * hd;
        let mut new_k = vec![0.0f32; total];
        let mut new_v = vec![0.0f32; total];

        // Copy existing data from [heads, old_cap, hd] to [heads, new_cap, hd]
        if old_len > 0 {
            for h in 0..nkv {
                let src_base = h * old_cap * hd;
                let dst_base = h * new_cap * hd;
                let copy_bytes = old_len * hd;
                new_k[dst_base..dst_base + copy_bytes]
                    .copy_from_slice(&self.k[src_base..src_base + copy_bytes]);
                new_v[dst_base..dst_base + copy_bytes]
                    .copy_from_slice(&self.v[src_base..src_base + copy_bytes]);
            }
        }

        self.k = new_k;
        self.v = new_v;
        self.capacity = new_cap;
    }

    /// Append new K, V of shape [num_kv_heads, seq_len, head_dim].
    pub fn append(&mut self, new_k: &[f32], new_v: &[f32], seq_len: usize) {
        let new_len = self.len + seq_len;
        self.ensure_capacity(new_len);

        let hd = self.head_dim;
        let cap = self.capacity;
        let old_len = self.len;

        for h in 0..self.num_kv_heads {
            let dst_base = h * cap * hd + old_len * hd;
            let src_base = h * seq_len * hd;
            self.k[dst_base..dst_base + seq_len * hd]
                .copy_from_slice(&new_k[src_base..src_base + seq_len * hd]);
            self.v[dst_base..dst_base + seq_len * hd]
                .copy_from_slice(&new_v[src_base..src_base + seq_len * hd]);
        }

        self.len = new_len;
    }

    /// Rollback cache to `new_len` tokens (for speculative decoding rejection).
    /// Panics if `new_len > self.len`.
    pub fn rollback_to(&mut self, new_len: usize) {
        assert!(new_len <= self.len, "rollback_to: new_len {} > current len {}", new_len, self.len);
        // Just truncate logically — data beyond new_len is stale but capacity stays
        self.len = new_len;
    }

    /// Return the stride between heads (= capacity * head_dim).
    /// This is needed because the buffer may have capacity > len.
    #[inline]
    pub fn head_stride(&self) -> usize {
        self.capacity * self.head_dim
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
    // Use head_stride to handle capacity-based layout
    let k_stride = k_cache.head_stride();
    let v_stride = v_cache.head_stride();

    debug_assert_eq!(q.len(), num_q_heads * seq_q * head_dim);
    debug_assert_eq!(output.len(), seq_q * num_q_heads * head_dim);

    for qh in 0..num_q_heads {
        let kvh = qh / heads_per_group;

        for qt in 0..seq_q {
            let q_off = qh * seq_q * head_dim + qt * head_dim;
            let q_slice = &q[q_off..q_off + head_dim];

            // Compute scores against all cached K positions
            let mut scores = vec![0.0f32; total_len];
            let k_base = kvh * k_stride;

            for kt in 0..total_len {
                if is_masked(mask, qt, kt, total_len) {
                    scores[kt] = f32::NEG_INFINITY;
                } else {
                    let k_off = k_base + kt * head_dim;
                    let k_slice = &k_cache.k[k_off..k_off + head_dim];

                    // NEON-accelerated dot product
                    #[cfg(target_arch = "aarch64")]
                    let dot = {
                        use std::arch::aarch64::*;
                        let chunks4 = head_dim / 4;
                        let mut acc = unsafe { vdupq_n_f32(0.0) };
                        for c in 0..chunks4 {
                            let off = c * 4;
                            unsafe {
                                let vq = vld1q_f32(q_slice.as_ptr().add(off));
                                let vk = vld1q_f32(k_slice.as_ptr().add(off));
                                acc = vfmaq_f32(acc, vq, vk);
                            }
                        }
                        let mut d: f32 = unsafe { vaddvq_f32(acc) };
                        for i in (chunks4 * 4)..head_dim {
                            d += q_slice[i] * k_slice[i];
                        }
                        d
                    };
                    #[cfg(not(target_arch = "aarch64"))]
                    let dot = {
                        let mut d = 0.0f32;
                        for i in 0..head_dim {
                            d += q_slice[i] * k_slice[i];
                        }
                        d
                    };

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
            let v_base = kvh * v_stride;
            let out_off = qt * num_q_heads * head_dim + qh * head_dim;
            for kt in 0..total_len {
                let w = scores[kt];
                if w > 0.0 {
                    let v_off = v_base + kt * head_dim;

                    // NEON-accelerated weighted accumulation
                    #[cfg(target_arch = "aarch64")]
                    {
                        use std::arch::aarch64::*;
                        let vw = unsafe { vdupq_n_f32(w) };
                        let chunks4 = head_dim / 4;
                        for c in 0..chunks4 {
                            let off = c * 4;
                            unsafe {
                                let vo = vld1q_f32(output.as_ptr().add(out_off + off));
                                let vv = vld1q_f32(v_cache.v.as_ptr().add(v_off + off));
                                vst1q_f32(output.as_mut_ptr().add(out_off + off), vfmaq_f32(vo, vw, vv));
                            }
                        }
                        for d in (chunks4 * 4)..head_dim {
                            output[out_off + d] += w * v_cache.v[v_off + d];
                        }
                    }
                    #[cfg(not(target_arch = "aarch64"))]
                    {
                        for d in 0..head_dim {
                            output[out_off + d] += w * v_cache.v[v_off + d];
                        }
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
        // With capacity-based layout, k.len() >= 2 * 3 * 4
        assert!(cache.k.len() >= 2 * 3 * 4);

        // Verify data is accessible at correct offsets
        let stride = cache.head_stride();
        for h in 0..2 {
            for t in 0..3 {
                let base = h * stride + t * 4;
                for d in 0..4 {
                    assert_eq!(cache.k[base + d], 1.0);
                    assert_eq!(cache.v[base + d], 2.0);
                }
            }
        }

        // Append 1 more token
        let k2 = vec![3.0; 2 * 1 * 4];
        let v2 = vec![4.0; 2 * 1 * 4];
        cache.append(&k2, &v2, 1);
        assert_eq!(cache.len, 4);

        // Verify new data
        let stride = cache.head_stride();
        for h in 0..2 {
            let base = h * stride + 3 * 4;
            for d in 0..4 {
                assert_eq!(cache.k[base + d], 3.0);
                assert_eq!(cache.v[base + d], 4.0);
            }
        }

        // Rollback to 3
        cache.rollback_to(3);
        assert_eq!(cache.len, 3);

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
