/// Tests for core GQA attention module.
use peregrine::attention::*;

/// Naive single-head attention for reference.
fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
    offset: usize,
) -> Vec<f32> {
    let mut output = vec![0.0f32; seq_q * head_dim];

    for qt in 0..seq_q {
        let mut scores = vec![0.0f32; seq_kv];
        for kt in 0..seq_kv {
            if causal && kt > offset + qt {
                scores[kt] = f32::NEG_INFINITY;
            } else {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[qt * head_dim + d] * k[kt * head_dim + d];
                }
                scores[kt] = dot * scale;
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

        for kt in 0..seq_kv {
            let w = scores[kt];
            if w > 0.0 {
                for d in 0..head_dim {
                    output[qt * head_dim + d] += w * v[kt * head_dim + d];
                }
            }
        }
    }

    output
}

#[test]
fn test_gqa_matches_naive_mha() {
    // MHA fallback: num_q_heads == num_kv_heads
    let head_dim = 8;
    let num_heads = 2;
    let seq_q = 3;
    let seq_kv = 5;

    // Build Q, K, V with deterministic values
    let q: Vec<f32> = (0..num_heads * seq_q * head_dim)
        .map(|i| ((i as f32 * 0.1).sin()))
        .collect();

    let mut cache = StandardKVCache::new(num_heads, head_dim);
    // Build KV data: [num_heads, seq_kv, head_dim]
    let k: Vec<f32> = (0..num_heads * seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.2).cos()))
        .collect();
    let v: Vec<f32> = (0..num_heads * seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * num_heads * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        num_heads,
        num_heads,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::None,
        &PostScoreTransform::None,
        &mut output,
    );

    // Compare per-head against naive
    for h in 0..num_heads {
        let q_head: Vec<f32> = (0..seq_q)
            .flat_map(|t| {
                let off = h * seq_q * head_dim + t * head_dim;
                q[off..off + head_dim].to_vec()
            })
            .collect();
        let k_head = &cache.k[h * seq_kv * head_dim..(h + 1) * seq_kv * head_dim];
        let v_head = &cache.v[h * seq_kv * head_dim..(h + 1) * seq_kv * head_dim];

        let expected = naive_attention(
            &q_head, k_head, v_head, seq_q, seq_kv, head_dim, scale, false, 0,
        );

        for t in 0..seq_q {
            for d in 0..head_dim {
                let got = output[t * num_heads * head_dim + h * head_dim + d];
                let exp = expected[t * head_dim + d];
                assert!(
                    (got - exp).abs() < 1e-5,
                    "MHA mismatch at head={}, t={}, d={}: got={}, expected={}",
                    h, t, d, got, exp
                );
            }
        }
    }
}

#[test]
fn test_gqa_with_grouping() {
    // GQA: 4 Q heads, 2 KV heads (ratio=2)
    let head_dim = 4;
    let num_q_heads = 4;
    let num_kv_heads = 2;
    let seq_q = 2;
    let seq_kv = 3;

    let q: Vec<f32> = (0..num_q_heads * seq_q * head_dim)
        .map(|i| ((i as f32 * 0.1).sin()))
        .collect();

    let mut cache = StandardKVCache::new(num_kv_heads, head_dim);
    let k: Vec<f32> = (0..num_kv_heads * seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.2).cos()))
        .collect();
    let v: Vec<f32> = (0..num_kv_heads * seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * num_q_heads * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        num_q_heads,
        num_kv_heads,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::None,
        &PostScoreTransform::None,
        &mut output,
    );

    // Q heads 0,1 should use KV head 0; Q heads 2,3 should use KV head 1
    // Since Q heads 0 and 1 use same KV, their outputs differ only due to different Q values
    // Verify by computing naive for each pair
    for qh in 0..num_q_heads {
        let kvh = qh / 2;
        let q_head: Vec<f32> = (0..seq_q)
            .flat_map(|t| {
                let off = qh * seq_q * head_dim + t * head_dim;
                q[off..off + head_dim].to_vec()
            })
            .collect();
        let k_head = &cache.k[kvh * seq_kv * head_dim..(kvh + 1) * seq_kv * head_dim];
        let v_head = &cache.v[kvh * seq_kv * head_dim..(kvh + 1) * seq_kv * head_dim];

        let expected = naive_attention(
            &q_head, k_head, v_head, seq_q, seq_kv, head_dim, scale, false, 0,
        );

        for t in 0..seq_q {
            for d in 0..head_dim {
                let got = output[t * num_q_heads * head_dim + qh * head_dim + d];
                let exp = expected[t * head_dim + d];
                assert!(
                    (got - exp).abs() < 1e-5,
                    "GQA mismatch at qh={}, t={}, d={}: got={}, expected={}",
                    qh, t, d, got, exp
                );
            }
        }
    }
}

#[test]
fn test_mqa_fallback() {
    // MQA: 4 Q heads, 1 KV head
    let head_dim = 4;
    let num_q_heads = 4;
    let num_kv_heads = 1;
    let seq_q = 2;
    let seq_kv = 3;

    let q: Vec<f32> = (0..num_q_heads * seq_q * head_dim)
        .map(|i| ((i as f32 * 0.1).sin()))
        .collect();

    let mut cache = StandardKVCache::new(num_kv_heads, head_dim);
    let k: Vec<f32> = (0..num_kv_heads * seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.2).cos()))
        .collect();
    let v: Vec<f32> = (0..num_kv_heads * seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * num_q_heads * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        num_q_heads,
        num_kv_heads,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::None,
        &PostScoreTransform::None,
        &mut output,
    );

    // All Q heads should share the single KV head
    for qh in 0..num_q_heads {
        let q_head: Vec<f32> = (0..seq_q)
            .flat_map(|t| {
                let off = qh * seq_q * head_dim + t * head_dim;
                q[off..off + head_dim].to_vec()
            })
            .collect();

        let expected = naive_attention(
            &q_head, &cache.k, &cache.v, seq_q, seq_kv, head_dim, scale, false, 0,
        );

        for t in 0..seq_q {
            for d in 0..head_dim {
                let got = output[t * num_q_heads * head_dim + qh * head_dim + d];
                let exp = expected[t * head_dim + d];
                assert!(
                    (got - exp).abs() < 1e-5,
                    "MQA mismatch at qh={}, t={}, d={}",
                    qh, t, d
                );
            }
        }
    }
}

#[test]
fn test_causal_mask() {
    let head_dim = 4;
    let seq_q = 3;
    let seq_kv = 3;
    let offset = 0;

    let q: Vec<f32> = (0..seq_q * head_dim)
        .map(|i| ((i as f32 * 0.1).sin()))
        .collect();

    let mut cache = StandardKVCache::new(1, head_dim);
    let k: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.2).cos()))
        .collect();
    let v: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        1, 1,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::Causal { offset },
        &PostScoreTransform::None,
        &mut output,
    );

    let expected = naive_attention(&q, &k, &v, seq_q, seq_kv, head_dim, scale, true, offset);

    for i in 0..output.len() {
        assert!(
            (output[i] - expected[i]).abs() < 1e-5,
            "Causal mask mismatch at {}: got={}, expected={}",
            i, output[i], expected[i]
        );
    }
}

#[test]
fn test_causal_mask_with_offset() {
    // Simulate decoding after a prompt of 5 tokens
    let head_dim = 4;
    let seq_q = 1;
    let seq_kv = 6; // 5 prompt + 1 current
    let offset = 5; // query position starts at 5

    let q: Vec<f32> = (0..seq_q * head_dim)
        .map(|i| ((i as f32 * 0.1).sin()))
        .collect();

    let mut cache = StandardKVCache::new(1, head_dim);
    let k: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.2).cos()))
        .collect();
    let v: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        1, 1,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::Causal { offset },
        &PostScoreTransform::None,
        &mut output,
    );

    // With offset=5, qt=0, query_pos=5. All kt <= 5 should be visible.
    // Since seq_kv=6, position 5 is the last one — all visible.
    let expected = naive_attention(&q, &k, &v, seq_q, seq_kv, head_dim, scale, true, offset);

    for i in 0..output.len() {
        assert!(
            (output[i] - expected[i]).abs() < 1e-5,
            "Causal+offset mismatch at {}: got={}, expected={}",
            i, output[i], expected[i]
        );
    }
}

#[test]
fn test_sliding_window_mask() {
    let head_dim = 4;
    let seq_q = 1;
    let seq_kv = 10;
    let offset = 9; // query is at position 9
    let window = 3;

    let q: Vec<f32> = (0..seq_q * head_dim)
        .map(|i| ((i as f32 * 0.1).sin()))
        .collect();

    let mut cache = StandardKVCache::new(1, head_dim);
    let k: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.2).cos()))
        .collect();
    let v: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0f32; seq_q * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        1, 1,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::CausalSlidingWindow {
            offset,
            window,
            sink_tokens: 0,
        },
        &PostScoreTransform::None,
        &mut output,
    );

    // Query pos=9, window=3: positions 7,8,9 should be visible (9-kt <= 3)
    // positions 0..6 should be masked
    // Verify by computing expected with only positions 7,8,9
    let mut expected = vec![0.0f32; head_dim];
    let mut scores = vec![f32::NEG_INFINITY; seq_kv];
    for kt in 0..seq_kv {
        if offset.saturating_sub(kt) <= window {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += q[d] * k[kt * head_dim + d];
            }
            scores[kt] = dot * scale;
        }
    }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for s in &mut scores {
        *s = (*s - max_s).exp();
        exp_sum += *s;
    }
    for s in &mut scores {
        *s /= exp_sum;
    }
    for kt in 0..seq_kv {
        let w = scores[kt];
        if w > 0.0 {
            for d in 0..head_dim {
                expected[d] += w * v[kt * head_dim + d];
            }
        }
    }

    for d in 0..head_dim {
        assert!(
            (output[d] - expected[d]).abs() < 1e-5,
            "Sliding window mismatch at d={}: got={}, expected={}",
            d, output[d], expected[d]
        );
    }
}

#[test]
fn test_logit_cap_transform() {
    let head_dim = 4;
    let seq_q = 1;
    let seq_kv = 3;

    let q: Vec<f32> = vec![1.0, 0.5, -0.5, 0.2];
    let mut cache = StandardKVCache::new(1, head_dim);
    let k: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.3).cos()))
        .collect();
    let v: Vec<f32> = (0..seq_kv * head_dim)
        .map(|i| ((i as f32 * 0.15).sin()))
        .collect();
    cache.append(&k, &v, seq_kv);

    let scale = 1.0 / (head_dim as f32).sqrt();
    let cap = 30.0f32;

    let mut output = vec![0.0f32; seq_q * head_dim];

    gqa_attention_cpu(
        &q,
        &cache,
        &cache,
        1, 1,
        seq_q,
        head_dim,
        scale,
        &AttentionMask::None,
        &PostScoreTransform::LogitCap { cap },
        &mut output,
    );

    // Compute expected with logit capping
    let mut scores = vec![0.0f32; seq_kv];
    for kt in 0..seq_kv {
        let mut dot = 0.0f32;
        for d in 0..head_dim {
            dot += q[d] * k[kt * head_dim + d];
        }
        let scaled = dot * scale;
        scores[kt] = cap * (scaled / cap).tanh();
    }
    let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exp_sum = 0.0f32;
    for s in &mut scores {
        *s = (*s - max_s).exp();
        exp_sum += *s;
    }
    for s in &mut scores {
        *s /= exp_sum;
    }
    let mut expected = vec![0.0f32; head_dim];
    for kt in 0..seq_kv {
        for d in 0..head_dim {
            expected[d] += scores[kt] * v[kt * head_dim + d];
        }
    }

    for d in 0..head_dim {
        assert!(
            (output[d] - expected[d]).abs() < 1e-5,
            "LogitCap mismatch at d={}: got={}, expected={}",
            d, output[d], expected[d]
        );
    }
}

#[test]
fn test_cache_rollback_integrity() {
    let head_dim = 4;
    let num_kv_heads = 2;

    let mut cache = StandardKVCache::new(num_kv_heads, head_dim);

    // Append 5 tokens
    let k1: Vec<f32> = (0..num_kv_heads * 5 * head_dim)
        .map(|i| i as f32 * 0.01)
        .collect();
    let v1: Vec<f32> = (0..num_kv_heads * 5 * head_dim)
        .map(|i| i as f32 * 0.02)
        .collect();
    cache.append(&k1, &v1, 5);

    // Save state at len=5
    let k_at_5 = cache.k.clone();
    let v_at_5 = cache.v.clone();

    // Append 3 more
    let k2: Vec<f32> = vec![99.0; num_kv_heads * 3 * head_dim];
    let v2: Vec<f32> = vec![99.0; num_kv_heads * 3 * head_dim];
    cache.append(&k2, &v2, 3);
    assert_eq!(cache.len, 8);

    // Rollback to 5
    cache.rollback_to(5);
    assert_eq!(cache.len, 5);
    assert_eq!(cache.k, k_at_5);
    assert_eq!(cache.v, v_at_5);
}
