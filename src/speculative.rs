/// Speculative decoding (Leviathan et al. 2023) with self-speculation support.
///
/// Draft model proposes N tokens, target model verifies in a single batched forward pass.
/// Accept/reject uses stochastic criterion: accept with probability min(1, q(x)/p(x)).

use crate::attention::StandardKVCache;

/// Trait for causal language models compatible with speculative decoding.
pub trait CausalLM {
    /// Run forward pass on token sequence, return logits [seq_len, vocab_size].
    fn forward(&self, tokens: &[usize], caches: &mut [StandardKVCache]) -> Vec<f32>;

    /// Create fresh KV caches for all layers.
    fn init_caches(&self) -> Vec<StandardKVCache>;

    /// Truncate all KV caches to `new_len` tokens.
    fn truncate_caches(&self, caches: &mut [StandardKVCache], new_len: usize) {
        for cache in caches.iter_mut() {
            cache.rollback_to(new_len);
        }
    }

    /// Vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Speculative decoding: draft model proposes `lookahead` tokens, target verifies.
///
/// Returns generated token sequence (excluding prompt).
///
/// When `lookahead == 0`, this degenerates to standard autoregressive decoding
/// using only the target model.
pub fn speculative_decode<D: CausalLM, T: CausalLM>(
    draft: &D,
    target: &T,
    draft_caches: &mut [StandardKVCache],
    target_caches: &mut [StandardKVCache],
    prompt: &[usize],
    max_tokens: usize,
    lookahead: usize,
    temperature: f32,
    eos_id: usize,
) -> Vec<usize> {
    let vocab = target.vocab_size();

    // Prefill both models with prompt
    let target_logits = target.forward(prompt, target_caches);
    let mut generated = Vec::new();

    if lookahead == 0 {
        // Standard autoregressive decoding (target only)
        let mut next = sample_from_logits(&target_logits, vocab, temperature);
        generated.push(next);

        while generated.len() < max_tokens && next != eos_id {
            let logits = target.forward(&[next], target_caches);
            next = sample_from_logits(&logits, vocab, temperature);
            generated.push(next);
        }
        return generated;
    }

    // Also prefill draft model
    let _ = draft.forward(prompt, draft_caches);

    // First token from target
    let mut next = sample_from_logits(&target_logits, vocab, temperature);
    generated.push(next);

    while generated.len() < max_tokens && next != eos_id {
        let cache_len_before = target_caches[0].len;
        let draft_cache_len_before = draft_caches[0].len;

        // 1. Draft proposes `lookahead` tokens
        let mut draft_tokens = Vec::with_capacity(lookahead);
        let mut draft_probs_per_token = Vec::with_capacity(lookahead);

        let mut draft_input = next;
        for _ in 0..lookahead {
            let draft_logits = draft.forward(&[draft_input], draft_caches);
            let probs = logits_to_probs(&draft_logits, vocab, temperature);
            let sampled = sample_from_probs(&probs);
            draft_tokens.push(sampled);
            draft_probs_per_token.push(probs);
            draft_input = sampled;
        }

        // 2. Target verifies: run all draft tokens (plus the one before) in single forward pass
        // We need to feed [next, draft_tokens[0], ..., draft_tokens[N-2]] to get logits
        // that predict [draft_tokens[0], draft_tokens[1], ..., draft_tokens[N-1], next_after]
        let mut verify_input = vec![next];
        verify_input.extend_from_slice(&draft_tokens);
        let target_logits = target.forward(&verify_input, target_caches);

        // target_logits is [lookahead+1, vocab] — position i has logits for predicting token i+1
        let mut accepted = 0;

        for i in 0..lookahead {
            let target_probs = logits_to_probs_at(&target_logits, vocab, temperature, i);
            let draft_p = draft_probs_per_token[i][draft_tokens[i]];
            let target_q = target_probs[draft_tokens[i]];

            // Accept with probability min(1, q/p)
            let accept_prob = if draft_p > 0.0 { (target_q / draft_p).min(1.0) } else { 0.0 };

            if random_uniform() < accept_prob {
                accepted += 1;
                generated.push(draft_tokens[i]);
                if draft_tokens[i] == eos_id || generated.len() >= max_tokens {
                    return generated;
                }
            } else {
                // Rejection: sample from adjusted distribution max(0, q - p) / Z
                let mut adjusted = vec![0.0f32; vocab];
                let mut adj_sum = 0.0f32;
                for j in 0..vocab {
                    adjusted[j] = (target_probs[j] - draft_probs_per_token[i][j]).max(0.0);
                    adj_sum += adjusted[j];
                }
                if adj_sum > 0.0 {
                    let inv = 1.0 / adj_sum;
                    for j in 0..vocab {
                        adjusted[j] *= inv;
                    }
                }
                next = sample_from_probs(&adjusted);
                generated.push(next);
                accepted += 1; // count the resampled token

                // Rollback caches: target keeps accepted+1 positions, draft keeps accepted
                let target_new_len = cache_len_before + accepted;
                let draft_new_len = draft_cache_len_before + accepted;
                target.truncate_caches(target_caches, target_new_len);
                draft.truncate_caches(draft_caches, draft_new_len);
                break;
            }
        }

        if accepted == lookahead {
            // All accepted — bonus token from target's last position
            let bonus_probs = logits_to_probs_at(&target_logits, vocab, temperature, lookahead);
            next = sample_from_probs(&bonus_probs);
            generated.push(next);

            // Draft cache is behind by `lookahead` tokens — run through draft to sync
            draft.truncate_caches(draft_caches, draft_cache_len_before);
            let mut sync_input = vec![draft_tokens[0]];
            sync_input.extend_from_slice(&draft_tokens[1..]);
            let _ = draft.forward(&sync_input, draft_caches);

            if next == eos_id || generated.len() >= max_tokens {
                return generated;
            }
        }
    }

    generated
}

/// Convert logits to probability distribution.
fn logits_to_probs(logits: &[f32], vocab: usize, temperature: f32) -> Vec<f32> {
    logits_to_probs_at(logits, vocab, temperature, logits.len() / vocab - 1)
}

/// Extract probs from logits at position `pos` (row of [seq_len, vocab]).
fn logits_to_probs_at(logits: &[f32], vocab: usize, temperature: f32, pos: usize) -> Vec<f32> {
    let row = &logits[pos * vocab..(pos + 1) * vocab];

    if temperature <= 0.0 {
        // Greedy: one-hot at argmax
        let mut best = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best = i;
            }
        }
        let mut probs = vec![0.0f32; vocab];
        probs[best] = 1.0;
        return probs;
    }

    let scaled: Vec<f32> = row.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let inv = 1.0 / sum;
    exps.iter().map(|&x| x * inv).collect()
}

/// Sample a token index from a probability distribution.
fn sample_from_probs(probs: &[f32]) -> usize {
    let r = random_uniform();
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i;
        }
    }
    probs.len() - 1
}

/// Sample a token from logits (last row).
fn sample_from_logits(logits: &[f32], vocab: usize, temperature: f32) -> usize {
    let probs = logits_to_probs(logits, vocab, temperature);
    sample_from_probs(&probs)
}

/// Simple pseudo-random uniform [0, 1) using system time.
fn random_uniform() -> f32 {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let mut rng = seed;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    (rng as f32) / (u32::MAX as f32)
}
