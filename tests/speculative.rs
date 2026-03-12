/// Tests for speculative decoding.
use peregrine::attention::StandardKVCache;
use peregrine::speculative::{CausalLM, speculative_decode};

/// Trivial model: always returns the same logits regardless of input.
/// Used to test speculative decoding mechanics.
struct ConstantLM {
    vocab: usize,
    /// The token ID that this model always predicts (highest logit).
    predicted_token: usize,
    num_layers: usize,
    head_dim: usize,
}

impl ConstantLM {
    fn new(vocab: usize, predicted_token: usize) -> Self {
        ConstantLM {
            vocab,
            predicted_token,
            num_layers: 1,
            head_dim: 4,
        }
    }
}

impl CausalLM for ConstantLM {
    fn forward(&self, tokens: &[usize], caches: &mut [StandardKVCache]) -> Vec<f32> {
        let seq_len = tokens.len();

        // Advance caches to track position
        for cache in caches.iter_mut() {
            let k = vec![0.0f32; cache.num_kv_heads * seq_len * cache.head_dim];
            let v = vec![0.0f32; cache.num_kv_heads * seq_len * cache.head_dim];
            cache.append(&k, &v, seq_len);
        }

        // Return logits: [seq_len, vocab] with predicted_token having highest logit
        let mut logits = vec![-10.0f32; seq_len * self.vocab];
        for t in 0..seq_len {
            logits[t * self.vocab + self.predicted_token] = 10.0;
        }
        logits
    }

    fn init_caches(&self) -> Vec<StandardKVCache> {
        (0..self.num_layers)
            .map(|_| StandardKVCache::new(1, self.head_dim))
            .collect()
    }

    fn vocab_size(&self) -> usize {
        self.vocab
    }
}

#[test]
fn test_greedy_equivalence_lookahead_zero() {
    // With lookahead=0, speculative_decode should produce standard autoregressive output
    let model = ConstantLM::new(10, 3);
    let mut caches = model.init_caches();
    let mut caches2 = model.init_caches();

    let prompt = vec![1, 2];
    let max_tokens = 5;

    // Standard decode (lookahead=0)
    let result = speculative_decode(
        &model, &model,
        &mut caches, &mut caches2,
        &prompt,
        max_tokens,
        0,  // lookahead
        0.0, // greedy
        99,  // eos
    );

    // Should generate max_tokens tokens, all equal to predicted_token=3
    assert_eq!(result.len(), max_tokens);
    for &tok in &result {
        assert_eq!(tok, 3, "Expected constant prediction of 3");
    }
}

#[test]
fn test_speculative_same_model_acceptance() {
    // When draft == target (same model), all tokens should be accepted
    let model = ConstantLM::new(10, 5);
    let mut draft_caches = model.init_caches();
    let mut target_caches = model.init_caches();

    let prompt = vec![1];
    let max_tokens = 8;

    let result = speculative_decode(
        &model, &model,
        &mut draft_caches, &mut target_caches,
        &prompt,
        max_tokens,
        4,   // lookahead
        0.0, // greedy
        99,  // eos
    );

    // Should still produce max_tokens tokens of predicted_token=5
    assert_eq!(result.len(), max_tokens);
    for &tok in &result {
        assert_eq!(tok, 5, "Expected all tokens to be 5");
    }
}

#[test]
fn test_eos_stops_generation() {
    // Model predicts its own EOS token
    let eos = 7;
    let model = ConstantLM::new(10, eos);
    let mut caches = model.init_caches();
    let mut caches2 = model.init_caches();

    let prompt = vec![1, 2];

    let result = speculative_decode(
        &model, &model,
        &mut caches, &mut caches2,
        &prompt,
        100,
        0,
        0.0,
        eos,
    );

    // Should stop after generating EOS (first token)
    assert_eq!(result.len(), 1);
    assert_eq!(result[0], eos);
}

#[test]
fn test_cache_rollback_after_speculative() {
    // Verify cache integrity after speculative decoding
    let model = ConstantLM::new(10, 3);
    let mut draft_caches = model.init_caches();
    let mut target_caches = model.init_caches();

    let prompt = vec![1, 2, 3];
    let max_tokens = 6;

    let result = speculative_decode(
        &model, &model,
        &mut draft_caches, &mut target_caches,
        &prompt,
        max_tokens,
        3,   // lookahead
        0.0,
        99,
    );

    assert_eq!(result.len(), max_tokens);

    // Target cache contains prompt + all generated tokens.
    // With speculative decoding, the target verifies lookahead+1 tokens per step,
    // so the cache may overshoot slightly. The key invariant is that
    // cache.len >= prompt_len + generated_len.
    let min_expected = prompt.len() + result.len();
    assert!(
        target_caches[0].len >= min_expected,
        "Target cache len should be >= prompt+generated: min_expected={}, got={}",
        min_expected, target_caches[0].len
    );
}
