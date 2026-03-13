/// Managed prefill/decode aggregation with priority scheduling.
///
/// The scheduler does NOT own the model — it returns `SchedulerAction` values
/// telling the caller what to forward through the model next.  This keeps it
/// reusable across llama, grok1, deepseek, gpt_oss, etc.

use crate::attention::StandardKVCache;

// ── Identifiers & enums ─────────────────────────────────────────────

/// Unique request identifier.
pub type RequestId = u64;

/// Request priority — variants are ordered low→high so derived Ord works.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Background = 0,
    Normal = 1,
    High = 2,
}

/// Lifecycle state of a request inside the scheduler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestState {
    /// Waiting to begin prefill.
    Queued,
    /// Prefilling; `chunk_offset` tokens have been processed so far.
    Prefilling { chunk_offset: usize },
    /// Prefill done, generating tokens one at a time.
    Decoding,
    /// Terminal — EOS seen or max tokens reached.
    Done,
}

// ── Request ─────────────────────────────────────────────────────────

/// A single inference request tracked by the scheduler.
pub struct Request {
    pub id: RequestId,
    pub priority: Priority,
    pub state: RequestState,
    pub prompt_tokens: Vec<usize>,
    pub generated_tokens: Vec<usize>,
    pub kv_caches: Vec<StandardKVCache>,
    pub eos_id: usize,
    pub max_tokens: usize,
    /// Logits from the most recent forward pass (set by `complete_step`).
    pub pending_logits: Option<Vec<f32>>,
}

// ── Chunked prefiller ───────────────────────────────────────────────

/// Standalone helper that slices a request's prompt into chunks.
pub struct ChunkedPrefiller;

impl ChunkedPrefiller {
    /// Return the next chunk of prompt tokens to prefill, or `None` if done.
    pub fn next_chunk(req: &Request, chunk_size: usize) -> Option<Vec<usize>> {
        let offset = match req.state {
            RequestState::Queued => 0,
            RequestState::Prefilling { chunk_offset } => chunk_offset,
            _ => return None,
        };
        let total = req.prompt_tokens.len();
        if offset >= total {
            return None;
        }
        let end = (offset + chunk_size).min(total);
        Some(req.prompt_tokens[offset..end].to_vec())
    }

    /// Advance the request state after a prefill chunk.
    /// Returns `true` when prefill is complete (request transitions to Decoding).
    pub fn step(req: &mut Request, chunk_size: usize) -> bool {
        let offset = match req.state {
            RequestState::Queued => 0,
            RequestState::Prefilling { chunk_offset } => chunk_offset,
            _ => return true,
        };
        let new_offset = (offset + chunk_size).min(req.prompt_tokens.len());
        if new_offset >= req.prompt_tokens.len() {
            req.state = RequestState::Decoding;
            true
        } else {
            req.state = RequestState::Prefilling {
                chunk_offset: new_offset,
            };
            false
        }
    }
}

// ── Scheduler config & stats ────────────────────────────────────────

/// Tunable parameters for the scheduler.
#[derive(Clone, Debug)]
pub struct SchedulerConfig {
    /// Target decode latency in ms — chunk size is tuned to stay under this.
    pub target_decode_ms: f64,
    /// Initial prefill chunk size.
    pub initial_chunk_size: usize,
    /// Minimum chunk size.
    pub min_chunk_size: usize,
    /// Maximum chunk size.
    pub max_chunk_size: usize,
    /// EMA smoothing factor (0–1, higher = more weight to recent samples).
    pub ema_alpha: f64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            target_decode_ms: 40.0,
            initial_chunk_size: 256,
            min_chunk_size: 32,
            max_chunk_size: 1024,
            ema_alpha: 0.3,
        }
    }
}

/// Aggregate scheduler statistics.
#[derive(Clone, Debug)]
pub struct SchedulerStats {
    pub decode_latency_ema_ms: f64,
    pub prefill_chunk_latency_ema_ms: f64,
    pub current_chunk_size: usize,
    pub total_decode_steps: u64,
    pub total_prefill_chunks: u64,
    pub total_completed: u64,
}

// ── Scheduler action ────────────────────────────────────────────────

/// The next thing the caller should feed through the model.
#[derive(Clone, Debug)]
pub enum SchedulerAction {
    /// Decode a single token for request `id`.
    Decode { id: RequestId, token: Vec<usize> },
    /// Prefill a chunk for request `id`.
    PrefillChunk { id: RequestId, tokens: Vec<usize> },
    /// All requests have reached Done.
    AllDone,
    /// No actionable work right now (shouldn't happen with well-formed input).
    Idle,
}

// ── Scheduler ───────────────────────────────────────────────────────

/// Priority scheduler for interleaved prefill/decode across multiple requests.
pub struct Scheduler {
    requests: Vec<Request>,
    next_id: RequestId,
    config: SchedulerConfig,
    stats: SchedulerStats,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig) -> Self {
        let chunk = config.initial_chunk_size;
        Scheduler {
            requests: Vec::new(),
            next_id: 0,
            stats: SchedulerStats {
                decode_latency_ema_ms: 0.0,
                prefill_chunk_latency_ema_ms: 0.0,
                current_chunk_size: chunk,
                total_decode_steps: 0,
                total_prefill_chunks: 0,
                total_completed: 0,
            },
            config,
        }
    }

    /// Add a new request. Returns the assigned `RequestId`.
    pub fn add_request(
        &mut self,
        prompt_tokens: Vec<usize>,
        kv_caches: Vec<StandardKVCache>,
        eos_id: usize,
        max_tokens: usize,
        priority: Priority,
    ) -> RequestId {
        let id = self.next_id;
        self.next_id += 1;
        self.requests.push(Request {
            id,
            priority,
            state: RequestState::Queued,
            prompt_tokens,
            generated_tokens: Vec::new(),
            kv_caches,
            eos_id,
            max_tokens,
            pending_logits: None,
        });
        id
    }

    /// Determine the next action according to priority policy:
    ///   1. High-priority decodes
    ///   2. Normal-priority decodes
    ///   3. Highest-priority prefill chunk
    ///   4. Background decodes
    ///   5. AllDone / Idle
    pub fn next_action(&self) -> SchedulerAction {
        let chunk_size = self.stats.current_chunk_size;

        // Collect active requests by category
        let mut high_decodes = Vec::new();
        let mut normal_decodes = Vec::new();
        let mut background_decodes = Vec::new();
        let mut prefills: Vec<(RequestId, Priority)> = Vec::new();
        let mut any_active = false;

        for req in &self.requests {
            match req.state {
                RequestState::Done => {}
                RequestState::Decoding => {
                    any_active = true;
                    match req.priority {
                        Priority::High => high_decodes.push(req),
                        Priority::Normal => normal_decodes.push(req),
                        Priority::Background => background_decodes.push(req),
                    }
                }
                RequestState::Queued | RequestState::Prefilling { .. } => {
                    any_active = true;
                    prefills.push((req.id, req.priority));
                }
            }
        }

        if !any_active {
            return if self.requests.is_empty() {
                SchedulerAction::Idle
            } else {
                SchedulerAction::AllDone
            };
        }

        // 1. High-priority decode
        if let Some(req) = high_decodes.first() {
            let token = req
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(*req.prompt_tokens.last().unwrap_or(&0));
            return SchedulerAction::Decode {
                id: req.id,
                token: vec![token],
            };
        }

        // 2. Normal-priority decode
        if let Some(req) = normal_decodes.first() {
            let token = req
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(*req.prompt_tokens.last().unwrap_or(&0));
            return SchedulerAction::Decode {
                id: req.id,
                token: vec![token],
            };
        }

        // 3. Highest-priority prefill chunk
        if !prefills.is_empty() {
            // Sort by priority descending
            let mut sorted = prefills;
            sorted.sort_by(|a, b| b.1.cmp(&a.1));
            let best_id = sorted[0].0;
            let req = self.requests.iter().find(|r| r.id == best_id).unwrap();
            if let Some(tokens) = ChunkedPrefiller::next_chunk(req, chunk_size) {
                return SchedulerAction::PrefillChunk {
                    id: best_id,
                    tokens,
                };
            }
        }

        // 4. Background decodes
        if let Some(req) = background_decodes.first() {
            let token = req
                .generated_tokens
                .last()
                .copied()
                .unwrap_or(*req.prompt_tokens.last().unwrap_or(&0));
            return SchedulerAction::Decode {
                id: req.id,
                token: vec![token],
            };
        }

        SchedulerAction::Idle
    }

    /// Call after the model forward pass completes for a given request.
    ///
    /// `sample_fn` turns logits into a token id (greedy, top-p, etc).
    /// Returns the sampled token (or `None` if this was a non-final prefill chunk).
    pub fn complete_step<F>(
        &mut self,
        id: RequestId,
        logits: Vec<f32>,
        vocab_size: usize,
        elapsed_ms: f64,
        sample_fn: F,
    ) -> Option<usize>
    where
        F: FnOnce(&[f32], usize) -> usize,
    {
        let idx = self
            .requests
            .iter()
            .position(|r| r.id == id)
            .expect("complete_step: unknown request id");

        let chunk_size = self.stats.current_chunk_size;
        let state = self.requests[idx].state;

        match state {
            RequestState::Queued | RequestState::Prefilling { .. } => {
                let done = ChunkedPrefiller::step(&mut self.requests[idx], chunk_size);
                self.stats.total_prefill_chunks += 1;
                self.update_prefill_ema(elapsed_ms);

                if done {
                    let token = sample_fn(&logits, vocab_size);
                    let req = &mut self.requests[idx];
                    req.generated_tokens.push(token);
                    req.pending_logits = Some(logits);
                    if token == req.eos_id || req.generated_tokens.len() >= req.max_tokens {
                        req.state = RequestState::Done;
                        self.stats.total_completed += 1;
                    } else {
                        req.state = RequestState::Decoding;
                    }
                    return Some(token);
                }
                self.requests[idx].pending_logits = Some(logits);
                None
            }
            RequestState::Decoding => {
                self.stats.total_decode_steps += 1;
                self.update_decode_ema(elapsed_ms);

                let token = sample_fn(&logits, vocab_size);
                let req = &mut self.requests[idx];
                req.generated_tokens.push(token);
                req.pending_logits = Some(logits);

                if token == req.eos_id || req.generated_tokens.len() >= req.max_tokens {
                    req.state = RequestState::Done;
                    self.stats.total_completed += 1;
                }
                Some(token)
            }
            RequestState::Done => None,
        }
    }

    /// Get mutable reference to a request's KV caches.
    pub fn caches_mut(&mut self, id: RequestId) -> &mut Vec<StandardKVCache> {
        &mut self
            .requests
            .iter_mut()
            .find(|r| r.id == id)
            .expect("caches_mut: unknown request id")
            .kv_caches
    }

    /// Get immutable reference to a request.
    pub fn request(&self, id: RequestId) -> Option<&Request> {
        self.requests.iter().find(|r| r.id == id)
    }

    /// Current scheduler statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    // ── Internal EMA + chunk tuning ─────────────────────────────────

    fn update_decode_ema(&mut self, elapsed_ms: f64) {
        let alpha = self.config.ema_alpha;
        if self.stats.total_decode_steps <= 1 {
            self.stats.decode_latency_ema_ms = elapsed_ms;
        } else {
            self.stats.decode_latency_ema_ms =
                alpha * elapsed_ms + (1.0 - alpha) * self.stats.decode_latency_ema_ms;
        }
    }

    fn update_prefill_ema(&mut self, elapsed_ms: f64) {
        let alpha = self.config.ema_alpha;
        if self.stats.total_prefill_chunks <= 1 {
            self.stats.prefill_chunk_latency_ema_ms = elapsed_ms;
        } else {
            self.stats.prefill_chunk_latency_ema_ms =
                alpha * elapsed_ms + (1.0 - alpha) * self.stats.prefill_chunk_latency_ema_ms;
        }
        self.tune_chunk_size();
    }

    /// Dynamic chunk-size tuning with dead band to prevent oscillation.
    /// Shrink at 90% of target, grow at 50% of target.
    fn tune_chunk_size(&mut self) {
        let ema = self.stats.prefill_chunk_latency_ema_ms;
        let target = self.config.target_decode_ms;

        if ema > target * 0.9 {
            // Too slow — shrink
            let new = (self.stats.current_chunk_size / 2).max(self.config.min_chunk_size);
            self.stats.current_chunk_size = new;
        } else if ema < target * 0.5 {
            // Plenty of headroom — grow
            let new = (self.stats.current_chunk_size * 2).min(self.config.max_chunk_size);
            self.stats.current_chunk_size = new;
        }
        // Dead band [50%, 90%] of target → no change
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_kv_caches(n: usize) -> Vec<StandardKVCache> {
        (0..n).map(|_| StandardKVCache::new(4, 64)).collect()
    }

    #[test]
    fn test_single_request_lifecycle() {
        let config = SchedulerConfig {
            initial_chunk_size: 4,
            min_chunk_size: 4,
            max_chunk_size: 4,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config);

        let prompt = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let id = sched.add_request(prompt, make_kv_caches(2), 0, 5, Priority::Normal);
        assert_eq!(id, 0);

        // First action: prefill chunk [1,2,3,4]
        match sched.next_action() {
            SchedulerAction::PrefillChunk { id: rid, tokens } => {
                assert_eq!(rid, 0);
                assert_eq!(tokens, vec![1, 2, 3, 4]);
            }
            other => panic!("expected PrefillChunk, got {:?}", other),
        }

        // Complete chunk — not done yet
        let logits = vec![0.0f32; 10];
        let result = sched.complete_step(id, logits, 10, 5.0, |_, _| 99);
        assert!(result.is_none());

        // Second chunk [5,6,7,8]
        match sched.next_action() {
            SchedulerAction::PrefillChunk { tokens, .. } => {
                assert_eq!(tokens, vec![5, 6, 7, 8]);
            }
            other => panic!("expected PrefillChunk, got {:?}", other),
        }
        let result = sched.complete_step(id, vec![0.0; 10], 10, 5.0, |_, _| 99);
        assert!(result.is_none());

        // Third chunk [9,10] — last, should transition to Decoding
        match sched.next_action() {
            SchedulerAction::PrefillChunk { tokens, .. } => {
                assert_eq!(tokens, vec![9, 10]);
            }
            other => panic!("expected PrefillChunk, got {:?}", other),
        }
        let result = sched.complete_step(id, vec![0.0; 10], 10, 5.0, |_, _| 42);
        assert_eq!(result, Some(42));

        // Now in decode — should get Decode action
        match sched.next_action() {
            SchedulerAction::Decode { id: rid, token } => {
                assert_eq!(rid, 0);
                assert_eq!(token, vec![42]);
            }
            other => panic!("expected Decode, got {:?}", other),
        }
    }

    #[test]
    fn test_priority_ordering() {
        let mut sched = Scheduler::new(SchedulerConfig::default());

        // Add a background request already decoding, and a high-priority one decoding
        let id_bg = sched.add_request(vec![1], make_kv_caches(1), 0, 10, Priority::Background);
        let id_hi = sched.add_request(vec![2], make_kv_caches(1), 0, 10, Priority::High);

        // Complete prefills for both (single-token prompt = immediate decode)
        sched.complete_step(id_bg, vec![0.0; 10], 10, 1.0, |_, _| 5);
        sched.complete_step(id_hi, vec![0.0; 10], 10, 1.0, |_, _| 6);

        // High-priority decode should come first
        match sched.next_action() {
            SchedulerAction::Decode { id, .. } => assert_eq!(id, id_hi),
            other => panic!("expected high-priority Decode, got {:?}", other),
        }
    }

    #[test]
    fn test_eos_terminates() {
        let mut sched = Scheduler::new(SchedulerConfig {
            initial_chunk_size: 100,
            ..Default::default()
        });
        let eos = 999;
        let id = sched.add_request(vec![1, 2], make_kv_caches(1), eos, 10, Priority::Normal);

        // Prefill completes immediately (chunk_size > prompt length)
        sched.complete_step(id, vec![0.0; 10], 10, 1.0, |_, _| 5);

        // Decode step that returns EOS
        sched.complete_step(id, vec![0.0; 10], 10, 1.0, |_, _| eos);

        assert_eq!(sched.request(id).unwrap().state, RequestState::Done);
        match sched.next_action() {
            SchedulerAction::AllDone => {}
            other => panic!("expected AllDone, got {:?}", other),
        }
    }

    #[test]
    fn test_max_tokens_terminates() {
        let mut sched = Scheduler::new(SchedulerConfig {
            initial_chunk_size: 100,
            ..Default::default()
        });
        let id = sched.add_request(vec![1], make_kv_caches(1), 999, 2, Priority::Normal);

        // Prefill: first generated token
        sched.complete_step(id, vec![0.0; 5], 5, 1.0, |_, _| 10);
        assert_eq!(sched.request(id).unwrap().state, RequestState::Decoding);

        // Second token hits max_tokens=2
        sched.complete_step(id, vec![0.0; 5], 5, 1.0, |_, _| 11);
        assert_eq!(sched.request(id).unwrap().state, RequestState::Done);
    }

    #[test]
    fn test_chunk_size_tuning() {
        let config = SchedulerConfig {
            target_decode_ms: 40.0,
            initial_chunk_size: 256,
            min_chunk_size: 32,
            max_chunk_size: 1024,
            ema_alpha: 1.0, // instant tracking for test
        };
        let mut sched = Scheduler::new(config);

        let id = sched.add_request(
            (0..2048).collect(),
            make_kv_caches(1),
            9999,
            10,
            Priority::Normal,
        );

        // Simulate slow prefill — should shrink chunk
        sched.complete_step(id, vec![0.0; 10], 10, 50.0, |_, _| 0); // > 36ms (90%)
        assert!(sched.stats().current_chunk_size < 256);

        // Simulate fast prefill — should grow chunk
        let chunk = sched.stats().current_chunk_size;
        sched.complete_step(id, vec![0.0; 10], 10, 10.0, |_, _| 0); // < 20ms (50%)
        assert!(sched.stats().current_chunk_size > chunk);
    }

    #[test]
    fn test_all_done_vs_idle() {
        let sched = Scheduler::new(SchedulerConfig::default());
        // No requests at all → Idle
        match sched.next_action() {
            SchedulerAction::Idle => {}
            other => panic!("expected Idle, got {:?}", other),
        }

        // With a completed request → AllDone
        let mut sched2 = Scheduler::new(SchedulerConfig {
            initial_chunk_size: 100,
            ..Default::default()
        });
        let id = sched2.add_request(vec![1], make_kv_caches(1), 0, 1, Priority::Normal);
        // Prefill returns EOS as first token (eos_id=0, sampled 0) → Done
        sched2.complete_step(id, vec![1.0, 0.0], 2, 1.0, |_, _| 0);
        match sched2.next_action() {
            SchedulerAction::AllDone => {}
            other => panic!("expected AllDone, got {:?}", other),
        }
    }

    #[test]
    fn test_prefill_before_background_decode() {
        let mut sched = Scheduler::new(SchedulerConfig {
            initial_chunk_size: 2,
            ..Default::default()
        });

        // Background request that's already decoding
        let id_bg = sched.add_request(vec![1], make_kv_caches(1), 999, 10, Priority::Background);
        sched.complete_step(id_bg, vec![0.0; 5], 5, 1.0, |_, _| 10);

        // Normal request still prefilling
        let _id_norm = sched.add_request(
            vec![1, 2, 3, 4],
            make_kv_caches(1),
            999,
            10,
            Priority::Normal,
        );

        // Prefill for normal should come before background decode
        match sched.next_action() {
            SchedulerAction::PrefillChunk { id, .. } => assert_eq!(id, _id_norm),
            other => panic!("expected PrefillChunk for normal, got {:?}", other),
        }
    }
}
