//! Recursive Language Model (RLM) example using the Llama backend.
//!
//! Usage:
//!   cargo run --example rlm --release -- <model.gguf> --task "question" --context-file input.txt
//!
//! Options:
//!   --task <TEXT>           Task/question to answer about the context
//!   --context-file <PATH>  File whose contents become the input context
//!   --context <TEXT>        Inline input context (alternative to --context-file)
//!   --max-depth <N>        Maximum recursion depth (default: 4)
//!   --max-tokens <N>       Token budget across all LM calls (default: 100000)
//!   --temperature <F>      Sampling temperature (default: 0.7)
//!   --top-p <F>            Nucleus sampling threshold (default: 0.9)

mod model;
mod tokenizer;
mod attention;
mod decoder;

use model::Llama;
use tokenizer::Tokenizer;
use peregrine::rlm::{GenerativeLM, GenerateConfig, RlmConfig, RlmOrchestrator};
use std::cell::RefCell;
use std::env;
use std::fs;
use std::time::Instant;

// ---------------------------------------------------------------------------
// Llama-backed GenerativeLM implementation
// ---------------------------------------------------------------------------

struct LlamaLM {
    model: Llama,
    tokenizer: Tokenizer,
    kv_caches: RefCell<Vec<crate::attention::KVCache>>,
    context_len: usize,
    vocab_size: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LlamaLM {
    fn new(
        model: Llama, tokenizer: Tokenizer,
        num_layers: usize, num_kv_heads: usize, head_dim: usize,
        context_len: usize, vocab_size: usize,
    ) -> Self {
        let kv_caches: Vec<crate::attention::KVCache> = (0..num_layers)
            .map(|_| crate::attention::KVCache::new(num_kv_heads, head_dim))
            .collect();
        LlamaLM {
            model,
            tokenizer,
            kv_caches: RefCell::new(kv_caches),
            context_len,
            vocab_size,
            num_kv_heads,
            head_dim,
        }
    }

    fn reset_kv(&self) {
        let mut caches = self.kv_caches.borrow_mut();
        for c in caches.iter_mut() {
            *c = crate::attention::KVCache::new(self.num_kv_heads, self.head_dim);
        }
    }
}

fn sample_token(logits_data: &[f32], vocab_size: usize, temperature: f32, top_p: f32) -> usize {
    let last_row = &logits_data[logits_data.len() - vocab_size..];

    if temperature <= 0.0 {
        // Greedy
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in last_row.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx;
    }

    let scaled: Vec<f32> = last_row.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut probs: Vec<(usize, f32)> = exps.iter().enumerate().map(|(i, &x)| (i, x / sum)).collect();

    if top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cumsum = 0.0;
        let mut cutoff = probs.len();
        for (i, &(_, p)) in probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= top_p {
                cutoff = i + 1;
                break;
            }
        }
        probs.truncate(cutoff);
        let new_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
        for item in probs.iter_mut() {
            item.1 /= new_sum;
        }
    }

    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    let mut rng = seed;
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    let r = (rng as f32) / (u32::MAX as f32);

    let mut cumsum = 0.0;
    for &(idx, p) in &probs {
        cumsum += p;
        if cumsum >= r {
            return idx;
        }
    }
    probs.last().map(|&(idx, _)| idx).unwrap_or(0)
}

impl GenerativeLM for LlamaLM {
    fn generate(&self, prompt: &str, max_tokens: usize, config: &GenerateConfig) -> String {
        self.generate_counted(prompt, max_tokens, config).0
    }

    fn generate_counted(&self, prompt: &str, max_tokens: usize, config: &GenerateConfig) -> (String, usize) {
        self.reset_kv();
        let tokens = self.tokenizer.encode(prompt);
        let mut all_tokens = tokens.clone();
        let mut generated = Vec::new();
        let mut total_tokens = tokens.len();

        // Prefill
        let mut caches = self.kv_caches.borrow_mut();
        let logits = self.model.forward(&all_tokens, &mut caches);
        let logits_data = logits.data();
        let tok = sample_token(&logits_data, self.vocab_size, config.temperature, config.top_p);
        generated.push(tok);
        all_tokens.push(tok);
        total_tokens += 1;

        // Decode
        for _ in 1..max_tokens {
            let logits = self.model.forward(&[*all_tokens.last().unwrap()], &mut caches);
            let logits_data = logits.data();
            let tok = sample_token(&logits_data, self.vocab_size, config.temperature, config.top_p);

            let text = self.tokenizer.decode(&[tok]);
            // Check stop sequences
            let so_far = self.tokenizer.decode(&generated);
            let combined = format!("{}{}", so_far, text);
            let mut should_stop = false;
            for stop in &config.stop_sequences {
                if combined.contains(stop.as_str()) {
                    should_stop = true;
                    break;
                }
            }

            generated.push(tok);
            all_tokens.push(tok);
            total_tokens += 1;

            if should_stop {
                break;
            }
            // EOS
            if tok == 128001 || tok == 128009 {
                break;
            }
        }

        let text = self.tokenizer.decode(&generated);
        (text, total_tokens)
    }

    fn context_window(&self) -> usize {
        self.context_len
    }

    fn count_tokens(&self, text: &str) -> usize {
        self.tokenizer.encode(text).len()
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn print_usage() {
    eprintln!("Usage: rlm <model.gguf> --task <TEXT> [--context-file <PATH> | --context <TEXT>]");
    eprintln!("       [--max-depth N] [--max-tokens N] [--temperature F] [--top-p F]");
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let model_path = &args[1];
    let mut task = String::new();
    let mut context_text = String::new();
    let mut max_depth = 4usize;
    let mut max_tokens = 100_000usize;
    let mut temperature = 0.7f32;
    let mut top_p = 0.9f32;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--task" => { i += 1; task = args[i].clone(); }
            "--context-file" => {
                i += 1;
                context_text = fs::read_to_string(&args[i])
                    .unwrap_or_else(|e| { eprintln!("Error reading {}: {}", args[i], e); std::process::exit(1); });
            }
            "--context" => { i += 1; context_text = args[i].clone(); }
            "--max-depth" => { i += 1; max_depth = args[i].parse().unwrap(); }
            "--max-tokens" => { i += 1; max_tokens = args[i].parse().unwrap(); }
            "--temperature" => { i += 1; temperature = args[i].parse().unwrap(); }
            "--top-p" => { i += 1; top_p = args[i].parse().unwrap(); }
            other => { eprintln!("Unknown argument: {}", other); print_usage(); std::process::exit(1); }
        }
        i += 1;
    }

    if task.is_empty() {
        eprintln!("Error: --task is required");
        print_usage();
        std::process::exit(1);
    }

    // Load model
    eprintln!("Loading model from {} ...", model_path);
    let t0 = Instant::now();
    let (model, gguf) = Llama::from_gguf(model_path);
    let config = crate::model::LlamaConfig::from_gguf(&gguf);
    let tokenizer = Tokenizer::from_gguf(&gguf);
    let num_layers = config.num_layers;
    let context_len = config.max_seq_len;
    let vocab_size = config.vocab_size;
    eprintln!("  {} layers, dim={}, vocab={}, ctx={}", num_layers, config.model_dim, vocab_size, context_len);
    eprintln!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let lm = LlamaLM::new(model, tokenizer, num_layers, num_kv_heads, head_dim, context_len, vocab_size);

    // Configure RLM
    let rlm_config = RlmConfig::default()
        .max_depth(max_depth)
        .max_total_tokens(max_tokens)
        .temperature(temperature)
        .top_p(top_p);

    let mut orchestrator = RlmOrchestrator::new(&lm, rlm_config);

    eprintln!("Running RLM: task=\"{}\" context_len={} chars", task, context_text.len());
    let t1 = Instant::now();
    match orchestrator.run(&context_text, &task) {
        Ok(answer) => {
            println!("{}", answer);
            let stats = orchestrator.stats();
            eprintln!("\n--- RLM Stats ---");
            eprintln!("  Total tokens: {}", stats.total_tokens);
            eprintln!("  LM calls:     {}", stats.call_count);
            eprintln!("  Max depth:     {}", stats.max_depth_reached);
            eprintln!("  Wall time:     {:.1}s", t1.elapsed().as_secs_f32());
        }
        Err(e) => {
            eprintln!("RLM error: {}", e);
            std::process::exit(1);
        }
    }
}
