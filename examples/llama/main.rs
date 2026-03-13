mod attention;
mod decoder;
mod model;
mod tokenizer;
mod wandb;

use model::Llama;
use tokenizer::Tokenizer;
use peregrine::thermal::{thermal_init, thermal_state, ThermalState};
use std::env;
use std::time::{Duration, Instant};

fn greedy_decode(logits_data: &[f32], vocab_size: usize) -> usize {
    let last_row = &logits_data[logits_data.len() - vocab_size..];
    let mut best_idx = 0;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in last_row.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx
}

fn sample_with_temperature(logits_data: &[f32], vocab_size: usize, temperature: f32, top_p: f32) -> usize {
    let last_row = &logits_data[logits_data.len() - vocab_size..];

    if temperature <= 0.0 {
        return greedy_decode(logits_data, vocab_size);
    }

    // Apply temperature and softmax
    let scaled: Vec<f32> = last_row.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let mut probs: Vec<(usize, f32)> = exps.iter().enumerate().map(|(i, &x)| (i, x / sum)).collect();

    // Top-p (nucleus) sampling
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
        // Renormalize
        let new_sum: f32 = probs.iter().map(|&(_, p)| p).sum();
        for item in probs.iter_mut() {
            item.1 /= new_sum;
        }
    }

    // Random sampling using system time as seed
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

/// Per-token record for sustained profiling.
#[allow(dead_code)]
struct TokenRecord {
    elapsed_s: f64,
    latency_ms: f64,
    thermal_state: ThermalState,
    tokens_generated: usize,
}

/// Run one decode pass: prefill + autoregressive decode up to max_tokens.
/// Returns the generated token count.
fn run_one_pass(
    model: &Llama,
    tok: &Tokenizer,
    prompt_tokens: &[usize],
    max_tokens: usize,
    temperature: f32,
    top_p: f32,
    records: &mut Vec<TokenRecord>,
    start_time: Instant,
) -> usize {
    let mut kv_caches = model.init_kv_caches();
    let vocab_size = model.config.vocab_size;
    let eos_id = tok.eos_id;

    // Prefill
    let logits = model.forward(prompt_tokens, &mut kv_caches);
    let logits_data = logits.data();
    let mut next_token = sample_with_temperature(&logits_data, vocab_size, temperature, top_p);
    let mut count = 1;

    // Autoregressive decode
    for _ in 0..max_tokens - 1 {
        if next_token == eos_id {
            break;
        }

        let t0 = Instant::now();
        let logits = model.forward(&[next_token], &mut kv_caches);
        let latency_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let logits_data = logits.data();
        next_token = sample_with_temperature(&logits_data, vocab_size, temperature, top_p);
        count += 1;

        records.push(TokenRecord {
            elapsed_s: start_time.elapsed().as_secs_f64(),
            latency_ms,
            thermal_state: thermal_state(),
            tokens_generated: records.len() + 1,
        });

        if next_token == eos_id {
            break;
        }
    }

    count
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_sustained_stats(records: &[TokenRecord], total_secs: f64) {
    if records.is_empty() {
        eprintln!("No token records collected.");
        return;
    }

    let mut latencies: Vec<f64> = records.iter().map(|r| r.latency_ms).collect();
    latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p50 = percentile(&latencies, 50.0);
    let p95 = percentile(&latencies, 95.0);
    let p99 = percentile(&latencies, 99.0);
    let total_tokens = records.len();
    let throughput = total_tokens as f64 / total_secs;

    eprintln!("\n--- Sustained Stats ({:.1}s) ---", total_secs);
    eprintln!("  Total tokens:   {}", total_tokens);
    eprintln!("  Throughput:     {:.1} tok/s", throughput);
    eprintln!("  Latency p50:    {:.2}ms", p50);
    eprintln!("  Latency p95:    {:.2}ms", p95);
    eprintln!("  Latency p99:    {:.2}ms", p99);

    // Throughput grouped by thermal state
    let states = [
        ThermalState::Nominal,
        ThermalState::Moderate,
        ThermalState::Heavy,
        ThermalState::Trapping,
        ThermalState::Sleeping,
    ];

    eprintln!("  Thermal distribution:");
    for &state in &states {
        let state_records: Vec<&TokenRecord> =
            records.iter().filter(|r| r.thermal_state == state).collect();
        if state_records.is_empty() {
            continue;
        }
        let count = state_records.len();
        let pct = count as f64 / total_tokens as f64 * 100.0;
        let mut state_latencies: Vec<f64> = state_records.iter().map(|r| r.latency_ms).collect();
        state_latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let state_p50 = percentile(&state_latencies, 50.0);
        let avg_tps = 1000.0 / state_p50;
        eprintln!(
            "    {:>10}: {:>5.1}% ({:>5} tokens, p50={:.2}ms, ~{:.1} tok/s)",
            state.as_str(),
            pct,
            count,
            state_p50,
            avg_tps
        );
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse flags
    let max_tokens = args
        .windows(2)
        .find(|w| w[0] == "--max-tokens")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(128);

    let temperature = args
        .windows(2)
        .find(|w| w[0] == "--temperature")
        .and_then(|w| w[1].parse::<f32>().ok())
        .unwrap_or(0.0);

    let top_p = args
        .windows(2)
        .find(|w| w[0] == "--top-p")
        .and_then(|w| w[1].parse::<f32>().ok())
        .unwrap_or(0.9);

    let sustained_secs = args
        .windows(2)
        .find(|w| w[0] == "--sustained")
        .and_then(|w| w[1].parse::<u64>().ok());

    let use_wandb = args.iter().any(|a| a == "--wandb");

    #[allow(unused_variables)]
    let use_gpu = args.iter().any(|a| a == "--gpu");

    // Filter out flags to get positional args: <model.gguf> <prompt>
    let positional: Vec<&String> = {
        let mut result = Vec::new();
        let mut skip_next = false;
        for arg in args.iter().skip(1) {
            if skip_next {
                skip_next = false;
                continue;
            }
            if arg == "--max-tokens"
                || arg == "--temperature"
                || arg == "--top-p"
                || arg == "--sustained"
            {
                skip_next = true;
                continue;
            }
            if arg.starts_with("--") {
                continue;
            }
            result.push(arg);
        }
        result
    };

    if positional.len() < 2 {
        eprintln!("Usage: llama [--gpu] [--temperature T] [--top-p P] [--max-tokens N] [--sustained SECS] [--wandb] <model.gguf> <prompt>");
        std::process::exit(1);
    }

    let model_path = positional[0].as_str();
    let prompt_str = positional[1].as_str();

    // Initialize thermal monitoring
    if let Err(e) = thermal_init() {
        eprintln!("Warning: thermal monitoring unavailable: {}", e);
    }

    // Load model from GGUF
    let t0 = Instant::now();
    let (model, gguf) = Llama::from_gguf(model_path);
    let load_secs = t0.elapsed().as_secs_f64();
    eprintln!("Model loaded in {:.2}s", load_secs);

    // Load tokenizer from GGUF embedded vocab
    let tok = Tokenizer::from_gguf(&gguf);
    eprintln!("Tokenizer: {} vocab, BOS={}, EOS={}", tok.vocab_size(), tok.bos_id, tok.eos_id);

    // Tokenize prompt
    let mut tokens: Vec<usize> = vec![tok.bos_id];
    tokens.extend(tok.encode(prompt_str));
    eprintln!("Prompt: \"{}\"", prompt_str);
    eprintln!("Tokens ({}): {:?}", tokens.len(), &tokens[..tokens.len().min(32)]);

    // --- Sustained mode ---
    if let Some(duration_secs) = sustained_secs {
        eprintln!("Sustained mode: running for {}s", duration_secs);
        let duration = Duration::from_secs(duration_secs);
        let mut records = Vec::new();
        let start = Instant::now();
        let mut pass = 0u64;

        let mut wb = if use_wandb {
            Some(wandb::WandbRun::init("peregrine-llama"))
        } else {
            None
        };

        while start.elapsed() < duration {
            pass += 1;
            let before = records.len();
            run_one_pass(
                &model,
                &tok,
                &tokens,
                max_tokens,
                temperature,
                top_p,
                &mut records,
                start,
            );
            let after = records.len();

            // Log to wandb
            if let Some(ref mut wb) = wb {
                for r in &records[before..after] {
                    let tok_per_s = 1000.0 / r.latency_ms;
                    wb.log_metrics(r.tokens_generated, &[
                        ("decode/latency_ms", r.latency_ms as f32),
                        ("decode/tok_per_s", tok_per_s as f32),
                        ("thermal/state", r.thermal_state as u8 as f32),
                    ]);
                }
            }

            eprintln!(
                "  pass {}: {} tokens, thermal={}",
                pass,
                after - before,
                thermal_state()
            );
        }

        let total_secs = start.elapsed().as_secs_f64();
        print_sustained_stats(&records, total_secs);

        if let Some(ref mut wb) = wb {
            wb.finish();
        }

        return;
    }

    // --- Normal (single-pass) mode ---
    let mut kv_caches = model.init_kv_caches();

    // Prefill: process all prompt tokens at once
    let t_gen = Instant::now();
    let logits = model.forward(&tokens, &mut kv_caches);
    let logits_data = logits.data();
    let prefill_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    let ttft = prefill_ms;
    eprintln!(
        "Prefill: {} tokens in {:.1}ms ({:.1} tok/s) — TTFT={:.1}ms",
        tokens.len(),
        prefill_ms,
        tokens.len() as f64 / (prefill_ms / 1000.0),
        ttft
    );

    let vocab_size = model.config.vocab_size;
    let mut generated = Vec::new();
    let eos_id = tok.eos_id;
    let mut next_token = sample_with_temperature(&logits_data, vocab_size, temperature, top_p);
    generated.push(next_token);

    // Print first token
    eprint!("{}", tok.decode(&[next_token]));

    // Autoregressive decode
    let decode_start = Instant::now();
    for _step in 0..max_tokens - 1 {
        if next_token == eos_id {
            break;
        }

        let logits = model.forward(&[next_token], &mut kv_caches);
        let logits_data = logits.data();
        next_token = sample_with_temperature(&logits_data, vocab_size, temperature, top_p);
        generated.push(next_token);

        // Stream decoded text
        eprint!("{}", tok.decode(&[next_token]));

        if next_token == eos_id {
            break;
        }
    }
    eprintln!(); // newline after streaming

    let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    let total_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    let decode_tokens = generated.len().saturating_sub(1); // first token is from prefill

    eprintln!("--- Stats ---");
    eprintln!("  TTFT:           {:.1}ms", ttft);
    eprintln!("  Decode:         {} tokens in {:.1}ms ({:.1} tok/s)",
        decode_tokens,
        decode_ms,
        if decode_ms > 0.0 { decode_tokens as f64 / (decode_ms / 1000.0) } else { 0.0 }
    );
    eprintln!("  Total:          {} tokens in {:.1}ms ({:.1} tok/s)",
        tokens.len() + generated.len(),
        total_ms,
        (tokens.len() + generated.len()) as f64 / (total_ms / 1000.0)
    );

    // Final output
    println!("{}", tok.decode(&generated));
}
