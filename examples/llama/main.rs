mod attention;
mod decoder;
mod model;
mod tokenizer;

use model::Llama;
use tokenizer::Tokenizer;
use std::env;
use std::time::Instant;

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
            if arg == "--max-tokens" || arg == "--temperature" || arg == "--top-p" {
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
        eprintln!("Usage: llama [--gpu] [--temperature T] [--top-p P] [--max-tokens N] <model.gguf> <prompt>");
        std::process::exit(1);
    }

    let model_path = positional[0].as_str();
    let prompt_str = positional[1].as_str();

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

    // Initialize KV caches
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
