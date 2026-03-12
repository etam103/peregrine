mod attention;
mod decoder;
mod moe;
mod model;
mod tokenizer;

use crate::tokenizer::Tokenizer;
use model::{GptOss, GptOssConfig};
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

fn sample_with_temperature(logits_data: &[f32], vocab_size: usize, temperature: f32) -> usize {
    let last_row = &logits_data[logits_data.len() - vocab_size..];

    if temperature <= 0.0 {
        return greedy_decode(logits_data, vocab_size);
    }

    // Apply temperature and softmax
    let scaled: Vec<f32> = last_row.iter().map(|&x| x / temperature).collect();
    let max_val = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&x| x / sum).collect();

    // Simple random sampling using system time as seed
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
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if cumsum >= r {
            return i;
        }
    }
    vocab_size - 1
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let is_small = args.iter().any(|a| a == "--small");
    let is_20b = args.iter().any(|a| a == "--20b");
    let is_quantized = args.iter().any(|a| a == "--quantized");

    // Parse --max-tokens N
    let max_tokens = args
        .windows(2)
        .find(|w| w[0] == "--max-tokens")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(32);

    // Parse --temperature T
    let temperature = args
        .windows(2)
        .find(|w| w[0] == "--temperature")
        .and_then(|w| w[1].parse::<f32>().ok())
        .unwrap_or(0.0);

    // Parse --speculative N (lookahead for speculative decoding)
    let speculative_lookahead = args
        .windows(2)
        .find(|w| w[0] == "--speculative")
        .and_then(|w| w[1].parse::<usize>().ok())
        .unwrap_or(0);

    // Parse --tokenizer PATH
    let tokenizer_path = args
        .windows(2)
        .find(|w| w[0] == "--tokenizer")
        .map(|w| w[1].clone());

    // Filter out flags and their values to get positional args
    let positional: Vec<&String> = {
        let mut result = Vec::new();
        let mut skip_next = false;
        for arg in args.iter().skip(1) {
            if skip_next {
                skip_next = false;
                continue;
            }
            if arg == "--max-tokens" || arg == "--temperature" || arg == "--tokenizer" || arg == "--speculative" {
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

    if !is_small && positional.len() < 2 {
        eprintln!("Usage: gpt_oss [--small|--20b] [--quantized] [weights.bin] \"prompt text\"");
        eprintln!("       gpt_oss --small [--quantized] \"prompt text\"");
        eprintln!("Options:");
        eprintln!("  --small              Use small test config with random weights");
        eprintln!("  --20b                Use gpt-oss-20b config (21B params, 16GB)");
        eprintln!("  --quantized          Use MXFP4 quantized MoE expert weights");
        eprintln!("  --max-tokens N       Maximum tokens to generate (default: 32)");
        eprintln!("  --temperature T      Sampling temperature, 0=greedy (default: 0)");
        eprintln!("  --tokenizer PATH     Path to tokenizer.json");
        eprintln!("  --speculative N      Speculative decoding lookahead (default: 0 = off)");
        std::process::exit(1);
    }

    if is_small && positional.is_empty() {
        eprintln!("Usage: gpt_oss --small \"prompt text\"");
        std::process::exit(1);
    }

    // Load tokenizer if available
    let tok = if let Some(ref path) = tokenizer_path {
        Some(Tokenizer::load(path))
    } else {
        let defaults = ["tokenizer.json", "weights/tokenizer.json"];
        let mut found = None;
        for p in &defaults {
            if std::path::Path::new(p).exists() {
                found = Some(Tokenizer::load(p));
                break;
            }
        }
        found
    };

    let mut config = if is_small {
        GptOssConfig::small()
    } else if is_20b {
        GptOssConfig::medium()
    } else {
        GptOssConfig::full()
    };

    // If tokenizer is loaded, match vocab size to it
    if let Some(ref tok) = tok {
        if config.vocab_size != tok.vocab_size() {
            eprintln!(
                "Adjusting vocab_size: {} -> {} (from tokenizer)",
                config.vocab_size,
                tok.vocab_size()
            );
            config.vocab_size = tok.vocab_size();
        }
    }

    eprintln!(
        "GPT-OSS config: {} layers, {} dim, {} experts (top-{}), {} q_heads, {} kv_heads, sliding_window={}",
        config.num_layers,
        config.model_dim,
        config.num_experts,
        config.top_k_experts,
        config.num_q_heads,
        config.num_kv_heads,
        config.sliding_window,
    );
    if speculative_lookahead > 0 {
        eprintln!("Speculative decoding: lookahead={}", speculative_lookahead);
    }

    let t0 = Instant::now();
    let mut model = GptOss::new(config.clone());

    // Determine if a weights file was provided (first positional arg is a file path)
    let has_weights_file = if is_small {
        // With --small, a weights file is optional; detect by checking if first positional looks like a path
        positional.len() >= 2 && (positional[0].ends_with(".bin") || positional[0].contains('/'))
    } else {
        true
    };

    if is_small && !has_weights_file && is_quantized {
        eprintln!("Initializing random MXFP4 weights (small quantized test mode)...");
        model.init_random_quantized();
    } else if is_small && !has_weights_file {
        eprintln!("Initializing random weights (small test mode)...");
        model.init_random();
    } else if is_quantized {
        let weights_path = positional[0].as_str();
        eprintln!("Loading quantized weights from {}...", weights_path);
        model.load_weights_quantized(weights_path);
    } else {
        let weights_path = positional[0].as_str();
        eprintln!("Loading weights from {}...", weights_path);
        model.load_weights(weights_path);
    }
    eprintln!("Model initialized in {:.2}s", t0.elapsed().as_secs_f64());

    // Get prompt text (always last positional arg)
    let prompt_str = positional.last().unwrap().as_str();

    // Tokenize
    let tokens: Vec<usize> = if let Some(ref tok) = tok {
        let mut ids = vec![tok.bos_id()];
        ids.extend(tok.encode(prompt_str));
        ids
    } else {
        parse_prompt_fallback(prompt_str, config.vocab_size)
    };

    if tok.is_some() {
        eprintln!("Prompt: \"{}\"", prompt_str);
        eprintln!("Tokens ({}): {:?}", tokens.len(), &tokens);
    } else {
        eprintln!("No tokenizer loaded (use --tokenizer to specify path)");
        eprintln!("Input tokens: {:?}", tokens);
    }

    // Initialize KV caches
    let mut kv_caches = model.init_kv_caches();

    // Prefill: process all prompt tokens at once
    let t_gen = Instant::now();
    let logits = model.forward(&tokens, &mut kv_caches);
    let logits_data = logits.data();
    let prefill_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    eprintln!(
        "Prefill: {} tokens in {:.1}ms ({:.1} tok/s)",
        tokens.len(),
        prefill_ms,
        tokens.len() as f64 / (prefill_ms / 1000.0)
    );

    let mut generated = Vec::new();
    let eos_id = tok.as_ref().map(|t| t.eos_id()).unwrap_or(199999);
    let mut next_token = if temperature > 0.0 {
        sample_with_temperature(&logits_data, config.vocab_size, temperature)
    } else {
        greedy_decode(&logits_data, config.vocab_size)
    };
    generated.push(next_token);

    // Print first token immediately if we have a tokenizer
    if let Some(ref tok) = tok {
        eprint!("{}", tok.decode(&[next_token]));
    }

    // Autoregressive generation
    for step in 0..max_tokens - 1 {
        if next_token == eos_id {
            break;
        }

        let step_start = Instant::now();
        let logits = model.forward(&[next_token], &mut kv_caches);
        let logits_data = logits.data();
        next_token = if temperature > 0.0 {
            sample_with_temperature(&logits_data, config.vocab_size, temperature)
        } else {
            greedy_decode(&logits_data, config.vocab_size)
        };
        generated.push(next_token);

        // Stream decoded text
        if let Some(ref tok) = tok {
            eprint!("{}", tok.decode(&[next_token]));
        }

        let step_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        if tok.is_none() && (step < 3 || (step + 1) % 10 == 0) {
            eprintln!(
                "  Step {}: token {} ({:.1}ms)",
                step + 1,
                next_token,
                step_ms
            );
        }
    }

    if tok.is_some() {
        eprintln!(); // newline after streaming
    }

    let total_ms = t_gen.elapsed().as_secs_f64() * 1000.0;
    let total_tokens = tokens.len() + generated.len();
    eprintln!(
        "Generated {} tokens in {:.1}ms ({:.1} tok/s)",
        generated.len(),
        total_ms,
        total_tokens as f64 / (total_ms / 1000.0)
    );

    // Final output
    if let Some(ref tok) = tok {
        println!("{}", tok.decode(&generated));
    } else {
        println!("Input tokens:     {:?}", tokens);
        println!("Generated tokens: {:?}", generated);
    }
}

/// Fallback prompt parsing when no tokenizer is available.
fn parse_prompt_fallback(s: &str, vocab_size: usize) -> Vec<usize> {
    let parts: Vec<&str> = s.split_whitespace().collect();
    let parsed: Vec<Option<usize>> = parts.iter().map(|p| p.parse::<usize>().ok()).collect();
    if !parts.is_empty() && parsed.iter().all(|p| p.is_some()) {
        let ids: Vec<usize> = parsed.into_iter().map(|p| p.unwrap()).collect();
        if ids.iter().all(|&id| id < vocab_size) {
            return ids;
        }
    }
    s.bytes().map(|b| b as usize % vocab_size).collect()
}
