use pyo3::prelude::*;
use peregrine::models::llama::{Llama, KVCache, Tokenizer};
use std::path::Path;

/// A streaming token iterator returned by TextGenerator.generate().
#[pyclass(unsendable)]
pub struct TokenIterator {
    model: *const Llama,
    tokenizer: *const Tokenizer,
    kv_caches: *mut Vec<KVCache>,
    next_token: usize,
    eos_id: usize,
    vocab_size: usize,
    remaining: usize,
    temperature: f32,
    top_p: f32,
    done: bool,
}

// Safety: TokenIterator is unsendable and only used on the creating thread.
// The raw pointers reference data owned by the TextGenerator which must outlive the iterator.

#[pymethods]
impl TokenIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self) -> Option<String> {
        if self.done || self.remaining == 0 {
            return None;
        }

        if self.next_token == self.eos_id {
            self.done = true;
            return None;
        }

        // Safety: pointers are valid for the lifetime of the TextGenerator
        let model = unsafe { &*self.model };
        let tokenizer = unsafe { &*self.tokenizer };
        let kv_caches = unsafe { &mut *self.kv_caches };

        let logits = model.forward(&[self.next_token], kv_caches);
        let logits_data = logits.data();
        self.next_token = sample(&logits_data, self.vocab_size, self.temperature, self.top_p);
        self.remaining -= 1;

        if self.next_token == self.eos_id {
            self.done = true;
            return None;
        }

        Some(tokenizer.decode(&[self.next_token]))
    }
}

/// High-level text generation interface for Llama models.
#[pyclass(unsendable)]
pub struct TextGenerator {
    model: Llama,
    tokenizer: Tokenizer,
    kv_caches: Vec<KVCache>,
}

#[pymethods]
impl TextGenerator {
    /// Generate tokens from a prompt. Returns a streaming iterator.
    #[pyo3(signature = (prompt, max_tokens=128, temperature=0.0, top_p=0.9))]
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> TokenIterator {
        // Reset KV caches for new generation
        self.kv_caches = self.model.init_kv_caches();

        let mut tokens: Vec<usize> = vec![self.tokenizer.bos_id];
        tokens.extend(self.tokenizer.encode(prompt));

        let vocab_size = self.model.config.vocab_size;
        let eos_id = self.tokenizer.eos_id;

        // Prefill: process all prompt tokens
        let logits = self.model.forward(&tokens, &mut self.kv_caches);
        let logits_data = logits.data();
        let first_token = sample(&logits_data, vocab_size, temperature, top_p);

        TokenIterator {
            model: &self.model as *const Llama,
            tokenizer: &self.tokenizer as *const Tokenizer,
            kv_caches: &mut self.kv_caches as *mut Vec<KVCache>,
            next_token: first_token,
            eos_id,
            vocab_size,
            remaining: max_tokens,
            temperature,
            top_p,
            done: false,
        }
    }

    /// Encode text to token IDs.
    fn encode(&self, text: &str) -> Vec<usize> {
        self.tokenizer.encode(text)
    }

    /// Decode token IDs to text.
    fn decode(&self, token_ids: Vec<usize>) -> String {
        self.tokenizer.decode(&token_ids)
    }

    fn __repr__(&self) -> String {
        let c = &self.model.config;
        format!(
            "peregrine.TextGenerator(layers={}, dim={}, vocab={}, heads={}/{})",
            c.num_layers, c.model_dim, c.vocab_size, c.num_q_heads, c.num_kv_heads
        )
    }
}

/// Load a model and return a TextGenerator.
/// Accepts a GGUF file path, a safetensors directory, or an HF repo "org/repo".
#[pyfunction]
#[pyo3(signature = (model_path, quantize=None))]
pub fn load_model(model_path: &str, quantize: Option<&str>) -> PyResult<TextGenerator> {
    let path = Path::new(model_path);

    let (model, tokenizer) = if model_path.ends_with(".gguf") {
        let (model, gguf) = Llama::from_gguf(model_path);
        let tok = Tokenizer::from_gguf(&gguf);
        (model, tok)
    } else if path.is_dir() && path.join("config.json").exists() {
        let model = Llama::from_safetensors(model_path);
        let tok_path = path.join("tokenizer.json");
        let tok = if tok_path.exists() {
            let json = std::fs::read_to_string(&tok_path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
            Tokenizer::from_hf_json(&json)
        } else {
            return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                "tokenizer.json not found in model directory",
            ));
        };
        (model, tok)
    } else if model_path.contains('/') && !path.exists() {
        // HF Hub download
        #[cfg(feature = "hf")]
        {
            let repo = peregrine::hf_hub::HfRepo::new(model_path)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
            let dir = peregrine::hf_hub::ensure_model(&repo)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
            let dir_str = dir.to_str().unwrap();
            let model = Llama::from_safetensors(dir_str);
            let tok_path = dir.join("tokenizer.json");
            let tok = if tok_path.exists() {
                let json = std::fs::read_to_string(&tok_path)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{}", e)))?;
                Tokenizer::from_hf_json(&json)
            } else {
                return Err(pyo3::exceptions::PyFileNotFoundError::new_err(
                    "tokenizer.json not found after download",
                ));
            };
            (model, tok)
        }
        #[cfg(not(feature = "hf"))]
        {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "HF Hub support not compiled. Rebuild with --features hf",
            ));
        }
    } else {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Cannot determine model format for '{}'. Expected .gguf file, directory with config.json, or org/repo.",
            model_path
        )));
    };

    let _ = quantize; // TODO: implement int8 quantization on load

    let kv_caches = model.init_kv_caches();

    Ok(TextGenerator {
        model,
        tokenizer,
        kv_caches,
    })
}

/// Greedy or temperature sampling from logits.
fn sample(logits_data: &[f32], vocab_size: usize, temperature: f32, top_p: f32) -> usize {
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

    // Temperature + top-p sampling
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

    // Simple random sampling using system time
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
