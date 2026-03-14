use crate::gguf::GgufFile;
use crate::nn::{Embedding, RMSNorm};
use crate::safetensors::SafetensorsFile;
use crate::hf_config::ModelConfig;
use crate::tensor::Tensor;

use super::attention::KVCache;
use super::decoder::LlamaBlock;

use std::collections::HashMap;
use std::path::Path;

/// Llama model configuration — extracted from GGUF metadata.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub rope_base: f32,
    pub rms_eps: f32,
    pub max_seq_len: usize,
}

impl LlamaConfig {
    /// Extract config from GGUF metadata keys.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let model_dim = gguf.get_u32("llama.embedding_length").expect("missing llama.embedding_length") as usize;
        let num_layers = gguf.get_u32("llama.block_count").expect("missing llama.block_count") as usize;
        let num_q_heads = gguf.get_u32("llama.attention.head_count").expect("missing llama.attention.head_count") as usize;
        let num_kv_heads = gguf.get_u32("llama.attention.head_count_kv").unwrap_or(num_q_heads as u32) as usize;
        let ffn_dim = gguf.get_u32("llama.feed_forward_length").expect("missing llama.feed_forward_length") as usize;
        let rope_base = gguf.get_f32("llama.rope.freq_base").unwrap_or(500000.0);
        let rms_eps = gguf.get_f32("llama.attention.layer_norm_rms_epsilon").unwrap_or(1e-5);
        let max_seq_len = gguf.get_u32("llama.context_length").unwrap_or(8192) as usize;
        let head_dim = model_dim / num_q_heads;

        // Vocab size from tokenizer metadata
        let vocab_size = gguf
            .get_metadata("tokenizer.ggml.tokens")
            .map(|v| match v {
                crate::gguf::MetadataValue::Array(arr) => arr.len(),
                _ => 128256,
            })
            .unwrap_or(128256);

        LlamaConfig {
            vocab_size,
            model_dim,
            num_layers,
            num_q_heads,
            num_kv_heads,
            head_dim,
            ffn_dim,
            rope_base,
            rms_eps,
            max_seq_len,
        }
    }
}

/// Top-level Llama model.
pub struct Llama {
    pub embedding: Embedding,
    pub layers: Vec<LlamaBlock>,
    pub final_norm: RMSNorm,
    pub output_weight: Tensor, // [model_dim, vocab_size]
    pub config: LlamaConfig,
}

impl Llama {
    pub fn new(config: LlamaConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(LlamaBlock::new(
                config.model_dim,
                config.num_q_heads,
                config.num_kv_heads,
                config.head_dim,
                config.ffn_dim,
                config.max_seq_len,
                config.rope_base,
                config.rms_eps,
            ));
        }

        Llama {
            embedding: Embedding::new(config.vocab_size, config.model_dim),
            layers,
            final_norm: RMSNorm::new(config.model_dim, config.rms_eps),
            output_weight: Tensor::zeros(&[config.model_dim, config.vocab_size], false),
            config,
        }
    }

    /// Load a Llama model from a GGUF file.
    pub fn from_gguf(path: &str) -> (Self, GgufFile) {
        eprintln!("Parsing GGUF file: {}", path);
        let gguf = GgufFile::load(path).expect("failed to parse GGUF file");

        let config = LlamaConfig::from_gguf(&gguf);
        eprintln!(
            "Llama config: {} layers, {} dim, {} q_heads, {} kv_heads, head_dim={}, ffn={}, vocab={}",
            config.num_layers, config.model_dim, config.num_q_heads,
            config.num_kv_heads, config.head_dim, config.ffn_dim, config.vocab_size
        );

        let mut model = Llama::new(config.clone());

        // Load embedding: GGUF name = "token_embd.weight", shape [vocab, dim]
        // GGUF stores as [dim, vocab] (reversed), so we get [vocab, dim] after reversal
        let emb_data = gguf.load_tensor_f32("token_embd.weight");
        let emb_shape = gguf.tensor_shape("token_embd.weight"); // [vocab, dim]
        model.embedding.weight = Tensor::new(emb_data, emb_shape, false);

        // Load output projection: "output.weight" [vocab, dim] → we need [dim, vocab] for x @ W
        if gguf.tensors.contains_key("output.weight") {
            let out_data = gguf.load_tensor_f32("output.weight");
            let out_shape = gguf.tensor_shape("output.weight"); // [vocab, dim]
            // Transpose: [vocab, dim] → [dim, vocab]
            let vocab = out_shape[0];
            let dim = out_shape[1];
            let mut transposed = vec![0.0f32; dim * vocab];
            for v in 0..vocab {
                for d in 0..dim {
                    transposed[d * vocab + v] = out_data[v * dim + d];
                }
            }
            model.output_weight = Tensor::new(transposed, vec![dim, vocab], false);
        } else {
            // Tied weights: use embedding
            let emb_data = model.embedding.weight.data();
            let vocab = config.vocab_size;
            let dim = config.model_dim;
            let mut transposed = vec![0.0f32; dim * vocab];
            for v in 0..vocab {
                for d in 0..dim {
                    transposed[d * vocab + v] = emb_data[v * dim + d];
                }
            }
            model.output_weight = Tensor::new(transposed, vec![dim, vocab], false);
        }

        // Final norm: "output_norm.weight"
        let norm_data = gguf.load_tensor_f32("output_norm.weight");
        model.final_norm.weight = Tensor::new(norm_data.clone(), vec![norm_data.len()], false);

        // Load each layer
        for i in 0..config.num_layers {
            let layer = &mut model.layers[i];

            // Attention norms
            let attn_norm_data = gguf.load_tensor_f32(&format!("blk.{}.attn_norm.weight", i));
            layer.attn_norm.weight = Tensor::new(attn_norm_data.clone(), vec![attn_norm_data.len()], false);

            let ffn_norm_data = gguf.load_tensor_f32(&format!("blk.{}.ffn_norm.weight", i));
            layer.ffn_norm.weight = Tensor::new(ffn_norm_data.clone(), vec![ffn_norm_data.len()], false);

            // Attention projections: GGUF [out, in] → Peregrine [in, out]
            // Q: GGUF "blk.{i}.attn_q.weight" [n_q_heads*head_dim, model_dim]
            //    → Peregrine [model_dim, n_q_heads*head_dim]
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.attn_q.weight", i),
                &mut layer.attention.q_proj,
            );
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.attn_k.weight", i),
                &mut layer.attention.k_proj,
            );
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.attn_v.weight", i),
                &mut layer.attention.v_proj,
            );
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.attn_output.weight", i),
                &mut layer.attention.o_proj,
            );

            // SwiGLU FFN projections
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.ffn_gate.weight", i),
                &mut layer.gate_proj,
            );
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.ffn_up.weight", i),
                &mut layer.up_proj,
            );
            load_transposed_weight(
                &gguf,
                &format!("blk.{}.ffn_down.weight", i),
                &mut layer.down_proj,
            );

            if i == 0 || (i + 1) % 4 == 0 || i + 1 == config.num_layers {
                eprintln!("  Loaded layer {}/{}", i + 1, config.num_layers);
            }
        }

        (model, gguf)
    }

    /// Initialize KV caches for all layers.
    pub fn init_kv_caches(&self) -> Vec<KVCache> {
        (0..self.config.num_layers)
            .map(|_| KVCache::new(self.config.num_kv_heads, self.config.head_dim))
            .collect()
    }

    /// Forward pass: tokens → logits [seq_len, vocab_size].
    pub fn forward(&self, tokens: &[usize], kv_caches: &mut [KVCache]) -> Tensor {
        // Embed tokens
        let mut x = self.embedding.forward(tokens);

        // Decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mut kv_caches[i]);
        }

        // Final norm
        x = self.final_norm.forward(&x);

        // Output projection: [seq_len, model_dim] @ [model_dim, vocab_size]
        x.matmul(&self.output_weight)
    }
}

/// Load a GGUF weight tensor and store transposed into a Peregrine Tensor.
/// GGUF linear weights are [out_features, in_features] (row-major).
/// Peregrine convention is [in_features, out_features].
fn load_transposed_weight(gguf: &GgufFile, name: &str, target: &mut Tensor) {
    let data = gguf.load_tensor_f32(name);
    let shape = gguf.tensor_shape(name); // reversed: [out, in] in Peregrine ordering
    assert_eq!(shape.len(), 2, "expected 2D weight for {}", name);
    let (rows, cols) = (shape[0], shape[1]);
    // data is [rows, cols] row-major = [out, in]
    // transpose to [in, out] = [cols, rows]
    let mut transposed = vec![0.0f32; cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    *target = Tensor::new(transposed, vec![cols, rows], false);
}

/// Map HuggingFace weight name to Peregrine's internal naming convention.
/// Returns (mapped_name, is_2d_weight) — 2D weights need transposing.
fn map_hf_name(hf_name: &str) -> Option<(String, bool)> {
    // Strip "model." prefix if present
    let name = hf_name.strip_prefix("model.").unwrap_or(hf_name);

    // Embedding
    if name == "embed_tokens.weight" {
        return Some(("embed_tokens.weight".to_string(), false));
    }

    // LM head
    if hf_name == "lm_head.weight" {
        return Some(("lm_head.weight".to_string(), true));
    }

    // Final norm
    if name == "norm.weight" {
        return Some(("norm.weight".to_string(), false));
    }

    // Layer weights: "layers.{i}.self_attn.{q,k,v,o}_proj.weight"
    //                "layers.{i}.mlp.{gate,up,down}_proj.weight"
    //                "layers.{i}.input_layernorm.weight"
    //                "layers.{i}.post_attention_layernorm.weight"
    if let Some(rest) = name.strip_prefix("layers.") {
        // Extract layer index
        let dot = rest.find('.')?;
        let layer_idx = &rest[..dot];
        let suffix = &rest[dot + 1..];

        let (mapped, is_2d) = match suffix {
            "self_attn.q_proj.weight" => (format!("blk.{}.attn_q.weight", layer_idx), true),
            "self_attn.k_proj.weight" => (format!("blk.{}.attn_k.weight", layer_idx), true),
            "self_attn.v_proj.weight" => (format!("blk.{}.attn_v.weight", layer_idx), true),
            "self_attn.o_proj.weight" => (format!("blk.{}.attn_output.weight", layer_idx), true),
            "mlp.gate_proj.weight" => (format!("blk.{}.ffn_gate.weight", layer_idx), true),
            "mlp.up_proj.weight" => (format!("blk.{}.ffn_up.weight", layer_idx), true),
            "mlp.down_proj.weight" => (format!("blk.{}.ffn_down.weight", layer_idx), true),
            "input_layernorm.weight" => (format!("blk.{}.attn_norm.weight", layer_idx), false),
            "post_attention_layernorm.weight" => (format!("blk.{}.ffn_norm.weight", layer_idx), false),
            _ => return None,
        };
        return Some((mapped, is_2d));
    }

    None
}

impl Llama {
    /// Load a Llama model from a directory of safetensors files (HuggingFace format).
    /// The directory should contain config.json and one or more .safetensors files.
    pub fn from_safetensors(dir: &str) -> Self {
        let dir_path = Path::new(dir);

        // Load config.json
        let config_path = dir_path.join("config.json");
        let config_json = std::fs::read_to_string(&config_path)
            .unwrap_or_else(|e| panic!("failed to read config.json at {:?}: {}", config_path, e));
        let hf_config = ModelConfig::from_json(&config_json);

        let config = LlamaConfig {
            vocab_size: hf_config.vocab_size,
            model_dim: hf_config.hidden_size,
            num_layers: hf_config.num_hidden_layers,
            num_q_heads: hf_config.num_attention_heads,
            num_kv_heads: hf_config.num_key_value_heads,
            head_dim: hf_config.head_dim,
            ffn_dim: hf_config.intermediate_size,
            rope_base: hf_config.rope_theta as f32,
            rms_eps: hf_config.rms_norm_eps as f32,
            max_seq_len: hf_config.max_position_embeddings,
        };

        eprintln!(
            "Llama config (safetensors): {} layers, {} dim, {} q_heads, {} kv_heads, head_dim={}, ffn={}, vocab={}",
            config.num_layers, config.model_dim, config.num_q_heads,
            config.num_kv_heads, config.head_dim, config.ffn_dim, config.vocab_size
        );

        let mut model = Llama::new(config.clone());

        // Determine safetensors files to load
        let st_files = list_safetensors_in_dir(dir_path);
        if st_files.is_empty() {
            panic!("no safetensors files found in {}", dir);
        }
        eprintln!("Loading from {} safetensors file(s)", st_files.len());

        // Collect all tensor names → which file they come from
        let mut tensor_file_map: HashMap<String, usize> = HashMap::new();
        let mut files: Vec<SafetensorsFile> = Vec::new();

        for (idx, path) in st_files.iter().enumerate() {
            let path_str = path.to_str().unwrap();
            let sf = SafetensorsFile::open(path_str)
                .unwrap_or_else(|e| panic!("failed to open {:?}: {}", path, e));
            for name in sf.tensor_names() {
                tensor_file_map.insert(name.to_string(), idx);
            }
            files.push(sf);
        }

        let mut loaded_count = 0usize;

        // Load all tensors across all files
        for (hf_name, &file_idx) in &tensor_file_map {
            let sf = &files[file_idx];
            let mapped = match map_hf_name(hf_name) {
                Some(m) => m,
                None => continue,
            };
            let (mapped_name, is_2d) = mapped;

            let data = sf.load_tensor_f32(hf_name);
            let shape = sf.tensor_shape(hf_name).unwrap();

            // Route the tensor to the right model field
            if mapped_name == "embed_tokens.weight" {
                model.embedding.weight = Tensor::new(data, shape.to_vec(), false);
            } else if mapped_name == "lm_head.weight" {
                // [vocab, dim] → [dim, vocab]
                let vocab = shape[0];
                let dim = shape[1];
                let transposed = transpose_2d(&data, vocab, dim);
                model.output_weight = Tensor::new(transposed, vec![dim, vocab], false);
            } else if mapped_name == "norm.weight" {
                model.final_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
            } else if mapped_name.starts_with("blk.") {
                // Parse layer index from "blk.{i}.xxx"
                let rest = &mapped_name[4..]; // after "blk."
                let dot = rest.find('.').unwrap();
                let layer_idx: usize = rest[..dot].parse().unwrap();
                let suffix = &rest[dot + 1..];
                let layer = &mut model.layers[layer_idx];

                if is_2d && shape.len() == 2 {
                    // HF stores [out, in], Peregrine needs [in, out]
                    let (rows, cols) = (shape[0], shape[1]);
                    let transposed = transpose_2d(&data, rows, cols);
                    let tensor = Tensor::new(transposed, vec![cols, rows], false);
                    match suffix {
                        "attn_q.weight" => layer.attention.q_proj = tensor,
                        "attn_k.weight" => layer.attention.k_proj = tensor,
                        "attn_v.weight" => layer.attention.v_proj = tensor,
                        "attn_output.weight" => layer.attention.o_proj = tensor,
                        "ffn_gate.weight" => layer.gate_proj = tensor,
                        "ffn_up.weight" => layer.up_proj = tensor,
                        "ffn_down.weight" => layer.down_proj = tensor,
                        _ => continue,
                    }
                } else {
                    // 1D norm weights
                    let tensor = Tensor::new(data.clone(), vec![data.len()], false);
                    match suffix {
                        "attn_norm.weight" => layer.attn_norm.weight = tensor,
                        "ffn_norm.weight" => layer.ffn_norm.weight = tensor,
                        _ => continue,
                    }
                }
            } else {
                continue;
            }

            loaded_count += 1;
        }

        // Handle tied embeddings: if lm_head wasn't loaded, use embedding
        if !tensor_file_map.keys().any(|n| n == "lm_head.weight") && hf_config.tie_word_embeddings {
            eprintln!("  Using tied embeddings for lm_head");
            let emb_data = model.embedding.weight.data();
            let vocab = config.vocab_size;
            let dim = config.model_dim;
            let transposed = transpose_2d(&emb_data, vocab, dim);
            model.output_weight = Tensor::new(transposed, vec![dim, vocab], false);
            loaded_count += 1;
        }

        eprintln!("  Loaded {} tensors from safetensors", loaded_count);
        model
    }
}

/// Transpose a 2D matrix from [rows, cols] to [cols, rows].
fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut transposed = vec![0.0f32; cols * rows];
    for r in 0..rows {
        for c in 0..cols {
            transposed[c * rows + r] = data[r * cols + c];
        }
    }
    transposed
}

/// List safetensors files in a directory.
/// Uses the index file if present, otherwise falls back to the single file.
fn list_safetensors_in_dir(dir: &Path) -> Vec<std::path::PathBuf> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        if let Ok(index_str) = std::fs::read_to_string(&index_path) {
            let filenames = parse_weight_map_filenames(&index_str);
            if !filenames.is_empty() {
                return filenames.iter().map(|f| dir.join(f)).collect();
            }
        }
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return vec![single];
    }
    vec![]
}

/// Parse unique filenames from a safetensors index JSON weight_map.
fn parse_weight_map_filenames(index_json: &str) -> Vec<String> {
    let mut filenames = Vec::new();
    let mut seen = std::collections::HashSet::new();

    if let Some(wm_idx) = index_json.find("\"weight_map\"") {
        let after = &index_json[wm_idx + "\"weight_map\"".len()..];
        if let Some(brace) = after.find('{') {
            let map_str = &after[brace..];
            let bytes = map_str.as_bytes();
            let mut pos = 1;

            while pos < bytes.len() {
                if bytes[pos] == b'}' {
                    break;
                }
                if bytes[pos] == b'"' {
                    // Skip key
                    pos += 1;
                    while pos < bytes.len() && bytes[pos] != b'"' {
                        if bytes[pos] == b'\\' { pos += 1; }
                        pos += 1;
                    }
                    pos += 1;

                    // Skip to colon
                    while pos < bytes.len() && bytes[pos] != b':' { pos += 1; }
                    pos += 1;

                    // Skip whitespace
                    while pos < bytes.len() && matches!(bytes[pos], b' ' | b'\n' | b'\r' | b'\t') {
                        pos += 1;
                    }

                    // Parse value string
                    if pos < bytes.len() && bytes[pos] == b'"' {
                        pos += 1;
                        let start = pos;
                        while pos < bytes.len() && bytes[pos] != b'"' {
                            if bytes[pos] == b'\\' { pos += 1; }
                            pos += 1;
                        }
                        let filename = std::str::from_utf8(&bytes[start..pos])
                            .unwrap_or("")
                            .to_string();
                        if !filename.is_empty() && seen.insert(filename.clone()) {
                            filenames.push(filename);
                        }
                        pos += 1;
                    }
                } else {
                    pos += 1;
                }
            }
        }
    }
    filenames
}
