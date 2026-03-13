use peregrine::gguf::GgufFile;
use peregrine::nn::{Embedding, RMSNorm};
use peregrine::tensor::Tensor;

use crate::attention::KVCache;
use crate::decoder::LlamaBlock;

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
                peregrine::gguf::MetadataValue::Array(arr) => arr.len(),
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
