use peregrine::nn::{Embedding, RMSNorm};
use peregrine::serial::load_model;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::KVCache;
use crate::decoder::DecoderLayer;

/// Grok-1 model configuration.
#[derive(Clone)]
pub struct Grok1Config {
    pub vocab_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub ffn_dim: usize,
    pub rope_base: f32,
    pub max_seq_len: usize,
    pub embedding_mult: f32,
    pub output_mult: f32,
    pub attn_output_mult: f32,
    pub attn_logit_cap: f32,
}

impl Grok1Config {
    /// Full 314B parameter config matching run.py.
    pub fn full() -> Self {
        Grok1Config {
            vocab_size: 131072,
            model_dim: 6144,
            num_layers: 64,
            num_q_heads: 48,
            num_kv_heads: 8,
            head_dim: 128,
            num_experts: 8,
            top_k_experts: 2,
            ffn_dim: 32768,
            rope_base: 10000.0,
            max_seq_len: 8192,
            embedding_mult: 78.38367176906169,
            output_mult: 0.5773502691896257,
            attn_output_mult: 0.08838834764831845,
            attn_logit_cap: 30.0,
        }
    }

    /// Small test config with random weights.
    pub fn small() -> Self {
        Grok1Config {
            vocab_size: 1024,
            model_dim: 256,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            num_experts: 4,
            top_k_experts: 2,
            ffn_dim: 512,
            rope_base: 10000.0,
            max_seq_len: 256,
            embedding_mult: 78.384,
            output_mult: 0.577,
            attn_output_mult: 0.088,
            attn_logit_cap: 30.0,
        }
    }
}

/// Top-level Grok-1 model.
pub struct Grok1 {
    pub embedding: Embedding,
    pub layers: Vec<DecoderLayer>,
    pub final_norm: RMSNorm,
    pub config: Grok1Config,
}

impl Grok1 {
    pub fn new(config: Grok1Config) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(DecoderLayer::new(
                config.model_dim,
                config.num_q_heads,
                config.num_kv_heads,
                config.head_dim,
                config.ffn_dim,
                config.num_experts,
                config.top_k_experts,
                config.max_seq_len,
                config.rope_base,
                config.attn_output_mult,
                config.attn_logit_cap,
            ));
        }

        Grok1 {
            embedding: Embedding::new(config.vocab_size, config.model_dim),
            layers,
            final_norm: RMSNorm::new(config.model_dim, 1e-5),
            config,
        }
    }

    /// Initialize KV caches for all layers.
    pub fn init_kv_caches(&self) -> Vec<KVCache> {
        (0..self.config.num_layers)
            .map(|_| KVCache::new(self.config.num_kv_heads, self.config.head_dim))
            .collect()
    }

    /// Forward pass: tokens -> logits.
    /// tokens: slice of token IDs
    /// kv_caches: mutable KV caches for each layer
    /// Returns logits [seq_len, vocab_size]
    pub fn forward(&self, tokens: &[usize], kv_caches: &mut [KVCache]) -> Tensor {
        // Embed and scale
        let mut x = self.embedding.forward(tokens);
        x = x.scale(self.config.embedding_mult);

        // Decoder layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mut kv_caches[i]);
        }

        // Final norm
        x = self.final_norm.forward(&x);

        // Tied embedding logits: x @ embedding.T * output_mult
        let emb_weight = self.embedding.weight.data();
        let x_data = x.data();
        let seq_len = x.shape()[0];
        let model_dim = self.config.model_dim;
        let vocab_size = self.config.vocab_size;

        let mut logits = vec![0.0f32; seq_len * vocab_size];
        for t in 0..seq_len {
            for v in 0..vocab_size {
                let mut dot = 0.0f32;
                let x_off = t * model_dim;
                let e_off = v * model_dim;
                for d in 0..model_dim {
                    dot += x_data[x_off + d] * emb_weight[e_off + d];
                }
                logits[t * vocab_size + v] = dot * self.config.output_mult;
            }
        }

        Tensor::new(logits, vec![seq_len, vocab_size], false)
    }

    /// Load weights from Peregrine binary format.
    pub fn load_weights(&mut self, path: &str) {
        let raw = load_model(path).expect("Failed to load model weights");
        let mut params: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, shape, data) in raw {
            params.insert(name, (shape, data));
        }

        eprintln!("Loaded {} weight tensors", params.len());

        // Embedding
        if let Some((shape, data)) = params.get("embedding") {
            self.embedding.weight = Tensor::new(data.clone(), shape.clone(), false);
        }

        // Final norm
        if let Some((_s, data)) = params.get("final_norm.weight") {
            self.final_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }

        // Layers
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{}", i);
            self.layers[i].load_weights(&params, &prefix);
        }
    }

    /// Initialize with random weights (for --small testing).
    pub fn init_random(&mut self) {
        let c = &self.config;
        self.embedding.weight = Tensor::randn(&[c.vocab_size, c.model_dim], false);
        // Scale down random weights to avoid numerical issues
        let emb_data: Vec<f32> = self.embedding.weight.data().iter().map(|x| x * 0.02).collect();
        self.embedding.weight = Tensor::new(emb_data, vec![c.vocab_size, c.model_dim], false);

        self.final_norm.weight = Tensor::ones(&[c.model_dim], false);

        for layer in &mut self.layers {
            layer.pre_attn_norm.weight = Tensor::ones(&[c.model_dim], false);
            layer.post_attn_norm.weight = Tensor::ones(&[c.model_dim], false);
            layer.pre_moe_norm.weight = Tensor::ones(&[c.model_dim], false);
            layer.post_moe_norm.weight = Tensor::ones(&[c.model_dim], false);

            // Attention projections (small random)
            let init = |rows: usize, cols: usize| -> Tensor {
                let data: Vec<f32> = Tensor::randn(&[rows, cols], false)
                    .data()
                    .iter()
                    .map(|x| x * 0.02)
                    .collect();
                Tensor::new(data, vec![rows, cols], false)
            };

            layer.attention.q_proj = init(c.model_dim, c.num_q_heads * c.head_dim);
            layer.attention.k_proj = init(c.model_dim, c.num_kv_heads * c.head_dim);
            layer.attention.v_proj = init(c.model_dim, c.num_kv_heads * c.head_dim);
            layer.attention.o_proj = init(c.num_q_heads * c.head_dim, c.model_dim);

            // Router
            let router_data: Vec<f32> = Tensor::randn(&[c.model_dim, c.num_experts], false)
                .data()
                .iter()
                .map(|x| x * 0.02)
                .collect();
            layer.moe.router =
                Tensor::new(router_data, vec![c.model_dim, c.num_experts], false);

            // Experts
            for expert in &mut layer.moe.experts {
                expert.linear_gate = init(c.model_dim, c.ffn_dim);
                expert.linear_v = init(c.model_dim, c.ffn_dim);
                expert.linear_out = init(c.ffn_dim, c.model_dim);
            }
        }
    }
}
