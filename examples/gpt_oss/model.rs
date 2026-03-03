use peregrine::nn::{Embedding, RMSNorm};
use peregrine::serial::load_model;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::KVCache;
use crate::decoder::TransformerBlock;

/// GPT-OSS model configuration.
#[derive(Clone)]
pub struct GptOssConfig {
    pub vocab_size: usize,
    pub model_dim: usize,
    pub num_layers: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub top_k_experts: usize,
    pub intermediate: usize,
    pub max_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub initial_ctx: usize,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub sliding_window: usize,
    pub swiglu_alpha: f32,
    pub swiglu_clamp: f32,
}

impl GptOssConfig {
    /// Full 117B parameter config (gpt-oss-120b).
    pub fn full() -> Self {
        GptOssConfig {
            vocab_size: 201088,
            model_dim: 2880,
            num_layers: 36,
            num_q_heads: 64,
            num_kv_heads: 8,
            head_dim: 64,
            num_experts: 128,
            top_k_experts: 4,
            intermediate: 2880,
            max_seq_len: 4096,
            rope_theta: 150000.0,
            rope_factor: 32.0,
            initial_ctx: 4096,
            beta_fast: 32,
            beta_slow: 1,
            sliding_window: 128,
            swiglu_alpha: 1.702,
            swiglu_clamp: 7.0,
        }
    }

    /// 21B parameter config (gpt-oss-20b). Fits in 16GB with MXFP4.
    pub fn medium() -> Self {
        GptOssConfig {
            vocab_size: 201088,
            model_dim: 2880,
            num_layers: 24,
            num_q_heads: 64,
            num_kv_heads: 8,
            head_dim: 64,
            num_experts: 32,
            top_k_experts: 4,
            intermediate: 2880,
            max_seq_len: 4096,
            rope_theta: 150000.0,
            rope_factor: 32.0,
            initial_ctx: 4096,
            beta_fast: 32,
            beta_slow: 1,
            sliding_window: 128,
            swiglu_alpha: 1.702,
            swiglu_clamp: 7.0,
        }
    }

    /// Small test config with random weights.
    pub fn small() -> Self {
        GptOssConfig {
            vocab_size: 1024,
            model_dim: 256,
            num_layers: 2,
            num_q_heads: 4,
            num_kv_heads: 2,
            head_dim: 64,
            num_experts: 8,
            top_k_experts: 2,
            intermediate: 256,
            max_seq_len: 256,
            rope_theta: 150000.0,
            rope_factor: 32.0,
            initial_ctx: 256,
            beta_fast: 32,
            beta_slow: 1,
            sliding_window: 128,
            swiglu_alpha: 1.702,
            swiglu_clamp: 7.0,
        }
    }
}

/// Top-level GPT-OSS model.
pub struct GptOss {
    pub embedding: Embedding,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: RMSNorm,
    pub unembed: Tensor, // [model_dim, vocab_size] — separate from embedding
    pub config: GptOssConfig,
}

impl GptOss {
    pub fn new(config: GptOssConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            layers.push(TransformerBlock::new(
                i,
                config.model_dim,
                config.num_q_heads,
                config.num_kv_heads,
                config.head_dim,
                config.intermediate,
                config.num_experts,
                config.top_k_experts,
                config.max_seq_len,
                config.rope_theta,
                config.rope_factor,
                config.initial_ctx,
                config.beta_fast,
                config.beta_slow,
                config.sliding_window,
                config.swiglu_alpha,
                config.swiglu_clamp,
            ));
        }

        GptOss {
            embedding: Embedding::new(config.vocab_size, config.model_dim),
            layers,
            final_norm: RMSNorm::new(config.model_dim, 1e-5),
            unembed: Tensor::zeros(&[config.model_dim, config.vocab_size], false),
            config,
        }
    }

    /// Initialize KV caches for all layers.
    pub fn init_kv_caches(&self) -> Vec<KVCache> {
        (0..self.config.num_layers)
            .map(|_| KVCache::new(self.config.num_kv_heads, self.config.head_dim))
            .collect()
    }

    /// Forward pass: tokens -> logits [seq_len, vocab_size].
    pub fn forward(&self, tokens: &[usize], kv_caches: &mut [KVCache]) -> Tensor {
        let mut x = self.embedding.forward(tokens);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mut kv_caches[i]);
        }

        x = self.final_norm.forward(&x);

        // Logits: x @ unembed
        let x_data = x.data();
        let seq_len = x.shape()[0];
        let model_dim = self.config.model_dim;
        let vocab_size = self.config.vocab_size;
        let unembed_data = self.unembed.data();

        let mut logits = vec![0.0f32; seq_len * vocab_size];
        for t in 0..seq_len {
            for v in 0..vocab_size {
                let mut dot = 0.0f32;
                let x_off = t * model_dim;
                let u_off = v * model_dim;
                for d in 0..model_dim {
                    dot += x_data[x_off + d] * unembed_data[u_off + d];
                }
                logits[t * vocab_size + v] = dot;
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

        // Unembed
        if let Some((shape, data)) = params.get("unembed") {
            self.unembed = Tensor::new(data.clone(), shape.clone(), false);
        }

        // Layers
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{}", i);
            self.layers[i].load_weights(&params, &prefix);
        }
    }

    /// Load weights from Peregrine binary format (MXFP4 quantized MoE experts).
    pub fn load_weights_quantized(&mut self, path: &str) {
        // Initialize MXFP4 expert structs before loading
        for layer in &mut self.layers {
            layer.moe.init_mxfp4_experts(
                self.config.intermediate,
                self.config.swiglu_alpha,
                self.config.swiglu_clamp,
            );
        }

        let raw = load_model(path).expect("Failed to load model weights");
        let mut params: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, shape, data) in raw {
            params.insert(name, (shape, data));
        }

        eprintln!("Loaded {} weight tensors (quantized mode)", params.len());

        // Embedding
        if let Some((shape, data)) = params.get("embedding") {
            self.embedding.weight = Tensor::new(data.clone(), shape.clone(), false);
        }

        // Final norm
        if let Some((_s, data)) = params.get("final_norm.weight") {
            self.final_norm.weight = Tensor::new(data.clone(), vec![data.len()], false);
        }

        // Unembed
        if let Some((shape, data)) = params.get("unembed") {
            self.unembed = Tensor::new(data.clone(), shape.clone(), false);
        }

        // Layers — load_weights handles quantized dispatch internally
        for i in 0..self.config.num_layers {
            let prefix = format!("layers.{}", i);
            self.layers[i].load_weights(&params, &prefix);
        }
    }

    /// Initialize with random weights (for --small testing).
    pub fn init_random(&mut self) {
        let c = &self.config;
        let init = |rows: usize, cols: usize| -> Tensor {
            let data: Vec<f32> = Tensor::randn(&[rows, cols], false)
                .data()
                .iter()
                .map(|x| x * 0.02)
                .collect();
            Tensor::new(data, vec![rows, cols], false)
        };
        let init_1d = |n: usize| -> Tensor {
            let data: Vec<f32> = Tensor::randn(&[n], false)
                .data()
                .iter()
                .map(|x| x * 0.02)
                .collect();
            Tensor::new(data, vec![n], false)
        };

        // Embedding
        let emb_data: Vec<f32> = Tensor::randn(&[c.vocab_size, c.model_dim], false)
            .data()
            .iter()
            .map(|x| x * 0.02)
            .collect();
        self.embedding.weight = Tensor::new(emb_data, vec![c.vocab_size, c.model_dim], false);

        self.final_norm.weight = Tensor::ones(&[c.model_dim], false);
        self.unembed = init(c.model_dim, c.vocab_size);

        let total_qkv = (c.num_q_heads + 2 * c.num_kv_heads) * c.head_dim;

        for layer in &mut self.layers {
            layer.attn_norm.weight = Tensor::ones(&[c.model_dim], false);
            layer.ffn_norm.weight = Tensor::ones(&[c.model_dim], false);

            // Attention weights
            layer.attn.qkv_weight = init(c.model_dim, total_qkv);
            layer.attn.qkv_bias = init_1d(total_qkv);
            layer.attn.o_weight = init(c.num_q_heads * c.head_dim, c.model_dim);
            layer.attn.o_bias = init_1d(c.model_dim);
            layer.attn.sinks = vec![0.0; c.num_q_heads];

            // MoE gate
            layer.moe.gate = init(c.model_dim, c.num_experts);
            layer.moe.gate_bias = init_1d(c.num_experts);

            // Experts
            for expert in &mut layer.moe.experts {
                expert.mlp1 = init(c.model_dim, c.intermediate * 2);
                expert.mlp1_bias = init_1d(c.intermediate * 2);
                expert.mlp2 = init(c.intermediate, c.model_dim);
                expert.mlp2_bias = init_1d(c.model_dim);
            }
        }
    }

    /// Initialize with random MXFP4-quantized weights (for --small --quantized testing).
    pub fn init_random_quantized(&mut self) {
        let c = &self.config;
        let init = |rows: usize, cols: usize| -> Tensor {
            let data: Vec<f32> = Tensor::randn(&[rows, cols], false)
                .data()
                .iter()
                .map(|x| x * 0.02)
                .collect();
            Tensor::new(data, vec![rows, cols], false)
        };
        let init_1d = |n: usize| -> Tensor {
            let data: Vec<f32> = Tensor::randn(&[n], false)
                .data()
                .iter()
                .map(|x| x * 0.02)
                .collect();
            Tensor::new(data, vec![n], false)
        };

        // Embedding
        let emb_data: Vec<f32> = Tensor::randn(&[c.vocab_size, c.model_dim], false)
            .data()
            .iter()
            .map(|x| x * 0.02)
            .collect();
        self.embedding.weight = Tensor::new(emb_data, vec![c.vocab_size, c.model_dim], false);

        self.final_norm.weight = Tensor::ones(&[c.model_dim], false);
        self.unembed = init(c.model_dim, c.vocab_size);

        let total_qkv = (c.num_q_heads + 2 * c.num_kv_heads) * c.head_dim;

        for layer in &mut self.layers {
            layer.attn_norm.weight = Tensor::ones(&[c.model_dim], false);
            layer.ffn_norm.weight = Tensor::ones(&[c.model_dim], false);

            // Attention weights
            layer.attn.qkv_weight = init(c.model_dim, total_qkv);
            layer.attn.qkv_bias = init_1d(total_qkv);
            layer.attn.o_weight = init(c.num_q_heads * c.head_dim, c.model_dim);
            layer.attn.o_bias = init_1d(c.model_dim);
            layer.attn.sinks = vec![0.0; c.num_q_heads];

            // MoE gate
            layer.moe.gate = init(c.model_dim, c.num_experts);
            layer.moe.gate_bias = init_1d(c.num_experts);

            // MXFP4 experts with random blocks/scales
            layer
                .moe
                .init_mxfp4_experts(c.intermediate, c.swiglu_alpha, c.swiglu_clamp);
            for expert in &mut layer.moe.mxfp4_experts {
                // mlp1: weight [intermediate*2, hidden]
                //   blocks: out_dim * (in_dim/32) * 16 bytes per block
                //   scales: out_dim * (in_dim/32)
                let mlp1_out = c.intermediate * 2;
                let mlp1_in = c.model_dim;
                let mlp1_bpr = mlp1_in / 32; // blocks per row
                expert.mlp1_blocks = random_bytes(mlp1_out * mlp1_bpr * 16);
                expert.mlp1_scales = random_bytes(mlp1_out * mlp1_bpr);

                // mlp2: weight [hidden, intermediate]
                let mlp2_out = c.model_dim;
                let mlp2_in = c.intermediate;
                let mlp2_bpr = mlp2_in / 32;
                expert.mlp2_blocks = random_bytes(mlp2_out * mlp2_bpr * 16);
                expert.mlp2_scales = random_bytes(mlp2_out * mlp2_bpr);

                expert.mlp1_bias = Tensor::randn(&[c.intermediate * 2], false)
                    .data()
                    .iter()
                    .map(|x| x * 0.02)
                    .collect();
                expert.mlp2_bias = Tensor::randn(&[c.model_dim], false)
                    .data()
                    .iter()
                    .map(|x| x * 0.02)
                    .collect();
            }
        }
    }
}

/// Generate random bytes for MXFP4 test weights.
fn random_bytes(n: usize) -> Vec<u8> {
    // Simple xorshift RNG for deterministic random bytes
    let mut rng: u32 = 0xDEAD_BEEF;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        out.push((rng & 0xFF) as u8);
    }
    out
}
