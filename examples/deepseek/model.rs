use peregrine::nn::{Embedding, RMSNorm};
use peregrine::serial::load_model;
use peregrine::tensor::Tensor;
use std::collections::HashMap;

use crate::attention::KVCache;
use crate::decoder::Block;

/// DeepSeek-V3/R1 model configuration.
#[derive(Clone)]
pub struct DeepSeekConfig {
    pub vocab_size: usize,
    pub dim: usize,
    pub inter_dim: usize,
    pub moe_inter_dim: usize,
    pub n_layers: usize,
    pub n_dense_layers: usize,
    pub n_heads: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub n_activated_experts: usize,
    pub n_expert_groups: usize,
    pub n_limited_groups: usize,
    pub route_scale: f32,
    pub score_func_sigmoid: bool,
    pub q_lora_rank: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub max_seq_len: usize,
    pub original_seq_len: usize,
    pub rope_theta: f32,
    pub rope_factor: f32,
    pub beta_fast: usize,
    pub beta_slow: usize,
    pub mscale: f32,
}

impl DeepSeekConfig {
    /// Full 671B parameter config (DeepSeek-V3 / DeepSeek-R1).
    pub fn full() -> Self {
        DeepSeekConfig {
            vocab_size: 129280,
            dim: 7168,
            inter_dim: 18432,
            moe_inter_dim: 2048,
            n_layers: 61,
            n_dense_layers: 3,
            n_heads: 128,
            n_routed_experts: 256,
            n_shared_experts: 1,
            n_activated_experts: 8,
            n_expert_groups: 8,
            n_limited_groups: 4,
            route_scale: 2.5,
            score_func_sigmoid: true,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope_head_dim: 128,
            qk_rope_head_dim: 64,
            v_head_dim: 128,
            max_seq_len: 16384,
            original_seq_len: 4096,
            rope_theta: 10000.0,
            rope_factor: 40.0,
            beta_fast: 32,
            beta_slow: 1,
            mscale: 1.0,
        }
    }

    /// Small test config with random weights.
    pub fn small() -> Self {
        DeepSeekConfig {
            vocab_size: 1024,
            dim: 256,
            inter_dim: 512,
            moe_inter_dim: 128,
            n_layers: 2,
            n_dense_layers: 1,
            n_heads: 4,
            n_routed_experts: 8,
            n_shared_experts: 1,
            n_activated_experts: 2,
            n_expert_groups: 2,
            n_limited_groups: 1,
            route_scale: 2.5,
            score_func_sigmoid: true,
            q_lora_rank: 64,
            kv_lora_rank: 32,
            qk_nope_head_dim: 32,
            qk_rope_head_dim: 16,
            v_head_dim: 32,
            max_seq_len: 256,
            original_seq_len: 256,
            rope_theta: 10000.0,
            rope_factor: 1.0,
            beta_fast: 32,
            beta_slow: 1,
            mscale: 1.0,
        }
    }

    fn has_bias(&self) -> bool {
        self.dim == 7168
    }
}

/// Top-level DeepSeek-V3/R1 model.
pub struct DeepSeek {
    pub embedding: Embedding,
    pub layers: Vec<Block>,
    pub final_norm: RMSNorm,
    pub head: Tensor, // [dim, vocab_size] (separate output head, not tied)
    pub config: DeepSeekConfig,
}

impl DeepSeek {
    pub fn new(config: DeepSeekConfig) -> Self {
        let mut layers = Vec::with_capacity(config.n_layers);
        for i in 0..config.n_layers {
            layers.push(Block::new(
                i,
                config.dim,
                config.n_heads,
                config.inter_dim,
                config.moe_inter_dim,
                config.n_dense_layers,
                config.n_routed_experts,
                config.n_shared_experts,
                config.n_activated_experts,
                config.n_expert_groups,
                config.n_limited_groups,
                config.route_scale,
                config.score_func_sigmoid,
                config.has_bias(),
                config.q_lora_rank,
                config.kv_lora_rank,
                config.qk_nope_head_dim,
                config.qk_rope_head_dim,
                config.v_head_dim,
                config.max_seq_len,
                config.rope_theta,
                config.rope_factor,
                config.original_seq_len,
                config.beta_fast,
                config.beta_slow,
                config.mscale,
            ));
        }

        DeepSeek {
            embedding: Embedding::new(config.vocab_size, config.dim),
            layers,
            final_norm: RMSNorm::new(config.dim, 1e-6),
            head: Tensor::zeros(&[config.dim, config.vocab_size], false),
            config,
        }
    }

    pub fn init_kv_caches(&self) -> Vec<KVCache> {
        (0..self.config.n_layers)
            .map(|_| KVCache::new(self.config.kv_lora_rank, self.config.qk_rope_head_dim))
            .collect()
    }

    /// Forward pass: tokens -> logits (last token only).
    pub fn forward(&self, tokens: &[usize], kv_caches: &mut [KVCache]) -> Tensor {
        let mut x = self.embedding.forward(tokens);

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, &mut kv_caches[i]);
        }

        x = self.final_norm.forward(&x);

        // Only compute logits for last token
        let x_data = x.data();
        let seq_len = x.shape()[0];
        let dim = self.config.dim;
        let vocab_size = self.config.vocab_size;

        let last_token = &x_data[(seq_len - 1) * dim..seq_len * dim];
        let head_data = self.head.data();

        let mut logits = vec![0.0f32; vocab_size];
        for v in 0..vocab_size {
            let mut dot = 0.0f32;
            let h_off = v * dim;
            for d in 0..dim {
                dot += last_token[d] * head_data[h_off + d];
            }
            logits[v] = dot;
        }

        Tensor::new(logits, vec![1, vocab_size], false)
    }

    pub fn load_weights(&mut self, path: &str) {
        let raw = load_model(path).expect("Failed to load model weights");
        let mut params: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, shape, data) in raw {
            params.insert(name, (shape, data));
        }

        eprintln!("Loaded {} weight tensors", params.len());

        if let Some((s, d)) = params.get("embed.weight") {
            self.embedding.weight = Tensor::new(d.clone(), s.clone(), false);
        }
        if let Some((_s, d)) = params.get("norm.weight") {
            self.final_norm.weight = Tensor::new(d.clone(), vec![d.len()], false);
        }
        if let Some((s, d)) = params.get("head.weight") {
            self.head = Tensor::new(d.clone(), s.clone(), false);
        }

        for i in 0..self.config.n_layers {
            let prefix = format!("layers.{}", i);
            self.layers[i].load_weights(&params, &prefix);
        }
    }

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
        let init_1d = |n: usize| -> Vec<f32> {
            vec![1.0; n]
        };

        // Embedding
        let emb_data: Vec<f32> = Tensor::randn(&[c.vocab_size, c.dim], false)
            .data()
            .iter()
            .map(|x| x * 0.02)
            .collect();
        self.embedding.weight = Tensor::new(emb_data, vec![c.vocab_size, c.dim], false);

        self.final_norm.weight = Tensor::ones(&[c.dim], false);
        self.head = init(c.dim, c.vocab_size);

        for layer in &mut self.layers {
            layer.attn_norm.weight = Tensor::ones(&[c.dim], false);
            layer.ffn_norm.weight = Tensor::ones(&[c.dim], false);

            // MLA weights
            let attn = &mut layer.attn;
            let qk_head_dim = c.qk_nope_head_dim + c.qk_rope_head_dim;
            if c.q_lora_rank == 0 {
                attn.wq = Some(init(c.dim, c.n_heads * qk_head_dim));
            } else {
                attn.wq_a = Some(init(c.dim, c.q_lora_rank));
                attn.q_norm_weight = init_1d(c.q_lora_rank);
                attn.wq_b = Some(init(c.q_lora_rank, c.n_heads * qk_head_dim));
            }
            attn.wkv_a = init(c.dim, c.kv_lora_rank + c.qk_rope_head_dim);
            attn.kv_norm_weight = init_1d(c.kv_lora_rank);
            attn.wkv_b = init(c.kv_lora_rank, c.n_heads * (c.qk_nope_head_dim + c.v_head_dim));
            attn.wo = init(c.n_heads * c.v_head_dim, c.dim);

            // FFN weights
            match &mut layer.ffn {
                crate::decoder::FFN::Dense(mlp) => {
                    mlp.w1 = init(c.dim, c.inter_dim);
                    mlp.w2 = init(c.inter_dim, c.dim);
                    mlp.w3 = init(c.dim, c.inter_dim);
                }
                crate::decoder::FFN::Moe(moe) => {
                    // Gate
                    let gate_data: Vec<f32> = Tensor::randn(&[c.n_routed_experts, c.dim], false)
                        .data()
                        .iter()
                        .map(|x| x * 0.02)
                        .collect();
                    moe.gate.weight = Tensor::new(
                        gate_data,
                        vec![c.n_routed_experts, c.dim],
                        false,
                    );

                    for expert in &mut moe.experts {
                        expert.w1 = init(c.dim, c.moe_inter_dim);
                        expert.w2 = init(c.moe_inter_dim, c.dim);
                        expert.w3 = init(c.dim, c.moe_inter_dim);
                    }

                    moe.shared_experts.w1 = init(c.dim, c.n_shared_experts * c.moe_inter_dim);
                    moe.shared_experts.w2 = init(c.n_shared_experts * c.moe_inter_dim, c.dim);
                    moe.shared_experts.w3 = init(c.dim, c.n_shared_experts * c.moe_inter_dim);
                }
            }
        }
    }
}
