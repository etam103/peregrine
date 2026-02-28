// MUSt3R Decoder: ViT-Base decoder with cross-attention and memory
//
// Architecture: 12 CachedDecoderBlocks, each with:
//   1. Self-attention with 2D RoPE
//   2. Cross-attention (no RoPE) attending to memory from the other view
//   3. Feed-forward network (FFN)
//
// Two-view forward: img1 attends to img2 as memory, img2 attends to img1.

use std::collections::HashMap;
use std::time::Instant;

use peregrine::tensor::Tensor;

use super::rope2d::RoPE2D;

// --------------------------------------------------------------------------
// CachedDecoderBlock
// --------------------------------------------------------------------------

pub struct CachedDecoderBlock {
    // Self-attention (with 2D RoPE)
    pub norm1_weight: Tensor,      // [embed_dim]
    pub norm1_bias: Tensor,        // [embed_dim]
    pub qkv_weight: Tensor,       // [embed_dim, 3*embed_dim] (transposed for Peregrine matmul)
    pub qkv_bias: Tensor,         // [1, 3*embed_dim]
    pub proj_weight: Tensor,      // [embed_dim, embed_dim] (transposed)
    pub proj_bias: Tensor,        // [1, embed_dim]

    // Cross-attention (no RoPE, attends to memory)
    pub norm_y_weight: Tensor,    // [embed_dim] normalizes memory
    pub norm_y_bias: Tensor,      // [embed_dim]
    pub norm2_weight: Tensor,     // [embed_dim] normalizes query
    pub norm2_bias: Tensor,       // [embed_dim]
    pub cross_projq_weight: Tensor,  // [embed_dim, embed_dim] (transposed)
    pub cross_projq_bias: Tensor,    // [1, embed_dim]
    pub cross_projk_weight: Tensor,  // [embed_dim, embed_dim] (transposed)
    pub cross_projk_bias: Tensor,    // [1, embed_dim]
    pub cross_projv_weight: Tensor,  // [embed_dim, embed_dim] (transposed)
    pub cross_projv_bias: Tensor,    // [1, embed_dim]
    pub cross_proj_weight: Tensor,   // [embed_dim, embed_dim] output projection (transposed)
    pub cross_proj_bias: Tensor,     // [1, embed_dim]

    // FFN
    pub norm3_weight: Tensor,     // [embed_dim]
    pub norm3_bias: Tensor,       // [embed_dim]
    pub mlp_fc1_weight: Tensor,   // [embed_dim, ffn_dim] (transposed)
    pub mlp_fc1_bias: Tensor,     // [1, ffn_dim]
    pub mlp_fc2_weight: Tensor,   // [ffn_dim, embed_dim] (transposed)
    pub mlp_fc2_bias: Tensor,     // [1, embed_dim]

    embed_dim: usize,
}

impl CachedDecoderBlock {
    /// Create a new decoder block with zero-initialized weights.
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        let _ = num_heads; // used implicitly via dimensions
        let ffn_dim = embed_dim * 4; // 3072 for embed_dim=768

        CachedDecoderBlock {
            // Self-attention
            norm1_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm1_bias: Tensor::zeros(&[embed_dim], false),
            qkv_weight: Tensor::zeros(&[embed_dim, 3 * embed_dim], false),
            qkv_bias: Tensor::zeros(&[1, 3 * embed_dim], false),
            proj_weight: Tensor::zeros(&[embed_dim, embed_dim], false),
            proj_bias: Tensor::zeros(&[1, embed_dim], false),

            // Cross-attention
            norm_y_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm_y_bias: Tensor::zeros(&[embed_dim], false),
            norm2_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm2_bias: Tensor::zeros(&[embed_dim], false),
            cross_projq_weight: Tensor::zeros(&[embed_dim, embed_dim], false),
            cross_projq_bias: Tensor::zeros(&[1, embed_dim], false),
            cross_projk_weight: Tensor::zeros(&[embed_dim, embed_dim], false),
            cross_projk_bias: Tensor::zeros(&[1, embed_dim], false),
            cross_projv_weight: Tensor::zeros(&[embed_dim, embed_dim], false),
            cross_projv_bias: Tensor::zeros(&[1, embed_dim], false),
            cross_proj_weight: Tensor::zeros(&[embed_dim, embed_dim], false),
            cross_proj_bias: Tensor::zeros(&[1, embed_dim], false),

            // FFN
            norm3_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm3_bias: Tensor::zeros(&[embed_dim], false),
            mlp_fc1_weight: Tensor::zeros(&[embed_dim, ffn_dim], false),
            mlp_fc1_bias: Tensor::zeros(&[1, ffn_dim], false),
            mlp_fc2_weight: Tensor::zeros(&[ffn_dim, embed_dim], false),
            mlp_fc2_bias: Tensor::zeros(&[1, embed_dim], false),

            embed_dim,
        }
    }

    /// Self-attention with 2D RoPE.
    ///
    /// x: [batch * seq_len, embed_dim]
    /// positions: flattened [seq_len, 2] position data (y, x) for 2D RoPE
    /// Returns: x + self_attn(norm1(x))
    pub fn forward_self_attn(
        &self,
        x: &Tensor,
        positions: &[f32],
        rope: &RoPE2D,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Tensor {
        let embed_dim = self.embed_dim;
        let head_dim = embed_dim / num_heads;

        // Pre-norm
        let normed = x.layer_norm(&self.norm1_weight, &self.norm1_bias, embed_dim);

        // Fused QKV projection: [batch*seq, embed_dim] x [embed_dim, 3*embed_dim]
        let qkv = normed.matmul(&self.qkv_weight).add_bias(&self.qkv_bias);
        let qkv_data = qkv.data();

        let qkv_stride = 3 * embed_dim;

        // Fused QKV split + reshape: directly from [batch*seq, 3*embed_dim]
        // to Q,K,V each in [batch, num_heads, seq_len, head_dim] layout.
        let mut q_heads = vec![0.0f32; batch * num_heads * seq_len * head_dim];
        let mut k_heads = vec![0.0f32; batch * num_heads * seq_len * head_dim];
        let mut v_heads = vec![0.0f32; batch * num_heads * seq_len * head_dim];

        for b in 0..batch {
            for s in 0..seq_len {
                let src_token = b * seq_len + s;
                let qkv_base = src_token * qkv_stride;
                for h in 0..num_heads {
                    let dst_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let h_offset = h * head_dim;
                    q_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&qkv_data[qkv_base + h_offset..qkv_base + h_offset + head_dim]);
                    k_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&qkv_data[qkv_base + embed_dim + h_offset..qkv_base + embed_dim + h_offset + head_dim]);
                    v_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&qkv_data[qkv_base + 2 * embed_dim + h_offset..qkv_base + 2 * embed_dim + h_offset + head_dim]);
                }
            }
        }

        // Apply 2D RoPE to Q and K per batch element.
        // RoPE expects [num_heads, seq_len, head_dim] per batch.
        let per_batch = num_heads * seq_len * head_dim;
        for b in 0..batch {
            let start = b * per_batch;
            let end = start + per_batch;

            let q_rotated = rope.apply(
                &q_heads[start..end],
                positions,
                num_heads,
                seq_len,
                head_dim,
            );
            q_heads[start..end].copy_from_slice(&q_rotated);

            let k_rotated = rope.apply(
                &k_heads[start..end],
                positions,
                num_heads,
                seq_len,
                head_dim,
            );
            k_heads[start..end].copy_from_slice(&k_rotated);
        }

        // Multi-head attention (parallel for large sequences)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_bh = batch * num_heads;

        let attn_out_data = peregrine::tensor::multi_head_attention(
            &q_heads, &k_heads, &v_heads,
            total_bh, seq_len, seq_len, head_dim, scale,
        );

        // Direct transpose [batch, num_heads, seq_len, head_dim] -> [batch*seq_len, embed_dim]
        let mut attn_flat_data = vec![0.0f32; batch * seq_len * embed_dim];
        for b in 0..batch {
            for s in 0..seq_len {
                let dst_row = b * seq_len + s;
                for h in 0..num_heads {
                    let src_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let dst_idx = dst_row * embed_dim + h * head_dim;
                    attn_flat_data[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&attn_out_data[src_idx..src_idx + head_dim]);
                }
            }
        }
        let attn_flat = Tensor::new(attn_flat_data, vec![batch * seq_len, embed_dim], false);

        // Output projection
        let attn_proj = attn_flat.matmul(&self.proj_weight).add_bias(&self.proj_bias);

        // Residual connection
        x.add(&attn_proj)
    }

    /// Cross-attention: query attends to memory (no RoPE).
    ///
    /// x: [batch * seq_q, embed_dim] - query tokens
    /// memory: [batch * seq_kv, embed_dim] - memory tokens (from the other view)
    /// Returns: x + cross_attn(norm2(x), norm_y(memory))
    pub fn forward_cross_attn(
        &self,
        x: &Tensor,
        memory: &Tensor,
        batch: usize,
        seq_q: usize,
        seq_kv: usize,
        num_heads: usize,
    ) -> Tensor {
        let embed_dim = self.embed_dim;
        let head_dim = embed_dim / num_heads;

        // Normalize memory with norm_y
        let memory_normed =
            memory.layer_norm(&self.norm_y_weight, &self.norm_y_bias, embed_dim);

        // Normalize query with norm2
        let x_normed = x.layer_norm(&self.norm2_weight, &self.norm2_bias, embed_dim);

        // Separate Q, K, V projections (no fused QKV for cross-attention)
        let q = x_normed
            .matmul(&self.cross_projq_weight)
            .add_bias(&self.cross_projq_bias); // [batch*seq_q, embed_dim]
        let k = memory_normed
            .matmul(&self.cross_projk_weight)
            .add_bias(&self.cross_projk_bias); // [batch*seq_kv, embed_dim]
        let v = memory_normed
            .matmul(&self.cross_projv_weight)
            .add_bias(&self.cross_projv_bias); // [batch*seq_kv, embed_dim]

        let q_data = q.data();
        let k_data = k.data();
        let v_data = v.data();

        // Reshape to [batch, num_heads, seq, head_dim] for multi-head attention
        // Q: [batch*seq_q, embed_dim] -> [batch, num_heads, seq_q, head_dim]
        let mut q_heads = vec![0.0f32; batch * num_heads * seq_q * head_dim];
        for b in 0..batch {
            for s in 0..seq_q {
                let src_row = b * seq_q + s;
                for h in 0..num_heads {
                    let dst_idx = ((b * num_heads + h) * seq_q + s) * head_dim;
                    let src_idx = src_row * embed_dim + h * head_dim;
                    q_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&q_data[src_idx..src_idx + head_dim]);
                }
            }
        }

        // K, V: [batch*seq_kv, embed_dim] -> [batch, num_heads, seq_kv, head_dim]
        let mut k_heads = vec![0.0f32; batch * num_heads * seq_kv * head_dim];
        let mut v_heads = vec![0.0f32; batch * num_heads * seq_kv * head_dim];
        for b in 0..batch {
            for s in 0..seq_kv {
                let src_row = b * seq_kv + s;
                for h in 0..num_heads {
                    let dst_idx = ((b * num_heads + h) * seq_kv + s) * head_dim;
                    let src_idx = src_row * embed_dim + h * head_dim;
                    k_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&k_data[src_idx..src_idx + head_dim]);
                    v_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&v_data[src_idx..src_idx + head_dim]);
                }
            }
        }

        // Multi-head cross-attention (parallel for large sequences)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_bh = batch * num_heads;

        let attn_out_data = peregrine::tensor::multi_head_attention(
            &q_heads, &k_heads, &v_heads,
            total_bh, seq_q, seq_kv, head_dim, scale,
        );

        // Direct transpose [batch, num_heads, seq_q, head_dim] -> [batch*seq_q, embed_dim]
        let mut attn_flat_data = vec![0.0f32; batch * seq_q * embed_dim];
        for b in 0..batch {
            for s in 0..seq_q {
                let dst_row = b * seq_q + s;
                for h in 0..num_heads {
                    let src_idx = ((b * num_heads + h) * seq_q + s) * head_dim;
                    let dst_idx = dst_row * embed_dim + h * head_dim;
                    attn_flat_data[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&attn_out_data[src_idx..src_idx + head_dim]);
                }
            }
        }
        let attn_flat = Tensor::new(attn_flat_data, vec![batch * seq_q, embed_dim], false);

        // Output projection
        let attn_proj = attn_flat
            .matmul(&self.cross_proj_weight)
            .add_bias(&self.cross_proj_bias);

        // Residual connection
        x.add(&attn_proj)
    }

    /// Feed-forward network with pre-norm.
    ///
    /// x: [batch * seq, embed_dim]
    /// Returns: x + ffn(norm3(x))
    pub fn forward_ffn(&self, x: &Tensor) -> Tensor {
        let embed_dim = self.embed_dim;

        // Pre-norm
        let normed = x.layer_norm(&self.norm3_weight, &self.norm3_bias, embed_dim);

        // FFN: Linear -> GELU -> Linear
        let hidden = normed
            .matmul(&self.mlp_fc1_weight)
            .add_bias(&self.mlp_fc1_bias)
            .gelu();
        let output = hidden
            .matmul(&self.mlp_fc2_weight)
            .add_bias(&self.mlp_fc2_bias);

        // Residual connection
        x.add(&output)
    }
}

// --------------------------------------------------------------------------
// MUSt3RDecoder
// --------------------------------------------------------------------------

pub struct MUSt3RDecoder {
    pub feat_embed_weight: Tensor, // [enc_dim, dec_dim] (transposed for Peregrine matmul)
    pub feat_embed_bias: Tensor,   // [1, dec_dim]
    pub image2_embed: Tensor,      // [1, 1, 768] learnable parameter
    pub blocks: Vec<CachedDecoderBlock>,
    pub norm_weight: Tensor,       // [768] final LayerNorm
    pub norm_bias: Tensor,         // [768]
    pub rope: RoPE2D,
    num_heads: usize,
    embed_dim: usize,
}

impl MUSt3RDecoder {
    /// Create a new MUSt3R decoder with zero-initialized weights.
    ///
    /// - enc_embed_dim: encoder output dimension (1024 for ViT-Large encoder)
    /// - embed_dim: decoder embedding dimension (768 for ViT-Base decoder)
    /// - depth: number of decoder blocks (12)
    /// - num_heads: number of attention heads (12)
    pub fn new(enc_embed_dim: usize, embed_dim: usize, depth: usize, num_heads: usize) -> Self {
        let blocks = (0..depth)
            .map(|_| CachedDecoderBlock::new(embed_dim, num_heads))
            .collect();

        MUSt3RDecoder {
            feat_embed_weight: Tensor::zeros(&[enc_embed_dim, embed_dim], false),
            feat_embed_bias: Tensor::zeros(&[1, embed_dim], false),
            image2_embed: Tensor::zeros(&[1, 1, embed_dim], false),
            blocks,
            norm_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm_bias: Tensor::zeros(&[embed_dim], false),
            rope: RoPE2D::new(100.0),
            num_heads,
            embed_dim,
        }
    }

    /// Two-view forward pass.
    ///
    /// - enc_feat1: [batch * seq_len, enc_embed_dim] encoder output for image 1 (reference)
    /// - enc_feat2: [batch * seq_len, enc_embed_dim] encoder output for image 2 (target)
    /// - pos1: flattened [seq_len, 2] position data (y, x) for image 1
    /// - pos2: flattened [seq_len, 2] position data (y, x) for image 2
    /// - batch: batch size
    /// - seq_len: number of patches per image
    ///
    /// Returns: (decoded_feat1, decoded_feat2), each [batch * seq_len, embed_dim]
    pub fn forward(
        &self,
        enc_feat1: &Tensor,
        enc_feat2: &Tensor,
        pos1: &[f32],
        pos2: &[f32],
        batch: usize,
        seq_len: usize,
    ) -> (Tensor, Tensor) {
        let total_tokens = batch * seq_len;
        let embed_dim = self.embed_dim;

        // Batch feature embedding: stack both encoder outputs, project once
        let enc1_data = enc_feat1.data();
        let enc2_data = enc_feat2.data();
        let enc_dim = enc1_data.len() / total_tokens;
        let mut enc_both_data = Vec::with_capacity(enc1_data.len() + enc2_data.len());
        enc_both_data.extend_from_slice(&enc1_data);
        enc_both_data.extend_from_slice(&enc2_data);
        let enc_both = Tensor::new(enc_both_data, vec![2 * total_tokens, enc_dim], false);
        let feat_both = enc_both.matmul(&self.feat_embed_weight).add_bias(&self.feat_embed_bias);

        // Split and add image2_embed to feat2
        let feat_data = feat_both.data();
        let half = total_tokens * embed_dim;
        let feat1_data = &feat_data[..half];
        let img2_embed_data = self.image2_embed.data();
        let mut feat2_with_embed = feat_data[half..].to_vec();
        for t in 0..total_tokens {
            for d in 0..embed_dim {
                feat2_with_embed[t * embed_dim + d] += img2_embed_data[d];
            }
        }

        let mut feat1 = Tensor::new(feat1_data.to_vec(), vec![total_tokens, embed_dim], false);
        let mut feat2 = Tensor::new(feat2_with_embed, vec![total_tokens, embed_dim], false);

        let mut total_self_attn = 0.0f64;
        let mut total_cross_attn = 0.0f64;
        let mut total_ffn = 0.0f64;

        for block in &self.blocks {
            // Batched self-attention: stack feat1/feat2, run once with batch=2.
            // Both views use the same positions (same grid), so pos1 works for both.
            let t0 = Instant::now();
            let both = Self::stack_features(&feat1, &feat2, total_tokens, embed_dim);
            let both = block.forward_self_attn(
                &both, pos1, &self.rope, 2 * batch, seq_len, self.num_heads,
            );
            let (f1, f2) = Self::split_features(&both, total_tokens, embed_dim);
            feat1 = f1;
            feat2 = f2;
            total_self_attn += t0.elapsed().as_secs_f64() * 1000.0;

            // Cross-attention: img1 attends to img2 as memory, and vice versa.
            // Cannot batch — each view uses different memory (the other view).
            let t0 = Instant::now();
            let mem1 = feat1.clone();
            let mem2 = feat2.clone();
            feat1 = block.forward_cross_attn(
                &feat1, &mem2, batch, seq_len, seq_len, self.num_heads,
            );
            feat2 = block.forward_cross_attn(
                &feat2, &mem1, batch, seq_len, seq_len, self.num_heads,
            );
            total_cross_attn += t0.elapsed().as_secs_f64() * 1000.0;

            // Batched FFN: stack feat1/feat2, run once
            let t0 = Instant::now();
            let both = Self::stack_features(&feat1, &feat2, total_tokens, embed_dim);
            let both = block.forward_ffn(&both);
            let (f1, f2) = Self::split_features(&both, total_tokens, embed_dim);
            feat1 = f1;
            feat2 = f2;
            total_ffn += t0.elapsed().as_secs_f64() * 1000.0;
        }

        println!("    [Decoder profile] self_attn={:.1}ms cross_attn={:.1}ms ffn={:.1}ms",
                 total_self_attn, total_cross_attn, total_ffn);

        // Batched final layer norm
        let both = Self::stack_features(&feat1, &feat2, total_tokens, embed_dim);
        let normed = both.layer_norm(&self.norm_weight, &self.norm_bias, embed_dim);
        let (out1, out2) = Self::split_features(&normed, total_tokens, embed_dim);

        (out1, out2)
    }

    /// Stack two [tokens, dim] tensors into [2*tokens, dim].
    #[inline]
    fn stack_features(a: &Tensor, b: &Tensor, tokens: usize, dim: usize) -> Tensor {
        let a_data = a.data();
        let b_data = b.data();
        let mut out = Vec::with_capacity(2 * tokens * dim);
        out.extend_from_slice(&a_data);
        out.extend_from_slice(&b_data);
        Tensor::new(out, vec![2 * tokens, dim], false)
    }

    /// Split [2*tokens, dim] tensor into two [tokens, dim] tensors.
    #[inline]
    fn split_features(both: &Tensor, tokens: usize, dim: usize) -> (Tensor, Tensor) {
        let data = both.data();
        let half = tokens * dim;
        let a = Tensor::new(data[..half].to_vec(), vec![tokens, dim], false);
        let b = Tensor::new(data[half..].to_vec(), vec![tokens, dim], false);
        (a, b)
    }

    /// Load pretrained weights from a parameter map.
    ///
    /// The parameter map keys follow the MUSt3R/CroCo naming convention:
    /// - "feat_embed_enc_to_dec.weight" / ".bias"
    /// - "image2_embed"
    /// - "blocks_dec.{i}.norm1.weight" / ".bias"
    /// - "blocks_dec.{i}.attn.qkv.weight" / ".bias"
    /// - "blocks_dec.{i}.attn.proj.weight" / ".bias"
    /// - "blocks_dec.{i}.norm_y.weight" / ".bias"
    /// - "blocks_dec.{i}.norm2.weight" / ".bias"
    /// - "blocks_dec.{i}.cross_attn.projq.weight" / ".bias"
    /// - "blocks_dec.{i}.cross_attn.projk.weight" / ".bias"
    /// - "blocks_dec.{i}.cross_attn.projv.weight" / ".bias"
    /// - "blocks_dec.{i}.cross_attn.proj.weight" / ".bias"
    /// - "blocks_dec.{i}.norm3.weight" / ".bias"
    /// - "blocks_dec.{i}.mlp.fc1.weight" / ".bias"
    /// - "blocks_dec.{i}.mlp.fc2.weight" / ".bias"
    /// - "norm_dec.weight" / ".bias"
    ///
    /// The converter (convert_must3r.py) transposes all 2D Linear weights from
    /// PyTorch [out_features, in_features] to Peregrine [in_features, out_features].
    /// Weights are loaded directly without runtime transposition.
    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>) {
        // Feature embedding (encoder -> decoder dimension)
        // Already transposed to [enc_dim, dec_dim] by converter
        if let Some((shape, data)) = params.get("feat_embed_enc_to_dec.weight") {
            self.feat_embed_weight = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((_s, data)) = params.get("feat_embed_enc_to_dec.bias") {
            self.feat_embed_bias = Tensor::new(data.clone(), vec![1, self.embed_dim], false);
        }

        // Image2 embedding (learnable parameter, not a Linear layer)
        if let Some((shape, data)) = params.get("image2_embed") {
            self.image2_embed = Tensor::new(data.clone(), shape.clone(), false);
        }

        // Decoder blocks
        let ffn_dim = 4 * self.embed_dim;
        for i in 0..self.blocks.len() {
            let prefix = format!("blocks_dec.{}", i);
            let block = &mut self.blocks[i];

            // --- Self-attention ---
            if let Some((_s, data)) = params.get(&format!("{}.norm1.weight", prefix)) {
                block.norm1_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm1.bias", prefix)) {
                block.norm1_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }

            // QKV (fused): already transposed to [embed_dim, 3*embed_dim] by converter
            if let Some((shape, data)) = params.get(&format!("{}.attn.qkv.weight", prefix)) {
                block.qkv_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.attn.qkv.bias", prefix)) {
                block.qkv_bias = Tensor::new(data.clone(), vec![1, 3 * self.embed_dim], false);
            }

            // Output projection: already transposed to [embed_dim, embed_dim] by converter
            if let Some((shape, data)) = params.get(&format!("{}.attn.proj.weight", prefix)) {
                block.proj_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.attn.proj.bias", prefix)) {
                block.proj_bias = Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }

            // --- Cross-attention ---
            if let Some((_s, data)) = params.get(&format!("{}.norm_y.weight", prefix)) {
                block.norm_y_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm_y.bias", prefix)) {
                block.norm_y_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm2.weight", prefix)) {
                block.norm2_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm2.bias", prefix)) {
                block.norm2_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }

            // Cross-attn Q projection: already transposed by converter
            if let Some((shape, data)) =
                params.get(&format!("{}.cross_attn.projq.weight", prefix))
            {
                block.cross_projq_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) =
                params.get(&format!("{}.cross_attn.projq.bias", prefix))
            {
                block.cross_projq_bias =
                    Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }

            // Cross-attn K projection: already transposed by converter
            if let Some((shape, data)) =
                params.get(&format!("{}.cross_attn.projk.weight", prefix))
            {
                block.cross_projk_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) =
                params.get(&format!("{}.cross_attn.projk.bias", prefix))
            {
                block.cross_projk_bias =
                    Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }

            // Cross-attn V projection: already transposed by converter
            if let Some((shape, data)) =
                params.get(&format!("{}.cross_attn.projv.weight", prefix))
            {
                block.cross_projv_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) =
                params.get(&format!("{}.cross_attn.projv.bias", prefix))
            {
                block.cross_projv_bias =
                    Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }

            // Cross-attn output projection: already transposed by converter
            if let Some((shape, data)) =
                params.get(&format!("{}.cross_attn.proj.weight", prefix))
            {
                block.cross_proj_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) =
                params.get(&format!("{}.cross_attn.proj.bias", prefix))
            {
                block.cross_proj_bias =
                    Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }

            // --- FFN ---
            if let Some((_s, data)) = params.get(&format!("{}.norm3.weight", prefix)) {
                block.norm3_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm3.bias", prefix)) {
                block.norm3_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }

            // MLP fc1: already transposed to [embed_dim, ffn_dim] by converter
            if let Some((shape, data)) = params.get(&format!("{}.mlp.fc1.weight", prefix)) {
                block.mlp_fc1_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.mlp.fc1.bias", prefix)) {
                block.mlp_fc1_bias = Tensor::new(data.clone(), vec![1, ffn_dim], false);
            }

            // MLP fc2: already transposed to [ffn_dim, embed_dim] by converter
            if let Some((shape, data)) = params.get(&format!("{}.mlp.fc2.weight", prefix)) {
                block.mlp_fc2_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.mlp.fc2.bias", prefix)) {
                block.mlp_fc2_bias =
                    Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }
        }

        // Final decoder norm
        if let Some((_s, data)) = params.get("norm_dec.weight") {
            self.norm_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
        }
        if let Some((_s, data)) = params.get("norm_dec.bias") {
            self.norm_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
        }
    }
}
