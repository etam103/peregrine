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

    /// Upload all weight tensors to GPU.
    #[cfg(feature = "metal")]
    pub fn to_gpu(&self) {
        // Self-attention
        self.norm1_weight.to_gpu();
        self.norm1_bias.to_gpu();
        self.qkv_weight.to_gpu();
        self.qkv_bias.to_gpu();
        self.proj_weight.to_gpu();
        self.proj_bias.to_gpu();
        // Cross-attention
        self.norm_y_weight.to_gpu();
        self.norm_y_bias.to_gpu();
        self.norm2_weight.to_gpu();
        self.norm2_bias.to_gpu();
        self.cross_projq_weight.to_gpu();
        self.cross_projq_bias.to_gpu();
        self.cross_projk_weight.to_gpu();
        self.cross_projk_bias.to_gpu();
        self.cross_projv_weight.to_gpu();
        self.cross_projv_bias.to_gpu();
        self.cross_proj_weight.to_gpu();
        self.cross_proj_bias.to_gpu();
        // FFN
        self.norm3_weight.to_gpu();
        self.norm3_bias.to_gpu();
        self.mlp_fc1_weight.to_gpu();
        self.mlp_fc1_bias.to_gpu();
        self.mlp_fc2_weight.to_gpu();
        self.mlp_fc2_bias.to_gpu();
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
        use_gpu: bool,
    ) -> Tensor {
        let embed_dim = self.embed_dim;
        let head_dim = embed_dim / num_heads;

        // Pre-norm
        let normed = x.layer_norm(&self.norm1_weight, &self.norm1_bias, embed_dim);

        // Fused QKV projection: [batch*seq, embed_dim] x [embed_dim, 3*embed_dim]
        let qkv = normed.matmul_bias(&self.qkv_weight, &self.qkv_bias);

        // GPU path: reshape + RoPE + SDPA all on GPU, no CPU round-trip
        #[cfg(feature = "metal")]
        if use_gpu {
            let total_bh = batch * num_heads;
            let head_size = total_bh * seq_len * head_dim;
            let scores_size = total_bh * seq_len * seq_len;

            // Precompute RoPE tables on CPU (small: seq_len * quarter floats)
            let (cos_y, sin_y, cos_x, sin_x) = rope.compute_tables(positions, seq_len, head_dim);

            // Dispatch all GPU work using the QKV buffer already on GPU
            let attn_flat = qkv.with_gpu_buf(|qkv_buf| {
                peregrine::metal::with_gpu(|gpu| {
                    let q_buf = gpu.alloc::<f32>(head_size);
                    let k_buf = gpu.alloc::<f32>(head_size);
                    let v_buf = gpu.alloc::<f32>(head_size);

                    gpu.dispatch_qkv_reshape(
                        qkv_buf, &q_buf, &k_buf, &v_buf,
                        batch as u32, seq_len as u32, num_heads as u32, head_dim as u32, embed_dim as u32,
                    );

                    let cos_y_buf = gpu.upload(&cos_y);
                    let sin_y_buf = gpu.upload(&sin_y);
                    let cos_x_buf = gpu.upload(&cos_x);
                    let sin_x_buf = gpu.upload(&sin_x);

                    gpu.dispatch_rope2d(&q_buf, &cos_y_buf, &sin_y_buf, &cos_x_buf, &sin_x_buf,
                        total_bh as u32, seq_len as u32, head_dim as u32);
                    gpu.dispatch_rope2d(&k_buf, &cos_y_buf, &sin_y_buf, &cos_x_buf, &sin_x_buf,
                        total_bh as u32, seq_len as u32, head_dim as u32);

                    let scores_buf = gpu.alloc::<f32>(scores_size);
                    let attn_out_buf = gpu.alloc::<f32>(head_size);
                    let scale = 1.0 / (head_dim as f32).sqrt();
                    gpu.dispatch_sdpa(&q_buf, &k_buf, &v_buf, &scores_buf, &attn_out_buf,
                        total_bh, seq_len, seq_len, head_dim, scale);

                    let attn_flat_buf = gpu.alloc::<f32>(batch * seq_len * embed_dim);
                    gpu.dispatch_attn_output_reshape(&attn_out_buf, &attn_flat_buf,
                        batch as u32, seq_len as u32, num_heads as u32, head_dim as u32, embed_dim as u32);

                    // Return GPU-resident tensor — no sync needed, matmul will use this buffer
                    Tensor::from_gpu(attn_flat_buf, vec![batch * seq_len, embed_dim])
                }).unwrap()
            });

            let attn_proj = attn_flat.matmul_bias(&self.proj_weight, &self.proj_bias);
            return x.add(&attn_proj);
        }

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
        let attn_proj = attn_flat.matmul_bias(&self.proj_weight, &self.proj_bias);

        // Suppress unused variable warning when metal feature is off
        let _ = use_gpu;

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
        use_gpu: bool,
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
            .matmul_bias(&self.cross_projq_weight, &self.cross_projq_bias); // [batch*seq_q, embed_dim]
        let k = memory_normed
            .matmul_bias(&self.cross_projk_weight, &self.cross_projk_bias); // [batch*seq_kv, embed_dim]
        let v = memory_normed
            .matmul_bias(&self.cross_projv_weight, &self.cross_projv_bias); // [batch*seq_kv, embed_dim]

        // GPU path: reshape + SDPA on GPU (no RoPE for cross-attention), no CPU round-trip
        #[cfg(feature = "metal")]
        if use_gpu {
            let total_bh = batch * num_heads;
            let q_head_size = total_bh * seq_q * head_dim;
            let kv_head_size = total_bh * seq_kv * head_dim;
            let scores_size = total_bh * seq_q * seq_kv;

            let attn_flat = q.with_gpu_buf(|q_flat_buf| {
                k.with_gpu_buf(|k_flat_buf| {
                    v.with_gpu_buf(|v_flat_buf| {
                        peregrine::metal::with_gpu(|gpu| {
                            let q_buf = gpu.alloc::<f32>(q_head_size);
                            let k_buf = gpu.alloc::<f32>(kv_head_size);
                            let v_buf = gpu.alloc::<f32>(kv_head_size);

                            gpu.dispatch_separate_reshape(q_flat_buf, &q_buf,
                                batch as u32, seq_q as u32, num_heads as u32, head_dim as u32, embed_dim as u32);
                            gpu.dispatch_separate_reshape(k_flat_buf, &k_buf,
                                batch as u32, seq_kv as u32, num_heads as u32, head_dim as u32, embed_dim as u32);
                            gpu.dispatch_separate_reshape(v_flat_buf, &v_buf,
                                batch as u32, seq_kv as u32, num_heads as u32, head_dim as u32, embed_dim as u32);

                            let scores_buf = gpu.alloc::<f32>(scores_size);
                            let attn_out_buf = gpu.alloc::<f32>(q_head_size);
                            let scale = 1.0 / (head_dim as f32).sqrt();
                            gpu.dispatch_sdpa(&q_buf, &k_buf, &v_buf, &scores_buf, &attn_out_buf,
                                total_bh, seq_q, seq_kv, head_dim, scale);

                            let attn_flat_buf = gpu.alloc::<f32>(batch * seq_q * embed_dim);
                            gpu.dispatch_attn_output_reshape(&attn_out_buf, &attn_flat_buf,
                                batch as u32, seq_q as u32, num_heads as u32, head_dim as u32, embed_dim as u32);

                            Tensor::from_gpu(attn_flat_buf, vec![batch * seq_q, embed_dim])
                        }).unwrap()
                    })
                })
            });

            let attn_proj = attn_flat
                .matmul_bias(&self.cross_proj_weight, &self.cross_proj_bias);
            return x.add(&attn_proj);
        }

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
            .matmul_bias(&self.cross_proj_weight, &self.cross_proj_bias);

        // Suppress unused variable warning when metal feature is off
        let _ = use_gpu;

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

        // FFN: Linear -> GELU -> Linear (fused on GPU)
        let hidden = normed.matmul_bias_gelu(&self.mlp_fc1_weight, &self.mlp_fc1_bias);
        let output = hidden
            .matmul_bias(&self.mlp_fc2_weight, &self.mlp_fc2_bias);

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

    /// Upload all weight tensors to GPU.
    #[cfg(feature = "metal")]
    pub fn to_gpu(&self) {
        self.feat_embed_weight.to_gpu();
        self.feat_embed_bias.to_gpu();
        // image2_embed is a small learnable param, upload it too
        self.image2_embed.to_gpu();
        for block in &self.blocks {
            block.to_gpu();
        }
        self.norm_weight.to_gpu();
        self.norm_bias.to_gpu();
    }

    /// Two-view forward pass.
    ///
    /// - enc_feat1: [batch * seq_len, enc_embed_dim] encoder output for image 1 (reference)
    /// - enc_feat2: [batch * seq_len, enc_embed_dim] encoder output for image 2 (target)
    /// - pos1: flattened [seq_len, 2] position data (y, x) for image 1
    /// - pos2: flattened [seq_len, 2] position data (y, x) for image 2
    /// - batch: batch size
    /// - seq_len: number of patches per image
    /// - use_gpu: whether GPU acceleration is active
    /// - pipeline: if true, overlap feat1 (GPU) and feat2 (CPU/AMX) via het_execute
    ///
    /// Returns: (decoded_feat1, decoded_feat2), each [batch * seq_len, embed_dim]
    pub fn forward(
        &self,
        enc_feat1: &Tensor,
        enc_feat2: &Tensor,
        pos1: &[f32],
        _pos2: &[f32],
        batch: usize,
        seq_len: usize,
        use_gpu: bool,
        pipeline: bool,
    ) -> (Tensor, Tensor) {
        let total_tokens = batch * seq_len;
        let embed_dim = self.embed_dim;

        // GPU path: project each view separately, no stack/split round-trips.
        // All tensors stay GPU-resident throughout the entire decoder.
        #[cfg(feature = "metal")]
        if use_gpu {
            // Project each encoder output to decoder dimension (stays on GPU)
            let mut feat1 = enc_feat1.matmul_bias(&self.feat_embed_weight, &self.feat_embed_bias);
            let mut feat2 = enc_feat2.matmul_bias(&self.feat_embed_weight, &self.feat_embed_bias);

            // Add image2_embed to feat2 via add_bias (image2_embed is [1,1,768] → reshape to [1,768])
            let img2_data = self.image2_embed.data();
            let img2_bias = Tensor::new(img2_data, vec![1, embed_dim], false);
            img2_bias.to_gpu();
            feat2 = feat2.add_bias(&img2_bias);

            let mut total_self_attn = 0.0f64;
            let mut total_cross_attn = 0.0f64;
            let mut total_ffn = 0.0f64;

            if pipeline {
                // Pipelined mode: overlap feat1 (GPU) with feat2 (CPU/AMX).
                // Bring feat2 to CPU for the CPU/AMX path — weights are
                // StorageModeShared so CPU can read them directly.
                peregrine::metal::gpu_sync();
                feat2.to_cpu();

                for block in &self.blocks {
                    // --- Self-attention: feat1 on GPU, feat2 on CPU/AMX ---
                    let t0 = Instant::now();
                    let (new_feat1, new_feat2) = peregrine::metal::het_execute(
                        || block.forward_self_attn(
                            &feat1, pos1, &self.rope, batch, seq_len, self.num_heads, true,
                        ),
                        || block.forward_self_attn(
                            &feat2, pos1, &self.rope, batch, seq_len, self.num_heads, false,
                        ),
                    );
                    feat1 = new_feat1;
                    feat2 = new_feat2;
                    total_self_attn += t0.elapsed().as_secs_f64() * 1000.0;

                    // --- Cross-attention: feat1(GPU) attends to mem2, feat2(CPU) attends to mem1 ---
                    let t0 = Instant::now();
                    // Get mem1 from GPU→CPU for feat2's CPU cross-attention
                    peregrine::metal::gpu_sync();
                    let mem1_data = feat1.data();
                    let mem1_cpu = Tensor::new(mem1_data, vec![total_tokens, embed_dim], false);
                    // Upload mem2 (CPU) to GPU for feat1's GPU cross-attention
                    let mem2_data = feat2.data();
                    let mem2_gpu = Tensor::new(mem2_data, vec![total_tokens, embed_dim], false);
                    mem2_gpu.to_gpu();
                    let (new_feat1, new_feat2) = peregrine::metal::het_execute(
                        || block.forward_cross_attn(
                            &feat1, &mem2_gpu, batch, seq_len, seq_len, self.num_heads, true,
                        ),
                        || block.forward_cross_attn(
                            &feat2, &mem1_cpu, batch, seq_len, seq_len, self.num_heads, false,
                        ),
                    );
                    feat1 = new_feat1;
                    feat2 = new_feat2;
                    total_cross_attn += t0.elapsed().as_secs_f64() * 1000.0;

                    // --- FFN: feat1 on GPU, feat2 on CPU/AMX ---
                    let t0 = Instant::now();
                    let (new_feat1, new_feat2) = peregrine::metal::het_execute(
                        || block.forward_ffn(&feat1),
                        || block.forward_ffn(&feat2),
                    );
                    feat1 = new_feat1;
                    feat2 = new_feat2;
                    total_ffn += t0.elapsed().as_secs_f64() * 1000.0;
                }

                eprintln!("    [Decoder profile (pipelined)] self_attn={:.1}ms cross_attn={:.1}ms ffn={:.1}ms",
                         total_self_attn, total_cross_attn, total_ffn);
            } else {
                for block in &self.blocks {
                    // Self-attention on each view separately (no stacking needed)
                    let t0 = Instant::now();
                    feat1 = block.forward_self_attn(
                        &feat1, pos1, &self.rope, batch, seq_len, self.num_heads, true,
                    );
                    feat2 = block.forward_self_attn(
                        &feat2, pos1, &self.rope, batch, seq_len, self.num_heads, true,
                    );
                    total_self_attn += t0.elapsed().as_secs_f64() * 1000.0;

                    // Cross-attention (already separate — each view attends to the other)
                    let t0 = Instant::now();
                    let mem1 = feat1.clone();
                    let mem2 = feat2.clone();
                    feat1 = block.forward_cross_attn(
                        &feat1, &mem2, batch, seq_len, seq_len, self.num_heads, true,
                    );
                    feat2 = block.forward_cross_attn(
                        &feat2, &mem1, batch, seq_len, seq_len, self.num_heads, true,
                    );
                    total_cross_attn += t0.elapsed().as_secs_f64() * 1000.0;

                    // FFN on each view separately (no stacking needed)
                    let t0 = Instant::now();
                    feat1 = block.forward_ffn(&feat1);
                    feat2 = block.forward_ffn(&feat2);
                    total_ffn += t0.elapsed().as_secs_f64() * 1000.0;
                }

                eprintln!("    [Decoder profile] self_attn={:.1}ms cross_attn={:.1}ms ffn={:.1}ms",
                         total_self_attn, total_cross_attn, total_ffn);
            }

            // Final layer norm on each view separately
            // If pipelined, bring feat2 back to GPU for consistent output
            if pipeline {
                feat2.to_gpu();
            }
            let out1 = feat1.layer_norm(&self.norm_weight, &self.norm_bias, embed_dim);
            let out2 = feat2.layer_norm(&self.norm_weight, &self.norm_bias, embed_dim);

            return (out1, out2);
        }

        // CPU path: stack views for batched operations
        let enc1_data = enc_feat1.data();
        let enc2_data = enc_feat2.data();
        let enc_dim = enc1_data.len() / total_tokens;
        let mut enc_both_data = Vec::with_capacity(enc1_data.len() + enc2_data.len());
        enc_both_data.extend_from_slice(&enc1_data);
        enc_both_data.extend_from_slice(&enc2_data);
        let enc_both = Tensor::new(enc_both_data, vec![2 * total_tokens, enc_dim], false);

        let feat_both = enc_both.matmul_bias(&self.feat_embed_weight, &self.feat_embed_bias);

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
            let t0 = Instant::now();
            let both = Self::stack_features(&feat1, &feat2, total_tokens, embed_dim, false);
            let both = block.forward_self_attn(
                &both, pos1, &self.rope, 2 * batch, seq_len, self.num_heads, false,
            );
            let (f1, f2) = Self::split_features(&both, total_tokens, embed_dim, false);
            feat1 = f1;
            feat2 = f2;
            total_self_attn += t0.elapsed().as_secs_f64() * 1000.0;

            // Cross-attention
            let t0 = Instant::now();
            let mem1 = feat1.clone();
            let mem2 = feat2.clone();
            feat1 = block.forward_cross_attn(
                &feat1, &mem2, batch, seq_len, seq_len, self.num_heads, false,
            );
            feat2 = block.forward_cross_attn(
                &feat2, &mem1, batch, seq_len, seq_len, self.num_heads, false,
            );
            total_cross_attn += t0.elapsed().as_secs_f64() * 1000.0;

            // Batched FFN: stack feat1/feat2, run once
            let t0 = Instant::now();
            let both = Self::stack_features(&feat1, &feat2, total_tokens, embed_dim, false);
            let both = block.forward_ffn(&both);
            let (f1, f2) = Self::split_features(&both, total_tokens, embed_dim, false);
            feat1 = f1;
            feat2 = f2;
            total_ffn += t0.elapsed().as_secs_f64() * 1000.0;
        }

        eprintln!("    [Decoder profile] self_attn={:.1}ms cross_attn={:.1}ms ffn={:.1}ms",
                 total_self_attn, total_cross_attn, total_ffn);

        // Suppress unused variable warning when metal feature is off
        let _ = pipeline;

        // Batched final layer norm
        let both = Self::stack_features(&feat1, &feat2, total_tokens, embed_dim, false);
        let normed = both.layer_norm(&self.norm_weight, &self.norm_bias, embed_dim);
        let (out1, out2) = Self::split_features(&normed, total_tokens, embed_dim, false);

        (out1, out2)
    }

    /// Stack two [tokens, dim] tensors into [2*tokens, dim].
    #[inline]
    fn stack_features(a: &Tensor, b: &Tensor, tokens: usize, dim: usize, use_gpu: bool) -> Tensor {
        let a_data = a.data();
        let b_data = b.data();
        let mut out = Vec::with_capacity(2 * tokens * dim);
        out.extend_from_slice(&a_data);
        out.extend_from_slice(&b_data);
        let t = Tensor::new(out, vec![2 * tokens, dim], false);
        #[cfg(feature = "metal")]
        if use_gpu { t.to_gpu(); }
        let _ = use_gpu;
        t
    }

    /// Split [2*tokens, dim] tensor into two [tokens, dim] tensors.
    #[inline]
    fn split_features(both: &Tensor, tokens: usize, dim: usize, use_gpu: bool) -> (Tensor, Tensor) {
        let data = both.data();
        let half = tokens * dim;
        let a = Tensor::new(data[..half].to_vec(), vec![tokens, dim], false);
        let b = Tensor::new(data[half..].to_vec(), vec![tokens, dim], false);
        #[cfg(feature = "metal")]
        if use_gpu { a.to_gpu(); b.to_gpu(); }
        let _ = use_gpu;
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
