/// DUSt3R / MUSt3R encoder: ViT-Large with 2D RoPE.
///
/// Architecture:
/// - PatchEmbed: Conv2d(3, 1024, kernel=16, stride=16) -> [batch, num_patches, 1024]
/// - 24 EncoderBlocks (pre-norm, fused QKV, 2D RoPE attention, GELU FFN)
/// - Final LayerNorm(1024)
///
/// All tensors use `requires_grad: false` (inference only).

use std::collections::HashMap;
use peregrine::tensor::Tensor;
use super::rope2d::RoPE2D;

// ---------------------------------------------------------------------------
// PatchEmbed: Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
// ---------------------------------------------------------------------------

pub struct PatchEmbed {
    pub weight: Tensor, // [embed_dim, 3, patch_size, patch_size]
    pub bias: Tensor,   // [embed_dim]
    patch_size: usize,
}

impl PatchEmbed {
    pub fn new(embed_dim: usize, patch_size: usize) -> Self {
        PatchEmbed {
            weight: Tensor::zeros(
                &[embed_dim, 3, patch_size, patch_size],
                false,
            ),
            bias: Tensor::zeros(&[embed_dim], false),
            patch_size,
        }
    }

    /// Forward pass for patch embedding.
    ///
    /// # Arguments
    /// * `img` - Tensor with data layout [batch, 3, H, W]
    /// * `batch` - batch size
    /// * `channels` - number of channels (3)
    /// * `height` - image height (must be divisible by patch_size)
    /// * `width` - image width (must be divisible by patch_size)
    ///
    /// # Returns
    /// Tensor of shape [batch * num_patches, embed_dim]
    /// where num_patches = (H / patch_size) * (W / patch_size)
    pub fn forward(
        &self,
        img: &Tensor,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
    ) -> Tensor {
        assert_eq!(channels, 3, "PatchEmbed expects 3 input channels");
        assert_eq!(
            height % self.patch_size,
            0,
            "Image height must be divisible by patch_size"
        );
        assert_eq!(
            width % self.patch_size,
            0,
            "Image width must be divisible by patch_size"
        );

        // Use conv2d_strided: [batch, 3, H, W] -> [batch, embed_dim, H/ps, W/ps]
        let conv_out = img.conv2d_strided(
            &self.weight,
            &self.bias,
            (self.patch_size, self.patch_size),
            (0, 0),
        );

        // conv_out shape: [batch, embed_dim, grid_h, grid_w]
        let conv_shape = conv_out.shape();
        let embed_dim = conv_shape[1];
        let grid_h = conv_shape[2];
        let grid_w = conv_shape[3];
        let num_patches = grid_h * grid_w;

        // Reshape to [batch, embed_dim, num_patches], transpose to [batch, num_patches, embed_dim],
        // then flatten to [batch * num_patches, embed_dim].
        conv_out
            .reshape(vec![batch, embed_dim, num_patches])
            .transpose(1, 2)
            .reshape(vec![batch * num_patches, embed_dim])
    }
}

// ---------------------------------------------------------------------------
// EncoderBlock: pre-norm attention + pre-norm FFN with residual connections
// ---------------------------------------------------------------------------

pub struct EncoderBlock {
    pub norm1_weight: Tensor, // [embed_dim] - LayerNorm gamma
    pub norm1_bias: Tensor,   // [embed_dim] - LayerNorm beta
    pub qkv_weight: Tensor,   // [embed_dim, 3 * embed_dim] (transposed for Peregrine matmul)
    pub qkv_bias: Tensor,     // [1, 3 * embed_dim]
    pub proj_weight: Tensor,  // [embed_dim, embed_dim] (transposed for Peregrine matmul)
    pub proj_bias: Tensor,    // [1, embed_dim]
    pub norm2_weight: Tensor, // [embed_dim]
    pub norm2_bias: Tensor,   // [embed_dim]
    pub mlp_fc1_weight: Tensor, // [embed_dim, mlp_dim] (transposed for Peregrine matmul)
    pub mlp_fc1_bias: Tensor,   // [1, mlp_dim]
    pub mlp_fc2_weight: Tensor, // [mlp_dim, embed_dim] (transposed for Peregrine matmul)
    pub mlp_fc2_bias: Tensor,   // [1, embed_dim]
    embed_dim: usize,
}

impl EncoderBlock {
    pub fn new(embed_dim: usize, mlp_dim: usize) -> Self {
        EncoderBlock {
            norm1_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm1_bias: Tensor::zeros(&[embed_dim], false),
            // Peregrine matmul: x [N, embed_dim] * w [embed_dim, 3*embed_dim] -> [N, 3*embed_dim]
            qkv_weight: Tensor::zeros(&[embed_dim, 3 * embed_dim], false),
            qkv_bias: Tensor::zeros(&[1, 3 * embed_dim], false),
            proj_weight: Tensor::zeros(&[embed_dim, embed_dim], false),
            proj_bias: Tensor::zeros(&[1, embed_dim], false),
            norm2_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm2_bias: Tensor::zeros(&[embed_dim], false),
            mlp_fc1_weight: Tensor::zeros(&[embed_dim, mlp_dim], false),
            mlp_fc1_bias: Tensor::zeros(&[1, mlp_dim], false),
            mlp_fc2_weight: Tensor::zeros(&[mlp_dim, embed_dim], false),
            mlp_fc2_bias: Tensor::zeros(&[1, embed_dim], false),
            embed_dim,
        }
    }

    /// Forward pass for one encoder block.
    ///
    /// # Arguments
    /// * `x` - input tensor [batch * seq_len, embed_dim]
    /// * `positions` - flattened [seq_len, 2] position data (y, x) for 2D RoPE
    /// * `rope` - RoPE2D instance
    /// * `batch` - batch size
    /// * `seq_len` - sequence length (number of patches)
    /// * `num_heads` - number of attention heads
    ///
    /// # Returns
    /// Tensor [batch * seq_len, embed_dim]
    pub fn forward(
        &self,
        x: &Tensor,
        positions: &[f32],
        rope: &RoPE2D,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> Tensor {
        let head_dim = self.embed_dim / num_heads;

        // --- Pre-norm + Self-Attention ---
        let normed = x.layer_norm(&self.norm1_weight, &self.norm1_bias, self.embed_dim);

        // Fused QKV projection: [batch*seq, embed_dim] * [embed_dim, 3*embed_dim] -> [batch*seq, 3*embed_dim]
        let qkv = normed.matmul(&self.qkv_weight).add_bias(&self.qkv_bias);
        let qkv_data = qkv.data();

        // Split into Q, K, V: each [batch * seq_len, embed_dim]
        let total_tokens = batch * seq_len;
        let qkv_stride = 3 * self.embed_dim;
        let mut q_data = Vec::with_capacity(total_tokens * self.embed_dim);
        let mut k_data = Vec::with_capacity(total_tokens * self.embed_dim);
        let mut v_data = Vec::with_capacity(total_tokens * self.embed_dim);

        for t in 0..total_tokens {
            let base = t * qkv_stride;
            q_data.extend_from_slice(&qkv_data[base..base + self.embed_dim]);
            k_data.extend_from_slice(&qkv_data[base + self.embed_dim..base + 2 * self.embed_dim]);
            v_data.extend_from_slice(&qkv_data[base + 2 * self.embed_dim..base + 3 * self.embed_dim]);
        }

        // Reshape to [batch, num_heads, seq_len, head_dim] for RoPE application.
        // Current layout: q_data is [batch * seq_len, embed_dim] (row-major).
        // We need [batch, num_heads, seq_len, head_dim] which we can obtain by:
        //   [batch, seq_len, num_heads, head_dim] -> transpose(1,2) -> [batch, num_heads, seq_len, head_dim]
        // But since RoPE operates on raw data in [num_heads, seq_len, head_dim] layout per batch,
        // we do the reshape manually.

        let mut q_heads = vec![0.0f32; batch * num_heads * seq_len * head_dim];
        let mut k_heads = vec![0.0f32; batch * num_heads * seq_len * head_dim];

        // q_data layout: [batch*seq_len, embed_dim] where embed_dim = num_heads * head_dim
        // We want: [batch, num_heads, seq_len, head_dim]
        for b in 0..batch {
            for s in 0..seq_len {
                let src_row = b * seq_len + s;
                for h in 0..num_heads {
                    let dst_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let src_idx = src_row * self.embed_dim + h * head_dim;
                    q_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&q_data[src_idx..src_idx + head_dim]);
                    k_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&k_data[src_idx..src_idx + head_dim]);
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

        // Reshape V to [batch, num_heads, seq_len, head_dim] (same reorder as Q/K)
        let mut v_heads = vec![0.0f32; batch * num_heads * seq_len * head_dim];
        for b in 0..batch {
            for s in 0..seq_len {
                let src_row = b * seq_len + s;
                for h in 0..num_heads {
                    let dst_idx = ((b * num_heads + h) * seq_len + s) * head_dim;
                    let src_idx = src_row * self.embed_dim + h * head_dim;
                    v_heads[dst_idx..dst_idx + head_dim]
                        .copy_from_slice(&v_data[src_idx..src_idx + head_dim]);
                }
            }
        }

        // Scaled dot-product attention per (batch, head)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut attn_out_data = Vec::with_capacity(batch * num_heads * seq_len * head_dim);

        for bh in 0..(batch * num_heads) {
            let q_off = bh * seq_len * head_dim;
            let k_off = bh * seq_len * head_dim;
            let v_off = bh * seq_len * head_dim;

            let q_slice = q_heads[q_off..q_off + seq_len * head_dim].to_vec();
            let k_slice = k_heads[k_off..k_off + seq_len * head_dim].to_vec();
            let v_slice = v_heads[v_off..v_off + seq_len * head_dim].to_vec();

            // Q: [seq_len, head_dim], K: [seq_len, head_dim], V: [seq_len, head_dim]
            let q_t = Tensor::new(q_slice, vec![seq_len, head_dim], false);
            let k_t = Tensor::new(k_slice, vec![seq_len, head_dim], false);
            let v_t = Tensor::new(v_slice, vec![seq_len, head_dim], false);

            // scores = Q * K^T * scale: [seq_len, seq_len]
            let k_transposed = k_t.transpose(0, 1);
            let scores = q_t.matmul(&k_transposed).scale(scale);
            let attn_weights = scores.softmax(-1);

            // context = attn_weights * V: [seq_len, head_dim]
            let context = attn_weights.matmul(&v_t);
            attn_out_data.extend(context.data());
        }

        // attn_out_data is [batch, num_heads, seq_len, head_dim]
        // Need to transpose back to [batch, seq_len, num_heads, head_dim] then flatten.
        let attn_out = Tensor::new(
            attn_out_data,
            vec![batch, num_heads, seq_len, head_dim],
            false,
        );
        let attn_out = attn_out.transpose(1, 2); // [batch, seq_len, num_heads, head_dim]
        let attn_flat = attn_out.reshape(vec![batch * seq_len, self.embed_dim]);

        // Output projection
        let attn_proj = attn_flat.matmul(&self.proj_weight).add_bias(&self.proj_bias);

        // Residual connection
        let x = x.add(&attn_proj);

        // --- Pre-norm + FFN ---
        let normed = x.layer_norm(&self.norm2_weight, &self.norm2_bias, self.embed_dim);
        let h = normed.matmul(&self.mlp_fc1_weight).add_bias(&self.mlp_fc1_bias).gelu();
        let ffn_out = h.matmul(&self.mlp_fc2_weight).add_bias(&self.mlp_fc2_bias);

        // Residual connection
        x.add(&ffn_out)
    }
}

// ---------------------------------------------------------------------------
// Dust3rEncoder: Full ViT-Large encoder
// ---------------------------------------------------------------------------

pub struct Dust3rEncoder {
    pub patch_embed: PatchEmbed,
    pub blocks: Vec<EncoderBlock>,
    pub norm_weight: Tensor, // final LayerNorm gamma [embed_dim]
    pub norm_bias: Tensor,   // final LayerNorm beta  [embed_dim]
    pub rope: RoPE2D,
    num_heads: usize,
    embed_dim: usize,
    patch_size: usize,
}

impl Dust3rEncoder {
    /// Create a new encoder with zero-initialized weights.
    ///
    /// # Arguments
    /// * `embed_dim` - embedding dimension (1024 for ViT-Large)
    /// * `depth` - number of encoder blocks (24 for ViT-Large)
    /// * `num_heads` - number of attention heads (16 for ViT-Large)
    /// * `patch_size` - patch size for embedding (16)
    pub fn new(embed_dim: usize, depth: usize, num_heads: usize, patch_size: usize) -> Self {
        let mlp_dim = 4 * embed_dim; // 4096 for ViT-Large

        Dust3rEncoder {
            patch_embed: PatchEmbed::new(embed_dim, patch_size),
            blocks: (0..depth)
                .map(|_| EncoderBlock::new(embed_dim, mlp_dim))
                .collect(),
            norm_weight: Tensor::new(vec![1.0; embed_dim], vec![embed_dim], false),
            norm_bias: Tensor::zeros(&[embed_dim], false),
            rope: RoPE2D::new(100.0),
            num_heads,
            embed_dim,
            patch_size,
        }
    }

    /// Run encoder forward pass.
    ///
    /// # Arguments
    /// * `img` - image tensor [batch, 3, height, width]
    /// * `batch` - batch size
    /// * `height` - image height
    /// * `width` - image width
    ///
    /// # Returns
    /// (features, positions) where:
    /// - features: Tensor [batch * seq_len, embed_dim]
    /// - positions: Vec<f32> flattened [seq_len, 2] with (y, x) per patch
    pub fn forward(
        &self,
        img: &Tensor,
        batch: usize,
        height: usize,
        width: usize,
    ) -> (Tensor, Vec<f32>) {
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let seq_len = grid_h * grid_w;

        // Compute 2D grid positions: (y, x) for each patch
        let mut positions = Vec::with_capacity(seq_len * 2);
        for y in 0..grid_h {
            for x in 0..grid_w {
                positions.push(y as f32);
                positions.push(x as f32);
            }
        }

        // Patch embedding: [batch, 3, H, W] -> [batch * seq_len, embed_dim]
        let mut x = self.patch_embed.forward(img, batch, 3, height, width);

        // Encoder blocks
        for block in &self.blocks {
            x = block.forward(&x, &positions, &self.rope, batch, seq_len, self.num_heads);
        }

        // Final LayerNorm
        let x = x.layer_norm(&self.norm_weight, &self.norm_bias, self.embed_dim);

        (x, positions)
    }

    /// Load weights from a parameter dictionary.
    ///
    /// Expected key format (matching PyTorch DUSt3R checkpoint):
    /// - "patch_embed.proj.weight" -> [embed_dim, 3, patch_size, patch_size]
    /// - "patch_embed.proj.bias" -> [embed_dim]
    /// - "blocks_enc.{i}.norm1.weight" -> [embed_dim]
    /// - "blocks_enc.{i}.norm1.bias" -> [embed_dim]
    /// - "blocks_enc.{i}.attn.qkv.weight" -> [3*embed_dim, embed_dim] (PyTorch layout)
    /// - "blocks_enc.{i}.attn.qkv.bias" -> [3*embed_dim]
    /// - "blocks_enc.{i}.attn.proj.weight" -> [embed_dim, embed_dim] (PyTorch layout)
    /// - "blocks_enc.{i}.attn.proj.bias" -> [embed_dim]
    /// - "blocks_enc.{i}.norm2.weight" -> [embed_dim]
    /// - "blocks_enc.{i}.norm2.bias" -> [embed_dim]
    /// - "blocks_enc.{i}.mlp.fc1.weight" -> [mlp_dim, embed_dim] (PyTorch layout)
    /// - "blocks_enc.{i}.mlp.fc1.bias" -> [mlp_dim]
    /// - "blocks_enc.{i}.mlp.fc2.weight" -> [embed_dim, mlp_dim] (PyTorch layout)
    /// - "blocks_enc.{i}.mlp.fc2.bias" -> [embed_dim]
    /// - "norm_enc.weight" -> [embed_dim]
    /// - "norm_enc.bias" -> [embed_dim]
    ///
    /// PyTorch Linear stores weights as [out_features, in_features].
    /// Peregrine matmul expects [in_features, out_features].
    /// So all Linear weights are transposed during loading.
    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>) {
        // --- Patch embedding ---
        if let Some((shape, data)) = params.get("patch_embed.proj.weight") {
            // Conv2d weight: [embed_dim, 3, ps, ps] - no transpose needed for conv2d_strided
            self.patch_embed.weight = Tensor::new(data.clone(), shape.clone(), false);
        }
        if let Some((_shape, data)) = params.get("patch_embed.proj.bias") {
            self.patch_embed.bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
        }

        // --- Encoder blocks ---
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks_enc.{}", i);

            // norm1
            if let Some((_s, data)) = params.get(&format!("{}.norm1.weight", prefix)) {
                block.norm1_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm1.bias", prefix)) {
                block.norm1_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }

            // QKV (fused): PyTorch [3*embed_dim, embed_dim] -> transpose to [embed_dim, 3*embed_dim]
            if let Some((shape, data)) = params.get(&format!("{}.attn.qkv.weight", prefix)) {
                let out_f = shape[0]; // 3 * embed_dim
                let in_f = shape[1];  // embed_dim
                let pt_tensor = Tensor::new(data.clone(), vec![out_f, in_f], false);
                block.qkv_weight = pt_tensor.transpose(0, 1);
            }
            if let Some((_s, data)) = params.get(&format!("{}.attn.qkv.bias", prefix)) {
                block.qkv_bias = Tensor::new(data.clone(), vec![1, 3 * self.embed_dim], false);
            }

            // Output projection: PyTorch [embed_dim, embed_dim] -> transpose
            if let Some((shape, data)) = params.get(&format!("{}.attn.proj.weight", prefix)) {
                let out_f = shape[0];
                let in_f = shape[1];
                let pt_tensor = Tensor::new(data.clone(), vec![out_f, in_f], false);
                block.proj_weight = pt_tensor.transpose(0, 1);
            }
            if let Some((_s, data)) = params.get(&format!("{}.attn.proj.bias", prefix)) {
                block.proj_bias = Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }

            // norm2
            if let Some((_s, data)) = params.get(&format!("{}.norm2.weight", prefix)) {
                block.norm2_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm2.bias", prefix)) {
                block.norm2_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }

            // MLP fc1: PyTorch [mlp_dim, embed_dim] -> transpose to [embed_dim, mlp_dim]
            let mlp_dim = 4 * self.embed_dim;
            if let Some((shape, data)) = params.get(&format!("{}.mlp.fc1.weight", prefix)) {
                let out_f = shape[0]; // mlp_dim
                let in_f = shape[1];  // embed_dim
                let pt_tensor = Tensor::new(data.clone(), vec![out_f, in_f], false);
                block.mlp_fc1_weight = pt_tensor.transpose(0, 1);
            }
            if let Some((_s, data)) = params.get(&format!("{}.mlp.fc1.bias", prefix)) {
                block.mlp_fc1_bias = Tensor::new(data.clone(), vec![1, mlp_dim], false);
            }

            // MLP fc2: PyTorch [embed_dim, mlp_dim] -> transpose to [mlp_dim, embed_dim]
            if let Some((shape, data)) = params.get(&format!("{}.mlp.fc2.weight", prefix)) {
                let out_f = shape[0]; // embed_dim
                let in_f = shape[1];  // mlp_dim
                let pt_tensor = Tensor::new(data.clone(), vec![out_f, in_f], false);
                block.mlp_fc2_weight = pt_tensor.transpose(0, 1);
            }
            if let Some((_s, data)) = params.get(&format!("{}.mlp.fc2.bias", prefix)) {
                block.mlp_fc2_bias = Tensor::new(data.clone(), vec![1, self.embed_dim], false);
            }
        }

        // --- Final LayerNorm ---
        if let Some((_s, data)) = params.get("norm_enc.weight") {
            self.norm_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
        }
        if let Some((_s, data)) = params.get("norm_enc.bias") {
            self.norm_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
        }
    }
}
