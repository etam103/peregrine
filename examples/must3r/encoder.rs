/// DUSt3R / MUSt3R encoder: ViT-Large with 2D RoPE.
///
/// Architecture:
/// - PatchEmbed: Conv2d(3, 1024, kernel=16, stride=16) -> [batch, num_patches, 1024]
/// - 24 EncoderBlocks (pre-norm, fused QKV, 2D RoPE attention, GELU FFN)
/// - Final LayerNorm(1024)
///
/// All tensors use `requires_grad: false` (inference only).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;
use peregrine::tensor::Tensor;
use super::rope2d::RoPE2D;

static ENCODER_PROFILE_PRINTED: AtomicBool = AtomicBool::new(false);

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

    #[cfg(feature = "metal")]
    pub fn to_gpu(&self) {
        self.weight.to_gpu();
        self.bias.to_gpu();
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

    #[cfg(feature = "metal")]
    pub fn to_gpu(&self) {
        self.norm1_weight.to_gpu();
        self.norm1_bias.to_gpu();
        self.qkv_weight.to_gpu();
        self.qkv_bias.to_gpu();
        self.proj_weight.to_gpu();
        self.proj_bias.to_gpu();
        self.norm2_weight.to_gpu();
        self.norm2_bias.to_gpu();
        self.mlp_fc1_weight.to_gpu();
        self.mlp_fc1_bias.to_gpu();
        self.mlp_fc2_weight.to_gpu();
        self.mlp_fc2_bias.to_gpu();
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
    /// * `use_gpu` - whether to upload attention output back to GPU
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
        use_gpu: bool,
    ) -> Tensor {
        let head_dim = self.embed_dim / num_heads;
        let embed_dim = self.embed_dim;
        let t_block = Instant::now();

        // --- Pre-norm + Self-Attention ---
        let t0 = Instant::now();
        let normed = x.layer_norm(&self.norm1_weight, &self.norm1_bias, self.embed_dim);
        let dt_norm1 = t0.elapsed();

        // Fused QKV projection: [batch*seq, embed_dim] * [embed_dim, 3*embed_dim] -> [batch*seq, 3*embed_dim]
        let t0 = Instant::now();
        let qkv = normed.matmul(&self.qkv_weight).add_bias(&self.qkv_bias);
        let dt_qkv = t0.elapsed();

        // GPU path: reshape + RoPE + SDPA all on GPU, no CPU round-trip
        let t0 = Instant::now();
        #[cfg(feature = "metal")]
        let gpu_attn_result = if use_gpu {
            let total_bh = batch * num_heads;
            let head_size = total_bh * seq_len * head_dim;
            let scores_size = total_bh * seq_len * seq_len;

            // Precompute RoPE tables on CPU (small)
            let (cos_y, sin_y, cos_x, sin_x) = rope.compute_tables(positions, seq_len, head_dim);

            Some(qkv.with_gpu_buf(|qkv_buf| {
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

                    Tensor::from_gpu(attn_flat_buf, vec![batch * seq_len, embed_dim])
                }).unwrap()
            }))
        } else {
            None
        };

        #[cfg(feature = "metal")]
        let attn_flat = if let Some(flat) = gpu_attn_result {
            flat
        } else {
            // CPU fallback (code below)
            Self::cpu_self_attn(&qkv, rope, batch, seq_len, num_heads, head_dim, embed_dim, positions)
        };

        #[cfg(not(feature = "metal"))]
        let attn_flat = Self::cpu_self_attn(&qkv, rope, batch, seq_len, num_heads, head_dim, embed_dim, positions);

        let dt_attn = t0.elapsed();

        // Output projection + residual
        let t0 = Instant::now();
        let attn_proj = attn_flat.matmul(&self.proj_weight).add_bias(&self.proj_bias);
        let x = x.add(&attn_proj);
        let dt_proj = t0.elapsed();

        // --- Pre-norm + FFN ---
        let t0 = Instant::now();
        let normed = x.layer_norm(&self.norm2_weight, &self.norm2_bias, self.embed_dim);
        let h = normed.matmul(&self.mlp_fc1_weight).add_bias(&self.mlp_fc1_bias).gelu();
        let ffn_out = h.matmul(&self.mlp_fc2_weight).add_bias(&self.mlp_fc2_bias);

        // Residual connection
        let out = x.add(&ffn_out);
        let dt_ffn = t0.elapsed();

        // Print profile for first encoder block only
        if !ENCODER_PROFILE_PRINTED.swap(true, Ordering::Relaxed) {
            let total = t_block.elapsed();
            eprintln!("    [Encoder block 0 profile] total={:.1}ms", total.as_secs_f64() * 1000.0);
            eprintln!("      norm1:    {:.2}ms", dt_norm1.as_secs_f64() * 1000.0);
            eprintln!("      qkv:     {:.2}ms", dt_qkv.as_secs_f64() * 1000.0);
            eprintln!("      attn:    {:.2}ms", dt_attn.as_secs_f64() * 1000.0);
            eprintln!("      proj:    {:.2}ms", dt_proj.as_secs_f64() * 1000.0);
            eprintln!("      ffn:     {:.2}ms", dt_ffn.as_secs_f64() * 1000.0);
        }

        // Suppress unused variable warning when metal feature is off
        let _ = use_gpu;

        out
    }

    /// CPU self-attention fallback: QKV reshape + RoPE + MHA + output reshape.
    fn cpu_self_attn(
        qkv: &Tensor,
        rope: &RoPE2D,
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
        embed_dim: usize,
        positions: &[f32],
    ) -> Tensor {
        let qkv_data = qkv.data();
        let qkv_stride = 3 * embed_dim;
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

        let per_batch = num_heads * seq_len * head_dim;
        for b in 0..batch {
            let start = b * per_batch;
            let end = start + per_batch;
            let q_rotated = rope.apply(&q_heads[start..end], positions, num_heads, seq_len, head_dim);
            q_heads[start..end].copy_from_slice(&q_rotated);
            let k_rotated = rope.apply(&k_heads[start..end], positions, num_heads, seq_len, head_dim);
            k_heads[start..end].copy_from_slice(&k_rotated);
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        let total_bh = batch * num_heads;
        let attn_out_data = peregrine::tensor::multi_head_attention(
            &q_heads, &k_heads, &v_heads, total_bh, seq_len, seq_len, head_dim, scale,
        );

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
        Tensor::new(attn_flat_data, vec![batch * seq_len, embed_dim], false)
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

    /// Upload all weight tensors to GPU.
    #[cfg(feature = "metal")]
    pub fn to_gpu(&self) {
        self.patch_embed.to_gpu();
        for block in &self.blocks {
            block.to_gpu();
        }
        self.norm_weight.to_gpu();
        self.norm_bias.to_gpu();
    }

    /// Run encoder forward pass.
    ///
    /// # Arguments
    /// * `img` - image tensor [batch, 3, height, width]
    /// * `batch` - batch size
    /// * `height` - image height
    /// * `width` - image width
    /// * `use_gpu` - whether GPU acceleration is active
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
        use_gpu: bool,
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

        // Upload patch embeddings to GPU if in GPU mode
        #[cfg(feature = "metal")]
        if use_gpu { x.to_gpu(); }

        // Encoder blocks
        for block in &self.blocks {
            x = block.forward(&x, &positions, &self.rope, batch, seq_len, self.num_heads, use_gpu);
        }

        // Final LayerNorm
        let x = x.layer_norm(&self.norm_weight, &self.norm_bias, self.embed_dim);

        // Suppress unused variable warning when metal feature is off
        let _ = use_gpu;

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
    /// The converter (convert_must3r.py) transposes all 2D Linear weights from
    /// PyTorch [out_features, in_features] to Peregrine [in_features, out_features].
    /// Weights are loaded directly without runtime transposition.
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

            // norm2
            if let Some((_s, data)) = params.get(&format!("{}.norm2.weight", prefix)) {
                block.norm2_weight = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.norm2.bias", prefix)) {
                block.norm2_bias = Tensor::new(data.clone(), vec![self.embed_dim], false);
            }

            // MLP fc1: already transposed to [embed_dim, mlp_dim] by converter
            let mlp_dim = 4 * self.embed_dim;
            if let Some((shape, data)) = params.get(&format!("{}.mlp.fc1.weight", prefix)) {
                block.mlp_fc1_weight = Tensor::new(data.clone(), shape.clone(), false);
            }
            if let Some((_s, data)) = params.get(&format!("{}.mlp.fc1.bias", prefix)) {
                block.mlp_fc1_bias = Tensor::new(data.clone(), vec![1, mlp_dim], false);
            }

            // MLP fc2: already transposed to [mlp_dim, embed_dim] by converter
            if let Some((shape, data)) = params.get(&format!("{}.mlp.fc2.weight", prefix)) {
                block.mlp_fc2_weight = Tensor::new(data.clone(), shape.clone(), false);
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
