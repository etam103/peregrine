// MUSt3R output head: LinearHead with unpatchify and postprocessing.
//
// The head projects decoder features to per-patch predictions, then rearranges
// patches back into full-resolution spatial maps (unpatchify). Each pixel gets
// 7 channels: 3 (xyz points) + 3 (local xyz) + 1 (confidence).

use std::collections::HashMap;

use peregrine::tensor::Tensor;

// --------------------------------------------------------------------------
// Pointmap output
// --------------------------------------------------------------------------

/// 3D reconstruction output for a single image.
pub struct Pointmap {
    /// 3D point coordinates [H, W, 3] (after norm_exp activation).
    pub pts3d: Vec<f32>,
    /// Local 3D point coordinates [H, W, 3] (raw from head output).
    pub pts3d_local: Vec<f32>,
    /// Confidence values [H, W] (after 1 + exp activation).
    pub conf: Vec<f32>,
    pub height: usize,
    pub width: usize,
}

// --------------------------------------------------------------------------
// LinearHead
// --------------------------------------------------------------------------

/// Linear projection head that maps decoder features to per-pixel predictions.
///
/// Output channels = patch_size * patch_size * 7 = 16 * 16 * 7 = 1792
/// 7 channels per pixel: 3 (pts3d xyz) + 3 (pts3d_local xyz) + 1 (confidence)
pub struct LinearHead {
    pub weight: Tensor, // stored as [embed_dim, out_dim] for Peregrine matmul
    pub bias: Tensor,   // [1, out_dim] where out_dim = patch_size^2 * 7 = 1792
    patch_size: usize,  // 16
}

impl LinearHead {
    /// Create a new LinearHead with zero-initialized weights.
    ///
    /// - embed_dim: decoder embedding dimension (768)
    /// - patch_size: patch size used by the ViT (16)
    /// - num_channels: output channels per pixel (7)
    pub fn new(embed_dim: usize, patch_size: usize, num_channels: usize) -> Self {
        let out_dim = patch_size * patch_size * num_channels;
        LinearHead {
            weight: Tensor::zeros(&[embed_dim, out_dim], false),
            bias: Tensor::zeros(&[1, out_dim], false),
            patch_size,
        }
    }

    /// Forward pass: project and unpatchify.
    ///
    /// - x: [batch * num_patches, embed_dim] decoder output
    /// - batch: batch size
    /// - h_patches: number of patches along height
    /// - w_patches: number of patches along width
    ///
    /// Returns: [batch, H, W, 7] as a flat Tensor where H = h_patches * patch_size,
    ///          W = w_patches * patch_size.
    pub fn forward(
        &self,
        x: &Tensor,
        batch: usize,
        h_patches: usize,
        w_patches: usize,
    ) -> Tensor {
        let ps = self.patch_size;
        let num_channels = 7;
        let h = h_patches * ps;
        let w = w_patches * ps;
        let num_patches = h_patches * w_patches;

        // Linear projection: [batch * num_patches, embed_dim] x [embed_dim, out_dim]
        //                  -> [batch * num_patches, out_dim]
        let projected = x.matmul(&self.weight).add_bias(&self.bias);
        let proj_data = projected.data();

        // Unpatchify: rearrange from patches to spatial layout
        // Input layout per patch: [ps * ps * num_channels] (row-major within patch)
        // Output layout: [batch, H, W, num_channels]
        let out_size = batch * h * w * num_channels;
        let mut output = vec![0.0f32; out_size];

        for b in 0..batch {
            for ph in 0..h_patches {
                for pw in 0..w_patches {
                    let patch_idx = b * num_patches + ph * w_patches + pw;
                    let patch_base = patch_idx * (ps * ps * num_channels);

                    for py in 0..ps {
                        for px in 0..ps {
                            let pixel_y = ph * ps + py;
                            let pixel_x = pw * ps + px;

                            // Source: patch data at position (py, px) within the patch
                            let src_offset = (py * ps + px) * num_channels;

                            // Destination: [b, pixel_y, pixel_x, :]
                            let dst_offset =
                                ((b * h + pixel_y) * w + pixel_x) * num_channels;

                            output[dst_offset..dst_offset + num_channels].copy_from_slice(
                                &proj_data[patch_base + src_offset
                                    ..patch_base + src_offset + num_channels],
                            );
                        }
                    }
                }
            }
        }

        Tensor::new(output, vec![batch, h, w, num_channels], false)
    }

    /// Load pretrained weights from a parameter map.
    ///
    /// Weight keys (for image 1 head / image 2 head):
    /// - "head1.weight" / "head2.weight" (PyTorch shape [out, in], needs transpose)
    /// - "head1.bias" / "head2.bias"
    ///
    /// The caller should pass the appropriate prefix.
    pub fn load_weights(&mut self, params: &HashMap<String, (Vec<usize>, Vec<f32>)>, prefix: &str) {
        let weight_key = format!("{}.weight", prefix);
        let bias_key = format!("{}.bias", prefix);

        if let Some((shape, data)) = params.get(&weight_key) {
            // PyTorch Linear: [out_features, in_features] -> transpose for Peregrine
            let t = Tensor::new(data.clone(), shape.clone(), false);
            if shape.len() == 2 {
                self.weight = t.transpose(0, 1);
            } else {
                self.weight = t;
            }
        } else {
            panic!("Missing weight: {}", weight_key);
        }

        if let Some((_shape, data)) = params.get(&bias_key) {
            // Ensure bias is [1, out_dim] for add_bias compatibility
            self.bias = Tensor::new(data.clone(), vec![1, data.len()], false);
        } else {
            panic!("Missing weight: {}", bias_key);
        }
    }
}

// --------------------------------------------------------------------------
// Postprocessing
// --------------------------------------------------------------------------

/// Postprocess the raw head output into a Pointmap.
///
/// - pointmap_data: flat array from the head output for a single image,
///   layout [H, W, 7] where 7 = 3 (pts3d) + 3 (pts3d_local) + 1 (conf).
/// - h: image height in pixels
/// - w: image width in pixels
///
/// Activations:
/// - pts3d: norm_exp -- normalize direction to unit vector, then scale by exp(norm)
///   to get distance. Specifically for each pixel's (x, y, z):
///     norm = sqrt(x^2 + y^2 + z^2)
///     direction = (x/norm, y/norm, z/norm)
///     pts3d = direction * exp(norm)
/// - conf: 1 + exp(raw_conf) (always positive, minimum 1.0)
pub fn postprocess(pointmap_data: &[f32], h: usize, w: usize) -> Pointmap {
    let num_pixels = h * w;
    let num_channels = 7;
    assert_eq!(
        pointmap_data.len(),
        num_pixels * num_channels,
        "postprocess: expected {} elements ({}x{}x7), got {}",
        num_pixels * num_channels,
        h,
        w,
        pointmap_data.len()
    );

    let mut pts3d = vec![0.0f32; num_pixels * 3];
    let mut pts3d_local = vec![0.0f32; num_pixels * 3];
    let mut conf = vec![0.0f32; num_pixels];

    for i in 0..num_pixels {
        let base = i * num_channels;

        // Raw pts3d channels [0..3]
        let px = pointmap_data[base];
        let py = pointmap_data[base + 1];
        let pz = pointmap_data[base + 2];

        // norm_exp activation: normalize direction, scale by exp(norm)
        let norm = (px * px + py * py + pz * pz).sqrt();
        if norm > 1e-8 {
            let scale = norm.exp() / norm;
            pts3d[i * 3] = px * scale;
            pts3d[i * 3 + 1] = py * scale;
            pts3d[i * 3 + 2] = pz * scale;
        } else {
            // Near-zero vector: exp(0) = 1, direction is arbitrary
            pts3d[i * 3] = 1.0;
            pts3d[i * 3 + 1] = 0.0;
            pts3d[i * 3 + 2] = 0.0;
        }

        // Raw pts3d_local channels [3..6] (no activation, kept raw)
        pts3d_local[i * 3] = pointmap_data[base + 3];
        pts3d_local[i * 3 + 1] = pointmap_data[base + 4];
        pts3d_local[i * 3 + 2] = pointmap_data[base + 5];

        // Confidence channel [6]: 1 + exp(raw_conf)
        let raw_conf = pointmap_data[base + 6];
        conf[i] = 1.0 + raw_conf.exp();
    }

    Pointmap {
        pts3d,
        pts3d_local,
        conf,
        height: h,
        width: w,
    }
}
