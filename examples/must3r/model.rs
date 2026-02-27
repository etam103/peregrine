use std::collections::HashMap;
use peregrine::tensor::Tensor;
use peregrine::serial::load_model;
use super::encoder::Dust3rEncoder;
use super::decoder::MUSt3RDecoder;
use super::head::{LinearHead, Pointmap};

pub struct MUSt3R {
    pub encoder: Dust3rEncoder,
    pub decoder: MUSt3RDecoder,
    pub head: LinearHead,
    patch_size: usize,
}

impl MUSt3R {
    pub fn new() -> Self {
        // MUSt3R-224 configuration:
        // Encoder: ViT-Large (1024 dim, 24 blocks, 16 heads)
        // Decoder: ViT-Base (768 dim, 12 blocks, 12 heads)
        // Head: Linear(768 -> 16*16*7=1792) + unpatchify(16)
        // patch_size: 16
        // 7 output channels per pixel: 3 (pts3d) + 3 (pts3d_local) + 1 (conf)
        MUSt3R {
            encoder: Dust3rEncoder::new(1024, 24, 16, 16),
            decoder: MUSt3RDecoder::new(1024, 768, 12, 12),
            head: LinearHead::new(768, 16, 7),
            patch_size: 16,
        }
    }

    /// Load weights from Peregrine binary format (converted from PyTorch).
    pub fn load_weights(&mut self, path: &str) {
        let raw = load_model(path).expect("Failed to load model weights");

        // Build HashMap for efficient lookup
        let mut params: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for (name, shape, data) in raw {
            params.insert(name, (shape, data));
        }

        println!("Loaded {} parameters from {}", params.len(), path);

        self.encoder.load_weights(&params);
        self.decoder.load_weights(&params);
        self.head.load_weights(&params, "head_dec.proj");
    }

    /// Run two-view inference.
    /// img1, img2: [1, 3, H, W] tensors (normalized to [0,1], then ImageNet-standardized)
    /// Returns (pointmap1, pointmap2) for the two views.
    pub fn forward(&self, img1: &Tensor, img2: &Tensor, height: usize, width: usize) -> (Pointmap, Pointmap) {
        let batch = 1;
        let h_patches = height / self.patch_size;
        let w_patches = width / self.patch_size;
        let seq_len = h_patches * w_patches;

        println!("  Encoding image 1...");
        let (enc1, pos1) = self.encoder.forward(img1, batch, height, width);
        println!("  Encoding image 2...");
        let (enc2, pos2) = self.encoder.forward(img2, batch, height, width);

        println!("  Decoding...");
        let (dec1, dec2) = self.decoder.forward(&enc1, &enc2, &pos1, &pos2, batch, seq_len);

        println!("  Computing pointmaps...");
        let raw1 = self.head.forward(&dec1, batch, h_patches, w_patches);
        let raw2 = self.head.forward(&dec2, batch, h_patches, w_patches);

        let pm1 = super::head::postprocess(&raw1.data(), height, width);
        let pm2 = super::head::postprocess(&raw2.data(), height, width);

        (pm1, pm2)
    }
}
