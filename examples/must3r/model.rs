use std::collections::HashMap;
use std::time::Instant;
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

        eprintln!("Loaded {} parameters from {}", params.len(), path);

        self.encoder.load_weights(&params);
        self.decoder.load_weights(&params);
        self.head.load_weights(&params, "head_dec.proj");
    }

    /// Upload all weight tensors to GPU.
    #[cfg(feature = "metal")]
    pub fn to_gpu(&self) {
        self.encoder.to_gpu();
        self.decoder.to_gpu();
        self.head.to_gpu();
    }

    /// Run two-view inference.
    /// img1, img2: [1, 3, H, W] tensors (normalized to [0,1], then ImageNet-standardized)
    /// Returns (pointmap1, pointmap2) for the two views.
    pub fn forward(&self, img1: &Tensor, img2: &Tensor, height: usize, width: usize, use_gpu: bool, pipeline: bool) -> (Pointmap, Pointmap) {
        let h_patches = height / self.patch_size;
        let w_patches = width / self.patch_size;
        let seq_len = h_patches * w_patches;

        // Batch both images: [1,3,H,W] + [1,3,H,W] -> [2,3,H,W]
        // Run encoder once with batch=2 instead of twice — eliminates warmup
        // penalty and doubles GEMM sizes for better AMX utilization.
        eprintln!("  Encoding both images (batched)...");
        let t = Instant::now();
        let img1_data = img1.data();
        let img2_data = img2.data();
        let mut both_data = Vec::with_capacity(img1_data.len() + img2_data.len());
        both_data.extend_from_slice(&img1_data);
        both_data.extend_from_slice(&img2_data);
        let img_both = Tensor::new(both_data, vec![2, 3, height, width], false);

        // Upload input to GPU if weights are GPU-resident
        #[cfg(feature = "metal")]
        if use_gpu { img_both.to_gpu(); }

        let (enc_both, pos) = self.encoder.forward(&img_both, 2, height, width, use_gpu);
        let enc_ms = t.elapsed().as_secs_f64() * 1000.0;

        // Split encoder output: [2*seq_len, embed_dim] -> two [seq_len, embed_dim]
        let enc_shape = enc_both.shape();
        let embed_dim = enc_shape[1];
        let enc_data = enc_both.data();
        let half = seq_len * embed_dim;
        let enc1 = Tensor::new(enc_data[..half].to_vec(), vec![seq_len, embed_dim], false);
        let enc2 = Tensor::new(enc_data[half..].to_vec(), vec![seq_len, embed_dim], false);

        // Re-upload split tensors to GPU so decoder stays GPU-resident
        #[cfg(feature = "metal")]
        if use_gpu {
            enc1.to_gpu();
            enc2.to_gpu();
        }

        eprintln!("  Decoding...");
        let t = Instant::now();
        let (dec1, dec2) = self.decoder.forward(&enc1, &enc2, &pos, &pos, 1, seq_len, use_gpu, pipeline);
        let dec_ms = t.elapsed().as_secs_f64() * 1000.0;

        eprintln!("  Computing pointmaps...");
        let t = Instant::now();
        let raw1 = self.head.forward(&dec1, 1, h_patches, w_patches);
        let raw2 = self.head.forward(&dec2, 1, h_patches, w_patches);
        let pm1 = super::head::postprocess(&raw1.data(), height, width);
        let pm2 = super::head::postprocess(&raw2.data(), height, width);
        let head_ms = t.elapsed().as_secs_f64() * 1000.0;

        eprintln!();
        eprintln!("  [Profile] Encoder (batched): {:.1}ms", enc_ms);
        eprintln!("  [Profile] Decoder:           {:.1}ms", dec_ms);
        eprintln!("  [Profile] Head+postproc:     {:.1}ms", head_ms);

        // Suppress unused variable warning when metal feature is off
        let _ = use_gpu;

        (pm1, pm2)
    }
}
