mod rope2d;
mod encoder;
mod decoder;
mod head;
mod model;

use std::env;
use std::time::Instant;
use peregrine::tensor::Tensor;
use model::MUSt3R;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 4 {
        eprintln!("Usage: cargo run --example must3r --release -- <weights.bin> <image1> <image2>");
        eprintln!();
        eprintln!("Convert PyTorch weights first:");
        eprintln!("  python scripts/convert_must3r.py MUSt3R_224_cvpr.pth weights/must3r_224.bin");
        std::process::exit(1);
    }

    let weights_path = &args[1];
    let img1_path = &args[2];
    let img2_path = &args[3];

    // Default to 224x224 (MUSt3R_224 model)
    let img_size = 224usize;

    println!("MUSt3R 3D Reconstruction");
    println!("========================");
    println!("Model: MUSt3R-224 (ViT-L encoder + ViT-B decoder)");
    println!("Image size: {}x{}", img_size, img_size);
    println!();

    // Load and preprocess images
    println!("Loading images...");
    let img1 = load_image(img1_path, img_size);
    let img2 = load_image(img2_path, img_size);
    println!("  Image 1: {} ({}x{})", img1_path, img_size, img_size);
    println!("  Image 2: {} ({}x{})", img2_path, img_size, img_size);

    // Create model and load weights
    println!("Loading model...");
    let t0 = Instant::now();
    let mut model = MUSt3R::new();
    model.load_weights(weights_path);
    println!("  Loaded in {:.1}s", t0.elapsed().as_secs_f32());

    // Count parameters
    // MUSt3R-224: ~400M parameters
    println!();

    // Run inference
    println!("Running inference...");
    let t0 = Instant::now();
    let (pm1, pm2) = model.forward(&img1, &img2, img_size, img_size);
    let elapsed = t0.elapsed().as_secs_f32();
    println!("  Inference time: {:.2}s", elapsed);

    // Print results
    println!();
    println!("Results:");
    println!("  Pointmap 1: {}x{} ({} points)", pm1.height, pm1.width, pm1.height * pm1.width);
    println!("  Pointmap 2: {}x{} ({} points)", pm2.height, pm2.width, pm2.height * pm2.width);

    // Print some statistics about the 3D points
    print_pointmap_stats("View 1", &pm1);
    print_pointmap_stats("View 2", &pm2);

    // Save pointmaps as binary for visualization
    let out_path = "must3r_output.bin";
    save_pointmaps(&pm1, &pm2, out_path);
    println!();
    println!("Pointmaps saved to {}", out_path);
}

/// Load an image, resize to img_size x img_size, convert to [1, 3, H, W] tensor.
/// Applies ImageNet normalization: (pixel/255 - mean) / std
fn load_image(path: &str, img_size: usize) -> Tensor {
    // Read image file
    let img_bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        std::process::exit(1);
    });

    // Load RGB pixels from PPM or raw format
    let pixels = load_rgb_pixels(&img_bytes, path, img_size);

    // Apply ImageNet normalization
    let mean = [0.485f32, 0.456, 0.406];
    let std_dev = [0.229f32, 0.224, 0.225];

    let mut chw = vec![0.0f32; 3 * img_size * img_size];
    for y in 0..img_size {
        for x in 0..img_size {
            let idx = (y * img_size + x) * 3;
            for c in 0..3 {
                let val = pixels[idx + c] as f32 / 255.0;
                chw[c * img_size * img_size + y * img_size + x] = (val - mean[c]) / std_dev[c];
            }
        }
    }

    Tensor::new(chw, vec![1, 3, img_size, img_size], false)
}

/// Load RGB pixels from a file. Supports PPM (P6) format.
/// For other formats, assumes raw RGB bytes of correct size.
fn load_rgb_pixels(data: &[u8], path: &str, target_size: usize) -> Vec<u8> {
    // Try PPM format first
    if data.len() >= 2 && data[0] == b'P' && data[1] == b'6' {
        return load_ppm(data, target_size);
    }

    // If it's a raw RGB file of the right size
    let expected = target_size * target_size * 3;
    if data.len() == expected {
        return data.to_vec();
    }

    // Format not recognized
    eprintln!("Warning: {} format not recognized. Supported: PPM (P6) or raw RGB {}x{}x3",
              path, target_size, target_size);
    eprintln!("Convert with: convert input.jpg -resize {}x{} output.ppm", target_size, target_size);

    // Return mid-gray as fallback
    vec![128u8; expected]
}

/// Load PPM P6 format and resize to target_size using nearest neighbor.
fn load_ppm(data: &[u8], target_size: usize) -> Vec<u8> {
    // Parse PPM header manually to handle binary data after header
    let mut pos = 0;

    // Skip magic "P6"
    while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
    }
    pos += 1; // skip newline after P6

    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        pos += 1;
    }

    // Parse width and height
    let dim_start = pos;
    while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
    }
    let dim_str = std::str::from_utf8(&data[dim_start..pos]).unwrap_or("");
    pos += 1; // skip newline

    let dims: Vec<usize> = dim_str
        .split_whitespace()
        .filter_map(|s| s.parse().ok())
        .collect();
    let (src_w, src_h) = if dims.len() >= 2 {
        (dims[0], dims[1])
    } else {
        (target_size, target_size)
    };

    // Parse max value line
    while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
    }
    pos += 1; // skip newline after max value

    let pixel_data = &data[pos..];

    // Nearest-neighbor resize
    let mut result = vec![0u8; target_size * target_size * 3];
    for ty in 0..target_size {
        for tx in 0..target_size {
            let sy = (ty * src_h) / target_size;
            let sx = (tx * src_w) / target_size;
            let src_idx = (sy * src_w + sx) * 3;
            let dst_idx = (ty * target_size + tx) * 3;
            if src_idx + 2 < pixel_data.len() {
                result[dst_idx] = pixel_data[src_idx];
                result[dst_idx + 1] = pixel_data[src_idx + 1];
                result[dst_idx + 2] = pixel_data[src_idx + 2];
            }
        }
    }
    result
}

fn print_pointmap_stats(name: &str, pm: &head::Pointmap) {
    let n = pm.height * pm.width;

    // Compute point cloud bounding box
    let (mut min_x, mut min_y, mut min_z) = (f32::MAX, f32::MAX, f32::MAX);
    let (mut max_x, mut max_y, mut max_z) = (f32::MIN, f32::MIN, f32::MIN);
    let mut mean_conf = 0.0f32;

    for i in 0..n {
        let x = pm.pts3d[i * 3];
        let y = pm.pts3d[i * 3 + 1];
        let z = pm.pts3d[i * 3 + 2];
        min_x = min_x.min(x); max_x = max_x.max(x);
        min_y = min_y.min(y); max_y = max_y.max(y);
        min_z = min_z.min(z); max_z = max_z.max(z);
        mean_conf += pm.conf[i];
    }
    mean_conf /= n as f32;

    println!("  {}: bbox=[{:.2}..{:.2}, {:.2}..{:.2}, {:.2}..{:.2}], mean_conf={:.3}",
             name, min_x, max_x, min_y, max_y, min_z, max_z, mean_conf);
}

fn save_pointmaps(pm1: &head::Pointmap, pm2: &head::Pointmap, path: &str) {
    use std::io::Write;
    let mut f = std::fs::File::create(path).expect("Failed to create output file");

    // Simple binary format: [H: u32][W: u32][pts3d: f32*H*W*3][conf: f32*H*W] x2
    f.write_all(&(pm1.height as u32).to_le_bytes()).unwrap();
    f.write_all(&(pm1.width as u32).to_le_bytes()).unwrap();
    for &v in &pm1.pts3d { f.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &pm1.conf { f.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &pm2.pts3d { f.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &pm2.conf { f.write_all(&v.to_le_bytes()).unwrap(); }
}
