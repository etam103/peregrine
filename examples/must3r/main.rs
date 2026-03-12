mod rope2d;
mod encoder;
mod decoder;
mod head;
mod model;

use std::env;
use std::time::Instant;
use std::io::{self, BufRead, Write};
use peregrine::tensor::Tensor;
use model::MUSt3R;

fn main() {
    let args: Vec<String> = env::args().collect();

    // Check for flags in args
    let has_server = args.iter().any(|a| a == "--server");
    let has_gpu = args.iter().any(|a| a == "--gpu");
    let has_pipeline = args.iter().any(|a| a == "--pipeline");

    // Filter out flags for positional arg parsing
    let positional: Vec<&String> = args.iter().skip(1).filter(|a| !a.starts_with("--")).collect();

    if has_server {
        // Server mode: <weights.bin> --server [--gpu]
        if positional.is_empty() {
            eprintln!("Usage: must3r <weights.bin> --server [--gpu]");
            std::process::exit(1);
        }
        let weights_path = positional[0];
        run_server(weights_path, has_gpu, has_pipeline);
    } else {
        // Single-pair mode: <weights.bin> <image1> <image2> [size] [--gpu] [--pipeline]
        if positional.len() < 3 {
            eprintln!("Usage: must3r <weights.bin> <image1> <image2> [size] [--gpu] [--pipeline]");
            eprintln!("       must3r <weights.bin> --server [--gpu] [--pipeline]");
            eprintln!();
            eprintln!("Convert PyTorch weights first:");
            eprintln!("  python scripts/convert_must3r.py MUSt3R_224_cvpr.pth weights/must3r_224.bin");
            eprintln!();
            eprintln!("Flags:");
            eprintln!("  --server    Server mode: read pairs from stdin, write binary to stdout");
            eprintln!("  --gpu       Use Metal GPU acceleration (requires --features metal)");
            eprintln!("  --pipeline  Overlap GPU+CPU decoder (requires --gpu)");
            std::process::exit(1);
        }
        run_single_pair(&positional, has_gpu, has_pipeline);
    }
}

fn init_model(weights_path: &str, use_gpu: bool) -> MUSt3R {
    #[cfg(feature = "metal")]
    if use_gpu {
        peregrine::metal::init_gpu().expect("Failed to initialize Metal GPU");
        eprintln!("Metal GPU initialized");
    }
    #[cfg(not(feature = "metal"))]
    if use_gpu {
        eprintln!("Warning: --gpu flag ignored (not compiled with --features metal)");
    }

    let t0 = Instant::now();
    let mut model = MUSt3R::new();
    model.load_weights(weights_path);
    eprintln!("Model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    #[cfg(feature = "metal")]
    if use_gpu {
        let t0 = Instant::now();
        model.to_gpu();
        eprintln!("Weights uploaded to GPU in {:.1}s", t0.elapsed().as_secs_f32());
    }

    model
}

fn run_single_pair(positional: &[&String], use_gpu: bool, pipeline: bool) {
    let weights_path = positional[0];
    let img1_path = positional[1];
    let img2_path = positional[2];

    // Image size: WxH format, default 224x224
    let (img_w, img_h) = if positional.len() > 3 {
        let size_arg = positional[3].as_str();
        if let Some(idx) = size_arg.find('x') {
            let w: usize = size_arg[..idx].parse().unwrap_or(224);
            let h: usize = size_arg[idx + 1..].parse().unwrap_or(224);
            (w, h)
        } else {
            let s: usize = size_arg.parse().unwrap_or(224);
            (s, s)
        }
    } else {
        (224, 224)
    };

    println!("MUSt3R 3D Reconstruction");
    println!("========================");
    println!("Model: MUSt3R (ViT-L encoder + ViT-B decoder)");
    println!("Image size: {}x{}", img_w, img_h);
    if use_gpu { println!("GPU: enabled"); }
    if pipeline { println!("Pipeline: enabled (GPU+CPU overlap)"); }
    println!();

    // Load and preprocess images
    println!("Loading images...");
    let img1 = load_image(img1_path, img_w, img_h);
    let img2 = load_image(img2_path, img_w, img_h);
    println!("  Image 1: {} ({}x{})", img1_path, img_w, img_h);
    println!("  Image 2: {} ({}x{})", img2_path, img_w, img_h);

    // Create model and load weights
    println!("Loading model...");
    let model = init_model(weights_path, use_gpu);

    // Count parameters
    // MUSt3R-224: ~400M parameters
    println!();

    // Run inference
    println!("Running inference...");
    let t0 = Instant::now();
    let (pm1, pm2) = model.forward(&img1, &img2, img_h, img_w, use_gpu, pipeline);
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
    save_pointmaps_to_file(&pm1, &pm2, out_path);
    println!();
    println!("Pointmaps saved to {}", out_path);
}

fn run_server(weights_path: &str, use_gpu: bool, pipeline: bool) {
    eprintln!("MUSt3R server mode");
    let model = init_model(weights_path, use_gpu);
    eprintln!("Ready, reading pairs from stdin...");

    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }

        // Parse: <img1_path>\t<img2_path>\t<width>\t<height>
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() < 4 {
            eprintln!("ERROR: expected 4 tab-separated fields, got {}", parts.len());
            continue;
        }

        let img1_path = parts[0];
        let img2_path = parts[1];
        let width: usize = parts[2].parse().unwrap_or(224);
        let height: usize = parts[3].parse().unwrap_or(224);

        eprintln!("Processing: {} + {} ({}x{})", img1_path, img2_path, width, height);
        let t0 = Instant::now();

        let img1 = load_image(img1_path, width, height);
        let img2 = load_image(img2_path, width, height);
        let (pm1, pm2) = model.forward(&img1, &img2, height, width, use_gpu, pipeline);

        let elapsed = t0.elapsed().as_secs_f32();
        eprintln!("  Done in {:.2}s", elapsed);

        // Write binary response: [u32 H][u32 W][f32*H*W*3 pts1][f32*H*W conf1][f32*H*W*3 pts2][f32*H*W conf2]
        write_pointmaps_binary(&mut stdout, &pm1, &pm2);
        // Sentinel newline
        stdout.write_all(b"\n").unwrap();
        stdout.flush().unwrap();
    }

    eprintln!("Server shutting down.");
}

fn write_pointmaps_binary(w: &mut dyn Write, pm1: &head::Pointmap, pm2: &head::Pointmap) {
    w.write_all(&(pm1.height as u32).to_le_bytes()).unwrap();
    w.write_all(&(pm1.width as u32).to_le_bytes()).unwrap();
    for &v in &pm1.pts3d { w.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &pm1.conf { w.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &pm2.pts3d { w.write_all(&v.to_le_bytes()).unwrap(); }
    for &v in &pm2.conf { w.write_all(&v.to_le_bytes()).unwrap(); }
}

/// Load an image, resize to img_size x img_size, convert to [1, 3, H, W] tensor.
/// Applies ImageNet normalization: (pixel/255 - mean) / std
fn load_image(path: &str, width: usize, height: usize) -> Tensor {
    // Read image file
    let img_bytes = std::fs::read(path).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", path, e);
        std::process::exit(1);
    });

    // Load RGB pixels from PPM or raw format
    let pixels = load_rgb_pixels(&img_bytes, path, width, height);

    // Apply ImageNet normalization
    let mean = [0.485f32, 0.456, 0.406];
    let std_dev = [0.229f32, 0.224, 0.225];

    let mut chw = vec![0.0f32; 3 * height * width];
    for y in 0..height {
        for x in 0..width {
            let idx = (y * width + x) * 3;
            for c in 0..3 {
                let val = pixels[idx + c] as f32 / 255.0;
                chw[c * height * width + y * width + x] = (val - mean[c]) / std_dev[c];
            }
        }
    }

    Tensor::new(chw, vec![1, 3, height, width], false)
}

/// Load RGB pixels from a file. Supports PPM (P6) format.
/// For other formats, assumes raw RGB bytes of correct size.
fn load_rgb_pixels(data: &[u8], path: &str, target_w: usize, target_h: usize) -> Vec<u8> {
    // Try PPM format first
    if data.len() >= 2 && data[0] == b'P' && data[1] == b'6' {
        return load_ppm(data, target_w, target_h);
    }

    // If it's a raw RGB file of the right size
    let expected = target_w * target_h * 3;
    if data.len() == expected {
        return data.to_vec();
    }

    // Format not recognized
    eprintln!("Warning: {} format not recognized. Supported: PPM (P6) or raw RGB {}x{}x3",
              path, target_w, target_h);
    eprintln!("Convert with: magick input.jpg -resize {}x{}! output.ppm", target_w, target_h);

    // Return mid-gray as fallback
    vec![128u8; expected]
}

/// Load PPM P6 format and resize to target_size using nearest neighbor.
fn load_ppm(data: &[u8], target_w: usize, target_h: usize) -> Vec<u8> {
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
        (target_w, target_h)
    };

    // Parse max value line
    while pos < data.len() && data[pos] != b'\n' {
        pos += 1;
    }
    pos += 1; // skip newline after max value

    let pixel_data = &data[pos..];

    // Nearest-neighbor resize
    let mut result = vec![0u8; target_w * target_h * 3];
    for ty in 0..target_h {
        for tx in 0..target_w {
            let sy = (ty * src_h) / target_h;
            let sx = (tx * src_w) / target_w;
            let src_idx = (sy * src_w + sx) * 3;
            let dst_idx = (ty * target_w + tx) * 3;
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

fn save_pointmaps_to_file(pm1: &head::Pointmap, pm2: &head::Pointmap, path: &str) {
    let mut f = std::fs::File::create(path).expect("Failed to create output file");
    write_pointmaps_binary(&mut f, pm1, pm2);
}
