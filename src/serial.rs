//! Model weight serialization (save/load).
//!
//! Binary format per tensor: `[ndim: u32][shape[0]: u32]...[data: f32 * N]`
//! Model file: `[num_tensors: u32][name_len: u32][name: utf8]...[tensor]...`

use crate::tensor::Tensor;
use std::io::{BufReader, BufWriter, Read, Write};

/// Save named parameters to a binary file.
pub fn save_model(
    params: &[(String, &Tensor)],
    path: &str,
) -> std::io::Result<()> {
    let mut file = BufWriter::new(std::fs::File::create(path)?);

    // Number of tensors
    file.write_all(&(params.len() as u32).to_le_bytes())?;

    for (name, tensor) in params {
        // Name
        let name_bytes = name.as_bytes();
        file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
        file.write_all(name_bytes)?;

        // Shape
        let shape = tensor.shape();
        file.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &s in &shape {
            file.write_all(&(s as u32).to_le_bytes())?;
        }

        // Data — bulk write via byte reinterpret
        let data = tensor.data();
        let byte_slice = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        file.write_all(byte_slice)?;
    }

    Ok(())
}

/// Load named parameters from a binary file. Returns (name, shape, data) triples.
pub fn load_model(path: &str) -> std::io::Result<Vec<(String, Vec<usize>, Vec<f32>)>> {
    let mut file = BufReader::new(std::fs::File::open(path)?);
    let mut buf4 = [0u8; 4];

    // Number of tensors
    file.read_exact(&mut buf4)?;
    let num_tensors = u32::from_le_bytes(buf4) as usize;

    let mut result = Vec::with_capacity(num_tensors);

    for _ in 0..num_tensors {
        // Name
        file.read_exact(&mut buf4)?;
        let name_len = u32::from_le_bytes(buf4) as usize;
        let mut name_bytes = vec![0u8; name_len];
        file.read_exact(&mut name_bytes)?;
        let name = String::from_utf8(name_bytes)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        // Shape
        file.read_exact(&mut buf4)?;
        let ndim = u32::from_le_bytes(buf4) as usize;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            file.read_exact(&mut buf4)?;
            shape.push(u32::from_le_bytes(buf4) as usize);
        }

        // Data — bulk read via byte reinterpret
        let num_elements: usize = shape.iter().product();
        let mut byte_buf = vec![0u8; num_elements * 4];
        file.read_exact(&mut byte_buf)?;
        let data: Vec<f32> = byte_buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        result.push((name, shape, data));
    }

    Ok(result)
}

/// Export model weights to NumPy-compatible format for coremltools/ONNX conversion.
/// Writes one .npy file per parameter plus a manifest JSON.
pub fn export_for_coreml(
    params: &[(String, &Tensor)],
    output_dir: &str,
) -> std::io::Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let mut manifest = String::from("{\n  \"parameters\": [\n");

    for (i, (name, tensor)) in params.iter().enumerate() {
        let shape = tensor.shape();
        let data = tensor.data();

        // Write raw f32 binary (can be loaded with np.fromfile + reshape)
        let filename = format!("{}.bin", name.replace('.', "_"));
        let filepath = format!("{}/{}", output_dir, filename);

        let mut file = std::fs::File::create(&filepath)?;
        for &v in &data {
            file.write_all(&v.to_le_bytes())?;
        }

        // Manifest entry
        let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        if i > 0 { manifest.push_str(",\n"); }
        manifest.push_str(&format!(
            "    {{\"name\": \"{}\", \"file\": \"{}\", \"shape\": [{}], \"dtype\": \"float32\"}}",
            name, filename, shape_str.join(", ")
        ));
    }

    manifest.push_str("\n  ]\n}\n");
    std::fs::write(format!("{}/manifest.json", output_dir), manifest)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_save_load_roundtrip() {
        let t1 = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let t2 = Tensor::new(vec![5.0, 6.0], vec![2], true);

        let params = vec![
            ("layer1.weight".to_string(), &t1),
            ("layer1.bias".to_string(), &t2),
        ];

        let path = "/tmp/peregrine_test_model.bin";
        save_model(&params, path).unwrap();
        let loaded = load_model(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, "layer1.weight");
        assert_eq!(loaded[0].1, vec![2, 2]);
        assert_eq!(loaded[0].2, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].0, "layer1.bias");
        assert_eq!(loaded[1].1, vec![2]);
        assert_eq!(loaded[1].2, vec![5.0, 6.0]);

        std::fs::remove_file(path).ok();
    }
}
