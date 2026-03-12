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

/// Write a quantized tensor to a binary stream.
/// Format: [dtype=1u8][ndim: u32][shape[0]: u32]...[i8 data...][f32 scales (cols)]
pub fn write_quantized_tensor(
    file: &mut impl Write,
    name: &str,
    qt: &crate::quant::QuantizedTensor,
) -> std::io::Result<()> {
    // Name
    let name_bytes = name.as_bytes();
    file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    file.write_all(name_bytes)?;

    // Dtype tag: 1 = i8 quantized
    file.write_all(&[1u8])?;

    // Shape [rows, cols]
    file.write_all(&(2u32).to_le_bytes())?;
    file.write_all(&(qt.rows as u32).to_le_bytes())?;
    file.write_all(&(qt.cols as u32).to_le_bytes())?;

    // i8 data
    let byte_slice = unsafe {
        std::slice::from_raw_parts(qt.data_i8.as_ptr() as *const u8, qt.data_i8.len())
    };
    file.write_all(byte_slice)?;

    // f32 scales (one per column)
    let scale_bytes = unsafe {
        std::slice::from_raw_parts(qt.scales.as_ptr() as *const u8, qt.scales.len() * 4)
    };
    file.write_all(scale_bytes)?;

    Ok(())
}

/// Read a quantized tensor from a binary stream.
/// Expects: [dtype=1u8][ndim=2: u32][rows: u32][cols: u32][i8 data][f32 scales]
pub fn read_quantized_tensor(
    file: &mut impl Read,
) -> std::io::Result<(String, crate::quant::QuantizedTensor)> {
    let mut buf4 = [0u8; 4];

    // Name
    file.read_exact(&mut buf4)?;
    let name_len = u32::from_le_bytes(buf4) as usize;
    let mut name_bytes = vec![0u8; name_len];
    file.read_exact(&mut name_bytes)?;
    let name = String::from_utf8(name_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Dtype tag
    let mut dtype_byte = [0u8; 1];
    file.read_exact(&mut dtype_byte)?;
    assert_eq!(dtype_byte[0], 1, "expected dtype=1 (i8) for quantized tensor");

    // Shape
    file.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    assert_eq!(ndim, 2, "quantized tensors must be 2D");
    file.read_exact(&mut buf4)?;
    let rows = u32::from_le_bytes(buf4) as usize;
    file.read_exact(&mut buf4)?;
    let cols = u32::from_le_bytes(buf4) as usize;

    // i8 data
    let num_elements = rows * cols;
    let mut i8_bytes = vec![0u8; num_elements];
    file.read_exact(&mut i8_bytes)?;
    let data_i8: Vec<i8> = i8_bytes.iter().map(|&b| b as i8).collect();

    // f32 scales
    let mut scale_bytes = vec![0u8; cols * 4];
    file.read_exact(&mut scale_bytes)?;
    let scales: Vec<f32> = scale_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    Ok((name, crate::quant::QuantizedTensor {
        data_i8,
        scales,
        rows,
        cols,
        #[cfg(feature = "metal")]
        gpu_data_i8: None,
        #[cfg(feature = "metal")]
        gpu_scales: None,
    }))
}

/// Write a 2:4 sparse tensor to a binary stream.
/// Format: [dtype=2u8][ndim: u32=2][K: u32][N: u32][f32 values K/2*N][u8 indices K/4*N]
pub fn write_sparse_tensor_24(
    file: &mut impl Write,
    name: &str,
    st: &crate::sparse::SparseTensor24,
) -> std::io::Result<()> {
    // Name
    let name_bytes = name.as_bytes();
    file.write_all(&(name_bytes.len() as u32).to_le_bytes())?;
    file.write_all(name_bytes)?;

    // Dtype tag: 2 = 2:4 sparse f32
    file.write_all(&[2u8])?;

    // Shape [K, N]
    file.write_all(&(2u32).to_le_bytes())?;
    file.write_all(&(st.rows as u32).to_le_bytes())?;
    file.write_all(&(st.cols as u32).to_le_bytes())?;

    // f32 values: K/2 * N elements
    let val_bytes = unsafe {
        std::slice::from_raw_parts(st.values.as_ptr() as *const u8, st.values.len() * 4)
    };
    file.write_all(val_bytes)?;

    // u8 indices: K/4 * N bytes
    file.write_all(&st.indices)?;

    Ok(())
}

/// Read a 2:4 sparse tensor from a binary stream.
/// Expects: [dtype=2u8][ndim=2: u32][K: u32][N: u32][f32 values][u8 indices]
pub fn read_sparse_tensor_24(
    file: &mut impl Read,
) -> std::io::Result<(String, crate::sparse::SparseTensor24)> {
    let mut buf4 = [0u8; 4];

    // Name
    file.read_exact(&mut buf4)?;
    let name_len = u32::from_le_bytes(buf4) as usize;
    let mut name_bytes = vec![0u8; name_len];
    file.read_exact(&mut name_bytes)?;
    let name = String::from_utf8(name_bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

    // Dtype tag
    let mut dtype_byte = [0u8; 1];
    file.read_exact(&mut dtype_byte)?;
    assert_eq!(dtype_byte[0], 2, "expected dtype=2 (2:4 sparse) for sparse tensor");

    // Shape
    file.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    assert_eq!(ndim, 2, "sparse tensors must be 2D");
    file.read_exact(&mut buf4)?;
    let rows = u32::from_le_bytes(buf4) as usize;
    file.read_exact(&mut buf4)?;
    let cols = u32::from_le_bytes(buf4) as usize;

    assert_eq!(rows % 4, 0, "K must be divisible by 4 for 2:4 sparsity");

    let groups = rows / 4;

    // f32 values: groups * 2 * cols elements
    let num_vals = groups * 2 * cols;
    let mut val_bytes = vec![0u8; num_vals * 4];
    file.read_exact(&mut val_bytes)?;
    let values: Vec<f32> = val_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();

    // u8 indices: groups * cols bytes
    let num_idx = groups * cols;
    let mut indices = vec![0u8; num_idx];
    file.read_exact(&mut indices)?;

    Ok((name, crate::sparse::SparseTensor24 {
        values,
        indices,
        rows,
        cols,
        #[cfg(feature = "metal")]
        gpu_values: None,
        #[cfg(feature = "metal")]
        gpu_indices: None,
    }))
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
