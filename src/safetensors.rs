//! Safetensors binary format parser for loading model weights.
//!
//! Supports mmap-based loading on Unix with fallback to `fs::read`.
//! Handles F32, F16, BF16, and other dtypes defined by the safetensors spec.

use std::collections::HashMap;
use std::fs;
use std::io;

/// Safetensors tensor element types.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SafetensorsDtype {
    Bool,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl SafetensorsDtype {
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "BOOL" => Some(Self::Bool),
            "U8" => Some(Self::U8),
            "I8" => Some(Self::I8),
            "I16" => Some(Self::I16),
            "U16" => Some(Self::U16),
            "F16" => Some(Self::F16),
            "BF16" => Some(Self::BF16),
            "I32" => Some(Self::I32),
            "U32" => Some(Self::U32),
            "F32" => Some(Self::F32),
            "F64" => Some(Self::F64),
            "I64" => Some(Self::I64),
            "U64" => Some(Self::U64),
            _ => None,
        }
    }

    /// Bytes per element for this dtype.
    pub fn element_size(&self) -> usize {
        match self {
            Self::Bool | Self::U8 | Self::I8 => 1,
            Self::I16 | Self::U16 | Self::F16 | Self::BF16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::F64 | Self::I64 | Self::U64 => 8,
        }
    }
}

/// Metadata for a single tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: SafetensorsDtype,
    pub shape: Vec<usize>,
    pub data_start: usize,
    pub data_end: usize,
}

impl TensorInfo {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product::<usize>().max(1)
    }

    pub fn data_size(&self) -> usize {
        self.data_end - self.data_start
    }
}

/// Storage for file data — either mmap'd or heap-allocated.
enum Storage {
    #[cfg(unix)]
    Mmap { ptr: *mut u8, len: usize },
    Owned(Vec<u8>),
}

// Safety: The mmap'd memory is read-only after creation and lives for the
// lifetime of the SafetensorsFile.
unsafe impl Send for Storage {}
unsafe impl Sync for Storage {}

impl Storage {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(unix)]
            Storage::Mmap { ptr, len } => unsafe { std::slice::from_raw_parts(*ptr, *len) },
            Storage::Owned(v) => v.as_slice(),
        }
    }
}

impl Drop for Storage {
    fn drop(&mut self) {
        match self {
            #[cfg(unix)]
            Storage::Mmap { ptr, len } => {
                unsafe {
                    libc::munmap(*ptr as *mut libc::c_void, *len);
                }
            }
            Storage::Owned(_) => {}
        }
    }
}

/// A parsed safetensors file.
pub struct SafetensorsFile {
    storage: Storage,
    header_size: usize,
    pub tensors: HashMap<String, TensorInfo>,
}

/// Convert BF16 (bfloat16) to f32: just shift left by 16 bits.
#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

impl SafetensorsFile {
    /// Open a safetensors file from disk.
    /// Uses mmap on Unix for zero-copy access, falls back to fs::read otherwise.
    pub fn open(path: &str) -> io::Result<Self> {
        let storage = Self::load_storage(path)?;
        Self::parse(storage)
    }

    #[cfg(unix)]
    fn load_storage(path: &str) -> io::Result<Storage> {
        use std::os::unix::io::AsRawFd;
        let file = fs::File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "empty file"));
        }
        let fd = file.as_raw_fd();
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                fd,
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            // Fallback to read
            drop(file);
            let data = fs::read(path)?;
            return Ok(Storage::Owned(data));
        }
        Ok(Storage::Mmap { ptr: ptr as *mut u8, len })
    }

    #[cfg(not(unix))]
    fn load_storage(path: &str) -> io::Result<Storage> {
        let data = fs::read(path)?;
        Ok(Storage::Owned(data))
    }

    fn parse(storage: Storage) -> io::Result<Self> {
        let data = storage.as_slice();
        if data.len() < 8 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // First 8 bytes: little-endian u64 header size
        let header_size = u64::from_le_bytes([
            data[0], data[1], data[2], data[3],
            data[4], data[5], data[6], data[7],
        ]) as usize;

        if header_size > data.len() - 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("header size {} exceeds file size {}", header_size, data.len()),
            ));
        }

        let header_json = std::str::from_utf8(&data[8..8 + header_size])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let tensors = parse_safetensors_header(header_json)?;

        Ok(SafetensorsFile {
            storage,
            header_size,
            tensors,
        })
    }

    /// Get the data offset base (8 bytes for header length + header_size).
    fn data_offset(&self) -> usize {
        8 + self.header_size
    }

    /// Get the raw bytes backing this file.
    fn data(&self) -> &[u8] {
        self.storage.as_slice()
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    /// Get tensor info by name.
    pub fn tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    /// Get tensor shape by name.
    pub fn tensor_shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|t| t.shape.as_slice())
    }

    /// Load a tensor as f32, dequantizing F16/BF16 if needed.
    pub fn load_tensor_f32(&self, name: &str) -> Vec<f32> {
        let info = self.tensors.get(name).unwrap_or_else(|| {
            panic!("tensor '{}' not found in safetensors file", name)
        });
        let base = self.data_offset();
        let raw = &self.data()[base + info.data_start..base + info.data_end];
        let num_elements = info.num_elements();

        match info.dtype {
            SafetensorsDtype::F32 => {
                let mut out = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let off = i * 4;
                    out.push(f32::from_le_bytes([
                        raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                    ]));
                }
                out
            }
            SafetensorsDtype::F16 => {
                let mut out = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let off = i * 2;
                    out.push(crate::gguf::f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]])));
                }
                out
            }
            SafetensorsDtype::BF16 => {
                let mut out = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let off = i * 2;
                    out.push(bf16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]])));
                }
                out
            }
            _ => panic!(
                "unsupported dtype {:?} for tensor '{}' — only F32, F16, BF16 supported",
                info.dtype, name
            ),
        }
    }
}

// --- Handwritten JSON header parser for safetensors ---
//
// The safetensors header is a JSON object like:
// {
//   "__metadata__": { ... },
//   "model.layers.0.self_attn.q_proj.weight": {
//     "dtype": "BF16",
//     "shape": [4096, 4096],
//     "data_offsets": [0, 33554432]
//   },
//   ...
// }

fn parse_safetensors_header(json: &str) -> io::Result<HashMap<String, TensorInfo>> {
    let mut tensors = HashMap::new();
    let bytes = json.as_bytes();
    let mut pos = 0;

    // Skip to opening brace
    skip_whitespace(bytes, &mut pos);
    if pos >= bytes.len() || bytes[pos] != b'{' {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '{'"));
    }
    pos += 1;

    loop {
        skip_whitespace(bytes, &mut pos);
        if pos >= bytes.len() {
            break;
        }
        if bytes[pos] == b'}' {
            break;
        }
        if bytes[pos] == b',' {
            pos += 1;
            continue;
        }

        // Parse key
        let key = parse_json_str(bytes, &mut pos)?;

        // Skip colon
        skip_whitespace(bytes, &mut pos);
        if pos >= bytes.len() || bytes[pos] != b':' {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "expected ':'"));
        }
        pos += 1;
        skip_whitespace(bytes, &mut pos);

        if key == "__metadata__" {
            // Skip the metadata value (an object)
            skip_json_value(bytes, &mut pos)?;
            continue;
        }

        // Parse tensor info object
        if pos >= bytes.len() || bytes[pos] != b'{' {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '{' for tensor info"));
        }
        pos += 1;

        let mut dtype: Option<SafetensorsDtype> = None;
        let mut shape: Vec<usize> = Vec::new();
        let mut data_start: usize = 0;
        let mut data_end: usize = 0;

        loop {
            skip_whitespace(bytes, &mut pos);
            if pos >= bytes.len() {
                break;
            }
            if bytes[pos] == b'}' {
                pos += 1;
                break;
            }
            if bytes[pos] == b',' {
                pos += 1;
                continue;
            }

            let field_key = parse_json_str(bytes, &mut pos)?;
            skip_whitespace(bytes, &mut pos);
            if pos >= bytes.len() || bytes[pos] != b':' {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "expected ':'"));
            }
            pos += 1;
            skip_whitespace(bytes, &mut pos);

            match field_key.as_str() {
                "dtype" => {
                    let dtype_str = parse_json_str(bytes, &mut pos)?;
                    dtype = SafetensorsDtype::from_str(&dtype_str);
                    if dtype.is_none() {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("unknown dtype '{}' for tensor '{}'", dtype_str, key),
                        ));
                    }
                }
                "shape" => {
                    shape = parse_json_usize_array(bytes, &mut pos)?;
                }
                "data_offsets" => {
                    let offsets = parse_json_usize_array(bytes, &mut pos)?;
                    if offsets.len() != 2 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("expected 2 data_offsets for tensor '{}', got {}", key, offsets.len()),
                        ));
                    }
                    data_start = offsets[0];
                    data_end = offsets[1];
                }
                _ => {
                    skip_json_value(bytes, &mut pos)?;
                }
            }
        }

        if let Some(dt) = dtype {
            tensors.insert(
                key.clone(),
                TensorInfo {
                    name: key,
                    dtype: dt,
                    shape,
                    data_start,
                    data_end,
                },
            );
        }
    }

    Ok(tensors)
}

fn skip_whitespace(bytes: &[u8], pos: &mut usize) {
    while *pos < bytes.len() && (bytes[*pos] == b' ' || bytes[*pos] == b'\n' || bytes[*pos] == b'\r' || bytes[*pos] == b'\t') {
        *pos += 1;
    }
}

fn parse_json_str(bytes: &[u8], pos: &mut usize) -> io::Result<String> {
    if *pos >= bytes.len() || bytes[*pos] != b'"' {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '\"'"));
    }
    *pos += 1;
    let mut result = String::new();
    while *pos < bytes.len() {
        let b = bytes[*pos];
        if b == b'\\' {
            *pos += 1;
            if *pos >= bytes.len() {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected end of string"));
            }
            match bytes[*pos] {
                b'"' => result.push('"'),
                b'\\' => result.push('\\'),
                b'n' => result.push('\n'),
                b'r' => result.push('\r'),
                b't' => result.push('\t'),
                b'/' => result.push('/'),
                b'u' => {
                    *pos += 1;
                    let mut hex = String::new();
                    for _ in 0..4 {
                        if *pos < bytes.len() {
                            hex.push(bytes[*pos] as char);
                            *pos += 1;
                        }
                    }
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(ch) = char::from_u32(code) {
                            result.push(ch);
                        }
                    }
                    continue; // pos already advanced
                }
                other => {
                    result.push('\\');
                    result.push(other as char);
                }
            }
            *pos += 1;
        } else if b == b'"' {
            *pos += 1;
            return Ok(result);
        } else {
            result.push(b as char);
            *pos += 1;
        }
    }
    Err(io::Error::new(io::ErrorKind::InvalidData, "unterminated string"))
}

fn parse_json_usize_array(bytes: &[u8], pos: &mut usize) -> io::Result<Vec<usize>> {
    if *pos >= bytes.len() || bytes[*pos] != b'[' {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "expected '['"));
    }
    *pos += 1;
    let mut result = Vec::new();

    loop {
        skip_whitespace(bytes, pos);
        if *pos >= bytes.len() {
            break;
        }
        if bytes[*pos] == b']' {
            *pos += 1;
            return Ok(result);
        }
        if bytes[*pos] == b',' {
            *pos += 1;
            continue;
        }
        // Parse number
        let start = *pos;
        while *pos < bytes.len() && bytes[*pos].is_ascii_digit() {
            *pos += 1;
        }
        let num_str = std::str::from_utf8(&bytes[start..*pos])
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let num: usize = num_str
            .parse()
            .map_err(|e: std::num::ParseIntError| io::Error::new(io::ErrorKind::InvalidData, e))?;
        result.push(num);
    }

    Err(io::Error::new(io::ErrorKind::InvalidData, "unterminated array"))
}

/// Skip a JSON value (string, number, object, array, bool, null).
fn skip_json_value(bytes: &[u8], pos: &mut usize) -> io::Result<()> {
    skip_whitespace(bytes, pos);
    if *pos >= bytes.len() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "unexpected end"));
    }

    match bytes[*pos] {
        b'"' => {
            parse_json_str(bytes, pos)?;
        }
        b'{' => {
            *pos += 1;
            let mut depth = 1u32;
            let mut in_string = false;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'"' if !in_string => in_string = true,
                    b'"' => in_string = false,
                    b'\\' if in_string => { *pos += 1; } // skip escaped char
                    b'{' if !in_string => depth += 1,
                    b'}' if !in_string => depth -= 1,
                    _ => {}
                }
                *pos += 1;
            }
        }
        b'[' => {
            *pos += 1;
            let mut depth = 1u32;
            let mut in_string = false;
            while *pos < bytes.len() && depth > 0 {
                match bytes[*pos] {
                    b'"' if !in_string => in_string = true,
                    b'"' => in_string = false,
                    b'\\' if in_string => { *pos += 1; }
                    b'[' if !in_string => depth += 1,
                    b']' if !in_string => depth -= 1,
                    _ => {}
                }
                *pos += 1;
            }
        }
        _ => {
            // number, bool, null — skip until delimiter
            while *pos < bytes.len()
                && bytes[*pos] != b','
                && bytes[*pos] != b'}'
                && bytes[*pos] != b']'
                && bytes[*pos] != b' '
                && bytes[*pos] != b'\n'
                && bytes[*pos] != b'\r'
                && bytes[*pos] != b'\t'
            {
                *pos += 1;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal safetensors file in memory.
    fn build_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        // Build data section first
        let mut data = Vec::new();
        let mut entries = Vec::new();
        for (name, dtype, shape, raw) in tensors {
            let start = data.len();
            data.extend_from_slice(raw);
            let end = data.len();
            entries.push((*name, *dtype, *shape, start, end));
        }

        // Build header JSON
        let mut header = String::from("{");
        for (i, (name, dtype, shape, start, end)) in entries.iter().enumerate() {
            if i > 0 {
                header.push(',');
            }
            let shape_str: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
            header.push_str(&format!(
                "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name,
                dtype,
                shape_str.join(","),
                start,
                end
            ));
        }
        header.push('}');

        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_len.to_le_bytes());
        file.extend_from_slice(header_bytes);
        file.extend_from_slice(&data);
        file
    }

    #[test]
    fn test_parse_f32_tensor() {
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let file_bytes = build_safetensors(&[("weights", "F32", &[2, 3], &raw)]);

        let storage = Storage::Owned(file_bytes);
        let sf = SafetensorsFile::parse(storage).unwrap();

        assert!(sf.tensors.contains_key("weights"));
        let info = &sf.tensors["weights"];
        assert_eq!(info.dtype, SafetensorsDtype::F32);
        assert_eq!(info.shape, vec![2, 3]);
        assert_eq!(info.num_elements(), 6);

        let loaded = sf.load_tensor_f32("weights");
        assert_eq!(loaded, values);
    }

    #[test]
    fn test_parse_bf16_tensor() {
        // BF16 for 1.0 = 0x3F80 (upper 16 bits of f32 1.0 = 0x3F800000)
        let bf16_one: u16 = 0x3F80;
        let bf16_two: u16 = 0x4000; // 2.0
        let raw: Vec<u8> = [bf16_one, bf16_two]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let file_bytes = build_safetensors(&[("x", "BF16", &[2], &raw)]);

        let storage = Storage::Owned(file_bytes);
        let sf = SafetensorsFile::parse(storage).unwrap();
        let loaded = sf.load_tensor_f32("x");
        assert_eq!(loaded.len(), 2);
        assert!((loaded[0] - 1.0).abs() < 1e-6);
        assert!((loaded[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_bf16_to_f32() {
        assert_eq!(bf16_to_f32(0x0000), 0.0);
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xBF80), -1.0);
        assert_eq!(bf16_to_f32(0x4000), 2.0);
    }

    #[test]
    fn test_tensor_names() {
        let raw = 0.0f32.to_le_bytes();
        let file_bytes = build_safetensors(&[
            ("a", "F32", &[1], &raw),
            ("b", "F32", &[1], &raw),
        ]);
        let storage = Storage::Owned(file_bytes);
        let sf = SafetensorsFile::parse(storage).unwrap();
        let mut names = sf.tensor_names();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn test_metadata_skipped() {
        let raw = 1.0f32.to_le_bytes();
        let header = r#"{"__metadata__":{"format":"pt"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let header_bytes = header.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut file = Vec::new();
        file.extend_from_slice(&header_len.to_le_bytes());
        file.extend_from_slice(header_bytes);
        file.extend_from_slice(&raw);

        let storage = Storage::Owned(file);
        let sf = SafetensorsFile::parse(storage).unwrap();
        assert_eq!(sf.tensors.len(), 1);
        assert!(sf.tensors.contains_key("w"));
    }

    #[test]
    fn test_empty_shape_scalar() {
        let raw = 42.0f32.to_le_bytes();
        let file_bytes = build_safetensors(&[("scalar", "F32", &[], &raw)]);
        let storage = Storage::Owned(file_bytes);
        let sf = SafetensorsFile::parse(storage).unwrap();
        let loaded = sf.load_tensor_f32("scalar");
        assert_eq!(loaded.len(), 1);
        assert!((loaded[0] - 42.0).abs() < 1e-6);
    }
}
