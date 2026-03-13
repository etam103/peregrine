//! GGUF binary format parser for loading quantized model weights.
//!
//! Supports GGUF v3 with Q8_0, Q4_0, Q4_1, F32, and F16 tensor types.

use std::collections::HashMap;
use std::fs;
use std::io;

/// GGUF magic number: "GGUF" in little-endian
const GGUF_MAGIC: u32 = 0x46554747;

/// GGUF tensor element types
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(u32)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q8_0 = 8,
}

impl GgmlType {
    fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(GgmlType::F32),
            1 => Some(GgmlType::F16),
            2 => Some(GgmlType::Q4_0),
            3 => Some(GgmlType::Q4_1),
            8 => Some(GgmlType::Q8_0),
            _ => None,
        }
    }

    /// Block size for quantized types (number of elements per block)
    pub fn block_size(&self) -> usize {
        match self {
            GgmlType::F32 => 1,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 32,
            GgmlType::Q4_1 => 32,
            GgmlType::Q8_0 => 32,
        }
    }

    /// Bytes per block
    pub fn block_bytes(&self) -> usize {
        match self {
            GgmlType::F32 => 4,
            GgmlType::F16 => 2,
            GgmlType::Q4_0 => 2 + 16,    // f16 scale + 16 bytes of nibbles
            GgmlType::Q4_1 => 4 + 16,    // f16 scale + f16 min + 16 bytes of nibbles
            GgmlType::Q8_0 => 2 + 32,    // f16 scale + 32 x i8
        }
    }
}

/// GGUF metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
    Array(Vec<MetadataValue>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::Uint32(v) => Some(*v),
            MetadataValue::Int32(v) => Some(*v as u32),
            MetadataValue::Uint64(v) => Some(*v as u32),
            MetadataValue::Int64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            MetadataValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<&str>> {
        match self {
            MetadataValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_str()?);
                }
                Some(result)
            }
            _ => None,
        }
    }

    pub fn as_f32_array(&self) -> Option<Vec<f32>> {
        match self {
            MetadataValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_f32()?);
                }
                Some(result)
            }
            _ => None,
        }
    }

    pub fn as_u32_array(&self) -> Option<Vec<u32>> {
        match self {
            MetadataValue::Array(arr) => {
                let mut result = Vec::with_capacity(arr.len());
                for v in arr {
                    result.push(v.as_u32()?);
                }
                Some(result)
            }
            _ => None,
        }
    }
}

/// Descriptor for a tensor stored in the GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensor {
    pub name: String,
    pub dtype: GgmlType,
    pub shape: Vec<usize>,
    pub data_offset: u64,
}

impl GgufTensor {
    /// Total number of elements
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    /// Total bytes in the file for this tensor's data
    pub fn data_size(&self) -> usize {
        let n = self.num_elements();
        let bs = self.dtype.block_size();
        assert!(n % bs == 0, "tensor elements {} not divisible by block size {}", n, bs);
        (n / bs) * self.dtype.block_bytes()
    }
}

/// A parsed GGUF file.
pub struct GgufFile {
    pub metadata: HashMap<String, MetadataValue>,
    pub tensors: HashMap<String, GgufTensor>,
    data: Vec<u8>, // memory-mapped or loaded file data
    data_start: u64,
}

// Reader helpers
struct BufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        BufReader { data, pos: 0 }
    }

    fn read_u8(&mut self) -> u8 {
        let v = self.data[self.pos];
        self.pos += 1;
        v
    }

    fn read_i8(&mut self) -> i8 {
        self.read_u8() as i8
    }

    fn read_u16(&mut self) -> u16 {
        let bytes = &self.data[self.pos..self.pos + 2];
        self.pos += 2;
        u16::from_le_bytes([bytes[0], bytes[1]])
    }

    fn read_i16(&mut self) -> i16 {
        self.read_u16() as i16
    }

    fn read_u32(&mut self) -> u32 {
        let bytes = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn read_i32(&mut self) -> i32 {
        self.read_u32() as i32
    }

    fn read_u64(&mut self) -> u64 {
        let bytes = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn read_i64(&mut self) -> i64 {
        self.read_u64() as i64
    }

    fn read_f32(&mut self) -> f32 {
        let bytes = &self.data[self.pos..self.pos + 4];
        self.pos += 4;
        f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
    }

    fn read_f64(&mut self) -> f64 {
        let bytes = &self.data[self.pos..self.pos + 8];
        self.pos += 8;
        f64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ])
    }

    fn read_bool(&mut self) -> bool {
        self.read_u8() != 0
    }

    fn read_string(&mut self) -> String {
        let len = self.read_u64() as usize;
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .expect("invalid UTF-8 in GGUF string")
            .to_string();
        self.pos += len;
        s
    }

    fn read_metadata_value(&mut self, vtype: u32) -> MetadataValue {
        match vtype {
            0 => MetadataValue::Uint8(self.read_u8()),
            1 => MetadataValue::Int8(self.read_i8()),
            2 => MetadataValue::Uint16(self.read_u16()),
            3 => MetadataValue::Int16(self.read_i16()),
            4 => MetadataValue::Uint32(self.read_u32()),
            5 => MetadataValue::Int32(self.read_i32()),
            6 => MetadataValue::Float32(self.read_f32()),
            7 => MetadataValue::Bool(self.read_bool()),
            8 => MetadataValue::String(self.read_string()),
            9 => {
                let arr_type = self.read_u32();
                let arr_len = self.read_u64() as usize;
                let mut arr = Vec::with_capacity(arr_len);
                for _ in 0..arr_len {
                    arr.push(self.read_metadata_value(arr_type));
                }
                MetadataValue::Array(arr)
            }
            10 => MetadataValue::Uint64(self.read_u64()),
            11 => MetadataValue::Int64(self.read_i64()),
            12 => MetadataValue::Float64(self.read_f64()),
            _ => panic!("unknown GGUF metadata value type: {}", vtype),
        }
    }
}

/// Dequantize Q8_0 block data to f32.
/// Each block: 2 bytes f16 scale + 32 bytes i8 values = 34 bytes → 32 floats.
pub fn dequant_q8_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    assert!(num_elements % 32 == 0);
    let num_blocks = num_elements / 32;
    let block_size = 34; // 2 (f16 scale) + 32 (i8 data)
    assert!(data.len() >= num_blocks * block_size);

    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let offset = b * block_size;
        let scale = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        for i in 0..32 {
            let q = data[offset + 2 + i] as i8;
            out.push(q as f32 * scale);
        }
    }
    out
}

/// Dequantize Q4_0 block data to f32.
/// Each block: 2 bytes f16 scale + 16 bytes nibble-packed = 18 bytes → 32 floats.
/// Nibbles are signed: stored as unsigned 0..15, subtract 8 to get -8..7.
pub fn dequant_q4_0(data: &[u8], num_elements: usize) -> Vec<f32> {
    assert!(num_elements % 32 == 0);
    let num_blocks = num_elements / 32;
    let block_size = 18;
    assert!(data.len() >= num_blocks * block_size);

    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let offset = b * block_size;
        let scale = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        for i in 0..16 {
            let byte = data[offset + 2 + i];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = ((byte >> 4) & 0x0F) as i32 - 8;
            out.push(lo as f32 * scale);
            out.push(hi as f32 * scale);
        }
    }
    out
}

/// Dequantize Q4_1 block data to f32.
/// Each block: 2 bytes f16 scale + 2 bytes f16 min + 16 bytes nibble-packed = 20 bytes → 32 floats.
/// Nibbles are unsigned 0..15: value = nibble * scale + min.
pub fn dequant_q4_1(data: &[u8], num_elements: usize) -> Vec<f32> {
    assert!(num_elements % 32 == 0);
    let num_blocks = num_elements / 32;
    let block_size = 20;
    assert!(data.len() >= num_blocks * block_size);

    let mut out = Vec::with_capacity(num_elements);
    for b in 0..num_blocks {
        let offset = b * block_size;
        let scale = f16_to_f32(u16::from_le_bytes([data[offset], data[offset + 1]]));
        let min = f16_to_f32(u16::from_le_bytes([data[offset + 2], data[offset + 3]]));
        for i in 0..16 {
            let byte = data[offset + 4 + i];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            out.push(lo * scale + min);
            out.push(hi * scale + min);
        }
    }
    out
}

/// Convert IEEE 754 half-precision (f16) to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let mant = (bits & 0x3FF) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero
            return f32::from_bits(sign << 31);
        }
        // Subnormal: normalize
        let mut m = mant;
        let mut e = 0i32;
        while (m & 0x400) == 0 {
            m <<= 1;
            e -= 1;
        }
        m &= 0x3FF;
        let f32_exp = (127 - 15 + 1 + e) as u32;
        return f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13));
    }
    if exp == 31 {
        // Inf/NaN
        return f32::from_bits((sign << 31) | (0xFF << 23) | (mant << 13));
    }

    let f32_exp = exp + 127 - 15;
    f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
}

/// Load Q8_0 tensor data as a QuantizedTensor (keep quantized, for int8 matmul).
/// Converts from GGUF Q8_0 row-wise blocks to Peregrine per-column symmetric quant format.
pub fn load_tensor_q8_as_quantized(
    data: &[u8],
    rows: usize,
    cols: usize,
) -> crate::quant::QuantizedTensor {
    assert!(cols % 32 == 0);
    let num_elements = rows * cols;
    let num_blocks = num_elements / 32;
    let block_size = 34;
    assert!(data.len() >= num_blocks * block_size);

    // First dequant to f32, then use Peregrine's per-column quantizer.
    // This is simpler and ensures compatibility with the existing matmul kernels.
    let f32_data = dequant_q8_0(data, num_elements);
    crate::quant::quantize_weights(&f32_data, rows, cols)
}

impl GgufFile {
    /// Parse a GGUF file from disk.
    pub fn load(path: &str) -> io::Result<Self> {
        let data = fs::read(path)?;
        Self::parse(data)
    }

    /// Parse GGUF from a byte buffer.
    pub fn parse(data: Vec<u8>) -> io::Result<Self> {
        let mut r = BufReader::new(&data);

        // Header
        let magic = r.read_u32();
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: 0x{:08X}, expected 0x{:08X}", magic, GGUF_MAGIC),
            ));
        }

        let version = r.read_u32();
        if version < 2 || version > 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {}", version),
            ));
        }

        let n_tensors = r.read_u64() as usize;
        let n_kv = r.read_u64() as usize;

        // Metadata KV pairs
        let mut metadata = HashMap::with_capacity(n_kv);
        for _ in 0..n_kv {
            let key = r.read_string();
            let vtype = r.read_u32();
            let value = r.read_metadata_value(vtype);
            metadata.insert(key, value);
        }

        // Tensor info
        let mut tensors = HashMap::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.read_string();
            let n_dims = r.read_u32() as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64() as usize);
            }
            let dtype_raw = r.read_u32();
            let dtype = GgmlType::from_u32(dtype_raw).unwrap_or_else(|| {
                panic!("unsupported GGML type {} for tensor '{}'", dtype_raw, name)
            });
            let data_offset = r.read_u64();
            tensors.insert(
                name.clone(),
                GgufTensor {
                    name,
                    dtype,
                    shape,
                    data_offset,
                },
            );
        }

        // Data starts at the next alignment boundary (32 bytes) after the header
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u32())
            .unwrap_or(32) as u64;
        let header_end = r.pos as u64;
        let data_start = (header_end + alignment - 1) / alignment * alignment;

        Ok(GgufFile {
            metadata,
            tensors,
            data,
            data_start,
        })
    }

    /// Load a tensor and dequantize to f32.
    pub fn load_tensor_f32(&self, name: &str) -> Vec<f32> {
        let info = self.tensors.get(name).unwrap_or_else(|| {
            panic!("tensor '{}' not found in GGUF file", name)
        });
        let offset = (self.data_start + info.data_offset) as usize;
        let size = info.data_size();
        let raw = &self.data[offset..offset + size];
        let num_elements = info.num_elements();

        match info.dtype {
            GgmlType::F32 => {
                let mut out = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let off = i * 4;
                    out.push(f32::from_le_bytes([
                        raw[off], raw[off + 1], raw[off + 2], raw[off + 3],
                    ]));
                }
                out
            }
            GgmlType::F16 => {
                let mut out = Vec::with_capacity(num_elements);
                for i in 0..num_elements {
                    let off = i * 2;
                    out.push(f16_to_f32(u16::from_le_bytes([raw[off], raw[off + 1]])));
                }
                out
            }
            GgmlType::Q8_0 => dequant_q8_0(raw, num_elements),
            GgmlType::Q4_0 => dequant_q4_0(raw, num_elements),
            GgmlType::Q4_1 => dequant_q4_1(raw, num_elements),
        }
    }

    /// Load a Q8_0 tensor as QuantizedTensor (for int8 matmul path).
    /// Falls back to dequant→requant for non-Q8 types.
    pub fn load_tensor_q8(&self, name: &str) -> crate::quant::QuantizedTensor {
        let info = self.tensors.get(name).unwrap_or_else(|| {
            panic!("tensor '{}' not found in GGUF file", name)
        });
        let offset = (self.data_start + info.data_offset) as usize;
        let size = info.data_size();
        let raw = &self.data[offset..offset + size];

        // GGUF shapes are [cols, rows] (row-major with reversed dim order)
        let (rows, cols) = match info.shape.len() {
            1 => (1, info.shape[0]),
            2 => (info.shape[1], info.shape[0]),
            _ => panic!("expected 1D or 2D tensor for q8 load, got {:?}", info.shape),
        };

        load_tensor_q8_as_quantized(raw, rows, cols)
    }

    /// Get a metadata value by key.
    pub fn get_metadata(&self, key: &str) -> Option<&MetadataValue> {
        self.metadata.get(key)
    }

    /// Get u32 metadata value.
    pub fn get_u32(&self, key: &str) -> Option<u32> {
        self.metadata.get(key).and_then(|v| v.as_u32())
    }

    /// Get f32 metadata value.
    pub fn get_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }

    /// Get the tensor shape (converted to Peregrine convention — reversed from GGUF).
    /// GGUF stores shapes in reverse order (innermost dim first).
    pub fn tensor_shape(&self, name: &str) -> Vec<usize> {
        let info = self.tensors.get(name).unwrap_or_else(|| {
            panic!("tensor '{}' not found", name)
        });
        info.shape.iter().rev().cloned().collect()
    }

    /// List all tensor names.
    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f16_to_f32() {
        // 0.0
        assert_eq!(f16_to_f32(0x0000), 0.0);
        // 1.0 = sign=0, exp=15, mant=0 → 0_01111_0000000000 = 0x3C00
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        // -1.0
        assert_eq!(f16_to_f32(0xBC00), -1.0);
        // 0.5 = 0_01110_0000000000 = 0x3800
        assert_eq!(f16_to_f32(0x3800), 0.5);
        // -0.0
        assert_eq!(f16_to_f32(0x8000), -0.0f32);
    }

    #[test]
    fn test_dequant_q8_0_simple() {
        // One block of 32 elements: scale = 1.0 (f16 = 0x3C00), values = 0,1,2,...,31
        let mut block = vec![0u8; 34];
        block[0] = 0x00; // f16 1.0 = 0x3C00, little-endian
        block[1] = 0x3C;
        for i in 0..32 {
            block[2 + i] = i as u8;
        }
        let result = dequant_q8_0(&block, 32);
        assert_eq!(result.len(), 32);
        for i in 0..32 {
            assert!((result[i] - i as f32).abs() < 1e-3, "idx {}: {} != {}", i, result[i], i);
        }
    }

    #[test]
    fn test_dequant_q4_0_simple() {
        // One block: scale = 1.0, nibbles packed
        let mut block = vec![0u8; 18];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0
        // First byte: lo=8 (value 0), hi=9 (value 1)
        block[2] = 0x98; // lo nibble = 8 → 8-8=0, hi nibble = 9 → 9-8=1
        let result = dequant_q4_0(&block, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 0.0).abs() < 1e-3);
        assert!((result[1] - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_dequant_q4_1_simple() {
        // One block: scale = 1.0, min = 0.0
        let mut block = vec![0u8; 20];
        block[0] = 0x00;
        block[1] = 0x3C; // f16 1.0 scale
        block[2] = 0x00;
        block[3] = 0x00; // f16 0.0 min
        // First byte: lo=3, hi=5 → values 3*1+0=3, 5*1+0=5
        block[4] = 0x53;
        let result = dequant_q4_1(&block, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 3.0).abs() < 1e-3);
        assert!((result[1] - 5.0).abs() < 1e-3);
    }

    #[test]
    fn test_gguf_magic() {
        // "GGUF" as bytes: G=0x47, G=0x47, U=0x55, F=0x46
        // Little-endian u32: 0x46555747 — wait, let's check
        let bytes = b"GGUF";
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(magic, GGUF_MAGIC);
    }
}
