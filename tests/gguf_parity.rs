//! GGUF loader integration test.
//! Verifies parsing, metadata extraction, and tensor shape/dequant correctness.

use peregrine::gguf::{dequant_q8_0, dequant_q4_0, dequant_q4_1, GgufFile};

#[test]
fn test_dequant_q8_0_roundtrip() {
    // Create a block: scale=0.5 (f16 ≈ 0x3800), values 1..=32
    let scale_f16: u16 = 0x3800; // f16 0.5
    let mut block = vec![0u8; 34];
    block[0] = (scale_f16 & 0xFF) as u8;
    block[1] = (scale_f16 >> 8) as u8;
    for i in 0..32 {
        block[2 + i] = (i as i8 + 1) as u8;
    }
    let result = dequant_q8_0(&block, 32);
    assert_eq!(result.len(), 32);
    for i in 0..32 {
        let expected = (i as f32 + 1.0) * 0.5;
        assert!(
            (result[i] - expected).abs() < 0.01,
            "q8_0 idx {}: got {} expected {}",
            i,
            result[i],
            expected
        );
    }
}

#[test]
fn test_dequant_q4_0_symmetry() {
    // Scale=2.0 (f16=0x4000), all nibbles = 8 → dequant to 0
    let scale_f16: u16 = 0x4000;
    let mut block = vec![0u8; 18];
    block[0] = (scale_f16 & 0xFF) as u8;
    block[1] = (scale_f16 >> 8) as u8;
    for i in 0..16 {
        block[2 + i] = 0x88; // lo=8, hi=8 → both map to 0
    }
    let result = dequant_q4_0(&block, 32);
    assert_eq!(result.len(), 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.abs() < 0.01,
            "q4_0 idx {}: expected 0.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_dequant_q4_1_with_min() {
    // Scale=1.0, min=10.0
    let scale_f16: u16 = 0x3C00; // f16 1.0
    let min_f16: u16 = 0x4900; // f16 10.0
    let mut block = vec![0u8; 20];
    block[0] = (scale_f16 & 0xFF) as u8;
    block[1] = (scale_f16 >> 8) as u8;
    block[2] = (min_f16 & 0xFF) as u8;
    block[3] = (min_f16 >> 8) as u8;
    // All nibbles = 0 → value = 0*1.0 + 10.0 = 10.0
    for i in 0..16 {
        block[4 + i] = 0x00;
    }
    let result = dequant_q4_1(&block, 32);
    assert_eq!(result.len(), 32);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - 10.0).abs() < 0.1,
            "q4_1 idx {}: expected 10.0, got {}",
            i,
            v
        );
    }
}

#[test]
fn test_gguf_invalid_magic() {
    let data = vec![0u8; 64]; // All zeros, wrong magic
    let result = GgufFile::parse(data);
    assert!(result.is_err());
    assert!(result.err().unwrap().to_string().contains("invalid GGUF magic"));
}

#[test]
fn test_gguf_minimal_valid() {
    // Construct a minimal valid GGUF v3 file: magic + version + 0 tensors + 0 kv
    let mut data = Vec::new();
    // Magic "GGUF" as u32 LE
    data.extend_from_slice(&0x46554747u32.to_le_bytes());
    // Version 3
    data.extend_from_slice(&3u32.to_le_bytes());
    // n_tensors = 0
    data.extend_from_slice(&0u64.to_le_bytes());
    // n_kv = 0
    data.extend_from_slice(&0u64.to_le_bytes());
    // Pad to alignment
    while data.len() < 64 {
        data.push(0);
    }
    let gguf = GgufFile::parse(data).expect("should parse minimal GGUF");
    assert!(gguf.metadata.is_empty());
    assert!(gguf.tensors.is_empty());
}
