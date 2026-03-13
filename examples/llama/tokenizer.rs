use peregrine::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;

/// BPE tokenizer loaded from GGUF embedded vocabulary.
/// Supports Llama 3 byte-level BPE with byte-fallback tokens.
pub struct Tokenizer {
    /// Token ID → string piece
    vocab: Vec<String>,
    /// Token ID → merge priority score (lower = merge first)
    scores: Vec<f32>,
    /// Token ID → type (1=normal, 2=unknown, 3=control, 4=user, 5=unused, 6=byte)
    token_types: Vec<u32>,
    /// String piece → token ID
    piece_to_id: HashMap<String, usize>,
    pub bos_id: usize,
    pub eos_id: usize,
    vocab_size: usize,
}

impl Tokenizer {
    /// Load tokenizer from GGUF metadata.
    pub fn from_gguf(gguf: &GgufFile) -> Self {
        let tokens_meta = gguf
            .get_metadata("tokenizer.ggml.tokens")
            .expect("missing tokenizer.ggml.tokens");
        let tokens: Vec<String> = match tokens_meta {
            MetadataValue::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    MetadataValue::String(s) => s.clone(),
                    _ => panic!("expected string in tokens array"),
                })
                .collect(),
            _ => panic!("expected array for tokenizer.ggml.tokens"),
        };

        let scores: Vec<f32> = gguf
            .get_metadata("tokenizer.ggml.scores")
            .and_then(|v| v.as_f32_array())
            .unwrap_or_else(|| vec![0.0; tokens.len()]);

        let token_types: Vec<u32> = gguf
            .get_metadata("tokenizer.ggml.token_type")
            .and_then(|v| v.as_u32_array())
            .unwrap_or_else(|| vec![1; tokens.len()]);

        let vocab_size = tokens.len();
        let mut piece_to_id = HashMap::with_capacity(vocab_size);
        for (i, t) in tokens.iter().enumerate() {
            piece_to_id.insert(t.clone(), i);
        }

        // Llama 3.2 special tokens
        let bos_id = gguf
            .get_u32("tokenizer.ggml.bos_token_id")
            .unwrap_or(128000) as usize;
        let eos_id = gguf
            .get_u32("tokenizer.ggml.eos_token_id")
            .unwrap_or(128001) as usize;

        Tokenizer {
            vocab: tokens,
            scores,
            token_types,
            piece_to_id,
            bos_id,
            eos_id,
            vocab_size,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode text to token IDs using byte-level BPE.
    /// Does NOT prepend BOS — caller should do that.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return vec![];
        }

        // Start with one token per byte (byte-fallback)
        let bytes = text.as_bytes();
        let mut tokens: Vec<usize> = Vec::with_capacity(bytes.len());

        // Try to match each byte to a byte-fallback token <0xHH>
        for &b in bytes {
            let byte_token = format!("<0x{:02X}>", b);
            if let Some(&id) = self.piece_to_id.get(&byte_token) {
                tokens.push(id);
            } else {
                // Try single-char lookup
                let ch = b as char;
                if let Some(&id) = self.piece_to_id.get(&ch.to_string()) {
                    tokens.push(id);
                } else {
                    tokens.push(0); // unknown
                }
            }
        }

        // Iteratively merge the best pair
        loop {
            if tokens.len() < 2 {
                break;
            }

            // Find the best merge (highest score = lowest merge rank)
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for i in 0..tokens.len() - 1 {
                let merged = format!("{}{}", self.vocab[tokens[i]], self.vocab[tokens[i + 1]]);
                if let Some(&merged_id) = self.piece_to_id.get(&merged) {
                    let score = self.scores[merged_id];
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break;
            }

            // Apply the merge
            let merged = format!(
                "{}{}",
                self.vocab[tokens[best_idx]],
                self.vocab[tokens[best_idx + 1]]
            );
            let merged_id = self.piece_to_id[&merged];
            tokens[best_idx] = merged_id;
            tokens.remove(best_idx + 1);
        }

        tokens
    }

    /// Decode token IDs to a string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            if id >= self.vocab_size {
                continue;
            }
            let piece = &self.vocab[id];

            // Check if it's a byte-fallback token <0xHH>
            if piece.starts_with("<0x") && piece.ends_with('>') && piece.len() == 6 {
                if let Ok(byte_val) = u8::from_str_radix(&piece[3..5], 16) {
                    bytes.push(byte_val);
                    continue;
                }
            }

            // Skip control tokens (BOS, EOS, etc.)
            if self.token_types.get(id).copied() == Some(3) {
                continue;
            }

            bytes.extend_from_slice(piece.as_bytes());
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}
