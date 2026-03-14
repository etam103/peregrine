use crate::gguf::{GgufFile, MetadataValue};
use std::collections::HashMap;

/// BPE tokenizer loaded from GGUF embedded vocabulary or HuggingFace tokenizer.json.
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
    /// BPE merges in priority order (for HF tokenizer.json format)
    merges: Vec<(String, String)>,
    /// Whether this tokenizer uses GPT-2 style byte-level BPE
    use_byte_level_bpe: bool,
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
            merges: Vec::new(),
            use_byte_level_bpe: false,
            bos_id,
            eos_id,
            vocab_size,
        }
    }

    /// Load tokenizer from a HuggingFace tokenizer.json string (GPT-2 style byte-level BPE).
    pub fn from_hf_json(json: &str) -> Self {
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut merges: Vec<(String, String)> = Vec::new();
        let mut added_tokens: HashMap<String, usize> = HashMap::new();

        // Parse vocab from "model" -> "vocab" section
        if let Some(vocab_start) = find_json_key(json, "\"vocab\"") {
            if let Some(obj_start) = json[vocab_start..].find('{') {
                let abs_start = vocab_start + obj_start;
                if let Some(obj_end) = find_matching_brace(&json[abs_start..]) {
                    let vocab_str = &json[abs_start + 1..abs_start + obj_end];
                    parse_hf_vocab_entries(vocab_str, &mut vocab);
                }
            }
        }

        // Parse merges
        if let Some(merges_start) = find_json_key(json, "\"merges\"") {
            if let Some(arr_start) = json[merges_start..].find('[') {
                let abs_start = merges_start + arr_start;
                if let Some(arr_end) = find_matching_bracket(&json[abs_start..]) {
                    let merges_str = &json[abs_start + 1..abs_start + arr_end];
                    parse_hf_merges(merges_str, &mut merges);
                }
            }
        }

        // Parse added_tokens
        if let Some(at_start) = find_json_key(json, "\"added_tokens\"") {
            if let Some(arr_start) = json[at_start..].find('[') {
                let abs_start = at_start + arr_start;
                if let Some(arr_end) = find_matching_bracket(&json[abs_start..]) {
                    let at_str = &json[abs_start + 1..abs_start + arr_end];
                    parse_hf_added_tokens(at_str, &mut added_tokens);
                }
            }
        }

        // Merge added tokens into vocab
        for (token, id) in &added_tokens {
            vocab.insert(token.clone(), *id);
        }

        let vocab_size = vocab.len();

        // Build vocab list and reverse map
        let mut vocab_list = vec![String::new(); vocab_size];
        for (token, &id) in &vocab {
            if id < vocab_size {
                vocab_list[id] = token.clone();
            }
        }

        let mut piece_to_id = HashMap::with_capacity(vocab_size);
        for (token, &id) in &vocab {
            piece_to_id.insert(token.clone(), id);
        }

        // Find BOS/EOS tokens
        let bos_id = added_tokens
            .get("<|begin_of_text|>")
            .or_else(|| vocab.get("<|begin_of_text|>"))
            .or_else(|| added_tokens.get("<s>"))
            .or_else(|| vocab.get("<s>"))
            .copied()
            .unwrap_or(128000);
        let eos_id = added_tokens
            .get("<|end_of_text|>")
            .or_else(|| vocab.get("<|end_of_text|>"))
            .or_else(|| added_tokens.get("</s>"))
            .or_else(|| vocab.get("</s>"))
            .copied()
            .unwrap_or(128001);

        eprintln!(
            "HF tokenizer: {} vocab, {} merges, BOS={}, EOS={}",
            vocab_size, merges.len(), bos_id, eos_id
        );

        Tokenizer {
            vocab: vocab_list,
            scores: vec![0.0; vocab_size],
            token_types: vec![1; vocab_size],
            piece_to_id,
            merges,
            use_byte_level_bpe: true,
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

        if self.use_byte_level_bpe {
            return self.encode_byte_level_bpe(text);
        }

        // SentencePiece-style BPE (GGUF format):
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

    /// GPT-2 style byte-level BPE encoding (for HuggingFace tokenizer.json).
    fn encode_byte_level_bpe(&self, text: &str) -> Vec<usize> {
        let bytes = text.as_bytes();
        let mut symbols: Vec<String> = bytes.iter().map(|&b| byte_to_unicode(b)).collect();

        // Apply BPE merges in priority order
        for (first, second) in &self.merges {
            let mut i = 0;
            while i + 1 < symbols.len() {
                if symbols[i] == *first && symbols[i + 1] == *second {
                    symbols[i] = format!("{}{}", first, second);
                    symbols.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        let mut ids = Vec::with_capacity(symbols.len());
        for sym in &symbols {
            if let Some(&id) = self.piece_to_id.get(sym) {
                ids.push(id);
            } else {
                for b in sym.as_bytes() {
                    let byte_token = byte_to_unicode(*b);
                    if let Some(&id) = self.piece_to_id.get(&byte_token) {
                        ids.push(id);
                    }
                }
            }
        }
        ids
    }

    /// Decode token IDs to a string.
    pub fn decode(&self, tokens: &[usize]) -> String {
        if self.use_byte_level_bpe {
            return self.decode_byte_level_bpe(tokens);
        }

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

    /// GPT-2 style byte-level BPE decoding.
    fn decode_byte_level_bpe(&self, tokens: &[usize]) -> String {
        let mut pieces = String::new();
        for &id in tokens {
            if id < self.vocab.len() {
                pieces.push_str(&self.vocab[id]);
            }
        }
        // Convert byte-level unicode back to actual bytes
        let bytes: Vec<u8> = pieces.chars().filter_map(unicode_to_byte).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

// === GPT-2 byte-level BPE helpers ===

/// GPT-2 style byte-to-unicode mapping.
fn byte_to_unicode(b: u8) -> String {
    let c = match b {
        b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => b as u32,
        _ => b as u32 + 256,
    };
    char::from_u32(c).unwrap().to_string()
}

/// Reverse of byte_to_unicode.
fn unicode_to_byte(c: char) -> Option<u8> {
    let code = c as u32;
    if code >= 256 + 256 {
        return None;
    }
    if code >= 256 {
        return Some((code - 256) as u8);
    }
    let b = code as u8;
    match b {
        b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF => Some(b),
        _ => None,
    }
}

// === JSON parsing helpers for HuggingFace tokenizer.json ===

fn find_json_key(s: &str, key: &str) -> Option<usize> {
    s.find(key).map(|pos| pos + key.len())
}

fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape { escape = false; continue; }
        if c == '\\' && in_string { escape = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        match c {
            '{' => depth += 1,
            '}' => { depth -= 1; if depth == 0 { return Some(i); } }
            _ => {}
        }
    }
    None
}

fn find_matching_bracket(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape { escape = false; continue; }
        if c == '\\' && in_string { escape = true; continue; }
        if c == '"' { in_string = !in_string; continue; }
        if in_string { continue; }
        match c {
            '[' => depth += 1,
            ']' => { depth -= 1; if depth == 0 { return Some(i); } }
            _ => {}
        }
    }
    None
}

fn parse_hf_json_string(s: &str) -> Option<(String, usize)> {
    let s = s.trim_start();
    if !s.starts_with('"') { return None; }
    let mut result = String::new();
    let mut chars = s[1..].char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                if let Some((_, escaped)) = chars.next() {
                    match escaped {
                        '"' => result.push('"'),
                        '\\' => result.push('\\'),
                        'n' => result.push('\n'),
                        'r' => result.push('\r'),
                        't' => result.push('\t'),
                        'u' => {
                            let mut hex = String::new();
                            for _ in 0..4 {
                                if let Some((_, h)) = chars.next() { hex.push(h); }
                            }
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(code) { result.push(ch); }
                            }
                        }
                        _ => { result.push('\\'); result.push(escaped); }
                    }
                }
            }
            '"' => return Some((result, i + 2)),
            _ => result.push(c),
        }
    }
    None
}

fn parse_hf_vocab_entries(s: &str, vocab: &mut HashMap<String, usize>) {
    let mut pos = 0;
    let bytes = s.as_bytes();
    while pos < s.len() {
        while pos < s.len() && bytes[pos] != b'"' { pos += 1; }
        if pos >= s.len() { break; }
        if let Some((key, advance)) = parse_hf_json_string(&s[pos..]) {
            pos += advance;
            while pos < s.len() && bytes[pos] != b':' { pos += 1; }
            pos += 1;
            while pos < s.len() && bytes[pos] == b' ' { pos += 1; }
            let num_start = pos;
            while pos < s.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b'-') { pos += 1; }
            if let Ok(id) = s[num_start..pos].parse::<usize>() {
                vocab.insert(key, id);
            }
        } else {
            pos += 1;
        }
    }
}

fn parse_hf_merges(s: &str, merges: &mut Vec<(String, String)>) {
    let mut pos = 0;
    while pos < s.len() {
        if let Some((merge_str, advance)) = parse_hf_json_string(&s[pos..]) {
            pos += advance;
            if let Some(space_pos) = merge_str.find(' ') {
                let first = merge_str[..space_pos].to_string();
                let second = merge_str[space_pos + 1..].to_string();
                merges.push((first, second));
            }
        } else {
            pos += 1;
        }
    }
}

fn parse_hf_added_tokens(s: &str, tokens: &mut HashMap<String, usize>) {
    let mut pos = 0;
    let bytes = s.as_bytes();
    while pos < s.len() {
        if bytes[pos] == b'{' {
            if let Some(end) = find_matching_brace(&s[pos..]) {
                let obj = &s[pos + 1..pos + end];
                let mut content = String::new();
                let mut id: Option<usize> = None;

                if let Some(ck) = find_json_key(obj, "\"content\"") {
                    let rest = &obj[ck..];
                    if let Some(colon) = rest.find(':') {
                        if let Some((val, _)) = parse_hf_json_string(rest[colon + 1..].trim_start()) {
                            content = val;
                        }
                    }
                }
                if let Some(ik) = find_json_key(obj, "\"id\"") {
                    let rest = &obj[ik..];
                    if let Some(colon) = rest.find(':') {
                        let after = rest[colon + 1..].trim_start();
                        let num_end = after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
                        if let Ok(n) = after[..num_end].parse::<usize>() {
                            id = Some(n);
                        }
                    }
                }

                if let Some(id) = id {
                    if !content.is_empty() {
                        tokens.insert(content, id);
                    }
                }
                pos += end + 1;
            } else {
                pos += 1;
            }
        } else {
            pos += 1;
        }
    }
}
