use std::collections::HashMap;
use std::fs;

/// Minimal HuggingFace BPE tokenizer (reads tokenizer.json).
pub struct Tokenizer {
    vocab: HashMap<String, usize>,
    vocab_rev: Vec<String>,
    merges: Vec<(String, String)>,
    bos_id: usize,
    eos_id: usize,
    vocab_size: usize,
}

impl Tokenizer {
    /// Load from a HuggingFace `tokenizer.json` file.
    pub fn load(path: &str) -> Self {
        let content = fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read tokenizer {}: {}", path, e);
            std::process::exit(1);
        });

        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut merges: Vec<(String, String)> = Vec::new();
        let mut added_tokens: HashMap<String, usize> = HashMap::new();

        // Parse vocab from "model" -> "vocab" section
        if let Some(vocab_start) = find_key(&content, "\"vocab\"") {
            if let Some(obj_start) = content[vocab_start..].find('{') {
                let abs_start = vocab_start + obj_start;
                if let Some(obj_end) = find_matching_brace(&content[abs_start..]) {
                    let vocab_str = &content[abs_start + 1..abs_start + obj_end];
                    parse_vocab_entries(vocab_str, &mut vocab);
                }
            }
        }

        // Parse merges from "model" -> "merges"
        if let Some(merges_start) = find_key(&content, "\"merges\"") {
            if let Some(arr_start) = content[merges_start..].find('[') {
                let abs_start = merges_start + arr_start;
                if let Some(arr_end) = find_matching_bracket(&content[abs_start..]) {
                    let merges_str = &content[abs_start + 1..abs_start + arr_end];
                    parse_merges(merges_str, &mut merges);
                }
            }
        }

        // Parse added_tokens
        if let Some(at_start) = find_key(&content, "\"added_tokens\"") {
            if let Some(arr_start) = content[at_start..].find('[') {
                let abs_start = at_start + arr_start;
                if let Some(arr_end) = find_matching_bracket(&content[abs_start..]) {
                    let at_str = &content[abs_start + 1..abs_start + arr_end];
                    parse_added_tokens(at_str, &mut added_tokens);
                }
            }
        }

        // Merge added tokens into vocab
        for (token, id) in &added_tokens {
            vocab.insert(token.clone(), *id);
        }

        let vocab_size = vocab.len();

        // Build reverse vocab
        let mut vocab_rev = vec![String::new(); vocab_size];
        for (token, &id) in &vocab {
            if id < vocab_size {
                vocab_rev[id] = token.clone();
            }
        }

        // Find BOS/EOS — gpt-oss uses <|startoftext|> and <|endoftext|>
        let bos_id = added_tokens
            .get("<|startoftext|>")
            .or_else(|| vocab.get("<|startoftext|>"))
            .or_else(|| added_tokens.get("<s>"))
            .or_else(|| vocab.get("<bos>"))
            .copied()
            .unwrap_or(199998);
        let eos_id = added_tokens
            .get("<|endoftext|>")
            .or_else(|| vocab.get("<|endoftext|>"))
            .or_else(|| added_tokens.get("</s>"))
            .or_else(|| vocab.get("<eos>"))
            .copied()
            .unwrap_or(199999);

        eprintln!(
            "Loaded tokenizer: {} vocab, {} merges, BOS={}, EOS={}",
            vocab_size,
            merges.len(),
            bos_id,
            eos_id
        );

        Tokenizer {
            vocab,
            vocab_rev,
            merges,
            bos_id,
            eos_id,
            vocab_size,
        }
    }

    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    pub fn eos_id(&self) -> usize {
        self.eos_id
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Encode text to token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return vec![];
        }

        // Byte-level BPE: convert text to bytes, each byte maps to a vocab token
        let bytes = text.as_bytes();
        let mut symbols: Vec<String> = bytes.iter().map(|&b| byte_to_unicode(b)).collect();

        // Apply BPE merges in order
        for (first, second) in &self.merges {
            let mut i = 0;
            while i + 1 < symbols.len() {
                if &symbols[i] == first && &symbols[i + 1] == second {
                    symbols[i] = format!("{}{}", first, second);
                    symbols.remove(i + 1);
                } else {
                    i += 1;
                }
            }
        }

        // Convert to IDs
        let mut ids = Vec::with_capacity(symbols.len());
        for sym in &symbols {
            if let Some(&id) = self.vocab.get(sym) {
                ids.push(id);
            } else {
                // Fallback: try individual bytes
                for b in sym.as_bytes() {
                    let byte_token = byte_to_unicode(*b);
                    if let Some(&id) = self.vocab.get(&byte_token) {
                        ids.push(id);
                    }
                }
            }
        }

        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut pieces = String::new();
        for &id in ids {
            if id < self.vocab_rev.len() {
                pieces.push_str(&self.vocab_rev[id]);
            }
        }

        // Convert byte-level unicode back to actual bytes
        let bytes: Vec<u8> = pieces.chars().filter_map(unicode_to_byte).collect();

        String::from_utf8_lossy(&bytes).into_owned()
    }
}

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

// === Minimal JSON parsing helpers ===

fn find_key(s: &str, key: &str) -> Option<usize> {
    s.find(key).map(|pos| pos + key.len())
}

fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;
    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
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
        if escape {
            escape = false;
            continue;
        }
        if c == '\\' && in_string {
            escape = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

fn parse_json_string(s: &str) -> Option<(String, usize)> {
    let s = s.trim_start();
    if !s.starts_with('"') {
        return None;
    }
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
                                if let Some((_, h)) = chars.next() {
                                    hex.push(h);
                                }
                            }
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(code) {
                                    result.push(ch);
                                }
                            }
                        }
                        _ => {
                            result.push('\\');
                            result.push(escaped);
                        }
                    }
                }
            }
            '"' => return Some((result, i + 2)),
            _ => result.push(c),
        }
    }
    None
}

fn parse_vocab_entries(s: &str, vocab: &mut HashMap<String, usize>) {
    let mut pos = 0;
    let bytes = s.as_bytes();
    while pos < s.len() {
        while pos < s.len() && bytes[pos] != b'"' {
            pos += 1;
        }
        if pos >= s.len() {
            break;
        }
        if let Some((key, advance)) = parse_json_string(&s[pos..]) {
            pos += advance;
            while pos < s.len() && bytes[pos] != b':' {
                pos += 1;
            }
            pos += 1;
            while pos < s.len() && bytes[pos] == b' ' {
                pos += 1;
            }
            let num_start = pos;
            while pos < s.len() && (bytes[pos].is_ascii_digit() || bytes[pos] == b'-') {
                pos += 1;
            }
            if let Ok(id) = s[num_start..pos].parse::<usize>() {
                vocab.insert(key, id);
            }
        } else {
            pos += 1;
        }
    }
}

fn parse_merges(s: &str, merges: &mut Vec<(String, String)>) {
    let mut pos = 0;
    while pos < s.len() {
        if let Some((merge_str, advance)) = parse_json_string(&s[pos..]) {
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

fn parse_added_tokens(s: &str, tokens: &mut HashMap<String, usize>) {
    let mut pos = 0;
    let bytes = s.as_bytes();
    while pos < s.len() {
        if bytes[pos] == b'{' {
            if let Some(end) = find_matching_brace(&s[pos..]) {
                let obj = &s[pos + 1..pos + end];
                let mut content = String::new();
                let mut id: Option<usize> = None;

                if let Some(ck) = find_key(obj, "\"content\"") {
                    let rest = &obj[ck..];
                    if let Some(colon) = rest.find(':') {
                        if let Some((val, _)) = parse_json_string(rest[colon + 1..].trim_start()) {
                            content = val;
                        }
                    }
                }
                if let Some(ik) = find_key(obj, "\"id\"") {
                    let rest = &obj[ik..];
                    if let Some(colon) = rest.find(':') {
                        let after = rest[colon + 1..].trim_start();
                        let num_end =
                            after.find(|c: char| !c.is_ascii_digit()).unwrap_or(after.len());
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
