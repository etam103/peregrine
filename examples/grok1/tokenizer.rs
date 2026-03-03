use std::collections::HashMap;
use std::fs;

/// Piece type from the SentencePiece protobuf.
#[derive(Debug, Clone, Copy, PartialEq)]
enum PieceType {
    Normal,  // 1 - standard BPE piece
    Unknown, // 2 - unknown token
    Control, // 3 - control tokens (BOS, EOS, PAD)
    UserDef, // 4 - user-defined (newlines, digits, etc.)
    Unused,  // 5
    Byte,    // 6 - byte fallback <0xHH>
}

/// A single vocabulary piece.
#[derive(Debug, Clone)]
struct Piece {
    text: String,
    score: f32,
    ptype: PieceType,
}

/// SentencePiece BPE tokenizer (read-only, for inference).
pub struct Tokenizer {
    pieces: Vec<Piece>,
    piece_to_id: HashMap<String, usize>,
    bos_id: usize,
    eos_id: usize,
}

/// The SentencePiece "▁" space marker (U+2581).
const SPACE_MARKER: &str = "\u{2581}";

impl Tokenizer {
    /// Load a SentencePiece `.model` file (protobuf wire format).
    pub fn load(path: &str) -> Self {
        let data = fs::read(path).unwrap_or_else(|e| {
            eprintln!("Failed to read tokenizer model {}: {}", path, e);
            std::process::exit(1);
        });

        let mut pieces = Vec::new();
        let mut pos = 0;

        while pos < data.len() {
            // Read protobuf tag
            let (tag, new_pos) = read_varint(&data, pos);
            pos = new_pos;
            let field = tag >> 3;
            let wire = tag & 7;

            if field == 1 && wire == 2 {
                // Repeated SentencePiece submessage
                let (len, new_pos) = read_varint(&data, pos);
                pos = new_pos;
                let len = len as usize;
                let sub = &data[pos..pos + len];
                pos += len;

                let (text, score, ptype) = parse_piece(sub);
                pieces.push(Piece { text, score, ptype });
            } else if wire == 2 {
                // Skip other length-delimited fields (trainer_spec, etc.)
                let (len, new_pos) = read_varint(&data, pos);
                pos = new_pos;
                pos += len as usize;
            } else if wire == 0 {
                // Skip varint field
                let (_, new_pos) = read_varint(&data, pos);
                pos = new_pos;
            } else if wire == 5 {
                pos += 4; // 32-bit
            } else if wire == 1 {
                pos += 8; // 64-bit
            } else {
                break;
            }
        }

        // Build lookup
        let mut piece_to_id = HashMap::with_capacity(pieces.len());
        for (i, p) in pieces.iter().enumerate() {
            piece_to_id.insert(p.text.clone(), i);
        }

        // Find BOS/EOS
        let bos_id = pieces
            .iter()
            .position(|p| p.text == "[BOS]" && p.ptype == PieceType::Control)
            .unwrap_or(1);
        let eos_id = pieces
            .iter()
            .position(|p| p.text == "[EOS]" && p.ptype == PieceType::Control)
            .unwrap_or(2);

        eprintln!(
            "Loaded tokenizer: {} pieces, BOS={}, EOS={}",
            pieces.len(),
            bos_id,
            eos_id
        );

        Tokenizer {
            pieces,
            piece_to_id,
            bos_id,
            eos_id,
        }
    }

    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    pub fn eos_id(&self) -> usize {
        self.eos_id
    }

    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Encode text to token IDs using BPE.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if text.is_empty() {
            return vec![];
        }

        // SentencePiece convention: prepend space marker, replace spaces with marker
        let normalized = format!("{}{}", SPACE_MARKER, text.replace(' ', SPACE_MARKER));

        // Start with character-level segmentation
        let chars: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        let mut symbols: Vec<String> = chars;

        // Greedy BPE merging: repeatedly merge the pair with the best (highest) score
        loop {
            if symbols.len() < 2 {
                break;
            }

            // Find the best merge
            let mut best_score = f32::NEG_INFINITY;
            let mut best_idx = usize::MAX;

            for i in 0..symbols.len() - 1 {
                let merged = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some(&id) = self.piece_to_id.get(&merged) {
                    let score = self.pieces[id].score;
                    if score > best_score {
                        best_score = score;
                        best_idx = i;
                    }
                }
            }

            if best_idx == usize::MAX {
                break; // No more merges possible
            }

            // Apply the merge
            let merged = format!("{}{}", symbols[best_idx], symbols[best_idx + 1]);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        // Convert symbols to IDs
        let mut ids = Vec::with_capacity(symbols.len());
        for sym in &symbols {
            if let Some(&id) = self.piece_to_id.get(sym) {
                ids.push(id);
            } else {
                // Fall back to byte encoding
                for byte in sym.as_bytes() {
                    let byte_piece = format!("<0x{:02X}>", byte);
                    if let Some(&id) = self.piece_to_id.get(&byte_piece) {
                        ids.push(id);
                    } else {
                        ids.push(3); // UNK
                    }
                }
            }
        }

        ids
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut result = String::new();

        for &id in ids {
            if id >= self.pieces.len() {
                continue;
            }
            let piece = &self.pieces[id];
            match piece.ptype {
                PieceType::Control => continue, // Skip BOS, EOS, PAD
                PieceType::Byte => {
                    // Parse <0xHH> format
                    if let Some(byte_val) = parse_byte_piece(&piece.text) {
                        result.push(byte_val as char);
                    }
                }
                _ => {
                    result.push_str(&piece.text);
                }
            }
        }

        // Replace space markers with actual spaces
        result = result.replace(SPACE_MARKER, " ");

        // Strip leading space (artifact of the ▁ prepend)
        if result.starts_with(' ') {
            result = result[1..].to_string();
        }

        result
    }
}

/// Parse a `<0xHH>` byte piece, returning the byte value.
fn parse_byte_piece(s: &str) -> Option<u8> {
    if s.starts_with("<0x") && s.ends_with('>') && s.len() == 6 {
        u8::from_str_radix(&s[3..5], 16).ok()
    } else {
        None
    }
}

/// Read a protobuf varint from `data` at `pos`. Returns (value, new_pos).
fn read_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if pos >= data.len() {
            break;
        }
        let b = data[pos];
        pos += 1;
        result |= ((b & 0x7F) as u64) << shift;
        shift += 7;
        if b & 0x80 == 0 {
            break;
        }
    }
    (result, pos)
}

/// Parse a SentencePiece submessage, returning (text, score, type).
fn parse_piece(data: &[u8]) -> (String, f32, PieceType) {
    let mut text = String::new();
    let mut score = 0.0f32;
    let mut ptype = PieceType::Normal;
    let mut pos = 0;

    while pos < data.len() {
        let (tag, new_pos) = read_varint(data, pos);
        pos = new_pos;
        let field = tag >> 3;
        let wire = tag & 7;

        match (field, wire) {
            (1, 2) => {
                // piece text (string)
                let (len, new_pos) = read_varint(data, pos);
                pos = new_pos;
                let len = len as usize;
                text = String::from_utf8_lossy(&data[pos..pos + len]).into_owned();
                pos += len;
            }
            (2, 5) => {
                // score (float32)
                let bytes: [u8; 4] = data[pos..pos + 4].try_into().unwrap();
                score = f32::from_le_bytes(bytes);
                pos += 4;
            }
            (3, 0) => {
                // type (enum as varint)
                let (val, new_pos) = read_varint(data, pos);
                pos = new_pos;
                ptype = match val {
                    1 => PieceType::Normal,
                    2 => PieceType::Unknown,
                    3 => PieceType::Control,
                    4 => PieceType::UserDef,
                    5 => PieceType::Unused,
                    6 => PieceType::Byte,
                    _ => PieceType::Normal,
                };
            }
            (_, 0) => {
                let (_, new_pos) = read_varint(data, pos);
                pos = new_pos;
            }
            (_, 2) => {
                let (len, new_pos) = read_varint(data, pos);
                pos = new_pos;
                pos += len as usize;
            }
            (_, 5) => pos += 4,
            (_, 1) => pos += 8,
            _ => break,
        }
    }

    (text, score, ptype)
}
