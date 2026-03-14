//! HuggingFace config.json parser for model configuration.
//!
//! Parses the standard HF `config.json` format into a generic `ModelConfig`
//! that examples can use to initialize their model architectures.

/// Model configuration parsed from a HuggingFace config.json.
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub tie_word_embeddings: bool,
}

impl ModelConfig {
    /// Parse a `ModelConfig` from a HuggingFace config.json string.
    pub fn from_json(json: &str) -> Self {
        let model_type = json_str_value(json, "model_type").unwrap_or_default();
        let hidden_size = json_u64_value(json, "hidden_size").unwrap_or(4096) as usize;
        let num_hidden_layers = json_u64_value(json, "num_hidden_layers").unwrap_or(32) as usize;
        let num_attention_heads = json_u64_value(json, "num_attention_heads").unwrap_or(32) as usize;
        let num_key_value_heads = json_u64_value(json, "num_key_value_heads")
            .unwrap_or(num_attention_heads as u64) as usize;
        let intermediate_size = json_u64_value(json, "intermediate_size").unwrap_or(11008) as usize;
        let vocab_size = json_u64_value(json, "vocab_size").unwrap_or(32000) as usize;
        let rope_theta = json_f64_value(json, "rope_theta").unwrap_or(500000.0);
        let rms_norm_eps = json_f64_value(json, "rms_norm_eps").unwrap_or(1e-5);
        let max_position_embeddings = json_u64_value(json, "max_position_embeddings")
            .unwrap_or(8192) as usize;
        let head_dim = json_u64_value(json, "head_dim")
            .map(|v| v as usize)
            .unwrap_or(hidden_size / num_attention_heads);
        let tie_word_embeddings = json_bool_value(json, "tie_word_embeddings").unwrap_or(false);

        ModelConfig {
            model_type,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            vocab_size,
            rope_theta,
            rms_norm_eps,
            max_position_embeddings,
            head_dim,
            tie_word_embeddings,
        }
    }
}

/// Extract a string value for a given key from flat JSON.
/// Looks for `"key": "value"` patterns.
pub fn json_str_value(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.trim_start();
    if !after_colon.starts_with(':') {
        return None;
    }
    let after_colon = after_colon[1..].trim_start();
    if !after_colon.starts_with('"') {
        return None;
    }
    // Parse quoted string
    let mut result = String::new();
    let mut chars = after_colon[1..].chars();
    loop {
        match chars.next()? {
            '\\' => {
                let escaped = chars.next()?;
                match escaped {
                    '"' => result.push('"'),
                    '\\' => result.push('\\'),
                    'n' => result.push('\n'),
                    'r' => result.push('\r'),
                    't' => result.push('\t'),
                    _ => {
                        result.push('\\');
                        result.push(escaped);
                    }
                }
            }
            '"' => return Some(result),
            c => result.push(c),
        }
    }
}

/// Extract an unsigned integer value for a given key from flat JSON.
pub fn json_u64_value(json: &str, key: &str) -> Option<u64> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.trim_start();
    if !after_colon.starts_with(':') {
        return None;
    }
    let after_colon = after_colon[1..].trim_start();
    // Parse number
    let end = after_colon
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(after_colon.len());
    if end == 0 {
        return None;
    }
    after_colon[..end].parse().ok()
}

/// Extract a floating-point value for a given key from flat JSON.
/// Handles scientific notation (e.g., 1e-5, 1.0e-6).
pub fn json_f64_value(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.trim_start();
    if !after_colon.starts_with(':') {
        return None;
    }
    let after_colon = after_colon[1..].trim_start();
    // Find end of number (digits, dot, e, E, +, -)
    let end = after_colon
        .find(|c: char| !c.is_ascii_digit() && c != '.' && c != 'e' && c != 'E' && c != '+' && c != '-')
        .unwrap_or(after_colon.len());
    if end == 0 {
        return None;
    }
    after_colon[..end].parse().ok()
}

/// Extract a boolean value for a given key from flat JSON.
pub fn json_bool_value(json: &str, key: &str) -> Option<bool> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let after_key = &json[idx + pattern.len()..];
    let after_colon = after_key.trim_start();
    if !after_colon.starts_with(':') {
        return None;
    }
    let after_colon = after_colon[1..].trim_start();
    if after_colon.starts_with("true") {
        Some(true)
    } else if after_colon.starts_with("false") {
        Some(false)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LLAMA_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "rope_theta": 500000.0,
        "rms_norm_eps": 1e-05,
        "max_position_embeddings": 131072,
        "head_dim": 64,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16"
    }"#;

    const MISTRAL_CONFIG: &str = r#"{
        "model_type": "mistral",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 32768,
        "tie_word_embeddings": false
    }"#;

    #[test]
    fn test_llama_config() {
        let config = ModelConfig::from_json(LLAMA_CONFIG);
        assert_eq!(config.model_type, "llama");
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 16);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.intermediate_size, 8192);
        assert_eq!(config.vocab_size, 128256);
        assert!((config.rope_theta - 500000.0).abs() < 1.0);
        assert!((config.rms_norm_eps - 1e-5).abs() < 1e-8);
        assert_eq!(config.max_position_embeddings, 131072);
        assert_eq!(config.head_dim, 64);
        assert!(config.tie_word_embeddings);
    }

    #[test]
    fn test_mistral_config() {
        let config = ModelConfig::from_json(MISTRAL_CONFIG);
        assert_eq!(config.model_type, "mistral");
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_hidden_layers, 32);
        assert_eq!(config.num_attention_heads, 32);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.intermediate_size, 14336);
        assert_eq!(config.vocab_size, 32000);
        assert!((config.rope_theta - 10000.0).abs() < 1.0);
        assert!(!config.tie_word_embeddings);
        // head_dim defaults to hidden_size / num_attention_heads = 128
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn test_json_helpers() {
        let json = r#"{"name": "test", "count": 42, "rate": 3.14, "enabled": true}"#;
        assert_eq!(json_str_value(json, "name"), Some("test".to_string()));
        assert_eq!(json_u64_value(json, "count"), Some(42));
        assert!((json_f64_value(json, "rate").unwrap() - 3.14).abs() < 1e-6);
        assert_eq!(json_bool_value(json, "enabled"), Some(true));
    }

    #[test]
    fn test_missing_keys() {
        let json = r#"{"hidden_size": 1024}"#;
        assert_eq!(json_str_value(json, "missing"), None);
        assert_eq!(json_u64_value(json, "missing"), None);
        assert_eq!(json_f64_value(json, "missing"), None);
        assert_eq!(json_bool_value(json, "missing"), None);
    }

    #[test]
    fn test_scientific_notation() {
        let json = r#"{"eps": 1e-05, "theta": 5.0e+5}"#;
        assert!((json_f64_value(json, "eps").unwrap() - 1e-5).abs() < 1e-10);
        assert!((json_f64_value(json, "theta").unwrap() - 500000.0).abs() < 1.0);
    }
}
