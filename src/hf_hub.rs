//! HuggingFace Hub integration for downloading and caching model files.
//!
//! Feature-gated behind `--features hf`. Downloads safetensors weights,
//! config.json, and tokenizer.json from HuggingFace Hub with local caching.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// A HuggingFace repository reference.
#[derive(Debug, Clone)]
pub struct HfRepo {
    pub org: String,
    pub repo: String,
    pub revision: String,
}

impl HfRepo {
    /// Parse a repo string like "meta-llama/Llama-3.2-1B" or "mistralai/Mistral-7B-v0.1".
    /// Optional revision after `@`, e.g., "meta-llama/Llama-3.2-1B@refs/pr/42".
    pub fn new(spec: &str) -> io::Result<Self> {
        let (repo_part, revision) = if let Some(at) = spec.find('@') {
            (&spec[..at], spec[at + 1..].to_string())
        } else {
            (spec, "main".to_string())
        };

        let parts: Vec<&str> = repo_part.split('/').collect();
        if parts.len() != 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("invalid repo spec '{}' — expected 'org/repo'", spec),
            ));
        }

        Ok(HfRepo {
            org: parts[0].to_string(),
            repo: parts[1].to_string(),
            revision,
        })
    }

    /// Full repo ID like "meta-llama/Llama-3.2-1B".
    pub fn repo_id(&self) -> String {
        format!("{}/{}", self.org, self.repo)
    }
}

/// Get the cache directory for a repo: `~/.peregrine/models/{org}/{repo}/`
fn cache_dir(repo: &HfRepo) -> io::Result<PathBuf> {
    let home = std::env::var("HOME")
        .map_err(|_| io::Error::new(io::ErrorKind::NotFound, "HOME not set"))?;
    let dir = PathBuf::from(home)
        .join(".peregrine")
        .join("models")
        .join(&repo.org)
        .join(&repo.repo);
    Ok(dir)
}

/// Get the HuggingFace token from env var or cached token file.
fn hf_token() -> Option<String> {
    if let Ok(token) = std::env::var("HF_TOKEN") {
        if !token.is_empty() {
            return Some(token);
        }
    }
    // Try ~/.cache/huggingface/token
    if let Ok(home) = std::env::var("HOME") {
        let token_path = PathBuf::from(home)
            .join(".cache")
            .join("huggingface")
            .join("token");
        if let Ok(token) = fs::read_to_string(&token_path) {
            let token = token.trim().to_string();
            if !token.is_empty() {
                return Some(token);
            }
        }
    }
    None
}

/// Download a single file from HuggingFace Hub if not already cached.
/// Returns the local path to the cached file.
pub fn ensure_file(repo: &HfRepo, filename: &str) -> io::Result<PathBuf> {
    let dir = cache_dir(repo)?;
    let local_path = dir.join(filename);

    if local_path.exists() {
        return Ok(local_path);
    }

    // Create parent directories
    if let Some(parent) = local_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let url = format!(
        "https://huggingface.co/{}/{}/resolve/{}/{}",
        repo.org, repo.repo, repo.revision, filename
    );

    eprintln!("Downloading {} from {}", filename, repo.repo_id());

    let mut request = ureq::get(&url);
    if let Some(token) = hf_token() {
        request = request.set("Authorization", &format!("Bearer {}", token));
    }

    let response = request
        .call()
        .map_err(|e| io::Error::new(io::ErrorKind::Other, format!("download failed: {}", e)))?;

    // Stream to a temp file, then rename
    let tmp_path = local_path.with_extension("tmp");
    let mut tmp_file = fs::File::create(&tmp_path)?;

    let content_length = response
        .header("content-length")
        .and_then(|s| s.parse::<u64>().ok());

    let mut reader = response.into_reader();
    let mut buf = vec![0u8; 8 * 1024 * 1024]; // 8MB buffer
    let mut downloaded: u64 = 0;
    let mut last_report: u64 = 0;

    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        io::Write::write_all(&mut tmp_file, &buf[..n])?;
        downloaded += n as u64;

        // Progress report every 100MB
        if downloaded - last_report >= 100_000_000 {
            if let Some(total) = content_length {
                eprintln!(
                    "  {}: {:.0}MB / {:.0}MB ({:.0}%)",
                    filename,
                    downloaded as f64 / 1e6,
                    total as f64 / 1e6,
                    downloaded as f64 / total as f64 * 100.0
                );
            } else {
                eprintln!("  {}: {:.0}MB", filename, downloaded as f64 / 1e6);
            }
            last_report = downloaded;
        }
    }

    drop(tmp_file);
    fs::rename(&tmp_path, &local_path)?;
    eprintln!("  Saved {} ({:.1}MB)", filename, downloaded as f64 / 1e6);

    Ok(local_path)
}

/// Parse a safetensors shard index to get the list of shard filenames.
/// The index JSON has a "weight_map" mapping tensor names to shard filenames.
fn parse_shard_filenames(index_json: &str) -> Vec<String> {
    let mut filenames = Vec::new();

    // Find "weight_map" key and extract values
    if let Some(wm_idx) = index_json.find("\"weight_map\"") {
        let after = &index_json[wm_idx + "\"weight_map\"".len()..];
        // Find opening brace
        if let Some(brace) = after.find('{') {
            let map_str = &after[brace..];
            // Extract all string values (filenames)
            let mut pos = 1;
            let bytes = map_str.as_bytes();
            let mut seen = std::collections::HashSet::new();

            while pos < bytes.len() {
                if bytes[pos] == b'}' {
                    break;
                }
                if bytes[pos] == b'"' {
                    // Skip key
                    pos += 1;
                    while pos < bytes.len() && bytes[pos] != b'"' {
                        if bytes[pos] == b'\\' {
                            pos += 1;
                        }
                        pos += 1;
                    }
                    pos += 1; // closing quote

                    // Skip to colon
                    while pos < bytes.len() && bytes[pos] != b':' {
                        pos += 1;
                    }
                    pos += 1;

                    // Skip whitespace
                    while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\n' || bytes[pos] == b'\r' || bytes[pos] == b'\t') {
                        pos += 1;
                    }

                    // Parse value string (the filename)
                    if pos < bytes.len() && bytes[pos] == b'"' {
                        pos += 1;
                        let start = pos;
                        while pos < bytes.len() && bytes[pos] != b'"' {
                            if bytes[pos] == b'\\' {
                                pos += 1;
                            }
                            pos += 1;
                        }
                        let filename = std::str::from_utf8(&bytes[start..pos])
                            .unwrap_or("")
                            .to_string();
                        if !filename.is_empty() && seen.insert(filename.clone()) {
                            filenames.push(filename);
                        }
                        pos += 1; // closing quote
                    }
                } else {
                    pos += 1;
                }
            }
        }
    }

    filenames
}

/// Download all model files (safetensors, config.json, tokenizer.json) and return
/// the local cache directory path.
pub fn ensure_model(repo: &HfRepo) -> io::Result<PathBuf> {
    let dir = cache_dir(repo)?;

    // Always download config.json first
    ensure_file(repo, "config.json")?;

    // Try tokenizer.json (not all models have it, but most HF models do)
    match ensure_file(repo, "tokenizer.json") {
        Ok(_) => {}
        Err(e) => eprintln!("Warning: tokenizer.json not found: {}", e),
    }

    // Check if single-shard or multi-shard
    let index_path = dir.join("model.safetensors.index.json");
    let single_path = dir.join("model.safetensors");

    if !single_path.exists() && !index_path.exists() {
        // Try downloading index first
        match ensure_file(repo, "model.safetensors.index.json") {
            Ok(_) => {} // multi-shard
            Err(_) => {
                // Try single shard
                ensure_file(repo, "model.safetensors")?;
                return Ok(dir);
            }
        }
    }

    // If we have an index, download all shards
    if index_path.exists() {
        let index_str = fs::read_to_string(&index_path)?;
        let shard_files = parse_shard_filenames(&index_str);
        eprintln!("Model has {} shards", shard_files.len());
        for shard in &shard_files {
            ensure_file(repo, shard)?;
        }
    }

    Ok(dir)
}

/// List safetensors shard files in a directory.
/// Returns single "model.safetensors" or shards from the index.
pub fn list_safetensors_files(dir: &Path) -> io::Result<Vec<PathBuf>> {
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index_str = fs::read_to_string(&index_path)?;
        let shard_files = parse_shard_filenames(&index_str);
        Ok(shard_files.iter().map(|f| dir.join(f)).collect())
    } else {
        let single = dir.join("model.safetensors");
        if single.exists() {
            Ok(vec![single])
        } else {
            Err(io::Error::new(
                io::ErrorKind::NotFound,
                "no safetensors files found",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hf_repo_parse() {
        let repo = HfRepo::new("meta-llama/Llama-3.2-1B").unwrap();
        assert_eq!(repo.org, "meta-llama");
        assert_eq!(repo.repo, "Llama-3.2-1B");
        assert_eq!(repo.revision, "main");
    }

    #[test]
    fn test_hf_repo_with_revision() {
        let repo = HfRepo::new("meta-llama/Llama-3.2-1B@refs/pr/42").unwrap();
        assert_eq!(repo.org, "meta-llama");
        assert_eq!(repo.repo, "Llama-3.2-1B");
        assert_eq!(repo.revision, "refs/pr/42");
    }

    #[test]
    fn test_hf_repo_invalid() {
        assert!(HfRepo::new("no-slash").is_err());
        assert!(HfRepo::new("a/b/c").is_err());
    }

    #[test]
    fn test_parse_shard_filenames() {
        let index = r#"{
            "metadata": {"total_size": 12345},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors"
            }
        }"#;
        let files = parse_shard_filenames(index);
        assert_eq!(files.len(), 2);
        assert!(files.contains(&"model-00001-of-00002.safetensors".to_string()));
        assert!(files.contains(&"model-00002-of-00002.safetensors".to_string()));
    }
}
