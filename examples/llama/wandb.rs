//! Minimal Weights & Biases client for logging decode metrics.
//! Set the WANDB_API_KEY environment variable to enable logging.

const API: &str = "https://api.wandb.ai";

pub struct WandbRun {
    inner: Option<Inner>,
}

struct Inner {
    auth: String,
    entity: String,
    project: String,
    run_id: String,
    history_offset: usize,
    pending: Vec<String>,
}

fn base64(input: &[u8]) -> String {
    const C: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in input.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(C[(n >> 18 & 0x3F) as usize] as char);
        out.push(C[(n >> 12 & 0x3F) as usize] as char);
        out.push(if chunk.len() > 1 { C[(n >> 6 & 0x3F) as usize] as char } else { '=' });
        out.push(if chunk.len() > 2 { C[(n & 0x3F) as usize] as char } else { '=' });
    }
    out
}

fn random_run_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let seed = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    let chars = b"abcdefghijklmnopqrstuvwxyz0123456789";
    let mut id = String::with_capacity(8);
    let mut n = seed;
    for _ in 0..8 {
        id.push(chars[(n % 36) as usize] as char);
        n = n.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    }
    id
}

impl WandbRun {
    /// Create a new wandb run. Returns a no-op logger if WANDB_API_KEY is unset.
    pub fn init(project: &str) -> Self {
        match Inner::connect(project) {
            Some(inner) => {
                eprintln!("wandb: https://wandb.ai/{}/{}/runs/{}",
                    inner.entity, inner.project, inner.run_id);
                Self { inner: Some(inner) }
            }
            None => {
                eprintln!("wandb: WANDB_API_KEY not set or connection failed — logging disabled");
                Self { inner: None }
            }
        }
    }

    pub fn log_metrics(&mut self, step: usize, metrics: &[(&str, f32)]) {
        if let Some(inner) = &mut self.inner {
            let mut entry = serde_json::json!({ "_step": step });
            for &(k, v) in metrics {
                entry[k] = serde_json::json!(v);
            }
            inner.append(entry);
        }
    }

    pub fn finish(&mut self) {
        if let Some(inner) = &mut self.inner {
            inner.flush();
            let url = format!("{}/files/{}/{}/{}/file_stream",
                API, inner.entity, inner.project, inner.run_id);
            let _ = ureq::post(&url)
                .set("Authorization", &inner.auth)
                .send_json(serde_json::json!({ "complete": true, "exitcode": 0 }));
            eprintln!("wandb: Run complete → https://wandb.ai/{}/{}/runs/{}",
                inner.entity, inner.project, inner.run_id);
        }
    }
}

impl Inner {
    fn connect(project: &str) -> Option<Self> {
        let api_key = std::env::var("WANDB_API_KEY").ok()?;
        let auth = format!("Basic {}", base64(format!("api:{}", api_key).as_bytes()));

        let resp = ureq::post(&format!("{}/graphql", API))
            .set("Authorization", &auth)
            .send_json(serde_json::json!({"query": "{ viewer { entity } }"}))
            .ok()?;
        let body: serde_json::Value = resp.into_json().ok()?;
        let entity = body["data"]["viewer"]["entity"].as_str()?.to_string();

        let run_id = random_run_id();
        let resp = ureq::post(&format!("{}/graphql", API))
            .set("Authorization", &auth)
            .send_json(serde_json::json!({
                "query": "mutation($e:String!,$m:String!,$n:String!){upsertBucket(input:{entityName:$e,modelName:$m,name:$n}){bucket{id name project{name entity{name}}}}}",
                "variables": { "e": &entity, "m": project, "n": &run_id }
            }))
            .ok()?;
        let body: serde_json::Value = resp.into_json().ok()?;
        body["data"]["upsertBucket"]["bucket"]["name"].as_str()?;

        Some(Self { auth, entity, project: project.to_string(), run_id, history_offset: 0, pending: Vec::new() })
    }

    fn append(&mut self, entry: serde_json::Value) {
        self.pending.push(entry.to_string());
        if self.pending.len() >= 50 {
            self.flush();
        }
    }

    fn flush(&mut self) {
        if self.pending.is_empty() { return; }
        let url = format!("{}/files/{}/{}/{}/file_stream",
            API, self.entity, self.project, self.run_id);
        if ureq::post(&url)
            .set("Authorization", &self.auth)
            .send_json(serde_json::json!({
                "files": {
                    "wandb-history.jsonl": {
                        "offset": self.history_offset,
                        "content": &self.pending
                    }
                }
            }))
            .is_ok()
        {
            self.history_offset += self.pending.len();
        }
        self.pending.clear();
    }
}
