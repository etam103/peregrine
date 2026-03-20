//! Recursive Language Models (RLMs) based on Zhang & Khattab (MIT).
//!
//! Provides an orchestrator that lets a generative language model recursively
//! decompose tasks using structured actions: sub-calls, peeking into context
//! variables, grep, partition-map, and summarization.  The parser is
//! dependency-free (no regex crate) and operates via simple string scanning.

use std::collections::HashMap;

// ============================================================================
// GenerateConfig + GenerativeLM trait
// ============================================================================

/// Configuration for a single generation call.
pub struct GenerateConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub stop_sequences: Vec<String>,
}

impl Default for GenerateConfig {
    fn default() -> Self {
        GenerateConfig {
            temperature: 0.7,
            top_p: 0.9,
            stop_sequences: vec![],
        }
    }
}

/// Trait that any generative language model must implement to be used with the
/// RLM orchestrator.
pub trait GenerativeLM {
    /// Generate text from a prompt, returning the generated text.
    fn generate(&self, prompt: &str, max_tokens: usize, config: &GenerateConfig) -> String;

    /// Generate text and also return the number of tokens consumed.
    fn generate_counted(
        &self,
        prompt: &str,
        max_tokens: usize,
        config: &GenerateConfig,
    ) -> (String, usize);

    /// Return the model's context window size in tokens.
    fn context_window(&self) -> usize;

    /// Count the number of tokens in a text string.
    fn count_tokens(&self, text: &str) -> usize;
}

// ============================================================================
// ReplContext — variable store
// ============================================================================

/// A string-keyed variable store used during recursive execution.
pub struct ReplContext {
    vars: HashMap<String, String>,
}

impl ReplContext {
    pub fn new() -> Self {
        ReplContext {
            vars: HashMap::new(),
        }
    }

    pub fn set(&mut self, key: &str, value: String) {
        self.vars.insert(key.to_string(), value);
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.vars.get(key).map(|s| s.as_str())
    }

    pub fn keys(&self) -> Vec<&str> {
        let mut ks: Vec<&str> = self.vars.keys().map(|s| s.as_str()).collect();
        ks.sort();
        ks
    }

    /// Format all variables for inclusion in a prompt.
    /// Format: `[variable_name]\n{value}\n\n` for each var (sorted by key).
    pub fn as_prompt_context(&self) -> String {
        let mut out = String::new();
        let mut keys: Vec<&String> = self.vars.keys().collect();
        keys.sort();
        for k in keys {
            let v = &self.vars[k];
            out.push('[');
            out.push_str(k);
            out.push_str("]\n");
            out.push_str(v);
            out.push_str("\n\n");
        }
        out
    }
}

// ============================================================================
// RlmAction — parsed actions from LM output
// ============================================================================

/// An action parsed from the model's generated text.
#[derive(Debug, PartialEq)]
pub enum RlmAction {
    /// Spawn a recursive sub-call with a new prompt, store result in variable.
    RecursiveCall { prompt: String, assign_to: String },
    /// View a slice of a context variable (by character offset).
    Peek {
        variable: String,
        start: usize,
        end: usize,
    },
    /// Search a context variable for a pattern (simple substring match).
    Grep { variable: String, pattern: String },
    /// Split a variable into chunks, run a sub-prompt on each, collect results.
    PartitionMap {
        variable: String,
        chunk_size: usize,
        sub_prompt: String,
        assign_to: String,
    },
    /// Recursively summarize a variable.
    Summarize {
        variable: String,
        assign_to: String,
    },
    /// Return a final answer string.
    Final { answer: String },
    /// Return the contents of a variable as the final answer.
    FinalVar { variable: String },
}

// ============================================================================
// RlmConfig
// ============================================================================

/// Configuration for the RLM orchestrator.
pub struct RlmConfig {
    pub max_depth: usize,
    pub max_total_tokens: usize,
    pub max_tokens_per_call: usize,
    pub partition_size: usize,
    pub temperature: f32,
    pub top_p: f32,
}

impl Default for RlmConfig {
    fn default() -> Self {
        RlmConfig {
            max_depth: 4,
            max_total_tokens: 100_000,
            max_tokens_per_call: 4096,
            partition_size: 2048,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

impl RlmConfig {
    pub fn max_depth(mut self, v: usize) -> Self {
        self.max_depth = v;
        self
    }
    pub fn max_total_tokens(mut self, v: usize) -> Self {
        self.max_total_tokens = v;
        self
    }
    pub fn max_tokens_per_call(mut self, v: usize) -> Self {
        self.max_tokens_per_call = v;
        self
    }
    pub fn partition_size(mut self, v: usize) -> Self {
        self.partition_size = v;
        self
    }
    pub fn temperature(mut self, v: f32) -> Self {
        self.temperature = v;
        self
    }
    pub fn top_p(mut self, v: f32) -> Self {
        self.top_p = v;
        self
    }
}

// ============================================================================
// RlmError
// ============================================================================

/// Errors that can occur during RLM orchestration.
#[derive(Debug)]
pub enum RlmError {
    MaxDepthExceeded { depth: usize, max: usize },
    TokenBudgetExhausted { used: usize, budget: usize },
    ParseError(String),
    ModelError(String),
}

impl std::fmt::Display for RlmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RlmError::MaxDepthExceeded { depth, max } => {
                write!(f, "max recursion depth exceeded: {} > {}", depth, max)
            }
            RlmError::TokenBudgetExhausted { used, budget } => {
                write!(f, "token budget exhausted: {} used of {} budget", used, budget)
            }
            RlmError::ParseError(msg) => write!(f, "parse error: {}", msg),
            RlmError::ModelError(msg) => write!(f, "model error: {}", msg),
        }
    }
}

impl std::error::Error for RlmError {}

// ============================================================================
// RlmStats
// ============================================================================

/// Accumulated statistics from an RLM execution run.
pub struct RlmStats {
    pub total_tokens: usize,
    pub call_count: usize,
    pub max_depth_reached: usize,
}

// ============================================================================
// ActionResult (internal)
// ============================================================================

enum ActionResult {
    Continue,
    Done(String),
}

// ============================================================================
// RlmOrchestrator — core engine
// ============================================================================

/// The RLM orchestrator: drives recursive language model execution.
pub struct RlmOrchestrator<'a> {
    model: &'a dyn GenerativeLM,
    config: RlmConfig,
    stats: RlmStats,
}

impl<'a> RlmOrchestrator<'a> {
    pub fn new(model: &'a dyn GenerativeLM, config: RlmConfig) -> Self {
        RlmOrchestrator {
            model,
            config,
            stats: RlmStats {
                total_tokens: 0,
                call_count: 0,
                max_depth_reached: 0,
            },
        }
    }

    /// Main entry point: process `input_context` with `task_prompt`.
    pub fn run(&mut self, input_context: &str, task_prompt: &str) -> Result<String, RlmError> {
        let mut ctx = ReplContext::new();
        ctx.set("input", input_context.to_string());
        self.execute_recursive(&mut ctx, task_prompt, 0)
    }

    /// Recursive dispatch: build prompt, call model, parse actions, execute.
    fn execute_recursive(
        &mut self,
        context: &mut ReplContext,
        prompt: &str,
        depth: usize,
    ) -> Result<String, RlmError> {
        if depth > self.config.max_depth {
            return Err(RlmError::MaxDepthExceeded {
                depth,
                max: self.config.max_depth,
            });
        }
        if self.stats.total_tokens > self.config.max_total_tokens {
            return Err(RlmError::TokenBudgetExhausted {
                used: self.stats.total_tokens,
                budget: self.config.max_total_tokens,
            });
        }

        let full_prompt = self.build_prompt(context, prompt);
        let gen_config = GenerateConfig {
            temperature: self.config.temperature,
            top_p: self.config.top_p,
            stop_sequences: vec!["[END]".to_string()],
        };

        let (output, tokens_used) = self.model.generate_counted(
            &full_prompt,
            self.config.max_tokens_per_call,
            &gen_config,
        );
        self.stats.total_tokens += tokens_used;
        self.stats.call_count += 1;
        if depth > self.stats.max_depth_reached {
            self.stats.max_depth_reached = depth;
        }

        let actions = Self::parse_output(&output)?;
        for action in actions {
            match self.execute_action(action, context, depth)? {
                ActionResult::Continue => {}
                ActionResult::Done(answer) => return Ok(answer),
            }
        }

        // If no Final action was found, return the raw output.
        Ok(output)
    }

    fn build_prompt(&self, context: &ReplContext, task: &str) -> String {
        format!(
            "You are a recursive language model. You can use the following actions:\n\
             - [CALL prompt=\"...\"] assign_to=\"...\" \u{2014} recursively process a sub-task\n\
             - [PEEK var=\"...\" start=N end=N] \u{2014} view a slice of a variable\n\
             - [GREP var=\"...\" pattern=\"...\"] \u{2014} search a variable\n\
             - [PARTITION_MAP var=\"...\" chunk_size=N prompt=\"...\" assign_to=\"...\"] \u{2014} chunk and map\n\
             - [SUMMARIZE var=\"...\" assign_to=\"...\"] \u{2014} summarize a variable\n\
             - [FINAL] answer text here [/FINAL] \u{2014} return final answer\n\
             - [FINAL_VAR var=\"...\"] \u{2014} return a variable as the answer\n\n\
             Current context:\n{}\n\
             Task: {}\n",
            context.as_prompt_context(),
            task,
        )
    }

    // ---- Parsing ----------------------------------------------------------

    /// Parse the model's output text into a sequence of actions.
    /// Uses simple string scanning (no regex dependency).
    pub fn parse_output(text: &str) -> Result<Vec<RlmAction>, RlmError> {
        let mut actions = Vec::new();
        let mut pos = 0;
        let bytes = text.as_bytes();

        while pos < bytes.len() {
            if bytes[pos] == b'[' {
                let remaining = &text[pos..];

                // Order matters: try [FINAL] before [FINAL_VAR] because
                // [FINAL_VAR starts with [FINAL but the space check keeps them
                // distinct.  However, [FINAL] is a strict prefix, so we check
                // [FINAL_VAR first to avoid a false match on the tag.
                if let Some((action, consumed)) = Self::try_parse_final_var(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
                if let Some((action, consumed)) = Self::try_parse_final(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
                if let Some((action, consumed)) = Self::try_parse_call(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
                if let Some((action, consumed)) = Self::try_parse_peek(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
                if let Some((action, consumed)) = Self::try_parse_grep(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
                if let Some((action, consumed)) = Self::try_parse_partition_map(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
                if let Some((action, consumed)) = Self::try_parse_summarize(remaining) {
                    actions.push(action);
                    pos += consumed;
                    continue;
                }
            }
            pos += 1;
        }

        Ok(actions)
    }

    fn try_parse_final(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[FINAL]") {
            return None;
        }
        let after = &text[7..];
        if let Some(end) = after.find("[/FINAL]") {
            let answer = after[..end].trim().to_string();
            Some((RlmAction::Final { answer }, 7 + end + 8))
        } else {
            let answer = after.trim().to_string();
            Some((RlmAction::Final { answer }, text.len()))
        }
    }

    fn try_parse_final_var(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[FINAL_VAR ") {
            return None;
        }
        let end = text.find(']')?;
        let attrs = &text[11..end];
        let variable = Self::extract_attr(attrs, "var")?;
        Some((RlmAction::FinalVar { variable }, end + 1))
    }

    fn try_parse_call(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[CALL ") {
            return None;
        }
        let end = text.find(']')?;
        let attrs = &text[6..end];
        let prompt = Self::extract_attr(attrs, "prompt")?;
        let assign_to = Self::extract_attr(attrs, "assign_to")?;
        Some((RlmAction::RecursiveCall { prompt, assign_to }, end + 1))
    }

    fn try_parse_peek(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[PEEK ") {
            return None;
        }
        let end = text.find(']')?;
        let attrs = &text[6..end];
        let variable = Self::extract_attr(attrs, "var")?;
        let start: usize = Self::extract_attr(attrs, "start")?.parse().ok()?;
        let end_val: usize = Self::extract_attr(attrs, "end")?.parse().ok()?;
        Some((
            RlmAction::Peek {
                variable,
                start,
                end: end_val,
            },
            end + 1,
        ))
    }

    fn try_parse_grep(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[GREP ") {
            return None;
        }
        let end = text.find(']')?;
        let attrs = &text[6..end];
        let variable = Self::extract_attr(attrs, "var")?;
        let pattern = Self::extract_attr(attrs, "pattern")?;
        Some((RlmAction::Grep { variable, pattern }, end + 1))
    }

    fn try_parse_partition_map(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[PARTITION_MAP ") {
            return None;
        }
        let end = text.find(']')?;
        let attrs = &text[15..end];
        let variable = Self::extract_attr(attrs, "var")?;
        let chunk_size: usize = Self::extract_attr(attrs, "chunk_size")?.parse().ok()?;
        let sub_prompt = Self::extract_attr(attrs, "sub_prompt")
            .or_else(|| Self::extract_attr(attrs, "prompt"))?;
        let assign_to = Self::extract_attr(attrs, "assign_to")?;
        Some((
            RlmAction::PartitionMap {
                variable,
                chunk_size,
                sub_prompt,
                assign_to,
            },
            end + 1,
        ))
    }

    fn try_parse_summarize(text: &str) -> Option<(RlmAction, usize)> {
        if !text.starts_with("[SUMMARIZE ") {
            return None;
        }
        let end = text.find(']')?;
        let attrs = &text[11..end];
        let variable = Self::extract_attr(attrs, "var")?;
        let assign_to = Self::extract_attr(attrs, "assign_to")?;
        Some((
            RlmAction::Summarize {
                variable,
                assign_to,
            },
            end + 1,
        ))
    }

    /// Extract a named attribute value from an attribute string.
    /// Handles both quoted (`key="value"`) and unquoted (`key=value`) forms.
    pub fn extract_attr(attrs: &str, key: &str) -> Option<String> {
        let quoted_pat = format!("{}=\"", key);
        let plain_pat = format!("{}=", key);

        // Try quoted form first.
        if let Some(start) = attrs.find(quoted_pat.as_str()) {
            let value_start = start + quoted_pat.len();
            if let Some(end) = attrs[value_start..].find('"') {
                return Some(attrs[value_start..value_start + end].to_string());
            }
        }

        // Try unquoted form.
        if let Some(start) = attrs.find(plain_pat.as_str()) {
            let value_start = start + plain_pat.len();
            // Make sure this isn't a substring of the quoted pattern we
            // already tried (i.e. the character before `=` is part of key).
            let rest = &attrs[value_start..];
            if rest.starts_with('"') {
                // This is actually the quoted form; the earlier branch
                // should have handled it.  Skip.
                return None;
            }
            let end = rest
                .find(|c: char| c.is_whitespace() || c == ']')
                .unwrap_or(rest.len());
            return Some(rest[..end].to_string());
        }

        None
    }

    // ---- Action execution -------------------------------------------------

    fn execute_action(
        &mut self,
        action: RlmAction,
        context: &mut ReplContext,
        depth: usize,
    ) -> Result<ActionResult, RlmError> {
        match action {
            RlmAction::Final { answer } => Ok(ActionResult::Done(answer)),

            RlmAction::FinalVar { variable } => {
                let val = context
                    .get(&variable)
                    .ok_or_else(|| {
                        RlmError::ParseError(format!("variable '{}' not found", variable))
                    })?
                    .to_string();
                Ok(ActionResult::Done(val))
            }

            RlmAction::RecursiveCall { prompt, assign_to } => {
                let result = self.execute_recursive(context, &prompt, depth + 1)?;
                context.set(&assign_to, result);
                Ok(ActionResult::Continue)
            }

            RlmAction::Peek {
                variable,
                start,
                end,
            } => {
                let val = context.get(&variable).ok_or_else(|| {
                    RlmError::ParseError(format!("variable '{}' not found", variable))
                })?;
                let s = start.min(val.len());
                let e = end.min(val.len());
                let slice = &val[s..e];
                context.set(&format!("{}_peek", variable), slice.to_string());
                Ok(ActionResult::Continue)
            }

            RlmAction::Grep { variable, pattern } => {
                let val = context.get(&variable).ok_or_else(|| {
                    RlmError::ParseError(format!("variable '{}' not found", variable))
                })?;
                let matches: Vec<&str> = val.lines().filter(|line| line.contains(&pattern)).collect();
                context.set(&format!("{}_grep", variable), matches.join("\n"));
                Ok(ActionResult::Continue)
            }

            RlmAction::PartitionMap {
                variable,
                chunk_size,
                sub_prompt,
                assign_to,
            } => {
                let val = context
                    .get(&variable)
                    .ok_or_else(|| {
                        RlmError::ParseError(format!("variable '{}' not found", variable))
                    })?
                    .to_string();
                let mut results = Vec::new();
                let mut start = 0;
                while start < val.len() {
                    let end = (start + chunk_size).min(val.len());
                    let chunk = &val[start..end];
                    let mut sub_ctx = ReplContext::new();
                    sub_ctx.set("chunk", chunk.to_string());
                    let result = self.execute_recursive(&mut sub_ctx, &sub_prompt, depth + 1)?;
                    results.push(result);
                    start = end;
                }
                context.set(&assign_to, results.join("\n---\n"));
                Ok(ActionResult::Continue)
            }

            RlmAction::Summarize {
                variable,
                assign_to,
            } => {
                let val = context
                    .get(&variable)
                    .ok_or_else(|| {
                        RlmError::ParseError(format!("variable '{}' not found", variable))
                    })?
                    .to_string();
                let sub_prompt =
                    format!("Summarize the following text concisely:\n\n{}", val);
                let result = self.execute_recursive(context, &sub_prompt, depth + 1)?;
                context.set(&assign_to, result);
                Ok(ActionResult::Continue)
            }
        }
    }

    /// Access accumulated execution statistics.
    pub fn stats(&self) -> &RlmStats {
        &self.stats
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Mock model ----------------------------------------------------------

    struct MockLM {
        responses: std::cell::RefCell<Vec<String>>,
        context_window: usize,
    }

    impl MockLM {
        fn new(responses: Vec<String>) -> Self {
            let mut r = responses;
            r.reverse();
            MockLM {
                responses: std::cell::RefCell::new(r),
                context_window: 4096,
            }
        }
    }

    impl GenerativeLM for MockLM {
        fn generate(&self, _prompt: &str, _max_tokens: usize, _config: &GenerateConfig) -> String {
            self.responses.borrow_mut().pop().unwrap_or_default()
        }

        fn generate_counted(
            &self,
            prompt: &str,
            max_tokens: usize,
            config: &GenerateConfig,
        ) -> (String, usize) {
            let text = self.generate(prompt, max_tokens, config);
            let tokens = text.split_whitespace().count();
            (text, tokens)
        }

        fn context_window(&self) -> usize {
            self.context_window
        }

        fn count_tokens(&self, text: &str) -> usize {
            text.split_whitespace().count()
        }
    }

    // -- ReplContext tests ---------------------------------------------------

    #[test]
    fn test_repl_context_set_get() {
        let mut ctx = ReplContext::new();
        assert!(ctx.get("foo").is_none());
        ctx.set("foo", "bar".to_string());
        assert_eq!(ctx.get("foo"), Some("bar"));
        ctx.set("foo", "baz".to_string());
        assert_eq!(ctx.get("foo"), Some("baz"));
    }

    #[test]
    fn test_repl_context_keys() {
        let mut ctx = ReplContext::new();
        ctx.set("beta", "2".to_string());
        ctx.set("alpha", "1".to_string());
        ctx.set("gamma", "3".to_string());
        let keys = ctx.keys();
        assert_eq!(keys, vec!["alpha", "beta", "gamma"]);
    }

    #[test]
    fn test_repl_context_as_prompt_context() {
        let mut ctx = ReplContext::new();
        ctx.set("name", "Alice".to_string());
        ctx.set("age", "30".to_string());
        let prompt = ctx.as_prompt_context();
        // Keys are sorted, so "age" comes before "name".
        assert!(prompt.contains("[age]\n30\n"));
        assert!(prompt.contains("[name]\nAlice\n"));
        // "age" should appear before "name"
        let age_pos = prompt.find("[age]").unwrap();
        let name_pos = prompt.find("[name]").unwrap();
        assert!(age_pos < name_pos);
    }

    // -- RlmConfig tests ----------------------------------------------------

    #[test]
    fn test_rlm_config_default() {
        let cfg = RlmConfig::default();
        assert_eq!(cfg.max_depth, 4);
        assert_eq!(cfg.max_total_tokens, 100_000);
        assert_eq!(cfg.max_tokens_per_call, 4096);
        assert_eq!(cfg.partition_size, 2048);
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
        assert!((cfg.top_p - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_rlm_config_builder() {
        let cfg = RlmConfig::default()
            .max_depth(8)
            .max_total_tokens(50_000)
            .max_tokens_per_call(2048)
            .partition_size(1024)
            .temperature(0.5)
            .top_p(0.95);
        assert_eq!(cfg.max_depth, 8);
        assert_eq!(cfg.max_total_tokens, 50_000);
        assert_eq!(cfg.max_tokens_per_call, 2048);
        assert_eq!(cfg.partition_size, 1024);
        assert!((cfg.temperature - 0.5).abs() < 1e-6);
        assert!((cfg.top_p - 0.95).abs() < 1e-6);
    }

    // -- Parser tests -------------------------------------------------------

    #[test]
    fn test_parse_final() {
        let actions = RlmOrchestrator::parse_output("[FINAL] The answer is 42 [/FINAL]").unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::Final { answer } => assert_eq!(answer, "The answer is 42"),
            other => panic!("expected Final, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_final_no_closing_tag() {
        let actions = RlmOrchestrator::parse_output("[FINAL] Trailing answer").unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::Final { answer } => assert_eq!(answer, "Trailing answer"),
            other => panic!("expected Final, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_final_var() {
        let actions = RlmOrchestrator::parse_output("[FINAL_VAR var=\"result\"]").unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::FinalVar { variable } => assert_eq!(variable, "result"),
            other => panic!("expected FinalVar, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_call() {
        let text = "[CALL prompt=\"do something\" assign_to=\"out\"]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::RecursiveCall { prompt, assign_to } => {
                assert_eq!(prompt, "do something");
                assert_eq!(assign_to, "out");
            }
            other => panic!("expected RecursiveCall, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_peek() {
        let text = "[PEEK var=\"input\" start=10 end=50]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::Peek {
                variable,
                start,
                end,
            } => {
                assert_eq!(variable, "input");
                assert_eq!(*start, 10);
                assert_eq!(*end, 50);
            }
            other => panic!("expected Peek, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_grep() {
        let text = "[GREP var=\"data\" pattern=\"error\"]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::Grep { variable, pattern } => {
                assert_eq!(variable, "data");
                assert_eq!(pattern, "error");
            }
            other => panic!("expected Grep, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_partition_map() {
        let text =
            "[PARTITION_MAP var=\"input\" chunk_size=512 sub_prompt=\"summarize\" assign_to=\"parts\"]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::PartitionMap {
                variable,
                chunk_size,
                sub_prompt,
                assign_to,
            } => {
                assert_eq!(variable, "input");
                assert_eq!(*chunk_size, 512);
                assert_eq!(sub_prompt, "summarize");
                assert_eq!(assign_to, "parts");
            }
            other => panic!("expected PartitionMap, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_partition_map_prompt_alias() {
        let text =
            "[PARTITION_MAP var=\"input\" chunk_size=256 prompt=\"process\" assign_to=\"out\"]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::PartitionMap { sub_prompt, .. } => {
                assert_eq!(sub_prompt, "process");
            }
            other => panic!("expected PartitionMap, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_summarize() {
        let text = "[SUMMARIZE var=\"doc\" assign_to=\"summary\"]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::Summarize {
                variable,
                assign_to,
            } => {
                assert_eq!(variable, "doc");
                assert_eq!(assign_to, "summary");
            }
            other => panic!("expected Summarize, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_multiple_actions() {
        let text = "Some preamble\n\
                     [PEEK var=\"input\" start=0 end=100]\n\
                     Then some reasoning...\n\
                     [GREP var=\"input\" pattern=\"fn main\"]\n\
                     [FINAL] Done analyzing [/FINAL]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 3);
        assert!(matches!(&actions[0], RlmAction::Peek { .. }));
        assert!(matches!(&actions[1], RlmAction::Grep { .. }));
        assert!(matches!(&actions[2], RlmAction::Final { .. }));
    }

    #[test]
    fn test_parse_no_actions() {
        let actions = RlmOrchestrator::parse_output("Just some regular text").unwrap();
        assert_eq!(actions.len(), 0);
    }

    // -- extract_attr tests -------------------------------------------------

    #[test]
    fn test_extract_attr_quoted() {
        let attrs = "var=\"hello\" start=5";
        assert_eq!(
            RlmOrchestrator::extract_attr(attrs, "var"),
            Some("hello".to_string())
        );
    }

    #[test]
    fn test_extract_attr_unquoted() {
        let attrs = "var=\"hello\" start=5 end=10";
        assert_eq!(
            RlmOrchestrator::extract_attr(attrs, "start"),
            Some("5".to_string())
        );
        assert_eq!(
            RlmOrchestrator::extract_attr(attrs, "end"),
            Some("10".to_string())
        );
    }

    #[test]
    fn test_extract_attr_missing() {
        let attrs = "var=\"hello\"";
        assert_eq!(RlmOrchestrator::extract_attr(attrs, "missing"), None);
    }

    #[test]
    fn test_extract_attr_empty_quoted() {
        let attrs = "var=\"\"";
        assert_eq!(
            RlmOrchestrator::extract_attr(attrs, "var"),
            Some(String::new())
        );
    }

    // -- Orchestrator tests -------------------------------------------------

    #[test]
    fn test_orchestrator_simple_final() {
        let model = MockLM::new(vec!["[FINAL] hello world [/FINAL]".to_string()]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("some context", "what is this?").unwrap();
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_orchestrator_final_var() {
        let model = MockLM::new(vec!["[FINAL_VAR var=\"input\"]".to_string()]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("the original context", "return the input").unwrap();
        assert_eq!(result, "the original context");
    }

    #[test]
    fn test_orchestrator_max_depth() {
        let responses: Vec<String> = (0..10)
            .map(|_| "[CALL prompt=\"recurse\" assign_to=\"x\"]".to_string())
            .collect();
        let model = MockLM::new(responses);
        let config = RlmConfig::default().max_depth(2);
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("ctx", "task");
        assert!(matches!(result, Err(RlmError::MaxDepthExceeded { .. })));
    }

    #[test]
    fn test_orchestrator_token_budget() {
        let model = MockLM::new(vec!["[FINAL] ok [/FINAL]".to_string()]);
        let config = RlmConfig::default().max_total_tokens(0);
        let mut orch = RlmOrchestrator::new(&model, config);
        // The first call will generate some tokens, which will push us past
        // the budget of 0 if we recursed, but the initial call at depth 0
        // still checks budget before calling.  The budget is 0, total is 0,
        // so 0 > 0 is false; we proceed.  After the call, total_tokens > 0.
        // We get a Final answer so no further recursion occurs.
        let result = orch.run("ctx", "task");
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), "ok");
    }

    #[test]
    fn test_orchestrator_peek() {
        // First call: peek at input chars 0..5, then second call: return the
        // peeked value via FINAL_VAR.
        let model = MockLM::new(vec![
            "[PEEK var=\"input\" start=0 end=5]\n[FINAL_VAR var=\"input_peek\"]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("Hello, World!", "peek at start").unwrap();
        assert_eq!(result, "Hello");
    }

    #[test]
    fn test_orchestrator_peek_clamped() {
        // Peek with end beyond the variable length should clamp.
        let model = MockLM::new(vec![
            "[PEEK var=\"input\" start=0 end=9999]\n[FINAL_VAR var=\"input_peek\"]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("short", "peek").unwrap();
        assert_eq!(result, "short");
    }

    #[test]
    fn test_orchestrator_grep() {
        let input = "line one: foo\nline two: bar\nline three: foo again";
        let model = MockLM::new(vec![
            "[GREP var=\"input\" pattern=\"foo\"]\n[FINAL_VAR var=\"input_grep\"]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run(input, "find foo lines").unwrap();
        assert_eq!(result, "line one: foo\nline three: foo again");
    }

    #[test]
    fn test_orchestrator_grep_no_match() {
        let input = "alpha\nbeta\ngamma";
        let model = MockLM::new(vec![
            "[GREP var=\"input\" pattern=\"delta\"]\n[FINAL_VAR var=\"input_grep\"]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run(input, "find delta").unwrap();
        assert_eq!(result, "");
    }

    #[test]
    fn test_orchestrator_recursive_call() {
        // First call returns a CALL that spawns a sub-task, then FINALs the result.
        // The sub-task returns a FINAL with a value.
        let model = MockLM::new(vec![
            "[CALL prompt=\"sub\" assign_to=\"r\"]\n[FINAL_VAR var=\"r\"]".to_string(),
            "[FINAL] sub-result [/FINAL]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("ctx", "task").unwrap();
        assert_eq!(result, "sub-result");
        assert_eq!(orch.stats().call_count, 2);
        assert_eq!(orch.stats().max_depth_reached, 1);
    }

    #[test]
    fn test_orchestrator_no_actions_returns_raw() {
        let model = MockLM::new(vec!["Just plain text, no actions.".to_string()]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("ctx", "task").unwrap();
        assert_eq!(result, "Just plain text, no actions.");
    }

    #[test]
    fn test_orchestrator_stats() {
        let model = MockLM::new(vec!["[FINAL] done [/FINAL]".to_string()]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let _ = orch.run("ctx", "task");
        assert_eq!(orch.stats().call_count, 1);
        assert!(orch.stats().total_tokens > 0);
        assert_eq!(orch.stats().max_depth_reached, 0);
    }

    #[test]
    fn test_orchestrator_partition_map() {
        // Input is "aabbcc", chunk_size=2 => 3 chunks: "aa", "bb", "cc".
        // Each sub-call returns [FINAL] echoing the chunk.
        let model = MockLM::new(vec![
            "[PARTITION_MAP var=\"input\" chunk_size=2 sub_prompt=\"echo\" assign_to=\"mapped\"]\n\
             [FINAL_VAR var=\"mapped\"]"
                .to_string(),
            "[FINAL] chunk:aa [/FINAL]".to_string(),
            "[FINAL] chunk:bb [/FINAL]".to_string(),
            "[FINAL] chunk:cc [/FINAL]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("aabbcc", "map it").unwrap();
        assert_eq!(result, "chunk:aa\n---\nchunk:bb\n---\nchunk:cc");
    }

    #[test]
    fn test_orchestrator_summarize() {
        let model = MockLM::new(vec![
            "[SUMMARIZE var=\"input\" assign_to=\"summary\"]\n[FINAL_VAR var=\"summary\"]"
                .to_string(),
            "[FINAL] Short version. [/FINAL]".to_string(),
        ]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("A very long text that needs summarization.", "summarize").unwrap();
        assert_eq!(result, "Short version.");
    }

    #[test]
    fn test_orchestrator_missing_var() {
        let model = MockLM::new(vec!["[FINAL_VAR var=\"nonexistent\"]".to_string()]);
        let config = RlmConfig::default();
        let mut orch = RlmOrchestrator::new(&model, config);
        let result = orch.run("ctx", "task");
        assert!(matches!(result, Err(RlmError::ParseError(_))));
    }

    // -- RlmError display tests ---------------------------------------------

    #[test]
    fn test_rlm_error_display() {
        let e1 = RlmError::MaxDepthExceeded { depth: 5, max: 4 };
        assert_eq!(
            format!("{}", e1),
            "max recursion depth exceeded: 5 > 4"
        );

        let e2 = RlmError::TokenBudgetExhausted {
            used: 150_000,
            budget: 100_000,
        };
        assert_eq!(
            format!("{}", e2),
            "token budget exhausted: 150000 used of 100000 budget"
        );

        let e3 = RlmError::ParseError("bad tag".to_string());
        assert_eq!(format!("{}", e3), "parse error: bad tag");

        let e4 = RlmError::ModelError("timeout".to_string());
        assert_eq!(format!("{}", e4), "model error: timeout");
    }

    #[test]
    fn test_rlm_error_is_error_trait() {
        let e: Box<dyn std::error::Error> =
            Box::new(RlmError::ParseError("test".to_string()));
        // Ensure the error trait object works.
        assert!(e.to_string().contains("parse error"));
    }

    // -- GenerateConfig tests -----------------------------------------------

    #[test]
    fn test_generate_config_default() {
        let cfg = GenerateConfig::default();
        assert!((cfg.temperature - 0.7).abs() < 1e-6);
        assert!((cfg.top_p - 0.9).abs() < 1e-6);
        assert!(cfg.stop_sequences.is_empty());
    }

    // -- Edge cases ---------------------------------------------------------

    #[test]
    fn test_parse_bracket_not_action() {
        // A bracket that doesn't match any known action should be skipped.
        let actions = RlmOrchestrator::parse_output("[UNKNOWN action] some text").unwrap();
        assert_eq!(actions.len(), 0);
    }

    #[test]
    fn test_parse_nested_brackets() {
        // Ensure nested brackets in text don't confuse the parser.
        let text = "text with [brackets] then [FINAL] answer [/FINAL]";
        let actions = RlmOrchestrator::parse_output(text).unwrap();
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            RlmAction::Final { answer } => assert_eq!(answer, "answer"),
            other => panic!("expected Final, got {:?}", other),
        }
    }

    #[test]
    fn test_build_prompt_contains_task() {
        let model = MockLM::new(vec![]);
        let config = RlmConfig::default();
        let orch = RlmOrchestrator::new(&model, config);
        let mut ctx = ReplContext::new();
        ctx.set("input", "hello".to_string());
        let prompt = orch.build_prompt(&ctx, "analyze this");
        assert!(prompt.contains("Task: analyze this"));
        assert!(prompt.contains("[input]"));
        assert!(prompt.contains("hello"));
        assert!(prompt.contains("[FINAL]"));
    }

    #[test]
    fn test_repl_context_empty() {
        let ctx = ReplContext::new();
        assert!(ctx.keys().is_empty());
        assert_eq!(ctx.as_prompt_context(), "");
    }
}
