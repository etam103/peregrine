use crate::tensor::Tensor;
use std::fmt::Write;

// ---------------------------------------------------------------------------
// Model Summary
// ---------------------------------------------------------------------------

/// Produces a PyTorch-style ASCII table summarising every named parameter.
pub fn model_summary(named_params: &[(String, &Tensor)]) -> String {
    let mut out = String::new();
    let name_w = named_params
        .iter()
        .map(|(n, _)| n.len())
        .max()
        .unwrap_or(9)
        .max(9);

    let _ = writeln!(
        out,
        "{:<name_w$}  {:<20}  {:>10}",
        "Parameter", "Shape", "Params",
        name_w = name_w
    );
    let rule_len = name_w + 2 + 20 + 2 + 10;
    for _ in 0..rule_len {
        out.push('\u{2500}');
    }
    out.push('\n');

    let mut total: usize = 0;
    for (name, tensor) in named_params {
        let shape = tensor.shape();
        let count = tensor.size();
        total += count;
        let shape_str = format!("{:?}", shape);
        let _ = writeln!(
            out,
            "{:<name_w$}  {:<20}  {:>10}",
            name,
            shape_str,
            format_count(count),
            name_w = name_w
        );
    }

    for _ in 0..rule_len {
        out.push('\u{2500}');
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "{} parameters{:>width$}",
        named_params.len(),
        format_count(total),
        width = rule_len - format!("{} parameters", named_params.len()).len()
    );
    out
}

fn format_count(n: usize) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, ch) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(ch);
    }
    result.chars().rev().collect()
}

// ---------------------------------------------------------------------------
// Training Health
// ---------------------------------------------------------------------------

/// Per-parameter statistics computed by `training_health`.
pub struct ParamHealth {
    pub name: String,
    pub w_mean: f32,
    pub w_std: f32,
    pub w_absmax: f32,
    pub g_mean: f32,
    pub g_std: f32,
    pub g_absmax: f32,
    pub grad_to_weight_ratio: f32,
    pub has_nan: bool,
    pub zero_grad: bool,
}

/// Global report returned by `training_health`.
pub struct TrainingHealthReport {
    pub params: Vec<ParamHealth>,
    pub global_grad_norm: f32,
    pub warnings: Vec<String>,
}

impl TrainingHealthReport {
    /// Flatten the report into `(key, value)` pairs suitable for wandb logging.
    pub fn to_metrics(&self) -> Vec<(&str, f32)> {
        let mut m = vec![("health/grad_norm", self.global_grad_norm)];
        let has_nan = self.params.iter().any(|p| p.has_nan);
        m.push(("health/has_nan", if has_nan { 1.0 } else { 0.0 }));
        let zero_count = self.params.iter().filter(|p| p.zero_grad).count();
        m.push(("health/zero_grad_params", zero_count as f32));
        let max_ratio = self
            .params
            .iter()
            .map(|p| p.grad_to_weight_ratio)
            .fold(0.0f32, f32::max);
        m.push(("health/max_grad_weight_ratio", max_ratio));
        m
    }

    /// Human-readable multi-line display string.
    pub fn display(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "Training Health  (grad_norm = {:.6})", self.global_grad_norm);

        let name_w = self
            .params
            .iter()
            .map(|p| p.name.len())
            .max()
            .unwrap_or(9)
            .max(9);

        let _ = writeln!(
            out,
            "  {:<name_w$}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "param", "w_mean", "w_std", "g_mean", "g_std", "g/w ratio",
            name_w = name_w,
        );

        for p in &self.params {
            let flag = if p.has_nan {
                " NaN!"
            } else if p.zero_grad {
                " ZERO"
            } else {
                ""
            };
            let _ = writeln!(
                out,
                "  {:<name_w$}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}{}",
                p.name, p.w_mean, p.w_std, p.g_mean, p.g_std, p.grad_to_weight_ratio, flag,
                name_w = name_w,
            );
        }

        if !self.warnings.is_empty() {
            let _ = writeln!(out, "  Warnings:");
            for w in &self.warnings {
                let _ = writeln!(out, "    - {}", w);
            }
        }
        out
    }
}

/// Analyse the health of every named parameter and its gradient.
pub fn training_health(named_params: &[(String, &Tensor)]) -> TrainingHealthReport {
    let mut params = Vec::with_capacity(named_params.len());
    let mut warnings = Vec::new();
    let mut global_grad_norm_sq: f32 = 0.0;

    for (name, tensor) in named_params {
        let data = tensor.data();
        let grad_opt = tensor.grad();

        let (w_mean, w_std, w_absmax) = stats(&data);

        let (g_mean, g_std, g_absmax, has_nan, zero_grad) = match &grad_opt {
            Some(g) => {
                let (gm, gs, ga) = stats(g);
                let has_nan = g.iter().any(|v| v.is_nan());
                let zero = g.iter().all(|v| *v == 0.0);
                (gm, gs, ga, has_nan, zero)
            }
            None => (0.0, 0.0, 0.0, false, true),
        };

        if let Some(g) = &grad_opt {
            let sq_sum: f32 = g.iter().map(|v| v * v).sum();
            global_grad_norm_sq += sq_sum;
        }

        let w_norm = w_std.max(1e-12);
        let grad_to_weight_ratio = g_std / w_norm;

        if has_nan {
            warnings.push(format!("{}: gradient contains NaN", name));
        }
        if grad_to_weight_ratio > 1.0 {
            warnings.push(format!(
                "{}: grad/weight ratio {:.4} — possible exploding gradients",
                name, grad_to_weight_ratio
            ));
        }
        if zero_grad && grad_opt.is_some() {
            warnings.push(format!("{}: gradient is all zeros — parameter may be stalled", name));
        }

        params.push(ParamHealth {
            name: name.clone(),
            w_mean,
            w_std,
            w_absmax,
            g_mean,
            g_std,
            g_absmax,
            grad_to_weight_ratio,
            has_nan,
            zero_grad,
        });
    }

    let global_grad_norm = global_grad_norm_sq.sqrt();

    TrainingHealthReport {
        params,
        global_grad_norm,
        warnings,
    }
}

fn stats(data: &[f32]) -> (f32, f32, f32) {
    if data.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let n = data.len() as f32;
    let mean: f32 = data.iter().sum::<f32>() / n;
    let var: f32 = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let absmax = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    (mean, var.sqrt(), absmax)
}
