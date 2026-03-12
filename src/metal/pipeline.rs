//! Pipeline builder for declaring fused op sequences on Metal GPU.
//!
//! Maps high-level op sequences to pre-compiled fused kernel dispatches,
//! eliminating intermediate device memory round-trips.

use crate::tensor::Tensor;

/// Fused operation types supported by the pipeline engine.
#[derive(Debug, Clone)]
pub enum FusedOp {
    /// Fused matmul + bias + GELU: out = gelu(input @ weight + bias)
    MatmulBiasGelu,
    /// Fused matmul + bias + ReLU: out = relu(input @ weight + bias)
    MatmulBiasRelu,
    /// Fused matmul + bias: out = input @ weight + bias
    MatmulBias,
    /// Fused bias + GELU: out = gelu(input + bias)
    BiasGelu,
    /// Fused residual add + layernorm: out = layernorm(x + residual)
    AddLayerNorm,
}

/// Builder for declaring fused op sequences.
///
/// Example:
/// ```ignore
/// let output = PipelineBuilder::new()
///     .matmul_bias_gelu(&input, &w1, &b1)
///     .matmul_bias(&hidden, &w2, &b2)
///     .build();
/// ```
pub struct PipelineBuilder {
    result: Option<Tensor>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder.
    pub fn new() -> Self {
        PipelineBuilder { result: None }
    }

    /// Fused matmul + bias + GELU.
    pub fn matmul_bias_gelu(mut self, input: &Tensor, weight: &Tensor, bias: &Tensor) -> Self {
        self.result = Some(input.matmul_bias_gelu(weight, bias));
        self
    }

    /// Fused matmul + bias + ReLU.
    pub fn matmul_bias_relu(mut self, input: &Tensor, weight: &Tensor, bias: &Tensor) -> Self {
        self.result = Some(input.matmul_bias_relu(weight, bias));
        self
    }

    /// Matmul + bias (no activation).
    pub fn matmul_bias(mut self, input: &Tensor, weight: &Tensor, bias: &Tensor) -> Self {
        self.result = Some(input.matmul(weight).add_bias(bias));
        self
    }

    /// Fused residual add + layernorm.
    pub fn add_layernorm(mut self, x: &Tensor, residual: &Tensor, gamma: &Tensor, beta: &Tensor, dim: usize) -> Self {
        self.result = Some(x.add_layer_norm(residual, gamma, beta, dim));
        self
    }

    /// Execute the pipeline and return the final result tensor.
    pub fn build(self) -> Tensor {
        self.result.expect("PipelineBuilder: no operations were added")
    }
}
