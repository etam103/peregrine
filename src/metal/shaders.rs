/// Embedded Metal shader source for tensor compute kernels.
pub const SHADER_SOURCE: &str = r#"
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------------------
// Element-wise binary ops
// ---------------------------------------------------------------------------

kernel void add_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] + b[idx];
}

kernel void sub_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] - b[idx];
}

kernel void mul_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] * b[idx];
}

kernel void div_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] / b[idx];
}

// ---------------------------------------------------------------------------
// Element-wise unary ops
// ---------------------------------------------------------------------------

kernel void neg_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = -a[idx];
}

kernel void exp_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = exp(a[idx]);
}

kernel void log_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = log(a[idx]);
}

kernel void sqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = sqrt(a[idx]);
}

kernel void relu_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = max(a[idx], 0.0f);
}

kernel void sigmoid_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = 1.0f / (1.0f + exp(-a[idx]));
}

kernel void tanh_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = tanh(a[idx]);
}

kernel void sin_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = sin(a[idx]);
}

kernel void cos_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = cos(a[idx]);
}

kernel void abs_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = abs(a[idx]);
}

kernel void reciprocal_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = 1.0f / a[idx];
}

kernel void square_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] * a[idx];
}

kernel void rsqrt_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = rsqrt(a[idx]);
}

kernel void floor_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = floor(a[idx]);
}

kernel void ceil_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = ceil(a[idx]);
}

kernel void round_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = rint(a[idx]);
}

kernel void sign_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] > 0.0f) ? 1.0f : (a[idx] < 0.0f) ? -1.0f : 0.0f;
}

kernel void expm1_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = exp(a[idx]) - 1.0f;
}

kernel void log2_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = log2(a[idx]);
}

kernel void log10_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = log10(a[idx]);
}

kernel void log1p_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = log(1.0f + a[idx]);
}

// Abramowitz & Stegun approximation (max error ~1.5e-7)
float erf_approx(float x) {
    float ax = abs(x);
    float t = 1.0f / (1.0f + 0.3275911f * ax);
    float t2 = t * t;
    float t3 = t2 * t;
    float t4 = t3 * t;
    float t5 = t4 * t;
    float poly = 0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3
               - 1.453152027f * t4 + 1.061405429f * t5;
    float result = 1.0f - poly * exp(-ax * ax);
    return x >= 0.0f ? result : -result;
}

kernel void erf_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = erf_approx(a[idx]);
}

float erfinv_approx(float x) {
    float w = -log((1.0f - x) * (1.0f + x));
    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    } else {
        w = sqrt(w) - 3.0f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

kernel void erfinv_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = erfinv_approx(a[idx]);
}

kernel void sinh_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = sinh(a[idx]);
}

kernel void cosh_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = cosh(a[idx]);
}

kernel void arcsin_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = asin(a[idx]);
}

kernel void arccos_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = acos(a[idx]);
}

kernel void arctan_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = atan(a[idx]);
}

kernel void arcsinh_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = asinh(a[idx]);
}

kernel void arccosh_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = acosh(a[idx]);
}

kernel void arctanh_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = atanh(a[idx]);
}

// ---------------------------------------------------------------------------
// Scale: out = a * scalar
// ---------------------------------------------------------------------------

kernel void scale_f32(
    device const float* a      [[buffer(0)]],
    device float* out           [[buffer(1)]],
    constant float& scalar      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = a[idx] * scalar;
}

// ---------------------------------------------------------------------------
// Tiled matmul + bias + relu: C = relu(A @ B + bias)
// Uses threadgroup shared memory for tile-based data reuse.
// Each threadgroup computes a TILE_SIZE x TILE_SIZE tile of C.
// ---------------------------------------------------------------------------

constant uint TILE_SIZE = 16;

struct MatmulParams {
    uint M;
    uint N;
    uint K;
    uint fuse_bias;   // 0 = no bias, 1 = add bias
    uint fuse_relu;   // 0 = no relu, 1 = relu
    uint trans_a;     // 0 = no transpose, 1 = transpose A
    uint trans_b;     // 0 = no transpose, 1 = transpose B
    uint fuse_gelu;   // 0 = no gelu, 1 = apply GELU after bias
};

kernel void matmul_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    device const float* bias    [[buffer(3)]],
    constant MatmulParams& p    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint row = gid.y;
    uint col = gid.x;

    threadgroup float As[TILE_SIZE][TILE_SIZE];
    threadgroup float Bs[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    uint num_tiles = (p.K + TILE_SIZE - 1) / TILE_SIZE;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A into shared memory
        uint a_k = t * TILE_SIZE + lid.x;
        if (row < p.M && a_k < p.K) {
            uint a_idx = p.trans_a ? (a_k * p.M + row) : (row * p.K + a_k);
            As[lid.y][lid.x] = A[a_idx];
        } else {
            As[lid.y][lid.x] = 0.0f;
        }

        // Load tile of B into shared memory
        uint b_k = t * TILE_SIZE + lid.y;
        if (b_k < p.K && col < p.N) {
            uint b_idx = p.trans_b ? (col * p.K + b_k) : (b_k * p.N + col);
            Bs[lid.y][lid.x] = B[b_idx];
        } else {
            Bs[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_SIZE; k++) {
            sum += As[lid.y][k] * Bs[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < p.M && col < p.N) {
        if (p.fuse_bias) sum += bias[col];
        if (p.fuse_relu) sum = max(sum, 0.0f);
        if (p.fuse_gelu) {
            float x3 = sum * sum * sum;
            float inner = 0.7978845608f * (sum + 0.044715f * x3);
            sum = 0.5f * sum * (1.0f + precise::tanh(inner));
        }
        C[row * p.N + col] = sum;
    }
}

// ---------------------------------------------------------------------------
// Parallel reduction: sum
// Dispatched with threadgroups; each threadgroup reduces a chunk
// ---------------------------------------------------------------------------

kernel void sum_f32(
    device const float* input       [[buffer(0)]],
    device float* partial_sums      [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];

    float val = (tid < count) ? input[tid] : 0.0f;
    shared[lid] = val;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Tree reduction
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0) {
        partial_sums[gid] = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Numerically stable softmax (fused max-sub-exp-sum-div)
// Each threadgroup processes one row of [batch, dim]
// ---------------------------------------------------------------------------

struct SoftmaxParams {
    uint batch;
    uint dim;
};

kernel void softmax_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant SoftmaxParams& p   [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    device const float* row_in = input + row * p.dim;
    device float* row_out = output + row * p.dim;

    threadgroup float shared[1024];

    // 1. Find max (initialize unused lanes to -INFINITY)
    float local_max = -INFINITY;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_max = max(local_max, row_in[i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute exp(x - max) and sum (initialize unused lanes to 0)
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        float e = exp(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Normalize
    for (uint i = lid; i < p.dim; i += group_size) {
        row_out[i] /= total;
    }
}
// ---------------------------------------------------------------------------
// Parallel reduction: max
// ---------------------------------------------------------------------------

kernel void max_f32(
    device const float* input       [[buffer(0)]],
    device float* partial_out       [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];

    float val = (tid < count) ? input[tid] : -INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partial_out[gid] = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Parallel reduction: min
// ---------------------------------------------------------------------------

kernel void min_f32(
    device const float* input       [[buffer(0)]],
    device float* partial_out       [[buffer(1)]],
    constant uint& count            [[buffer(2)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint gid [[threadgroup_position_in_grid]],
    uint group_size [[threads_per_threadgroup]])
{
    threadgroup float shared[1024];

    float val = (tid < count) ? input[tid] : INFINITY;
    shared[lid] = val;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            shared[lid] = min(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lid == 0) {
        partial_out[gid] = shared[0];
    }
}

// ---------------------------------------------------------------------------
// Transpose: out[j, i] = in[i, j] for [rows, cols] matrix
// ---------------------------------------------------------------------------

struct TransposeParams {
    uint rows;
    uint cols;
};

kernel void transpose_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant TransposeParams& p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= p.rows || col >= p.cols) return;
    output[col * p.rows + row] = input[row * p.cols + col];
}

// ---------------------------------------------------------------------------
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
// One threadgroup per instance (row). Fused mean/var/normalize.
// ---------------------------------------------------------------------------

struct LayerNormParams {
    uint batch;
    uint dim;
    float eps;
};

kernel void layernorm_f32(
    device const float* input   [[buffer(0)]],
    device const float* gamma   [[buffer(1)]],
    device const float* beta    [[buffer(2)]],
    device float* output        [[buffer(3)]],
    constant LayerNormParams& p [[buffer(4)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    device const float* row_in = input + row * p.dim;
    device float* row_out = output + row * p.dim;

    threadgroup float shared[1024];

    // 1. Compute mean
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_sum += row_in[i];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(p.dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute variance
    float local_var = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        float diff = row_in[i] - mean;
        local_var += diff * diff;
    }
    shared[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(p.dim) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Normalize with affine transform
    for (uint i = lid; i < p.dim; i += group_size) {
        row_out[i] = gamma[i] * (row_in[i] - mean) * inv_std + beta[i];
    }
}

// ---------------------------------------------------------------------------
// Backward kernels
// ---------------------------------------------------------------------------

// ReLU backward: out = (input > 0) ? grad : 0
kernel void relu_backward_f32(
    device const float* input   [[buffer(0)]],
    device const float* grad    [[buffer(1)]],
    device float* out           [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (input[idx] > 0.0f) ? grad[idx] : 0.0f;
}

// Sigmoid backward: out = grad * s * (1 - s), where s = sigmoid output
kernel void sigmoid_backward_f32(
    device const float* sigmoid_out [[buffer(0)]],
    device const float* grad        [[buffer(1)]],
    device float* out               [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float s = sigmoid_out[idx];
    out[idx] = grad[idx] * s * (1.0f - s);
}

// Tanh backward: out = grad * (1 - y*y), where y = tanh output
kernel void tanh_backward_f32(
    device const float* tanh_out [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* out            [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float y = tanh_out[idx];
    out[idx] = grad[idx] * (1.0f - y * y);
}

// GELU forward: 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
// Use precise:: math to prevent Metal compiler from pattern-matching
// and replacing with a fast GELU approximation that can produce NaN.
kernel void gelu_f32(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    float x = input[idx];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    float t = precise::tanh(inner);
    output[idx] = 0.5f * x * (1.0f + t);
}

// GELU backward: full derivative of GELU(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
kernel void gelu_backward_f32(
    device const float* input   [[buffer(0)]],
    device const float* grad    [[buffer(1)]],
    device float* out           [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float x = input[idx];
    float x3 = x * x * x;
    float sqrt_2_over_pi = 0.7978845608f; // sqrt(2/pi)
    float inner_val = sqrt_2_over_pi * (x + 0.044715f * x3);
    float tanh_val = precise::tanh(inner_val);
    float sech2 = 1.0f - tanh_val * tanh_val;
    float d_inner = sqrt_2_over_pi * (1.0f + 3.0f * 0.044715f * x * x);
    out[idx] = grad[idx] * (0.5f * (1.0f + tanh_val) + 0.5f * x * sech2 * d_inner);
}

// Accumulate in-place: a[i] += b[i]
kernel void accumulate_f32(
    device float* a             [[buffer(0)]],
    device const float* b       [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    a[idx] += b[idx];
}

// Fill buffer with scalar value
kernel void fill_f32(
    device float* out           [[buffer(0)]],
    constant float& value       [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = value;
}

// Softmax backward: grad_input = y * (grad - dot(grad, y)) per row
// One threadgroup per row, same as forward softmax
kernel void softmax_backward_f32(
    device const float* softmax_out [[buffer(0)]],
    device const float* grad        [[buffer(1)]],
    device float* grad_input        [[buffer(2)]],
    constant SoftmaxParams& p       [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    device const float* y = softmax_out + row * p.dim;
    device const float* g = grad + row * p.dim;
    device float* gi = grad_input + row * p.dim;

    threadgroup float shared[1024];

    // Compute dot(grad, y) for this row
    float local_dot = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_dot += g[i] * y[i];
    }
    shared[lid] = local_dot;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float dot_val = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // grad_input[i] = y[i] * (grad[i] - dot)
    for (uint i = lid; i < p.dim; i += group_size) {
        gi[i] = y[i] * (g[i] - dot_val);
    }
}

// LayerNorm backward: compute grad_input, grad_gamma, grad_beta
// One threadgroup per instance (row). Outputs to 3 buffers.
struct LayerNormBackwardParams {
    uint batch;
    uint dim;
};

kernel void layernorm_backward_f32(
    device const float* grad_out    [[buffer(0)]],
    device const float* normalized  [[buffer(1)]],
    device const float* gamma       [[buffer(2)]],
    device const float* inv_std_buf [[buffer(3)]],
    device float* grad_input        [[buffer(4)]],
    device float* grad_gamma        [[buffer(5)]],
    device float* grad_beta         [[buffer(6)]],
    constant LayerNormBackwardParams& p [[buffer(7)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    uint offset = row * p.dim;
    float istd = inv_std_buf[row];

    threadgroup float shared_dy[1024];
    threadgroup float shared_dy_xhat[1024];

    // Accumulate grad_gamma, grad_beta and compute mean_dy, mean_dy_xhat
    float local_dy = 0.0f;
    float local_dy_xhat = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        uint idx = offset + i;
        float g = grad_out[idx];
        float xhat = normalized[idx];
        // Atomically add to grad_gamma/grad_beta (coarse but correct)
        // For better perf we'd reduce per-row then sum, but this is simpler
        float dy = g * gamma[i];
        local_dy += dy;
        local_dy_xhat += dy * xhat;
    }

    // Reduce mean_dy
    shared_dy[lid] = local_dy;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared_dy[lid] += shared_dy[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_dy = shared_dy[0] / float(p.dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reduce mean_dy_xhat
    shared_dy_xhat[lid] = local_dy_xhat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared_dy_xhat[lid] += shared_dy_xhat[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean_dy_xhat = shared_dy_xhat[0] / float(p.dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write grad_input and accumulate grad_gamma/grad_beta
    for (uint i = lid; i < p.dim; i += group_size) {
        uint idx = offset + i;
        float g = grad_out[idx];
        float xhat = normalized[idx];
        float dy = g * gamma[i];
        grad_input[idx] = istd * (dy - mean_dy - xhat * mean_dy_xhat);
        // Atomic add for grad_gamma/grad_beta (multiple rows write to same gamma index)
        // Use atomic_fetch_add for device memory
        atomic_fetch_add_explicit((device atomic_float*)&grad_gamma[i], g * xhat, memory_order_relaxed);
        atomic_fetch_add_explicit((device atomic_float*)&grad_beta[i], g, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Gather: out[i] = input[indices[i]]
// ---------------------------------------------------------------------------

kernel void gather_f32(
    device const float* input   [[buffer(0)]],
    device const uint* indices  [[buffer(1)]],
    device float* out           [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = input[indices[idx]];
}

// Scatter-add: grad_input[indices[i]] += grad[i]  (atomic for safety)
kernel void scatter_add_f32(
    device const float* grad      [[buffer(0)]],
    device const uint* indices    [[buffer(1)]],
    device float* grad_input      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    atomic_fetch_add_explicit(
        (device atomic_float*)&grad_input[indices[idx]],
        grad[idx],
        memory_order_relaxed);
}

// ---------------------------------------------------------------------------
// Bias add: out[i] = input[i] + bias[i % cols]
// ---------------------------------------------------------------------------

kernel void bias_add_f32(
    device const float* input [[buffer(0)]],
    device const float* bias  [[buffer(1)]],
    device float* out         [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = input[idx] + bias[idx % cols];
}

// ---------------------------------------------------------------------------
// Bias gradient: sum grad over rows → bias_grad[col] = sum_r grad[r, col]
// ---------------------------------------------------------------------------

kernel void bias_grad_sum_f32(
    device const float* grad     [[buffer(0)]],
    device float* bias_grad      [[buffer(1)]],
    constant uint& rows          [[buffer(2)]],
    constant uint& cols          [[buffer(3)]],
    uint col [[thread_position_in_grid]])
{
    if (col >= cols) return;
    float sum = 0.0f;
    for (uint r = 0; r < rows; r++) {
        sum += grad[r * cols + col];
    }
    bias_grad[col] = sum;
}

// ---------------------------------------------------------------------------
// Log-softmax: out[i] = x[i] - max - log(sum(exp(x - max)))
// One threadgroup per row, same structure as softmax_f32
// ---------------------------------------------------------------------------

kernel void log_softmax_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant SoftmaxParams& p   [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    device const float* row_in = input + row * p.dim;
    device float* row_out = output + row * p.dim;

    threadgroup float shared[1024];

    // 1. Find max
    float local_max = -INFINITY;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_max = max(local_max, row_in[i]);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute sum(exp(x - max))
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_sum += exp(row_in[i] - row_max);
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float log_sum = log(shared[0]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Output: x - max - log(sum_exp)
    for (uint i = lid; i < p.dim; i += group_size) {
        row_out[i] = row_in[i] - row_max - log_sum;
    }
}

// Log-softmax backward: grad_input[i] = grad[i] - exp(output[i]) * sum(grad)
// One threadgroup per row
kernel void log_softmax_backward_f32(
    device const float* log_softmax_out [[buffer(0)]],
    device const float* grad            [[buffer(1)]],
    device float* grad_input            [[buffer(2)]],
    constant SoftmaxParams& p           [[buffer(3)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    device const float* y = log_softmax_out + row * p.dim;
    device const float* g = grad + row * p.dim;
    device float* gi = grad_input + row * p.dim;

    threadgroup float shared[1024];

    // Compute sum(grad) for this row
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_sum += g[i];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float sum_grad = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // grad_input[i] = grad[i] - exp(log_softmax[i]) * sum(grad)
    for (uint i = lid; i < p.dim; i += group_size) {
        gi[i] = g[i] - exp(y[i]) * sum_grad;
    }
}

// ---------------------------------------------------------------------------
// Scale-fill: dst[i] = src[0] * scalar (broadcast scalar from reduction)
// ---------------------------------------------------------------------------

kernel void scale_fill_f32(
    device const float* src [[buffer(0)]],
    device float* dst       [[buffer(1)]],
    constant float& scalar  [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    dst[idx] = src[0] * scalar;
}

// Fused Adam optimizer step
struct AdamParams {
    float lr;
    float beta1;
    float beta2;
    float eps;
    float bc1;          // 1 - beta1^t
    float bc2;          // 1 - beta2^t
    float weight_decay;
    uint  decoupled_wd; // 0 = L2, 1 = AdamW
};

kernel void adam_step_f32(
    device float* data      [[buffer(0)]],
    device const float* grad [[buffer(1)]],
    device float* m         [[buffer(2)]],
    device float* v         [[buffer(3)]],
    constant AdamParams& p  [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    float g = grad[idx];

    // L2 regularization (classic Adam)
    if (p.weight_decay > 0.0f && p.decoupled_wd == 0) {
        g += p.weight_decay * data[idx];
    }

    // Update moments
    m[idx] = p.beta1 * m[idx] + (1.0f - p.beta1) * g;
    v[idx] = p.beta2 * v[idx] + (1.0f - p.beta2) * g * g;

    // Bias-corrected estimates
    float m_hat = m[idx] / p.bc1;
    float v_hat = v[idx] / p.bc2;

    // Decoupled weight decay (AdamW)
    if (p.decoupled_wd != 0 && p.weight_decay > 0.0f) {
        data[idx] -= p.lr * p.weight_decay * data[idx];
    }

    data[idx] -= p.lr * m_hat / (sqrt(v_hat) + p.eps);
}

// ---------------------------------------------------------------------------
// Phase 1B: Binary Math Ops
// ---------------------------------------------------------------------------

kernel void maximum_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = max(a[idx], b[idx]);
}

kernel void minimum_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = min(a[idx], b[idx]);
}

kernel void power_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = pow(a[idx], b[idx]);
}

kernel void arctan2_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = atan2(a[idx], b[idx]);
}

kernel void logaddexp_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float m = max(a[idx], b[idx]);
    out[idx] = m + log(exp(a[idx] - m) + exp(b[idx] - m));
}

// ---------------------------------------------------------------------------
// Phase 1C: Clip, Where, NanToNum
// ---------------------------------------------------------------------------

struct ClipParams {
    float min_val;
    float max_val;
};

kernel void clip_f32(
    device const float* a   [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant ClipParams& p   [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = clamp(a[idx], p.min_val, p.max_val);
}

kernel void where_f32(
    device const float* cond [[buffer(0)]],
    device const float* x    [[buffer(1)]],
    device const float* y    [[buffer(2)]],
    device float* out        [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (cond[idx] != 0.0f) ? x[idx] : y[idx];
}

struct NanToNumParams {
    float nan_val;
    float posinf_val;
    float neginf_val;
};

kernel void nan_to_num_f32(
    device const float* a   [[buffer(0)]],
    device float* out        [[buffer(1)]],
    constant NanToNumParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float v = a[idx];
    out[idx] = isnan(v) ? p.nan_val : (isinf(v) ? (v > 0 ? p.posinf_val : p.neginf_val) : v);
}

// ---------------------------------------------------------------------------
// Phase 1D: Comparison / Logical Ops (binary)
// ---------------------------------------------------------------------------

kernel void equal_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] == b[idx]) ? 1.0f : 0.0f;
}

kernel void not_equal_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] != b[idx]) ? 1.0f : 0.0f;
}

kernel void greater_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] > b[idx]) ? 1.0f : 0.0f;
}

kernel void greater_equal_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] >= b[idx]) ? 1.0f : 0.0f;
}

kernel void less_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] < b[idx]) ? 1.0f : 0.0f;
}

kernel void less_equal_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] <= b[idx]) ? 1.0f : 0.0f;
}

kernel void logical_and_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] != 0.0f && b[idx] != 0.0f) ? 1.0f : 0.0f;
}

kernel void logical_or_f32(
    device const float* a [[buffer(0)]],
    device const float* b [[buffer(1)]],
    device float* out      [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] != 0.0f || b[idx] != 0.0f) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// Phase 1D: Comparison / Logical Ops (unary)
// ---------------------------------------------------------------------------

kernel void isnan_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = isnan(a[idx]) ? 1.0f : 0.0f;
}

kernel void isinf_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = isinf(a[idx]) ? 1.0f : 0.0f;
}

kernel void isfinite_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (!isnan(a[idx]) && !isinf(a[idx])) ? 1.0f : 0.0f;
}

kernel void logical_not_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = (a[idx] == 0.0f) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// Leaky ReLU: out = x > 0 ? x : alpha * x
// ---------------------------------------------------------------------------

kernel void leaky_relu_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant float& alpha  [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float x = a[idx];
    out[idx] = x > 0.0f ? x : alpha * x;
}

kernel void leaky_relu_backward_f32(
    device const float* input   [[buffer(0)]],
    device const float* grad    [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant float& alpha       [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = input[idx] > 0.0f ? grad[idx] : alpha * grad[idx];
}

// ---------------------------------------------------------------------------
// ELU: out = x > 0 ? x : alpha * (exp(x) - 1)
// ---------------------------------------------------------------------------

kernel void elu_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    constant float& alpha  [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    float x = a[idx];
    out[idx] = x > 0.0f ? x : alpha * (exp(x) - 1.0f);
}

kernel void elu_backward_f32(
    device const float* input   [[buffer(0)]],
    device const float* grad    [[buffer(1)]],
    device float* out           [[buffer(2)]],
    constant float& alpha       [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    float x = input[idx];
    out[idx] = x > 0.0f ? grad[idx] : grad[idx] * alpha * exp(x);
}

// ---------------------------------------------------------------------------
// Tril: zero elements above the k-th diagonal
// Treats tensor as batch of 2D matrices (last two dims are rows/cols)
// ---------------------------------------------------------------------------

struct TrilTriuParams {
    uint rows;
    uint cols;
    int k;
};

kernel void tril_f32(
    device const float* a [[buffer(0)]],
    device float* out     [[buffer(1)]],
    constant TrilTriuParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    uint mat_size = p.rows * p.cols;
    uint local_idx = idx % mat_size;
    uint row = local_idx / p.cols;
    uint col = local_idx % p.cols;
    out[idx] = ((int)col <= (int)row + p.k) ? a[idx] : 0.0f;
}

kernel void triu_f32(
    device const float* a [[buffer(0)]],
    device float* out     [[buffer(1)]],
    constant TrilTriuParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    uint mat_size = p.rows * p.cols;
    uint local_idx = idx % mat_size;
    uint row = local_idx / p.cols;
    uint col = local_idx % p.cols;
    out[idx] = ((int)col - (int)row >= p.k) ? a[idx] : 0.0f;
}

// ---------------------------------------------------------------------------
// Pad: map output indices to input indices, fill value for padding
// ---------------------------------------------------------------------------

struct PadParams {
    uint ndim;
    uint in_total;
    uint out_total;
};

kernel void pad_f32(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant PadParams& p      [[buffer(2)]],
    constant uint* in_shape    [[buffer(3)]],
    constant uint* out_shape   [[buffer(4)]],
    constant uint* pad_before  [[buffer(5)]],
    constant float& pad_value  [[buffer(6)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.out_total) return;

    // Decompose output flat index into coordinates
    uint rem = idx;
    bool in_bounds = true;
    uint in_flat = 0;
    uint in_stride = 1;
    uint out_stride = 1;

    // Pre-compute strides from the end
    // We compute on-the-fly from the last dimension
    for (uint d = p.ndim; d > 0; d--) {
        uint di = d - 1;
        uint out_s = 1;
        uint in_s = 1;
        for (uint j = di + 1; j < p.ndim; j++) {
            out_s *= out_shape[j];
            in_s *= in_shape[j];
        }
        uint coord = (idx / out_s) % out_shape[di];
        int in_coord = (int)coord - (int)pad_before[di];
        if (in_coord < 0 || (uint)in_coord >= in_shape[di]) {
            in_bounds = false;
            break;
        }
        in_flat += (uint)in_coord * in_s;
    }

    output[idx] = in_bounds ? input[in_flat] : pad_value;
}

// ---------------------------------------------------------------------------
// Repeat: map output index back to input index via modulo
// ---------------------------------------------------------------------------

struct RepeatParams {
    uint ndim;
    uint total;
};

kernel void repeat_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant RepeatParams& p    [[buffer(2)]],
    constant uint* in_shape     [[buffer(3)]],
    constant uint* out_shape    [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.total) return;

    uint in_flat = 0;
    for (uint d = p.ndim; d > 0; d--) {
        uint di = d - 1;
        uint out_s = 1;
        uint in_s = 1;
        for (uint j = di + 1; j < p.ndim; j++) {
            out_s *= out_shape[j];
            in_s *= in_shape[j];
        }
        uint coord = (idx / out_s) % out_shape[di];
        uint in_coord = coord % in_shape[di];
        in_flat += in_coord * in_s;
    }

    output[idx] = input[in_flat];
}

// ---------------------------------------------------------------------------
// Axis-aware reduction kernels
// ---------------------------------------------------------------------------

struct ReduceAxisParams {
    uint outer_size;
    uint reduce_size;
    uint inner_size;
};

// Sum along axis
kernel void sum_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = 0.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        acc += input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner];
    }
    output[idx] = acc;
}

// Mean along axis
kernel void mean_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = 0.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        acc += input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner];
    }
    output[idx] = acc / float(p.reduce_size);
}

// Max along axis
kernel void max_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = -INFINITY;
    for (uint r = 0; r < p.reduce_size; r++) {
        acc = max(acc, input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner]);
    }
    output[idx] = acc;
}

// Min along axis
kernel void min_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = INFINITY;
    for (uint r = 0; r < p.reduce_size; r++) {
        acc = min(acc, input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner]);
    }
    output[idx] = acc;
}

// Product along axis
kernel void prod_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = 1.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        acc *= input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner];
    }
    output[idx] = acc;
}

// Argmax along axis (returns float indices)
kernel void argmax_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float best = -INFINITY;
    uint best_r = 0;
    for (uint r = 0; r < p.reduce_size; r++) {
        float val = input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner];
        if (val > best) {
            best = val;
            best_r = r;
        }
    }
    output[idx] = float(best_r);
}

// Argmin along axis (returns float indices)
kernel void argmin_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float best = INFINITY;
    uint best_r = 0;
    for (uint r = 0; r < p.reduce_size; r++) {
        float val = input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner];
        if (val < best) {
            best = val;
            best_r = r;
        }
    }
    output[idx] = float(best_r);
}

// Cumulative sum along axis
// Each thread handles one (outer, inner) pair and writes reduce_size outputs
kernel void cumsum_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = 0.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        uint pos = outer * p.reduce_size * p.inner_size + r * p.inner_size + inner;
        acc += input[pos];
        output[pos] = acc;
    }
}

// Cumulative product along axis
kernel void cumprod_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    float acc = 1.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        uint pos = outer * p.reduce_size * p.inner_size + r * p.inner_size + inner;
        acc *= input[pos];
        output[pos] = acc;
    }
}

// Log-sum-exp along axis (numerically stable)
kernel void logsumexp_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant ReduceAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    // Find max for numerical stability
    float mx = -INFINITY;
    for (uint r = 0; r < p.reduce_size; r++) {
        mx = max(mx, input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner]);
    }
    float acc = 0.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        acc += exp(input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner] - mx);
    }
    output[idx] = mx + log(acc);
}

// Variance along axis
struct VarAxisParams {
    uint outer_size;
    uint reduce_size;
    uint inner_size;
    uint ddof;
};

kernel void var_axis_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant VarAxisParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    if (idx >= p.outer_size * p.inner_size) return;
    uint outer = idx / p.inner_size;
    uint inner = idx % p.inner_size;
    // Compute mean
    float sum = 0.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        sum += input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner];
    }
    float mean = sum / float(p.reduce_size);
    // Compute variance
    float var_acc = 0.0f;
    for (uint r = 0; r < p.reduce_size; r++) {
        float diff = input[outer * p.reduce_size * p.inner_size + r * p.inner_size + inner] - mean;
        var_acc += diff * diff;
    }
    output[idx] = var_acc / float(p.reduce_size - p.ddof);
}

// ---------------------------------------------------------------------------
// QKV reshape: split fused [batch*seq, 3*embed_dim] -> Q, K, V in
// [batch*heads, seq, head_dim] layout. 1 thread per output element.
// ---------------------------------------------------------------------------

struct QkvReshapeParams {
    uint batch;
    uint seq;
    uint heads;
    uint head_dim;
    uint embed_dim;
};

kernel void qkv_reshape_f32(
    device const float* qkv [[buffer(0)]],
    device float* q         [[buffer(1)]],
    device float* k         [[buffer(2)]],
    device float* v         [[buffer(3)]],
    constant QkvReshapeParams& p [[buffer(4)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = p.batch * p.heads * p.seq * p.head_dim;
    if (idx >= total) return;

    // Decompose flat index into (b, h, s, d) in output layout [batch*heads, seq, head_dim]
    uint d = idx % p.head_dim;
    uint rem = idx / p.head_dim;
    uint s = rem % p.seq;
    uint bh = rem / p.seq;
    uint b = bh / p.heads;
    uint h = bh % p.heads;

    // Source index in [batch*seq, 3*embed_dim] layout
    uint src_token = b * p.seq + s;
    uint qkv_stride = 3 * p.embed_dim;
    uint h_offset = h * p.head_dim + d;

    q[idx] = qkv[src_token * qkv_stride + h_offset];
    k[idx] = qkv[src_token * qkv_stride + p.embed_dim + h_offset];
    v[idx] = qkv[src_token * qkv_stride + 2 * p.embed_dim + h_offset];
}

// ---------------------------------------------------------------------------
// RoPE2D: in-place rotate-half on Q or K in [batch*heads, seq, head_dim]
// Takes precomputed cos/sin tables [seq * quarter].
// 1 thread per (batch*head, seq, quarter_idx).
// ---------------------------------------------------------------------------

struct Rope2dParams {
    uint batch_heads;
    uint seq;
    uint head_dim;
    uint quarter;  // head_dim / 4
};

kernel void rope2d_f32(
    device float* data         [[buffer(0)]],
    device const float* cos_y  [[buffer(1)]],
    device const float* sin_y  [[buffer(2)]],
    device const float* cos_x  [[buffer(3)]],
    device const float* sin_x  [[buffer(4)]],
    constant Rope2dParams& p   [[buffer(5)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = p.batch_heads * p.seq * p.quarter;
    if (idx >= total) return;

    uint i = idx % p.quarter;
    uint rem = idx / p.quarter;
    uint s = rem % p.seq;
    uint bh = rem / p.seq;

    uint half_dim = p.quarter * 2;
    uint base = (bh * p.seq + s) * p.head_dim;

    // Y-half: indices (base+i, base+quarter+i)
    uint y_table = s * p.quarter + i;
    float cy = cos_y[y_table];
    float sy = sin_y[y_table];
    float ya = data[base + i];
    float yb = data[base + p.quarter + i];
    data[base + i]            = ya * cy - yb * sy;
    data[base + p.quarter + i] = yb * cy + ya * sy;

    // X-half: indices (base+half_dim+i, base+half_dim+quarter+i)
    uint x_table = s * p.quarter + i;
    float cx = cos_x[x_table];
    float sx = sin_x[x_table];
    float xa = data[base + half_dim + i];
    float xb = data[base + half_dim + p.quarter + i];
    data[base + half_dim + i]            = xa * cx - xb * sx;
    data[base + half_dim + p.quarter + i] = xb * cx + xa * sx;
}

// ---------------------------------------------------------------------------
// Attention output reshape: [batch*heads, seq, head_dim] -> [batch*seq, embed_dim]
// ---------------------------------------------------------------------------

struct AttnReshapeParams {
    uint batch;
    uint seq;
    uint heads;
    uint head_dim;
    uint embed_dim;
};

kernel void attn_output_reshape_f32(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant AttnReshapeParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = p.batch * p.seq * p.embed_dim;
    if (idx >= total) return;

    // Decompose into (b, s, h, d) in output [batch*seq, embed_dim]
    uint d_all = idx % p.embed_dim;
    uint bs = idx / p.embed_dim;
    uint b = bs / p.seq;
    uint s = bs % p.seq;
    uint h = d_all / p.head_dim;
    uint d = d_all % p.head_dim;

    // Source: [batch*heads, seq, head_dim]
    uint src = ((b * p.heads + h) * p.seq + s) * p.head_dim + d;
    output[idx] = input[src];
}

// ---------------------------------------------------------------------------
// Separate reshape: [batch*seq, embed_dim] -> [batch*heads, seq, head_dim]
// Used for cross-attention (separate Q, K, V projections).
// ---------------------------------------------------------------------------

kernel void separate_reshape_f32(
    device const float* input [[buffer(0)]],
    device float* output      [[buffer(1)]],
    constant AttnReshapeParams& p [[buffer(2)]],
    uint idx [[thread_position_in_grid]])
{
    uint total = p.batch * p.heads * p.seq * p.head_dim;
    if (idx >= total) return;

    // Decompose into (b, h, s, d) in output [batch*heads, seq, head_dim]
    uint d = idx % p.head_dim;
    uint rem = idx / p.head_dim;
    uint s = rem % p.seq;
    uint bh = rem / p.seq;
    uint b = bh / p.heads;
    uint h = bh % p.heads;

    // Source: [batch*seq, embed_dim]
    uint src = (b * p.seq + s) * p.embed_dim + h * p.head_dim + d;
    output[idx] = input[src];
}

// ---------------------------------------------------------------------------
// Simdgroup matmul: uses Metal's simdgroup_matrix 8x8 hardware multiply.
// Cooperative threadgroup loads A/B tiles to shared memory, then simdgroups
// load from shared for data reuse. 4 simdgroups per threadgroup (128 threads),
// each threadgroup computes a 32x32 output tile.
// ---------------------------------------------------------------------------

constant uint STILE = 32;   // Threadgroup tile size (output)
constant uint KTILE = 32;   // K-dimension tile (loaded to shared per iteration)

kernel void matmul_simd_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    device const float* bias    [[buffer(3)]],
    constant MatmulParams& p    [[buffer(4)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]])
{
    // Each threadgroup computes a 32x32 tile of C.
    // 4 simdgroups, each computes a 16x16 sub-tile using 2x2 grid of 8x8.
    uint sg_row = simd_id / 2;  // 0 or 1
    uint sg_col = simd_id % 2;  // 0 or 1

    uint base_row = group_id.y * STILE;
    uint base_col = group_id.x * STILE;

    // Shared memory: cooperative load from global, simdgroup_load from here
    threadgroup float As[STILE][KTILE];   // 32 x 32
    threadgroup float Bs[KTILE][STILE];   // 32 x 32

    // 2x2 grid of 8x8 accumulators per simdgroup = 16x16 output
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    uint num_k_tiles = (p.K + KTILE - 1) / KTILE;

    for (uint tk = 0; tk < num_k_tiles; tk++) {
        uint k_base = tk * KTILE;

        // Cooperative load: 128 threads load 32x32 = 1024 elements (8 per thread)
        for (uint idx = tid; idx < STILE * KTILE; idx += 128) {
            uint lr = idx / KTILE;
            uint lc = idx % KTILE;
            uint gr = base_row + lr;
            uint gk = k_base + lc;
            if (gr < p.M && gk < p.K) {
                uint a_idx = p.trans_a ? (gk * p.M + gr) : (gr * p.K + gk);
                As[lr][lc] = A[a_idx];
            } else {
                As[lr][lc] = 0.0f;
            }
        }
        for (uint idx = tid; idx < KTILE * STILE; idx += 128) {
            uint lr = idx / STILE;
            uint lc = idx % STILE;
            uint gk = k_base + lr;
            uint gc = base_col + lc;
            if (gk < p.K && gc < p.N) {
                uint b_idx = p.trans_b ? (gc * p.K + gk) : (gk * p.N + gc);
                Bs[lr][lc] = B[b_idx];
            } else {
                Bs[lr][lc] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Each simdgroup processes its 16x16 sub-tile
        // Walk K in steps of 8 within the loaded KTILE
        for (uint kk = 0; kk < KTILE; kk += 8) {
            // Load 2 8x8 tiles from A shared memory
            simdgroup_float8x8 a_tile[2];
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a_tile[i],
                    (threadgroup float*)&As[sg_row * 16 + i * 8][kk],
                    KTILE);
            }

            // Load 2 8x8 tiles from B shared memory
            simdgroup_float8x8 b_tile[2];
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_tile[j],
                    (threadgroup float*)&Bs[kk][sg_col * 16 + j * 8],
                    STILE);
            }

            // 2x2 multiply-accumulate
            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results — each simdgroup stores its 2x2 grid of 8x8
    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            uint r = base_row + sg_row * 16 + i * 8;
            uint c = base_col + sg_col * 16 + j * 8;
            if (r < p.M && c < p.N) {
                simdgroup_store(acc[i][j], C + r * p.N + c, p.N);
            }
        }
    }

    // Fused bias + relu/gelu
    if (p.fuse_bias || p.fuse_relu || p.fuse_gelu) {
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < STILE * STILE; i += 128) {
            uint lr = i / STILE;
            uint lc = i % STILE;
            uint r = base_row + lr;
            uint c = base_col + lc;
            if (r < p.M && c < p.N) {
                uint idx = r * p.N + c;
                float val = C[idx];
                if (p.fuse_bias) val += bias[c];
                if (p.fuse_relu) val = max(val, 0.0f);
                if (p.fuse_gelu) {
                    float x3 = val * val * val;
                    float inner = 0.7978845608f * (val + 0.044715f * x3);
                    val = 0.5f * val * (1.0f + precise::tanh(inner));
                }
                C[idx] = val;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused bias + GELU: out[i] = gelu(input[i] + bias[i % cols])
// Eliminates a device memory round-trip between bias_add and gelu.
// ---------------------------------------------------------------------------

kernel void bias_gelu_f32(
    device const float* input [[buffer(0)]],
    device const float* bias  [[buffer(1)]],
    device float* out         [[buffer(2)]],
    constant uint& cols       [[buffer(3)]],
    uint idx [[thread_position_in_grid]])
{
    float x = input[idx] + bias[idx % cols];
    float x3 = x * x * x;
    float inner = 0.7978845608f * (x + 0.044715f * x3);
    out[idx] = 0.5f * x * (1.0f + precise::tanh(inner));
}

// ---------------------------------------------------------------------------
// Fused residual add + layernorm: out = layernorm(x + residual)
// Single pass: add, compute mean/var, normalize — eliminates intermediate buffer.
// One threadgroup per row (same structure as layernorm_f32).
// ---------------------------------------------------------------------------

struct AddLayerNormParams {
    uint batch;
    uint dim;
    float eps;
};

kernel void add_layernorm_f32(
    device const float* x        [[buffer(0)]],
    device const float* residual [[buffer(1)]],
    device const float* gamma    [[buffer(2)]],
    device const float* beta     [[buffer(3)]],
    device float* output         [[buffer(4)]],
    constant AddLayerNormParams& p [[buffer(5)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    device const float* row_x   = x + row * p.dim;
    device const float* row_res = residual + row * p.dim;
    device float* row_out       = output + row * p.dim;

    threadgroup float shared[1024];

    // 1. Compute mean of (x + residual)
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        local_sum += row_x[i] + row_res[i];
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mean = shared[0] / float(p.dim);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute variance
    float local_var = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        float val = row_x[i] + row_res[i];
        float diff = val - mean;
        local_var += diff * diff;
    }
    shared[lid] = local_var;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride) shared[lid] += shared[lid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float inv_std = rsqrt(shared[0] / float(p.dim) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Normalize with affine transform
    for (uint i = lid; i < p.dim; i += group_size) {
        float val = row_x[i] + row_res[i];
        row_out[i] = gamma[i] * (val - mean) * inv_std + beta[i];
    }
}

// ---------------------------------------------------------------------------
// Double-buffered simdgroup matmul: overlap next tile load with current compute.
// K-tile reduced to 16 to fit double buffers in 32KB threadgroup memory:
//   2 × As[32][16] + 2 × Bs[16][32] = 2×(32×16 + 16×32)×4 = 2×4096 = 8KB per slot × 2 = 16KB
// Actually fits well within 32KB, leaving room for the simdgroup accumulators.
// ---------------------------------------------------------------------------

constant uint DB_STILE = 32;   // Output tile size (same as single-buffered)
constant uint DB_KTILE = 16;   // K-dimension tile (halved for double-buffering)

kernel void matmul_simd_db_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    device const float* bias    [[buffer(3)]],
    constant MatmulParams& p    [[buffer(4)]],
    uint2 group_id              [[threadgroup_position_in_grid]],
    uint  tid                   [[thread_index_in_threadgroup]],
    uint  simd_id               [[simdgroup_index_in_threadgroup]])
{
    uint sg_row = simd_id / 2;
    uint sg_col = simd_id % 2;

    uint base_row = group_id.y * DB_STILE;
    uint base_col = group_id.x * DB_STILE;

    // Double-buffered shared memory: two slots for A and B tiles
    threadgroup float As[2][DB_STILE][DB_KTILE];   // 2 × 32 × 16
    threadgroup float Bs[2][DB_KTILE][DB_STILE];   // 2 × 16 × 32

    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_float8x8(0.0f);

    uint num_k_tiles = (p.K + DB_KTILE - 1) / DB_KTILE;
    uint cur = 0;  // current buffer slot

    // Pre-load first tile into slot 0
    {
        uint k_base = 0;
        for (uint idx = tid; idx < DB_STILE * DB_KTILE; idx += 128) {
            uint lr = idx / DB_KTILE;
            uint lc = idx % DB_KTILE;
            uint gr = base_row + lr;
            uint gk = k_base + lc;
            if (gr < p.M && gk < p.K) {
                uint a_idx = p.trans_a ? (gk * p.M + gr) : (gr * p.K + gk);
                As[0][lr][lc] = A[a_idx];
            } else {
                As[0][lr][lc] = 0.0f;
            }
        }
        for (uint idx = tid; idx < DB_KTILE * DB_STILE; idx += 128) {
            uint lr = idx / DB_STILE;
            uint lc = idx % DB_STILE;
            uint gk = k_base + lr;
            uint gc = base_col + lc;
            if (gk < p.K && gc < p.N) {
                uint b_idx = p.trans_b ? (gc * p.K + gk) : (gk * p.N + gc);
                Bs[0][lr][lc] = B[b_idx];
            } else {
                Bs[0][lr][lc] = 0.0f;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint tk = 0; tk < num_k_tiles; tk++) {
        uint nxt = 1 - cur;

        // Start loading next tile into alternate slot (if there is a next tile)
        if (tk + 1 < num_k_tiles) {
            uint k_base = (tk + 1) * DB_KTILE;
            for (uint idx = tid; idx < DB_STILE * DB_KTILE; idx += 128) {
                uint lr = idx / DB_KTILE;
                uint lc = idx % DB_KTILE;
                uint gr = base_row + lr;
                uint gk = k_base + lc;
                if (gr < p.M && gk < p.K) {
                    uint a_idx = p.trans_a ? (gk * p.M + gr) : (gr * p.K + gk);
                    As[nxt][lr][lc] = A[a_idx];
                } else {
                    As[nxt][lr][lc] = 0.0f;
                }
            }
            for (uint idx = tid; idx < DB_KTILE * DB_STILE; idx += 128) {
                uint lr = idx / DB_STILE;
                uint lc = idx % DB_STILE;
                uint gk = k_base + lr;
                uint gc = base_col + lc;
                if (gk < p.K && gc < p.N) {
                    uint b_idx = p.trans_b ? (gc * p.K + gk) : (gk * p.N + gc);
                    Bs[nxt][lr][lc] = B[b_idx];
                } else {
                    Bs[nxt][lr][lc] = 0.0f;
                }
            }
        }

        // Compute on current slot — walk K in steps of 8 within DB_KTILE
        for (uint kk = 0; kk < DB_KTILE; kk += 8) {
            simdgroup_float8x8 a_tile[2];
            for (uint i = 0; i < 2; i++) {
                simdgroup_load(a_tile[i],
                    (threadgroup float*)&As[cur][sg_row * 16 + i * 8][kk],
                    DB_KTILE);
            }

            simdgroup_float8x8 b_tile[2];
            for (uint j = 0; j < 2; j++) {
                simdgroup_load(b_tile[j],
                    (threadgroup float*)&Bs[cur][kk][sg_col * 16 + j * 8],
                    DB_STILE);
            }

            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        cur = nxt;
    }

    // Store results
    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            uint r = base_row + sg_row * 16 + i * 8;
            uint c = base_col + sg_col * 16 + j * 8;
            if (r < p.M && c < p.N) {
                simdgroup_store(acc[i][j], C + r * p.N + c, p.N);
            }
        }
    }

    // Fused bias + relu/gelu (same epilogue as single-buffered)
    if (p.fuse_bias || p.fuse_relu || p.fuse_gelu) {
        threadgroup_barrier(mem_flags::mem_device);
        for (uint i = tid; i < DB_STILE * DB_STILE; i += 128) {
            uint lr = i / DB_STILE;
            uint lc = i % DB_STILE;
            uint r = base_row + lr;
            uint c = base_col + lc;
            if (r < p.M && c < p.N) {
                uint idx = r * p.N + c;
                float val = C[idx];
                if (p.fuse_bias) val += bias[c];
                if (p.fuse_relu) val = max(val, 0.0f);
                if (p.fuse_gelu) {
                    float x3 = val * val * val;
                    float inner = 0.7978845608f * (val + 0.044715f * x3);
                    val = 0.5f * val * (1.0f + precise::tanh(inner));
                }
                C[idx] = val;
            }
        }
    }
}
// ---------------------------------------------------------------------------
// Int8 dequantizing matmul: C[M,N] = A_f32[M,K] @ W_i8[K,N]
// W is stored as int8 with per-column f32 scales.
// Loads i8 weights, dequantizes to f32 in registers, accumulates.
// ---------------------------------------------------------------------------

struct DequantMatmulParams {
    uint M;
    uint N;
    uint K;
};

constant uint DQ_TILE = 16;

kernel void matmul_dequant_i8(
    device const float* A            [[buffer(0)]],
    device const char* W_i8          [[buffer(1)]],
    device const float* W_scales     [[buffer(2)]],
    device float* C                  [[buffer(3)]],
    constant DequantMatmulParams& p  [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 lid [[thread_position_in_threadgroup]])
{
    uint row = gid.y;
    uint col = gid.x;

    threadgroup float As[DQ_TILE][DQ_TILE];
    threadgroup float Ws[DQ_TILE][DQ_TILE];

    float sum = 0.0f;
    uint num_tiles = (p.K + DQ_TILE - 1) / DQ_TILE;

    float w_scale = (col < p.N) ? W_scales[col] : 1.0f;

    for (uint t = 0; t < num_tiles; t++) {
        // Load tile of A
        uint a_k = t * DQ_TILE + lid.x;
        if (row < p.M && a_k < p.K) {
            As[lid.y][lid.x] = A[row * p.K + a_k];
        } else {
            As[lid.y][lid.x] = 0.0f;
        }

        // Load tile of W (dequantize i8 -> f32 in registers)
        uint w_k = t * DQ_TILE + lid.y;
        if (w_k < p.K && col < p.N) {
            float qi = float(W_i8[w_k * p.N + col]);
            Ws[lid.y][lid.x] = qi * w_scale;
        } else {
            Ws[lid.y][lid.x] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < DQ_TILE; k++) {
            sum += As[lid.y][k] * Ws[k][lid.x];
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < p.M && col < p.N) {
        C[row * p.N + col] = sum;
    }
}

// Simdgroup variant: 32x32 tiles with hardware 8x8 matrix multiply.
// Dequantizes i8 weights to f32 during tile load.
constant uint DQ_STILE = 32;

kernel void matmul_dequant_simd_i8(
    device const float* A            [[buffer(0)]],
    device const char* W_i8          [[buffer(1)]],
    device const float* W_scales     [[buffer(2)]],
    device float* C                  [[buffer(3)]],
    constant DequantMatmulParams& p  [[buffer(4)]],
    uint2 group_id [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sid [[simdgroup_index_in_threadgroup]])
{
    uint base_row = group_id.y * DQ_STILE;
    uint base_col = group_id.x * DQ_STILE;

    uint sg_row = sid / 2;
    uint sg_col = sid % 2;

    simdgroup_matrix<float, 8, 8> acc[2][2];
    for (uint i = 0; i < 2; i++)
        for (uint j = 0; j < 2; j++)
            acc[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);

    threadgroup float tileA[DQ_STILE][DQ_STILE];
    threadgroup float tileW[DQ_STILE][DQ_STILE];

    uint num_k_tiles = (p.K + DQ_STILE - 1) / DQ_STILE;

    for (uint t = 0; t < num_k_tiles; t++) {
        // Cooperative load of A tile
        for (uint i = tid; i < DQ_STILE * DQ_STILE; i += 128) {
            uint lr = i / DQ_STILE;
            uint lc = i % DQ_STILE;
            uint gr = base_row + lr;
            uint gk = t * DQ_STILE + lc;
            tileA[lr][lc] = (gr < p.M && gk < p.K) ? A[gr * p.K + gk] : 0.0f;
        }

        // Cooperative load of W tile (dequantize i8 -> f32)
        for (uint i = tid; i < DQ_STILE * DQ_STILE; i += 128) {
            uint lr = i / DQ_STILE;
            uint lc = i % DQ_STILE;
            uint gk = t * DQ_STILE + lr;
            uint gc = base_col + lc;
            if (gk < p.K && gc < p.N) {
                float qi = float(W_i8[gk * p.N + gc]);
                tileW[lr][lc] = qi * W_scales[gc];
            } else {
                tileW[lr][lc] = 0.0f;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 2x2 simdgroup matrix multiply from 16x16 sub-tiles
        for (uint tk = 0; tk < DQ_STILE; tk += 8) {
            simdgroup_matrix<float, 8, 8> a_tile[2];
            simdgroup_matrix<float, 8, 8> b_tile[2];

            simdgroup_load(a_tile[0], &tileA[sg_row * 16][tk], DQ_STILE);
            simdgroup_load(a_tile[1], &tileA[sg_row * 16 + 8][tk], DQ_STILE);
            simdgroup_load(b_tile[0], &tileW[tk][sg_col * 16], DQ_STILE);
            simdgroup_load(b_tile[1], &tileW[tk][sg_col * 16 + 8], DQ_STILE);

            for (uint i = 0; i < 2; i++)
                for (uint j = 0; j < 2; j++)
                    simdgroup_multiply_accumulate(acc[i][j], a_tile[i], b_tile[j], acc[i][j]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store results
    for (uint i = 0; i < 2; i++) {
        for (uint j = 0; j < 2; j++) {
            uint r = base_row + sg_row * 16 + i * 8;
            uint c = base_col + sg_col * 16 + j * 8;
            if (r < p.M && c < p.N) {
                simdgroup_store(acc[i][j], C + r * p.N + c, p.N);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Causal-masked softmax (fused into softmax — zero extra memory bandwidth)
// Mask type via bit flags: bit0=causal, bit1=sliding_window, bit2=sink_tokens
// Each threadgroup processes one row of [batch, dim]
// ---------------------------------------------------------------------------

struct CausalMaskSoftmaxParams {
    uint batch;     // number of rows (total_bh * seq_q)
    uint dim;       // row width (seq_kv)
    uint seq_q;     // query sequence length
    uint seq_kv;    // key/value sequence length
    uint mask_type; // bit0=causal, bit1=sliding_window, bit2=sink_tokens
    uint offset;    // position offset for causal masking
    uint window;    // sliding window size (used if bit1 set)
    uint sink_tokens; // number of sink tokens (used if bit2 set)
};

kernel void causal_mask_softmax_f32(
    device const float* input   [[buffer(0)]],
    device float* output        [[buffer(1)]],
    constant CausalMaskSoftmaxParams& p [[buffer(2)]],
    uint gid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint group_size [[threads_per_threadgroup]])
{
    uint row = gid;
    if (row >= p.batch) return;

    // Determine query head and query time position from row index
    // rows are laid out as [bh, seq_q, seq_kv], so row = bh * seq_q + qt
    uint qt = row % p.seq_q;
    uint query_pos = p.offset + qt;

    bool causal    = (p.mask_type & 1u) != 0;
    bool sliding   = (p.mask_type & 2u) != 0;
    bool use_sinks = (p.mask_type & 4u) != 0;

    device const float* row_in = input + row * p.dim;
    device float* row_out = output + row * p.dim;

    threadgroup float shared[1024];

    // 1. Find max (with masking)
    float local_max = -INFINITY;
    for (uint i = lid; i < p.dim; i += group_size) {
        bool masked = false;
        if (causal && i > query_pos) masked = true;
        if (!masked && sliding && !(use_sinks && i < p.sink_tokens)) {
            if (query_pos > i && (query_pos - i) > p.window) masked = true;
        }
        float val = masked ? -INFINITY : row_in[i];
        local_max = max(local_max, val);
    }
    shared[lid] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] = max(shared[lid], shared[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float row_max = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. Compute exp(x - max) and sum (masked positions get 0)
    float local_sum = 0.0f;
    for (uint i = lid; i < p.dim; i += group_size) {
        bool masked = false;
        if (causal && i > query_pos) masked = true;
        if (!masked && sliding && !(use_sinks && i < p.sink_tokens)) {
            if (query_pos > i && (query_pos - i) > p.window) masked = true;
        }
        float e = masked ? 0.0f : exp(row_in[i] - row_max);
        row_out[i] = e;
        local_sum += e;
    }
    shared[lid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = group_size / 2; stride > 0; stride >>= 1) {
        if (lid < stride && lid + stride < group_size) {
            shared[lid] += shared[lid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float total = shared[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 3. Normalize
    for (uint i = lid; i < p.dim; i += group_size) {
        row_out[i] = (total > 0.0f) ? row_out[i] / total : 0.0f;
    }
}
"#;
