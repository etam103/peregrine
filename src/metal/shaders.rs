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

kernel void erf_f32(
    device const float* a [[buffer(0)]],
    device float* out      [[buffer(1)]],
    uint idx [[thread_position_in_grid]])
{
    out[idx] = erf(a[idx]);
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
    float tanh_val = tanh(inner_val);
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
"#;
