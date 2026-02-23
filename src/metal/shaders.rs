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
// Fused matmul + bias + relu: C = relu(A @ B + bias)
// Naive implementation — dispatched as 2D grid [M, N]
// ---------------------------------------------------------------------------

struct MatmulParams {
    uint M;
    uint N;
    uint K;
    uint fuse_bias;   // 0 = no bias, 1 = add bias
    uint fuse_relu;   // 0 = no relu, 1 = relu
};

kernel void matmul_f32(
    device const float* A       [[buffer(0)]],
    device const float* B       [[buffer(1)]],
    device float* C             [[buffer(2)]],
    device const float* bias    [[buffer(3)]],
    constant MatmulParams& p    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint row = gid.y;
    uint col = gid.x;
    if (row >= p.M || col >= p.N) return;

    float sum = 0.0f;
    for (uint k = 0; k < p.K; k++) {
        sum += A[row * p.K + k] * B[k * p.N + col];
    }
    if (p.fuse_bias) {
        sum += bias[col];
    }
    if (p.fuse_relu) {
        sum = max(sum, 0.0f);
    }
    C[row * p.N + col] = sum;
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
    threadgroup float shared[256];

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

    threadgroup float shared[256];

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
    threadgroup float shared[256];

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
    threadgroup float shared[256];

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

    threadgroup float shared[256];

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
"#;
