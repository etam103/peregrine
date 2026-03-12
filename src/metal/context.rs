use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

use super::buffer::GpuBuffer;
use super::pool::BufferPool;
use super::shaders::SHADER_SOURCE;

/// Central Metal GPU context: device, command queue, compiled compute pipelines,
/// and a buffer pool for memory reuse.
///
/// Create once at startup and reuse for all GPU operations.
///
/// ```no_run
/// use peregrine::metal::GpuContext;
/// let gpu = GpuContext::new().expect("No Metal GPU");
/// println!("{}", gpu.device_name());
/// ```
pub struct GpuContext {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pipelines: HashMap<String, Retained<ProtocolObject<dyn MTLComputePipelineState>>>,
    pub(crate) pool: BufferPool,
    pending_cmd: RefCell<Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>>,
    event: Retained<ProtocolObject<dyn MTLSharedEvent>>,
    event_counter: Cell<u64>,
}

/// GPU attention mask enum (Rust-side mirror of Metal kernel bit flags).
#[derive(Clone, Debug)]
pub enum GpuAttentionMask {
    /// No masking (plain softmax).
    None,
    /// Causal mask only.
    Causal { offset: usize },
    /// Causal + sliding window + optional sink tokens.
    CausalSlidingWindow { offset: usize, window: usize, sink_tokens: usize },
}

impl GpuContext {
    /// Initialize Metal: get default device, compile shaders, create pipelines.
    pub fn new() -> Result<Self, String> {
        let device = MTLCreateSystemDefaultDevice()
            .ok_or_else(|| "No Metal device found".to_string())?;

        let queue = device
            .newCommandQueue()
            .ok_or_else(|| "Failed to create command queue".to_string())?;

        let source = NSString::from_str(SHADER_SOURCE);
        let library = device
            .newLibraryWithSource_options_error(&source, None)
            .map_err(|e| format!("Shader compilation failed: {e}"))?;

        // Build pipeline states for all kernel functions
        let kernel_names = [
            // Binary ops
            "add_f32", "sub_f32", "mul_f32", "div_f32",
            // Unary ops
            "neg_f32", "exp_f32", "log_f32", "sqrt_f32",
            "relu_f32", "sigmoid_f32", "tanh_f32", "gelu_f32",
            "sin_f32", "cos_f32", "abs_f32",
            "reciprocal_f32", "square_f32", "rsqrt_f32",
            "floor_f32", "ceil_f32", "round_f32", "sign_f32",
            "expm1_f32", "log2_f32", "log10_f32", "log1p_f32",
            "erf_f32", "erfinv_f32",
            "sinh_f32", "cosh_f32",
            "arcsin_f32", "arccos_f32", "arctan_f32",
            "arcsinh_f32", "arccosh_f32", "arctanh_f32",
            // Other
            "scale_f32", "matmul_f32", "sum_f32", "max_f32", "min_f32",
            "softmax_f32", "transpose_f32", "layernorm_f32",
            // Backward kernels
            "relu_backward_f32", "sigmoid_backward_f32", "tanh_backward_f32",
            "gelu_backward_f32", "accumulate_f32", "fill_f32",
            "softmax_backward_f32", "layernorm_backward_f32", "adam_step_f32",
            // New forward/backward kernels
            "bias_add_f32", "bias_grad_sum_f32",
            "log_softmax_f32", "log_softmax_backward_f32",
            "scale_fill_f32",
            "gather_f32", "scatter_add_f32",
            // Phase 1B: Binary math ops
            "maximum_f32", "minimum_f32", "power_f32", "arctan2_f32", "logaddexp_f32",
            // Phase 1C: Clip, Where, NanToNum
            "clip_f32", "where_f32", "nan_to_num_f32",
            // Phase 1D: Comparison / Logical ops (binary)
            "equal_f32", "not_equal_f32", "greater_f32", "greater_equal_f32",
            "less_f32", "less_equal_f32", "logical_and_f32", "logical_or_f32",
            // Phase 1D: Comparison / Logical ops (unary)
            "isnan_f32", "isinf_f32", "isfinite_f32", "logical_not_f32",
            // Activation kernels
            "leaky_relu_f32", "leaky_relu_backward_f32",
            "elu_f32", "elu_backward_f32",
            // Shape manipulation kernels
            "tril_f32", "triu_f32", "pad_f32", "repeat_f32",
            // Axis-aware reduction kernels
            "sum_axis_f32", "mean_axis_f32", "max_axis_f32", "min_axis_f32",
            "prod_axis_f32", "argmax_axis_f32", "argmin_axis_f32",
            "cumsum_f32", "cumprod_f32",
            "logsumexp_axis_f32", "var_axis_f32",
            // Simdgroup matmul
            "matmul_simd_f32",
            // Double-buffered simdgroup matmul
            "matmul_simd_db_f32",
            // Fused kernels
            "bias_gelu_f32", "add_layernorm_f32",
            // Attention reshape + RoPE kernels
            "qkv_reshape_f32", "rope2d_f32",
            "attn_output_reshape_f32", "separate_reshape_f32",
            // Int8 dequant matmul
            "matmul_dequant_i8", "matmul_dequant_simd_i8",
            // Causal-masked softmax
            "causal_mask_softmax_f32",
            // 2:4 structured sparse matmul
            "matmul_sparse_24", "matmul_sparse_simd_24",
        ];

        let mut pipelines = HashMap::new();
        for name in &kernel_names {
            let ns_name = NSString::from_str(name);
            let function = library
                .newFunctionWithName(&ns_name)
                .ok_or_else(|| format!("Kernel function '{}' not found", name))?;
            let pipeline = device
                .newComputePipelineStateWithFunction_error(&function)
                .map_err(|e| format!("Pipeline creation failed for '{}': {}", name, e))?;
            pipelines.insert(name.to_string(), pipeline);
        }

        let pool = BufferPool::new(&device);

        let event = device.newSharedEvent()
            .ok_or_else(|| "Failed to create MTLSharedEvent".to_string())?;

        Ok(GpuContext { device, queue, pipelines, pool, pending_cmd: RefCell::new(None), event, event_counter: Cell::new(0) })
    }

    /// Reference to the underlying MTLDevice.
    pub fn device(&self) -> &ProtocolObject<dyn MTLDevice> {
        &self.device
    }

    /// Device name (e.g. "Apple M2 Max").
    pub fn device_name(&self) -> String {
        self.device.name().to_string()
    }

    /// Whether this device has unified memory (always true on Apple Silicon).
    pub fn has_unified_memory(&self) -> bool {
        self.device.hasUnifiedMemory()
    }

    /// Recommended max working set size in bytes.
    pub fn max_working_set_size(&self) -> u64 {
        self.device.recommendedMaxWorkingSetSize()
    }

    /// Max threads per threadgroup for a given kernel.
    pub fn max_threads_per_threadgroup(&self, kernel: &str) -> usize {
        self.pipelines[kernel].maxTotalThreadsPerThreadgroup() as usize
    }

    /// Allocate a GPU buffer with `len` elements.
    pub fn alloc<T: Copy>(&self, len: usize) -> GpuBuffer<T> {
        GpuBuffer::new(&self.device, len)
    }

    /// Create a GPU buffer from a CPU slice.
    pub fn upload<T: Copy>(&self, data: &[T]) -> GpuBuffer<T> {
        GpuBuffer::from_slice(&self.device, data)
    }

    /// Allocate a GPU buffer from the pool.
    pub fn alloc_from_pool(&mut self, len: usize) -> GpuBuffer<f32> {
        let byte_size = len * std::mem::size_of::<f32>();
        let raw = self.pool.get(byte_size);
        GpuBuffer::from_raw(raw, len)
    }

    // --- Command batching ---

    /// Lazily create or return the pending command buffer.
    fn ensure_cmd(&self) -> std::cell::RefMut<'_, Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>> {
        let mut cmd = self.pending_cmd.borrow_mut();
        if cmd.is_none() {
            *cmd = Some(self.queue.commandBuffer().expect("command buffer"));
        }
        cmd
    }

    /// Commit the pending command buffer and wait for completion.
    /// No-op if no commands are pending.
    pub fn sync(&self) {
        let cmd = self.pending_cmd.borrow_mut().take();
        if let Some(cmd) = cmd {
            cmd.commit();
            cmd.waitUntilCompleted();
        }
    }

    /// Commit the pending command buffer with a signal event (non-blocking).
    /// Returns a ticket that can be used with `wait_for()` or `is_done()`.
    /// Unlike `sync()`, this does NOT wait for GPU completion — the CPU can
    /// do other work while the GPU executes.
    pub fn commit_and_signal(&self) -> u64 {
        let ticket = self.event_counter.get() + 1;
        self.event_counter.set(ticket);

        let cmd = self.pending_cmd.borrow_mut().take();
        if let Some(cmd) = cmd {
            // Upcast MTLSharedEvent → MTLEvent for the command buffer API
            let event_ref: &ProtocolObject<dyn MTLEvent> =
                ProtocolObject::from_ref(&*self.event);
            cmd.encodeSignalEvent_value(event_ref, ticket);
            cmd.commit();
        } else {
            // No pending work — signal immediately
            self.event.setSignaledValue(ticket);
        }
        ticket
    }

    /// Spin-wait until the GPU has signaled the given ticket.
    pub fn wait_for(&self, ticket: u64) {
        while self.event.signaledValue() < ticket {
            std::hint::spin_loop();
        }
    }

    /// Check whether the GPU has completed work for the given ticket (non-blocking).
    pub fn is_done(&self, ticket: u64) -> bool {
        self.event.signaledValue() >= ticket
    }

    // --- Dispatch helpers ---

    /// Dispatch a binary element-wise op: out[i] = op(a[i], b[i]).
    pub fn dispatch_binary(
        &self,
        kernel: &str,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
    ) {
        let n = a.len();
        assert_eq!(n, b.len());
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines[kernel];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch a unary element-wise op: out[i] = op(a[i]).
    pub fn dispatch_unary(
        &self,
        kernel: &str,
        a: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
    ) {
        let n = a.len();
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines[kernel];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 1);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch a backward unary op: out[i] = kernel(cached[i], grad[i]).
    /// Used for relu_backward, sigmoid_backward, tanh_backward, gelu_backward.
    pub fn dispatch_backward_unary(
        &self,
        kernel: &str,
        cached: &GpuBuffer<f32>,
        grad: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
    ) {
        let n = cached.len();
        assert_eq!(n, grad.len());
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines[kernel];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(cached.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch a unary element-wise op with one scalar parameter: out[i] = op(a[i], param).
    /// Used for leaky_relu, elu forward kernels.
    pub fn dispatch_unary_param(
        &self,
        kernel: &str,
        a: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        param: f32,
    ) {
        let n = a.len();
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines[kernel];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&param as *const _ as *mut _).unwrap(),
                std::mem::size_of::<f32>(),
                2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch a backward unary op with one scalar parameter: out[i] = kernel(cached[i], grad[i], param).
    /// Used for leaky_relu_backward, elu_backward.
    pub fn dispatch_backward_unary_param(
        &self,
        kernel: &str,
        cached: &GpuBuffer<f32>,
        grad: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        param: f32,
    ) {
        let n = cached.len();
        assert_eq!(n, grad.len());
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines[kernel];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(cached.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&param as *const _ as *mut _).unwrap(),
                std::mem::size_of::<f32>(),
                3,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// In-place accumulate: a[i] += b[i].
    pub fn dispatch_accumulate(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
    ) {
        let n = a.len();
        assert_eq!(n, b.len());

        let pipeline = &self.pipelines["accumulate_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Fill a buffer with a scalar value.
    pub fn dispatch_fill(
        &self,
        out: &GpuBuffer<f32>,
        value: f32,
    ) {
        let n = out.len();
        let pipeline = &self.pipelines["fill_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 0);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&value as *const _ as *mut _).unwrap(),
                std::mem::size_of::<f32>(),
                1,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch scale: out[i] = a[i] * scalar.
    pub fn dispatch_scale(
        &self,
        a: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        scalar: f32,
    ) {
        let n = a.len();
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines["scale_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&scalar as *const _ as *mut _).unwrap(),
                std::mem::size_of::<f32>(),
                2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch matmul with automatic kernel selection (legacy — no GELU fusion).
    /// Delegates to `dispatch_matmul_fused` with `fuse_gelu = false`.
    pub fn dispatch_matmul_auto(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
        trans_a: bool,
        trans_b: bool,
    ) {
        self.dispatch_matmul_fused(a, b, c, bias, m, n, k, fuse_relu, false, trans_a, trans_b);
    }

    /// Dispatch matmul with fusion-aware kernel selection.
    /// Routes to the best kernel based on matrix size and requested fusions:
    /// - Large matrices (M,N >= 64, K >= 64, M*N >= 1M): double-buffered simdgroup kernel
    /// - Large matrices (M,N >= 64, M*N >= 1M, K < 64): single-buffered simdgroup kernel
    /// - Small matrices: scalar 16x16 tiled kernel
    /// All kernel variants support fused bias + relu/gelu epilogue.
    pub fn dispatch_matmul_fused(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
        fuse_gelu: bool,
        trans_a: bool,
        trans_b: bool,
    ) {
        let large_enough = m >= 64 && n >= 64
            && (m as u64) * (n as u64) >= 1024 * 1024;

        if large_enough {
            // Simdgroup kernel with fused epilogue for large matrices.
            // Note: dispatch_matmul_simd_db is available for explicit use but
            // benchmarks show single-buffered performs equivalently on Apple Silicon
            // unified memory, so we default to the simpler single-buffered kernel.
            self.dispatch_matmul_simd(a, b, c, bias, m, n, k, fuse_relu, fuse_gelu, trans_a, trans_b);
        } else {
            self.dispatch_matmul_scalar(a, b, c, bias, m, n, k, fuse_relu, trans_a, trans_b, fuse_gelu);
        }
    }

    /// Dispatch matmul: C = op(A)[M,K] @ op(B)[K,N], optionally fused with bias and relu/gelu.
    /// trans_a/trans_b control whether A and B are transposed.
    /// Scalar 16x16 tiled kernel — works at all sizes, supports fused bias/relu/gelu.
    pub fn dispatch_matmul_scalar(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
        trans_a: bool,
        trans_b: bool,
        fuse_gelu: bool,
    ) {
        #[repr(C)]
        struct MatmulParams {
            m: u32,
            n: u32,
            k: u32,
            fuse_bias: u32,
            fuse_relu: u32,
            trans_a: u32,
            trans_b: u32,
            fuse_gelu: u32,
        }

        let params = MatmulParams {
            m, n, k,
            fuse_bias: if bias.is_some() { 1 } else { 0 },
            fuse_relu: if fuse_relu { 1 } else { 0 },
            trans_a: if trans_a { 1 } else { 0 },
            trans_b: if trans_b { 1 } else { 0 },
            fuse_gelu: if fuse_gelu { 1 } else { 0 },
        };

        let pipeline = &self.pipelines["matmul_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(c.raw()), 0, 2);
            if let Some(bias_buf) = bias {
                enc.setBuffer_offset_atIndex(Some(bias_buf.raw()), 0, 3);
            }
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<MatmulParams>(),
                4,
            );
        }

        // Tiled matmul: always TILE_SIZE x TILE_SIZE threadgroups
        // Kernel handles out-of-bounds threads via bounds checking
        let tile_size = 16usize;
        let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
        let num_groups = MTLSize {
            width: (n as usize + tile_size - 1) / tile_size,
            height: (m as usize + tile_size - 1) / tile_size,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch matmul with automatic kernel selection (convenience alias).
    pub fn dispatch_matmul(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
        trans_a: bool,
        trans_b: bool,
    ) {
        self.dispatch_matmul_fused(a, b, c, bias, m, n, k, fuse_relu, false, trans_a, trans_b);
    }

    /// Dispatch simdgroup matmul: uses hardware 8x8 simdgroup_matrix_multiply.
    /// Each threadgroup has 4 simdgroups (128 threads), computing a 32x32 output tile.
    /// Supports fused bias + relu/gelu epilogue.
    pub fn dispatch_matmul_simd(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
        fuse_gelu: bool,
        trans_a: bool,
        trans_b: bool,
    ) {
        #[repr(C)]
        struct MatmulParams {
            m: u32,
            n: u32,
            k: u32,
            fuse_bias: u32,
            fuse_relu: u32,
            trans_a: u32,
            trans_b: u32,
            fuse_gelu: u32,
        }

        let params = MatmulParams {
            m, n, k,
            fuse_bias: if bias.is_some() { 1 } else { 0 },
            fuse_relu: if fuse_relu { 1 } else { 0 },
            trans_a: if trans_a { 1 } else { 0 },
            trans_b: if trans_b { 1 } else { 0 },
            fuse_gelu: if fuse_gelu { 1 } else { 0 },
        };

        let pipeline = &self.pipelines["matmul_simd_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(c.raw()), 0, 2);
            if let Some(bias_buf) = bias {
                enc.setBuffer_offset_atIndex(Some(bias_buf.raw()), 0, 3);
            }
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<MatmulParams>(),
                4,
            );
        }

        // 4 simdgroups of 32 threads = 128 threads per threadgroup
        // Each threadgroup covers 32x32 output tile
        let threads_per_group = 128usize;
        let tile = 32usize;
        let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
        let num_groups = MTLSize {
            width: (n as usize + tile - 1) / tile,
            height: (m as usize + tile - 1) / tile,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch double-buffered simdgroup matmul: overlaps next tile load with current compute.
    /// K-tile = 16 (halved from 32) to fit double buffers in 32KB threadgroup memory.
    /// Same interface and epilogue as `dispatch_matmul_simd`.
    pub fn dispatch_matmul_simd_db(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
        fuse_gelu: bool,
        trans_a: bool,
        trans_b: bool,
    ) {
        #[repr(C)]
        struct MatmulParams {
            m: u32,
            n: u32,
            k: u32,
            fuse_bias: u32,
            fuse_relu: u32,
            trans_a: u32,
            trans_b: u32,
            fuse_gelu: u32,
        }

        let params = MatmulParams {
            m, n, k,
            fuse_bias: if bias.is_some() { 1 } else { 0 },
            fuse_relu: if fuse_relu { 1 } else { 0 },
            trans_a: if trans_a { 1 } else { 0 },
            trans_b: if trans_b { 1 } else { 0 },
            fuse_gelu: if fuse_gelu { 1 } else { 0 },
        };

        let pipeline = &self.pipelines["matmul_simd_db_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(b.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(c.raw()), 0, 2);
            if let Some(bias_buf) = bias {
                enc.setBuffer_offset_atIndex(Some(bias_buf.raw()), 0, 3);
            }
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<MatmulParams>(),
                4,
            );
        }

        let threads_per_group = 128usize;
        let tile = 32usize;
        let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
        let num_groups = MTLSize {
            width: (n as usize + tile - 1) / tile,
            height: (m as usize + tile - 1) / tile,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch fused bias + GELU: out[i] = gelu(input[i] + bias[i % cols]).
    pub fn dispatch_bias_gelu(
        &self,
        input: &GpuBuffer<f32>,
        bias: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        cols: u32,
    ) {
        let n = input.len();
        assert_eq!(n, out.len());
        assert_eq!(bias.len(), cols as usize);

        let pipeline = &self.pipelines["bias_gelu_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(bias.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&cols as *const _ as *mut _).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch fused residual add + layernorm: out = layernorm(x + residual).
    /// Input shapes: x=[batch, dim], residual=[batch, dim].
    pub fn dispatch_add_layernorm(
        &self,
        x: &GpuBuffer<f32>,
        residual: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32, dim: u32, eps: f32,
    ) {
        #[repr(C)]
        struct AddLayerNormParams { batch: u32, dim: u32, eps: f32 }
        let params = AddLayerNormParams { batch, dim, eps };

        let pipeline = &self.pipelines["add_layernorm_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(x.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(residual.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(gamma.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(beta.raw()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 4);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<AddLayerNormParams>(), 5,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );
        enc.endEncoding();
    }

    /// Dispatch softmax over rows: out[b][i] = softmax(in[b][:])[i].
    pub fn dispatch_softmax(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32,
        dim: u32,
    ) {
        #[repr(C)]
        struct SoftmaxParams {
            batch: u32,
            dim: u32,
        }

        let params = SoftmaxParams { batch, dim };

        let pipeline = &self.pipelines["softmax_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<SoftmaxParams>(),
                2,
            );
        }

        // One threadgroup per row — must be power of 2 for tree reduction
        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );

        enc.endEncoding();
    }

    /// Dispatch softmax backward.
    pub fn dispatch_softmax_backward(
        &self,
        softmax_out: &GpuBuffer<f32>,
        grad: &GpuBuffer<f32>,
        grad_input: &GpuBuffer<f32>,
        batch: u32,
        dim: u32,
    ) {
        #[repr(C)]
        struct SoftmaxParams {
            batch: u32,
            dim: u32,
        }

        let params = SoftmaxParams { batch, dim };

        let pipeline = &self.pipelines["softmax_backward_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(softmax_out.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(grad_input.raw()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<SoftmaxParams>(),
                3,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );

        enc.endEncoding();
    }

    /// Dispatch a reduction op (sum/max/min). Returns a single f32 result.
    pub fn dispatch_reduce(&self, kernel: &str, input: &GpuBuffer<f32>) -> f32 {
        let n = input.len();
        let pipeline = &self.pipelines[kernel];
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let group_size = max_threads.min(1024).min(n.next_power_of_two());
        let num_groups = (n + group_size - 1) / group_size;

        let partial: GpuBuffer<f32> = GpuBuffer::new(&self.device, num_groups);
        let count = n as u32;

        {
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");
            enc.setComputePipelineState(pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(partial.raw()), 0, 1);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&count as *const _ as *mut _).unwrap(),
                    4, 2,
                );
            }
            let grid = MTLSize { width: num_groups * group_size, height: 1, depth: 1 };
            let tg = MTLSize { width: group_size, height: 1, depth: 1 };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        } // drop RefMut guard before sync

        self.sync();

        // CPU-side final reduction of partial sums
        let partials = partial.read();
        match kernel {
            "sum_f32" => partials.iter().sum(),
            "max_f32" => partials.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            "min_f32" => partials.iter().cloned().fold(f32::INFINITY, f32::min),
            _ => partials.iter().sum(),
        }
    }

    /// Dispatch transpose: out[j,i] = input[i,j] for [rows, cols] matrix.
    pub fn dispatch_transpose(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        rows: u32, cols: u32,
    ) {
        #[repr(C)]
        struct TransposeParams { rows: u32, cols: u32 }
        let params = TransposeParams { rows, cols };

        let pipeline = &self.pipelines["transpose_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<TransposeParams>(), 2,
            );
        }

        let grid = MTLSize { width: cols as usize, height: rows as usize, depth: 1 };
        let tg = MTLSize { width: 16usize.min(cols as usize), height: 16usize.min(rows as usize), depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
    }

    /// Dispatch layernorm: y = gamma * (x - mean) / sqrt(var + eps) + beta.
    /// Input shape: [batch, dim].
    pub fn dispatch_layernorm(
        &self,
        input: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        beta: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32, dim: u32, eps: f32,
    ) {
        #[repr(C)]
        struct LayerNormParams { batch: u32, dim: u32, eps: f32 }
        let params = LayerNormParams { batch, dim, eps };

        let pipeline = &self.pipelines["layernorm_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(gamma.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(beta.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<LayerNormParams>(), 4,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );
        enc.endEncoding();
    }

    /// Dispatch layernorm backward.
    pub fn dispatch_layernorm_backward(
        &self,
        grad_out: &GpuBuffer<f32>,
        normalized: &GpuBuffer<f32>,
        gamma: &GpuBuffer<f32>,
        inv_std: &GpuBuffer<f32>,
        grad_input: &GpuBuffer<f32>,
        grad_gamma: &GpuBuffer<f32>,
        grad_beta: &GpuBuffer<f32>,
        batch: u32, dim: u32,
    ) {
        #[repr(C)]
        struct LayerNormBackwardParams { batch: u32, dim: u32 }
        let params = LayerNormBackwardParams { batch, dim };

        let pipeline = &self.pipelines["layernorm_backward_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(grad_out.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(normalized.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(gamma.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(inv_std.raw()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(grad_input.raw()), 0, 4);
            enc.setBuffer_offset_atIndex(Some(grad_gamma.raw()), 0, 5);
            enc.setBuffer_offset_atIndex(Some(grad_beta.raw()), 0, 6);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<LayerNormBackwardParams>(), 7,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );
        enc.endEncoding();
    }

    /// Dispatch fused Adam optimizer step.
    pub fn dispatch_adam_step(
        &self,
        data: &GpuBuffer<f32>,
        grad: &GpuBuffer<f32>,
        m: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        lr: f32, beta1: f32, beta2: f32, eps: f32,
        bc1: f32, bc2: f32,
        weight_decay: f32, decoupled_wd: bool,
    ) {
        #[repr(C)]
        struct AdamParams {
            lr: f32,
            beta1: f32,
            beta2: f32,
            eps: f32,
            bc1: f32,
            bc2: f32,
            weight_decay: f32,
            decoupled_wd: u32,
        }

        let params = AdamParams {
            lr, beta1, beta2, eps, bc1, bc2, weight_decay,
            decoupled_wd: if decoupled_wd { 1 } else { 0 },
        };

        let n = data.len();
        let pipeline = &self.pipelines["adam_step_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(m.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(v.raw()), 0, 3);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<AdamParams>(), 4,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    /// Dispatch bias add: out[i] = input[i] + bias[i % cols].
    pub fn dispatch_bias_add(
        &self,
        input: &GpuBuffer<f32>,
        bias: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        cols: u32,
    ) {
        let n = input.len();
        assert_eq!(n, out.len());
        assert_eq!(bias.len(), cols as usize);

        let pipeline = &self.pipelines["bias_add_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(bias.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&cols as *const _ as *mut _).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch bias gradient sum: bias_grad[col] = sum_r grad[r, col].
    pub fn dispatch_bias_grad_sum(
        &self,
        grad: &GpuBuffer<f32>,
        bias_grad: &GpuBuffer<f32>,
        rows: u32,
        cols: u32,
    ) {
        assert_eq!(grad.len(), (rows * cols) as usize);
        assert_eq!(bias_grad.len(), cols as usize);

        let pipeline = &self.pipelines["bias_grad_sum_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(bias_grad.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&rows as *const _ as *mut _).unwrap(),
                std::mem::size_of::<u32>(),
                2,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&cols as *const _ as *mut _).unwrap(),
                std::mem::size_of::<u32>(),
                3,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(cols as usize), height: 1, depth: 1 };
        let grid_size = MTLSize { width: cols as usize, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch log-softmax over rows: out[b][i] = log_softmax(in[b][:])[i].
    pub fn dispatch_log_softmax(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32,
        dim: u32,
    ) {
        #[repr(C)]
        struct SoftmaxParams {
            batch: u32,
            dim: u32,
        }

        let params = SoftmaxParams { batch, dim };

        let pipeline = &self.pipelines["log_softmax_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<SoftmaxParams>(),
                2,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );

        enc.endEncoding();
    }

    /// Dispatch log-softmax backward.
    pub fn dispatch_log_softmax_backward(
        &self,
        log_softmax_out: &GpuBuffer<f32>,
        grad: &GpuBuffer<f32>,
        grad_input: &GpuBuffer<f32>,
        batch: u32,
        dim: u32,
    ) {
        #[repr(C)]
        struct SoftmaxParams {
            batch: u32,
            dim: u32,
        }

        let params = SoftmaxParams { batch, dim };

        let pipeline = &self.pipelines["log_softmax_backward_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(log_softmax_out.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(grad_input.raw()), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<SoftmaxParams>(),
                3,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );

        enc.endEncoding();
    }

    /// Dispatch scale-fill: dst[i] = src[0] * scalar.
    pub fn dispatch_scale_fill(
        &self,
        src: &GpuBuffer<f32>,
        dst: &GpuBuffer<f32>,
        scalar: f32,
    ) {
        let n = dst.len();

        let pipeline = &self.pipelines["scale_fill_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(src.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(dst.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&scalar as *const _ as *mut _).unwrap(),
                std::mem::size_of::<f32>(),
                2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch gather: out[i] = input[indices[i]].
    pub fn dispatch_gather(
        &self,
        input: &GpuBuffer<f32>,
        indices: &GpuBuffer<u32>,
        out: &GpuBuffer<f32>,
    ) {
        let n = indices.len();
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines["gather_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(indices.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 2);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch an axis-aware reduction: one thread per output element.
    /// Kernels: sum_axis_f32, mean_axis_f32, max_axis_f32, min_axis_f32,
    /// prod_axis_f32, argmax_axis_f32, argmin_axis_f32, logsumexp_axis_f32,
    /// cumsum_f32, cumprod_f32.
    pub fn dispatch_reduce_axis(
        &self,
        kernel: &str,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        outer_size: u32,
        reduce_size: u32,
        inner_size: u32,
    ) {
        #[repr(C)]
        struct ReduceAxisParams {
            outer_size: u32,
            reduce_size: u32,
            inner_size: u32,
        }

        let params = ReduceAxisParams { outer_size, reduce_size, inner_size };

        // For cumsum/cumprod, output has same size as input (not reduced)
        let grid_n = if kernel == "cumsum_f32" || kernel == "cumprod_f32" {
            (outer_size * inner_size) as usize
        } else {
            output.len()
        };

        let pipeline = &self.pipelines[kernel];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<ReduceAxisParams>(),
                2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(grid_n.max(1)), height: 1, depth: 1 };
        let grid_size = MTLSize { width: grid_n.max(1), height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch variance along axis with ddof parameter.
    pub fn dispatch_var_axis(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        outer_size: u32,
        reduce_size: u32,
        inner_size: u32,
        ddof: u32,
    ) {
        #[repr(C)]
        struct VarAxisParams {
            outer_size: u32,
            reduce_size: u32,
            inner_size: u32,
            ddof: u32,
        }

        let params = VarAxisParams { outer_size, reduce_size, inner_size, ddof };

        let pipeline = &self.pipelines["var_axis_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<VarAxisParams>(),
                2,
            );
        }

        let n = output.len();
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n.max(1)), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n.max(1), height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch scatter-add: grad_input[indices[i]] += grad[i].
    /// `grad_input` must be pre-zeroed.
    pub fn dispatch_scatter_add(
        &self,
        grad: &GpuBuffer<f32>,
        indices: &GpuBuffer<u32>,
        grad_input: &GpuBuffer<f32>,
    ) {
        let n = indices.len();
        assert_eq!(n, grad.len());

        let pipeline = &self.pipelines["scatter_add_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(grad.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(indices.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(grad_input.raw()), 0, 2);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch clip: out[i] = clamp(a[i], min_val, max_val).
    pub fn dispatch_clip(
        &self,
        a: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        min_val: f32,
        max_val: f32,
    ) {
        let n = a.len();
        assert_eq!(n, out.len());

        #[repr(C)]
        struct ClipParams {
            min_val: f32,
            max_val: f32,
        }

        let params = ClipParams { min_val, max_val };

        let pipeline = &self.pipelines["clip_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<ClipParams>(),
                2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch ternary where: out[i] = (cond[i] != 0) ? x[i] : y[i].
    pub fn dispatch_ternary(
        &self,
        cond: &GpuBuffer<f32>,
        x: &GpuBuffer<f32>,
        y: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
    ) {
        let n = cond.len();
        assert_eq!(n, x.len());
        assert_eq!(n, y.len());
        assert_eq!(n, out.len());

        let pipeline = &self.pipelines["where_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(cond.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(x.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(y.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 3);
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch nan_to_num: replace NaN/Inf with specified values.
    pub fn dispatch_nan_to_num(
        &self,
        a: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        nan_val: f32,
        posinf_val: f32,
        neginf_val: f32,
    ) {
        let n = a.len();
        assert_eq!(n, out.len());

        #[repr(C)]
        struct NanToNumParams {
            nan_val: f32,
            posinf_val: f32,
            neginf_val: f32,
        }

        let params = NanToNumParams { nan_val, posinf_val, neginf_val };

        let pipeline = &self.pipelines["nan_to_num_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<NanToNumParams>(),
                2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
    }

    /// Dispatch tril: zero elements above the k-th diagonal.
    pub fn dispatch_tril(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        rows: u32, cols: u32, k: i32,
    ) {
        #[repr(C)]
        struct TrilTriuParams { rows: u32, cols: u32, k: i32 }
        let params = TrilTriuParams { rows, cols, k };
        let n = input.len();

        let pipeline = &self.pipelines["tril_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<TrilTriuParams>(), 2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    /// Dispatch triu: zero elements below the k-th diagonal.
    pub fn dispatch_triu(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        rows: u32, cols: u32, k: i32,
    ) {
        #[repr(C)]
        struct TrilTriuParams { rows: u32, cols: u32, k: i32 }
        let params = TrilTriuParams { rows, cols, k };
        let n = input.len();

        let pipeline = &self.pipelines["triu_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<TrilTriuParams>(), 2,
            );
        }

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(n), height: 1, depth: 1 };
        let grid_size = MTLSize { width: n, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    // --- Attention reshape + RoPE dispatch methods ---

    /// Split fused QKV [batch*seq, 3*embed_dim] -> Q, K, V in [batch*heads, seq, head_dim].
    pub fn dispatch_qkv_reshape(
        &self,
        qkv: &GpuBuffer<f32>,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        batch: u32, seq: u32, heads: u32, head_dim: u32, embed_dim: u32,
    ) {
        #[repr(C)]
        struct QkvReshapeParams { batch: u32, seq: u32, heads: u32, head_dim: u32, embed_dim: u32 }
        let params = QkvReshapeParams { batch, seq, heads, head_dim, embed_dim };
        let total = (batch * heads * seq * head_dim) as usize;

        let pipeline = &self.pipelines["qkv_reshape_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(qkv.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(q.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(k.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(v.raw()), 0, 3);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<QkvReshapeParams>(), 4,
            );
        }
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(total), height: 1, depth: 1 };
        let grid_size = MTLSize { width: total, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    /// In-place 2D RoPE on Q or K in [batch*heads, seq, head_dim] layout.
    pub fn dispatch_rope2d(
        &self,
        data: &GpuBuffer<f32>,
        cos_y: &GpuBuffer<f32>,
        sin_y: &GpuBuffer<f32>,
        cos_x: &GpuBuffer<f32>,
        sin_x: &GpuBuffer<f32>,
        batch_heads: u32, seq: u32, head_dim: u32,
    ) {
        #[repr(C)]
        struct Rope2dParams { batch_heads: u32, seq: u32, head_dim: u32, quarter: u32 }
        let quarter = head_dim / 4;
        let params = Rope2dParams { batch_heads, seq, head_dim, quarter };
        let total = (batch_heads * seq * quarter) as usize;

        let pipeline = &self.pipelines["rope2d_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(data.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(cos_y.raw()), 0, 1);
            enc.setBuffer_offset_atIndex(Some(sin_y.raw()), 0, 2);
            enc.setBuffer_offset_atIndex(Some(cos_x.raw()), 0, 3);
            enc.setBuffer_offset_atIndex(Some(sin_x.raw()), 0, 4);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<Rope2dParams>(), 5,
            );
        }
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(total), height: 1, depth: 1 };
        let grid_size = MTLSize { width: total, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    /// Reshape [batch*seq, embed_dim] -> [batch*heads, seq, head_dim] (for cross-attn).
    pub fn dispatch_separate_reshape(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32, seq: u32, heads: u32, head_dim: u32, embed_dim: u32,
    ) {
        #[repr(C)]
        struct AttnReshapeParams { batch: u32, seq: u32, heads: u32, head_dim: u32, embed_dim: u32 }
        let params = AttnReshapeParams { batch, seq, heads, head_dim, embed_dim };
        let total = (batch * heads * seq * head_dim) as usize;

        let pipeline = &self.pipelines["separate_reshape_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<AttnReshapeParams>(), 2,
            );
        }
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(total), height: 1, depth: 1 };
        let grid_size = MTLSize { width: total, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    /// Reshape [batch*heads, seq, head_dim] -> [batch*seq, embed_dim] (attn output).
    pub fn dispatch_attn_output_reshape(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32, seq: u32, heads: u32, head_dim: u32, embed_dim: u32,
    ) {
        #[repr(C)]
        struct AttnReshapeParams { batch: u32, seq: u32, heads: u32, head_dim: u32, embed_dim: u32 }
        let params = AttnReshapeParams { batch, seq, heads, head_dim, embed_dim };
        let total = (batch * seq * embed_dim) as usize;

        let pipeline = &self.pipelines["attn_output_reshape_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");
        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<AttnReshapeParams>(), 2,
            );
        }
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let threadgroup_size = MTLSize { width: max_threads.min(total), height: 1, depth: 1 };
        let grid_size = MTLSize { width: total, height: 1, depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);
        enc.endEncoding();
    }

    /// Composed SDPA: scale Q, batched Q@K^T, softmax, batched scores@V.
    /// All buffers in [total_bh, seq, head_dim] / [total_bh, seq_q, seq_kv] layout.
    pub fn dispatch_sdpa(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        scores: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        total_bh: usize, seq_q: usize, seq_kv: usize, head_dim: usize,
        scale: f32,
    ) {
        // 1. Scale Q into scratch buffer (avoids modifying caller's q)
        let q_len = total_bh * seq_q * head_dim;
        let q_scaled = self.alloc::<f32>(q_len);
        self.dispatch_scale(q, &q_scaled, scale);

        // 2. Batched Q @ K^T -> scores [total_bh, seq_q, seq_kv]
        let head_q_size = seq_q * head_dim;
        let head_kv_size = seq_kv * head_dim;
        let head_s_size = seq_q * seq_kv;

        {
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            #[repr(C)]
            struct MatmulParams { m: u32, n: u32, k: u32, fuse_bias: u32, fuse_relu: u32, trans_a: u32, trans_b: u32 }
            let params = MatmulParams {
                m: seq_q as u32, n: seq_kv as u32, k: head_dim as u32,
                fuse_bias: 0, fuse_relu: 0, trans_a: 0, trans_b: 1,
            };

            // Choose kernel based on size
            let use_simd = seq_q >= 64 && seq_kv >= 64
                && (seq_q as u64) * (seq_kv as u64) >= 1024 * 1024;
            let kernel_name = if use_simd { "matmul_simd_f32" } else { "matmul_f32" };
            let pipeline = &self.pipelines[kernel_name];
            enc.setComputePipelineState(pipeline);

            for bh in 0..total_bh {
                let q_offset = bh * head_q_size * 4; // byte offset
                let k_offset = bh * head_kv_size * 4;
                let s_offset = bh * head_s_size * 4;

                unsafe {
                    enc.setBuffer_offset_atIndex(Some(q_scaled.raw()), q_offset, 0);
                    enc.setBuffer_offset_atIndex(Some(k.raw()), k_offset, 1);
                    enc.setBuffer_offset_atIndex(Some(scores.raw()), s_offset, 2);
                    // dummy bias buffer (not used)
                    enc.setBuffer_offset_atIndex(Some(scores.raw()), 0, 3);
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                        std::mem::size_of::<MatmulParams>(), 4,
                    );
                }

                if use_simd {
                    let threads_per_group = 128usize;
                    let tile = 32usize;
                    let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
                    let num_groups = MTLSize {
                        width: (seq_kv + tile - 1) / tile,
                        height: (seq_q + tile - 1) / tile,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                } else {
                    let tile_size = 16usize;
                    let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
                    let num_groups = MTLSize {
                        width: (seq_kv + tile_size - 1) / tile_size,
                        height: (seq_q + tile_size - 1) / tile_size,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                }
            }
            enc.endEncoding();
        }

        // 3. Softmax over scores: each of total_bh * seq_q rows has seq_kv elements
        self.dispatch_softmax(
            scores, scores,
            (total_bh * seq_q) as u32, seq_kv as u32,
        );

        // 4. Batched scores @ V -> output [total_bh, seq_q, head_dim]
        {
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            #[repr(C)]
            struct MatmulParams { m: u32, n: u32, k: u32, fuse_bias: u32, fuse_relu: u32, trans_a: u32, trans_b: u32 }
            let params = MatmulParams {
                m: seq_q as u32, n: head_dim as u32, k: seq_kv as u32,
                fuse_bias: 0, fuse_relu: 0, trans_a: 0, trans_b: 0,
            };

            let use_simd = seq_q >= 64 && head_dim >= 64
                && (seq_q as u64) * (head_dim as u64) >= 1024 * 1024;
            let kernel_name = if use_simd { "matmul_simd_f32" } else { "matmul_f32" };
            let pipeline = &self.pipelines[kernel_name];
            enc.setComputePipelineState(pipeline);

            for bh in 0..total_bh {
                let s_offset = bh * head_s_size * 4;
                let v_offset = bh * head_kv_size * 4;
                let o_offset = bh * head_q_size * 4;

                unsafe {
                    enc.setBuffer_offset_atIndex(Some(scores.raw()), s_offset, 0);
                    enc.setBuffer_offset_atIndex(Some(v.raw()), v_offset, 1);
                    enc.setBuffer_offset_atIndex(Some(output.raw()), o_offset, 2);
                    enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                        std::mem::size_of::<MatmulParams>(), 4,
                    );
                }

                if use_simd {
                    let threads_per_group = 128usize;
                    let tile = 32usize;
                    let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
                    let num_groups = MTLSize {
                        width: (head_dim + tile - 1) / tile,
                        height: (seq_q + tile - 1) / tile,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                } else {
                    let tile_size = 16usize;
                    let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
                    let num_groups = MTLSize {
                        width: (head_dim + tile_size - 1) / tile_size,
                        height: (seq_q + tile_size - 1) / tile_size,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                }
            }
            enc.endEncoding();
        }
    }

    /// Dispatch causal-masked softmax kernel.
    pub fn dispatch_causal_mask_softmax(
        &self,
        input: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch: u32,
        dim: u32,
        seq_q: u32,
        seq_kv: u32,
        mask_type: u32,
        offset: u32,
        window: u32,
        sink_tokens: u32,
    ) {
        #[repr(C)]
        struct CausalMaskSoftmaxParams {
            batch: u32,
            dim: u32,
            seq_q: u32,
            seq_kv: u32,
            mask_type: u32,
            offset: u32,
            window: u32,
            sink_tokens: u32,
        }

        let params = CausalMaskSoftmaxParams {
            batch, dim, seq_q, seq_kv, mask_type, offset, window, sink_tokens,
        };

        let pipeline = &self.pipelines["causal_mask_softmax_f32"];
        let cmd_ref = self.ensure_cmd();
        let cmd = cmd_ref.as_ref().unwrap();
        let enc = cmd.computeCommandEncoder().expect("encoder");

        enc.setComputePipelineState(pipeline);
        unsafe {
            enc.setBuffer_offset_atIndex(Some(input.raw()), 0, 0);
            enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                std::mem::size_of::<CausalMaskSoftmaxParams>(),
                2,
            );
        }

        let threads_per_row = (dim as usize).next_power_of_two().min(1024);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );
        enc.endEncoding();
    }

    /// Composed SDPA with GQA support and optional causal masking.
    /// Q: [batch_size * num_q_heads, seq_q, head_dim]
    /// K: [batch_size * num_kv_heads, seq_kv, head_dim]
    /// V: [batch_size * num_kv_heads, seq_kv, head_dim]
    /// scores: [batch_size * num_q_heads, seq_q, seq_kv]
    /// output: [batch_size * num_q_heads, seq_q, head_dim]
    pub fn dispatch_sdpa_masked(
        &self,
        q: &GpuBuffer<f32>,
        k: &GpuBuffer<f32>,
        v: &GpuBuffer<f32>,
        scores: &GpuBuffer<f32>,
        output: &GpuBuffer<f32>,
        batch_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        seq_q: usize,
        seq_kv: usize,
        head_dim: usize,
        scale: f32,
        mask: &GpuAttentionMask,
    ) {
        let total_q_bh = batch_size * num_q_heads;
        let heads_per_group = num_q_heads / num_kv_heads;

        // 1. Scale Q
        let q_len = total_q_bh * seq_q * head_dim;
        let q_scaled = self.alloc::<f32>(q_len);
        self.dispatch_scale(q, &q_scaled, scale);

        // 2. Batched Q @ K^T -> scores [total_q_bh, seq_q, seq_kv]
        let head_q_size = seq_q * head_dim;
        let head_kv_size = seq_kv * head_dim;
        let head_s_size = seq_q * seq_kv;

        {
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            #[repr(C)]
            struct MatmulParams { m: u32, n: u32, k: u32, fuse_bias: u32, fuse_relu: u32, trans_a: u32, trans_b: u32 }
            let params = MatmulParams {
                m: seq_q as u32, n: seq_kv as u32, k: head_dim as u32,
                fuse_bias: 0, fuse_relu: 0, trans_a: 0, trans_b: 1,
            };

            let use_simd = seq_q >= 64 && seq_kv >= 64
                && (seq_q as u64) * (seq_kv as u64) >= 1024 * 1024;
            let kernel_name = if use_simd { "matmul_simd_f32" } else { "matmul_f32" };
            let pipeline = &self.pipelines[kernel_name];
            enc.setComputePipelineState(pipeline);

            for bh in 0..total_q_bh {
                // GQA: map Q head to KV head
                let b = bh / num_q_heads;
                let qh = bh % num_q_heads;
                let kv_bh = b * num_kv_heads + (qh / heads_per_group);

                let q_offset = bh * head_q_size * 4;
                let k_offset = kv_bh * head_kv_size * 4;
                let s_offset = bh * head_s_size * 4;

                unsafe {
                    enc.setBuffer_offset_atIndex(Some(q_scaled.raw()), q_offset, 0);
                    enc.setBuffer_offset_atIndex(Some(k.raw()), k_offset, 1);
                    enc.setBuffer_offset_atIndex(Some(scores.raw()), s_offset, 2);
                    enc.setBuffer_offset_atIndex(Some(scores.raw()), 0, 3);
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                        std::mem::size_of::<MatmulParams>(), 4,
                    );
                }

                if use_simd {
                    let threads_per_group = 128usize;
                    let tile = 32usize;
                    let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
                    let num_groups = MTLSize {
                        width: (seq_kv + tile - 1) / tile,
                        height: (seq_q + tile - 1) / tile,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                } else {
                    let tile_size = 16usize;
                    let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
                    let num_groups = MTLSize {
                        width: (seq_kv + tile_size - 1) / tile_size,
                        height: (seq_q + tile_size - 1) / tile_size,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                }
            }
            enc.endEncoding();
        }

        // 3. Softmax (with optional masking) over scores
        match mask {
            GpuAttentionMask::None => {
                self.dispatch_softmax(
                    scores, scores,
                    (total_q_bh * seq_q) as u32, seq_kv as u32,
                );
            }
            GpuAttentionMask::Causal { offset } => {
                self.dispatch_causal_mask_softmax(
                    scores, scores,
                    (total_q_bh * seq_q) as u32, seq_kv as u32,
                    seq_q as u32, seq_kv as u32,
                    1, // bit0 = causal
                    *offset as u32, 0, 0,
                );
            }
            GpuAttentionMask::CausalSlidingWindow { offset, window, sink_tokens } => {
                self.dispatch_causal_mask_softmax(
                    scores, scores,
                    (total_q_bh * seq_q) as u32, seq_kv as u32,
                    seq_q as u32, seq_kv as u32,
                    1 | 2 | if *sink_tokens > 0 { 4 } else { 0 },
                    *offset as u32, *window as u32, *sink_tokens as u32,
                );
            }
        }

        // 4. Batched scores @ V -> output [total_q_bh, seq_q, head_dim]
        {
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            #[repr(C)]
            struct MatmulParams { m: u32, n: u32, k: u32, fuse_bias: u32, fuse_relu: u32, trans_a: u32, trans_b: u32 }
            let params = MatmulParams {
                m: seq_q as u32, n: head_dim as u32, k: seq_kv as u32,
                fuse_bias: 0, fuse_relu: 0, trans_a: 0, trans_b: 0,
            };

            let use_simd = seq_q >= 64 && head_dim >= 64
                && (seq_q as u64) * (head_dim as u64) >= 1024 * 1024;
            let kernel_name = if use_simd { "matmul_simd_f32" } else { "matmul_f32" };
            let pipeline = &self.pipelines[kernel_name];
            enc.setComputePipelineState(pipeline);

            for bh in 0..total_q_bh {
                let b = bh / num_q_heads;
                let qh = bh % num_q_heads;
                let kv_bh = b * num_kv_heads + (qh / heads_per_group);

                let s_offset = bh * head_s_size * 4;
                let v_offset = kv_bh * head_kv_size * 4;
                let o_offset = bh * head_q_size * 4;

                unsafe {
                    enc.setBuffer_offset_atIndex(Some(scores.raw()), s_offset, 0);
                    enc.setBuffer_offset_atIndex(Some(v.raw()), v_offset, 1);
                    enc.setBuffer_offset_atIndex(Some(output.raw()), o_offset, 2);
                    enc.setBuffer_offset_atIndex(Some(output.raw()), 0, 3);
                    enc.setBytes_length_atIndex(
                        std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                        std::mem::size_of::<MatmulParams>(), 4,
                    );
                }

                if use_simd {
                    let threads_per_group = 128usize;
                    let tile = 32usize;
                    let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
                    let num_groups = MTLSize {
                        width: (head_dim + tile - 1) / tile,
                        height: (seq_q + tile - 1) / tile,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                } else {
                    let tile_size = 16usize;
                    let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
                    let num_groups = MTLSize {
                        width: (head_dim + tile_size - 1) / tile_size,
                        height: (seq_q + tile_size - 1) / tile_size,
                        depth: 1,
                    };
                    enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
                }
            }
            enc.endEncoding();
        }
    }

    /// Dispatch int8 dequantizing matmul: C[M,N] = A_f32[M,K] @ W_i8[K,N].
    /// Auto-selects scalar (16x16) or simdgroup (32x32) kernel based on matrix size.
    pub fn dispatch_matmul_dequant_i8(
        &self,
        a: &GpuBuffer<f32>,
        w_i8: &GpuBuffer<i8>,
        w_scales: &GpuBuffer<f32>,
        out: &GpuBuffer<f32>,
        m: u32, n: u32, k: u32,
    ) {
        #[repr(C)]
        struct DequantMatmulParams {
            m: u32,
            n: u32,
            k: u32,
        }

        let params = DequantMatmulParams { m, n, k };

        let large_enough = m >= 64 && n >= 64
            && (m as u64) * (n as u64) >= 1024 * 1024;

        if large_enough {
            // Simdgroup kernel
            let pipeline = &self.pipelines["matmul_dequant_simd_i8"];
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            enc.setComputePipelineState(pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(w_i8.raw()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(w_scales.raw()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 3);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    std::mem::size_of::<DequantMatmulParams>(),
                    4,
                );
            }

            let threads_per_group = 128usize;
            let tile = 32usize;
            let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
            let num_groups = MTLSize {
                width: (n as usize + tile - 1) / tile,
                height: (m as usize + tile - 1) / tile,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
            enc.endEncoding();
        } else {
            // Scalar 16x16 tiled kernel
            let pipeline = &self.pipelines["matmul_dequant_i8"];
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            enc.setComputePipelineState(pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(w_i8.raw()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(w_scales.raw()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 3);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    std::mem::size_of::<DequantMatmulParams>(),
                    4,
                );
            }

            let tile_size = 16usize;
            let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
            let num_groups = MTLSize {
                width: (n as usize + tile_size - 1) / tile_size,
                height: (m as usize + tile_size - 1) / tile_size,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
            enc.endEncoding();
        }
    }

    /// Dispatch 2:4 structured sparse matmul: C[M,N] = A_f32[M,K] @ W_sparse_24[K,N].
    /// Auto-selects scalar (per-element) or simdgroup (32x32) kernel based on matrix size.
    pub fn dispatch_matmul_sparse_24(
        &self,
        a: &GpuBuffer<f32>,
        w_vals: &GpuBuffer<f32>,
        w_idx: &GpuBuffer<u8>,
        out: &GpuBuffer<f32>,
        m: u32, n: u32, k: u32,
    ) {
        #[repr(C)]
        struct SparseMatmulParams24 {
            m: u32,
            n: u32,
            k: u32,
        }

        let params = SparseMatmulParams24 { m, n, k };

        let large_enough = m >= 64 && n >= 64
            && (m as u64) * (n as u64) >= 1024 * 1024;

        if large_enough {
            // Simdgroup kernel
            let pipeline = &self.pipelines["matmul_sparse_simd_24"];
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            enc.setComputePipelineState(pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(w_vals.raw()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(w_idx.raw()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 3);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    std::mem::size_of::<SparseMatmulParams24>(),
                    4,
                );
            }

            let threads_per_group = 128usize;
            let tile = 32usize;
            let threadgroup_size = MTLSize { width: threads_per_group, height: 1, depth: 1 };
            let num_groups = MTLSize {
                width: (n as usize + tile - 1) / tile,
                height: (m as usize + tile - 1) / tile,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
            enc.endEncoding();
        } else {
            // Scalar kernel — one thread per output element
            let pipeline = &self.pipelines["matmul_sparse_24"];
            let cmd_ref = self.ensure_cmd();
            let cmd = cmd_ref.as_ref().unwrap();
            let enc = cmd.computeCommandEncoder().expect("encoder");

            enc.setComputePipelineState(pipeline);
            unsafe {
                enc.setBuffer_offset_atIndex(Some(a.raw()), 0, 0);
                enc.setBuffer_offset_atIndex(Some(w_vals.raw()), 0, 1);
                enc.setBuffer_offset_atIndex(Some(w_idx.raw()), 0, 2);
                enc.setBuffer_offset_atIndex(Some(out.raw()), 0, 3);
                enc.setBytes_length_atIndex(
                    std::ptr::NonNull::new(&params as *const _ as *mut _).unwrap(),
                    std::mem::size_of::<SparseMatmulParams24>(),
                    4,
                );
            }

            let tile_size = 16usize;
            let threadgroup_size = MTLSize { width: tile_size, height: tile_size, depth: 1 };
            let num_groups = MTLSize {
                width: (n as usize + tile_size - 1) / tile_size,
                height: (m as usize + tile_size - 1) / tile_size,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(num_groups, threadgroup_size);
            enc.endEncoding();
        }
    }
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        self.sync();
    }
}

// ---------------------------------------------------------------------------
// Thread-local GPU singleton
// ---------------------------------------------------------------------------

thread_local! {
    static GPU_CONTEXT: RefCell<Option<GpuContext>> = const { RefCell::new(None) };
}

/// Initialize the thread-local GPU context. Call once at startup.
/// Returns Err if no Metal device is available.
pub fn init_gpu() -> Result<(), String> {
    GPU_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        if ctx.is_none() {
            *ctx = Some(GpuContext::new()?);
        }
        Ok(())
    })
}

/// Run a closure with shared (immutable) access to the GPU context.
/// Returns None if GPU is not initialized.
pub fn with_gpu<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&GpuContext) -> R,
{
    GPU_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.as_ref().map(f)
    })
}

/// Run a closure with mutable access to the GPU context (needed for pool operations).
/// Returns None if GPU is not initialized.
pub fn with_gpu_mut<F, R>(f: F) -> Option<R>
where
    F: FnOnce(&mut GpuContext) -> R,
{
    GPU_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        ctx.as_mut().map(f)
    })
}

/// Flush any pending GPU commands. No-op if no GPU or no pending commands.
pub fn gpu_sync() {
    GPU_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        if let Some(ref gpu) = *ctx {
            gpu.sync();
        }
    })
}

/// Commit pending GPU commands with a signal event (non-blocking).
/// Returns a ticket for `gpu_wait_for()` / `gpu_is_done()`.
pub fn gpu_commit_and_signal() -> u64 {
    GPU_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.as_ref().map(|gpu| gpu.commit_and_signal()).unwrap_or(0)
    })
}

/// Spin-wait until GPU work for the given ticket completes.
pub fn gpu_wait_for(ticket: u64) {
    GPU_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        if let Some(ref gpu) = *ctx {
            gpu.wait_for(ticket);
        }
    })
}

/// Check whether GPU work for the given ticket has completed (non-blocking).
pub fn gpu_is_done(ticket: u64) -> bool {
    GPU_CONTEXT.with(|ctx| {
        let ctx = ctx.borrow();
        ctx.as_ref().map(|gpu| gpu.is_done(ticket)).unwrap_or(true)
    })
}
