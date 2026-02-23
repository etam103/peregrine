use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::*;
use std::collections::HashMap;

use super::buffer::GpuBuffer;
use super::shaders::SHADER_SOURCE;

/// Central Metal GPU context: device, command queue, and compiled compute pipelines.
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
            "relu_f32", "sigmoid_f32", "tanh_f32",
            "sin_f32", "cos_f32", "abs_f32",
            // Other
            "scale_f32", "matmul_f32", "sum_f32", "max_f32", "min_f32",
            "softmax_f32", "transpose_f32", "layernorm_f32",
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

        Ok(GpuContext { device, queue, pipelines })
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
        let cmd = self.queue.commandBuffer().expect("command buffer");
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
        cmd.commit();
        cmd.waitUntilCompleted();
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
        let cmd = self.queue.commandBuffer().expect("command buffer");
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
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    /// Dispatch matmul: C = A[M,K] @ B[K,N], optionally fused with bias and relu.
    pub fn dispatch_matmul(
        &self,
        a: &GpuBuffer<f32>,
        b: &GpuBuffer<f32>,
        c: &GpuBuffer<f32>,
        bias: Option<&GpuBuffer<f32>>,
        m: u32, n: u32, k: u32,
        fuse_relu: bool,
    ) {
        #[repr(C)]
        struct MatmulParams {
            m: u32,
            n: u32,
            k: u32,
            fuse_bias: u32,
            fuse_relu: u32,
        }

        let params = MatmulParams {
            m, n, k,
            fuse_bias: if bias.is_some() { 1 } else { 0 },
            fuse_relu: if fuse_relu { 1 } else { 0 },
        };

        let pipeline = &self.pipelines["matmul_f32"];
        let cmd = self.queue.commandBuffer().expect("command buffer");
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

        let grid_size = MTLSize { width: n as usize, height: m as usize, depth: 1 };
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let side = (max_threads as f64).sqrt() as usize;
        let threadgroup_size = MTLSize {
            width: side.min(n as usize),
            height: side.min(m as usize),
            depth: 1,
        };
        enc.dispatchThreads_threadsPerThreadgroup(grid_size, threadgroup_size);

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
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
        let cmd = self.queue.commandBuffer().expect("command buffer");
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
        let threads_per_row = (dim as usize).next_power_of_two().min(256);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );

        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }

    /// Dispatch a reduction op (sum/max/min). Returns a single f32 result.
    pub fn dispatch_reduce(&self, kernel: &str, input: &GpuBuffer<f32>) -> f32 {
        let n = input.len();
        let pipeline = &self.pipelines[kernel];
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let group_size = max_threads.min(256).min(n.next_power_of_two());
        let num_groups = (n + group_size - 1) / group_size;

        let partial: GpuBuffer<f32> = GpuBuffer::new(&self.device, num_groups);
        let count = n as u32;

        let cmd = self.queue.commandBuffer().expect("command buffer");
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
        cmd.commit();
        cmd.waitUntilCompleted();

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
        let cmd = self.queue.commandBuffer().expect("command buffer");
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

        let max_threads = pipeline.maxTotalThreadsPerThreadgroup() as usize;
        let side = (max_threads as f64).sqrt() as usize;
        let grid = MTLSize { width: cols as usize, height: rows as usize, depth: 1 };
        let tg = MTLSize { width: side.min(cols as usize), height: side.min(rows as usize), depth: 1 };
        enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
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
        let cmd = self.queue.commandBuffer().expect("command buffer");
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

        let threads_per_row = (dim as usize).next_power_of_two().min(256);
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize { width: batch as usize, height: 1, depth: 1 },
            MTLSize { width: threads_per_row, height: 1, depth: 1 },
        );
        enc.endEncoding();
        cmd.commit();
        cmd.waitUntilCompleted();
    }
}
