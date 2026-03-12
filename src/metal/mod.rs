//! Metal GPU backend for Apple Silicon.
//!
//! Provides safe Rust wrappers around the Metal compute pipeline:
//! - [`GpuContext`]: Device, command queue, compiled pipelines
//! - [`GpuBuffer`]: Typed wrapper around MTLBuffer (shared memory, zero-copy)
//!
//! Enable with: `cargo build --features metal`

mod context;
mod buffer;
mod pool;
pub mod pipeline;
mod shaders;
pub mod sched;

pub use context::GpuContext;
pub use context::{init_gpu, with_gpu, with_gpu_mut, gpu_sync};
pub use context::{gpu_commit_and_signal, gpu_wait_for, gpu_is_done};
pub use buffer::GpuBuffer;
pub use pool::BufferPool;
pub use pipeline::{FusedOp, PipelineBuilder};
pub use sched::het_execute;
