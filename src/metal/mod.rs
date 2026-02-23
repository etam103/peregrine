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
mod shaders;

pub use context::GpuContext;
pub use buffer::GpuBuffer;
pub use pool::BufferPool;
