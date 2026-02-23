//! GPU buffer memory pool to reduce allocation overhead during training.
//!
//! Reuses MTLBuffers of the same size instead of allocating/deallocating
//! every iteration. This is critical for training loops where the same
//! tensor shapes are used repeatedly.

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::collections::HashMap;

/// A simple size-bucketed memory pool for GPU buffers.
///
/// Buffers are bucketed by byte size (rounded up to the next power of 2).
/// When a buffer is returned to the pool, it's cached for reuse.
pub struct BufferPool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    // bucket key = byte size (power of 2), value = free buffers
    free: HashMap<usize, Vec<Retained<ProtocolObject<dyn MTLBuffer>>>>,
    allocated_bytes: usize,
    peak_bytes: usize,
    alloc_count: usize,
    reuse_count: usize,
}

impl BufferPool {
    pub fn new(device: &ProtocolObject<dyn MTLDevice>) -> Self {
        // Clone the Retained by re-retaining the device
        let device = unsafe {
            Retained::retain(device as *const _ as *mut ProtocolObject<dyn MTLDevice>)
                .expect("device retain")
        };
        BufferPool {
            device,
            free: HashMap::new(),
            allocated_bytes: 0,
            peak_bytes: 0,
            alloc_count: 0,
            reuse_count: 0,
        }
    }

    /// Round up to the next power of 2 for bucketing.
    fn bucket_size(bytes: usize) -> usize {
        if bytes == 0 { return 64; } // minimum allocation
        bytes.next_power_of_two().max(64)
    }

    /// Get a buffer of at least `byte_size` bytes. Reuses from pool if available.
    pub fn get(&mut self, byte_size: usize) -> Retained<ProtocolObject<dyn MTLBuffer>> {
        let bucket = Self::bucket_size(byte_size);

        if let Some(free_list) = self.free.get_mut(&bucket) {
            if let Some(buf) = free_list.pop() {
                self.reuse_count += 1;
                return buf;
            }
        }

        // Allocate new buffer
        let buf = self.device
            .newBufferWithLength_options(bucket, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate MTLBuffer");
        self.alloc_count += 1;
        self.allocated_bytes += bucket;
        self.peak_bytes = self.peak_bytes.max(self.allocated_bytes);
        buf
    }

    /// Return a buffer to the pool for reuse.
    pub fn recycle(&mut self, buf: Retained<ProtocolObject<dyn MTLBuffer>>) {
        let byte_size = buf.length() as usize;
        let bucket = Self::bucket_size(byte_size);
        self.free.entry(bucket).or_default().push(buf);
    }

    /// Clear all cached buffers, releasing GPU memory.
    pub fn clear(&mut self) {
        let freed: usize = self.free.values()
            .flat_map(|v| v.iter())
            .map(|b| b.length() as usize)
            .sum();
        self.allocated_bytes = self.allocated_bytes.saturating_sub(freed);
        self.free.clear();
    }

    /// Current allocated bytes (including cached free buffers).
    pub fn allocated_bytes(&self) -> usize {
        self.allocated_bytes
    }

    /// Peak allocated bytes during this pool's lifetime.
    pub fn peak_bytes(&self) -> usize {
        self.peak_bytes
    }

    /// Total allocations made (not from cache).
    pub fn alloc_count(&self) -> usize {
        self.alloc_count
    }

    /// Total reuses from cache.
    pub fn reuse_count(&self) -> usize {
        self.reuse_count
    }

    /// Cache hit rate (reuse / (reuse + alloc)).
    pub fn hit_rate(&self) -> f32 {
        let total = self.alloc_count + self.reuse_count;
        if total == 0 { 0.0 } else { self.reuse_count as f32 / total as f32 }
    }

    /// Print pool statistics.
    pub fn stats(&self) -> String {
        format!(
            "BufferPool: allocated={:.1}MB peak={:.1}MB allocs={} reuses={} hit_rate={:.1}%",
            self.allocated_bytes as f64 / 1024.0 / 1024.0,
            self.peak_bytes as f64 / 1024.0 / 1024.0,
            self.alloc_count,
            self.reuse_count,
            self.hit_rate() * 100.0,
        )
    }
}
