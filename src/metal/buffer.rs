use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::marker::PhantomData;

/// Typed wrapper around MTLBuffer with StorageModeShared (CPU+GPU zero-copy).
pub struct GpuBuffer<T: Copy> {
    raw: Retained<objc2::runtime::ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _marker: PhantomData<T>,
}

impl<T: Copy> GpuBuffer<T> {
    /// Allocate a new GPU buffer with `len` elements.
    pub fn new(device: &objc2::runtime::ProtocolObject<dyn MTLDevice>, len: usize) -> Self {
        let byte_size = len * std::mem::size_of::<T>();
        let raw = device
            .newBufferWithLength_options(byte_size, MTLResourceOptions::StorageModeShared)
            .expect("Failed to allocate MTLBuffer");
        GpuBuffer { raw, len, _marker: PhantomData }
    }

    /// Create a GPU buffer from existing data.
    pub fn from_slice(
        device: &objc2::runtime::ProtocolObject<dyn MTLDevice>,
        data: &[T],
    ) -> Self {
        let buf = Self::new(device, data.len());
        buf.write(data);
        buf
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Write data from CPU slice into the buffer.
    pub fn write(&self, data: &[T]) {
        assert!(data.len() <= self.len, "data too large for buffer");
        let ptr = self.raw.contents().as_ptr() as *mut T;
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Read buffer contents back to a Vec.
    pub fn read(&self) -> Vec<T> {
        let ptr = self.raw.contents().as_ptr() as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.len) };
        slice.to_vec()
    }

    /// Get a reference to the raw MTLBuffer (for binding to encoders).
    pub fn raw(&self) -> &objc2::runtime::ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }

    /// Size in bytes.
    pub fn byte_size(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
}
