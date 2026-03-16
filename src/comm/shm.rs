use crate::comm::error::CommError;
use std::ptr;

pub struct ShmRegion {
    name: String,
    ptr: *mut u8,
    len: usize,
    is_owner: bool,
}

unsafe impl Send for ShmRegion {}
unsafe impl Sync for ShmRegion {}

impl ShmRegion {
    /// Create a new shared memory region (owner — will unlink on drop).
    pub fn create(name: &str, len: usize) -> Result<Self, CommError> {
        let c_name =
            std::ffi::CString::new(name).map_err(|e| CommError::ShmOpenFailed(e.to_string()))?;

        unsafe {
            let fd = libc::shm_open(
                c_name.as_ptr(),
                libc::O_CREAT | libc::O_RDWR,
                0o600,
            );
            if fd < 0 {
                return Err(CommError::ShmOpenFailed(format!(
                    "shm_open: {}",
                    std::io::Error::last_os_error()
                )));
            }

            if libc::ftruncate(fd, len as libc::off_t) != 0 {
                libc::close(fd);
                libc::shm_unlink(c_name.as_ptr());
                return Err(CommError::ShmOpenFailed(format!(
                    "ftruncate: {}",
                    std::io::Error::last_os_error()
                )));
            }

            let ptr = libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);

            if ptr == libc::MAP_FAILED {
                libc::shm_unlink(c_name.as_ptr());
                return Err(CommError::MmapFailed(format!(
                    "mmap: {}",
                    std::io::Error::last_os_error()
                )));
            }

            // Zero-initialize
            ptr::write_bytes(ptr as *mut u8, 0, len);

            Ok(ShmRegion {
                name: name.to_string(),
                ptr: ptr as *mut u8,
                len,
                is_owner: true,
            })
        }
    }

    /// Open an existing shared memory region (non-owner — will not unlink on drop).
    pub fn open(name: &str, len: usize) -> Result<Self, CommError> {
        let c_name =
            std::ffi::CString::new(name).map_err(|e| CommError::ShmOpenFailed(e.to_string()))?;

        unsafe {
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDWR, 0);
            if fd < 0 {
                return Err(CommError::ShmOpenFailed(format!(
                    "shm_open: {}",
                    std::io::Error::last_os_error()
                )));
            }

            let ptr = libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                fd,
                0,
            );
            libc::close(fd);

            if ptr == libc::MAP_FAILED {
                return Err(CommError::MmapFailed(format!(
                    "mmap: {}",
                    std::io::Error::last_os_error()
                )));
            }

            Ok(ShmRegion {
                name: name.to_string(),
                ptr: ptr as *mut u8,
                len,
                is_owner: false,
            })
        }
    }

    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }

    pub fn len(&self) -> usize {
        self.len
    }

    /// Interpret the region as a slice of T starting at byte offset.
    ///
    /// # Safety
    /// Caller must ensure the offset and count are within bounds and T is valid.
    pub unsafe fn as_slice<T>(&self, byte_offset: usize, count: usize) -> &[T] {
        let ptr = self.ptr.add(byte_offset) as *const T;
        std::slice::from_raw_parts(ptr, count)
    }

    /// Interpret the region as a mutable slice of T starting at byte offset.
    ///
    /// # Safety
    /// Caller must ensure the offset and count are within bounds, T is valid,
    /// and no aliasing violations occur.
    pub unsafe fn as_slice_mut<T>(&self, byte_offset: usize, count: usize) -> &mut [T] {
        let ptr = self.ptr.add(byte_offset) as *mut T;
        std::slice::from_raw_parts_mut(ptr, count)
    }
}

impl Drop for ShmRegion {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
            if self.is_owner {
                if let Ok(c_name) = std::ffi::CString::new(self.name.as_str()) {
                    libc::shm_unlink(c_name.as_ptr());
                }
            }
        }
    }
}
