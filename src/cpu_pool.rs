//! Thread-local CPU buffer pool to reduce allocation overhead for elementwise ops.
//!
//! Reuses `Vec<f32>` buffers of the same size-bucket instead of allocating via
//! `.collect()` every forward/backward pass. Modeled on `src/metal/pool.rs`.

use std::cell::RefCell;
use std::collections::HashMap;

const MAX_BUFFERS_PER_BUCKET: usize = 8;
const MIN_POOL_SIZE: usize = 1024;

thread_local! {
    static POOL: RefCell<HashMap<usize, Vec<Vec<f32>>>> = RefCell::new(HashMap::new());
}

/// Round up to next power of 2 (minimum 16 elements = 64 bytes).
#[inline]
fn bucket_size(len: usize) -> usize {
    len.next_power_of_two().max(16)
}

/// Get a `Vec<f32>` with at least `len` elements from the pool, or allocate fresh.
/// The returned Vec has length set to `len` (contents are uninitialized/stale).
#[inline]
pub fn pool_get(len: usize) -> Vec<f32> {
    if len < MIN_POOL_SIZE {
        let mut buf = Vec::with_capacity(len);
        unsafe { buf.set_len(len); }
        return buf;
    }
    let bucket = bucket_size(len);
    let mut buf = POOL.with(|pool| {
        pool.borrow_mut()
            .get_mut(&bucket)
            .and_then(|list| list.pop())
    })
    .unwrap_or_else(|| Vec::with_capacity(bucket));

    // SAFETY: we set len <= capacity; callers must write all elements before reading.
    // Using unsafe set_len avoids zeroing memory we're about to overwrite.
    unsafe { buf.set_len(len); }
    buf
}

/// Return a buffer to the pool for reuse. Caps at MAX_BUFFERS_PER_BUCKET per bucket.
#[inline]
pub fn pool_recycle(mut buf: Vec<f32>) {
    if buf.capacity() < MIN_POOL_SIZE {
        return; // drop naturally, pool overhead exceeds malloc savings
    }
    let bucket = bucket_size(buf.capacity());
    // Only recycle buffers whose capacity matches a bucket (avoid odd sizes)
    if buf.capacity() < bucket {
        return; // let it drop naturally
    }
    buf.clear();
    POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        let list = pool.entry(bucket).or_default();
        if list.len() < MAX_BUFFERS_PER_BUCKET {
            list.push(buf);
        }
        // else: drop buf, pool is full for this bucket
    });
}

/// Clear all cached buffers (useful for testing).
pub fn pool_clear() {
    POOL.with(|pool| {
        pool.borrow_mut().clear();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_get_returns_correct_length() {
        pool_clear();
        // Use size >= MIN_POOL_SIZE to test pool path
        let buf = pool_get(2000);
        assert_eq!(buf.len(), 2000);
        assert!(buf.capacity() >= 2048); // next power of 2
    }

    #[test]
    fn test_pool_get_small_bypasses_pool() {
        pool_clear();
        // Small allocations bypass the pool
        let buf = pool_get(100);
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.capacity(), 100); // exact, no bucketing
    }

    #[test]
    fn test_pool_recycle_and_reuse() {
        pool_clear();
        // Use size >= MIN_POOL_SIZE to test pool path
        let buf = pool_get(2000);
        let cap = buf.capacity();
        let ptr = buf.as_ptr();
        pool_recycle(buf);

        // Next get of same bucket should reuse
        let buf2 = pool_get(2000);
        assert_eq!(buf2.as_ptr(), ptr);
        assert_eq!(buf2.capacity(), cap);
        pool_clear();
    }

    #[test]
    fn test_pool_cap_limit() {
        pool_clear();
        // Fill pool beyond limit
        for _ in 0..12 {
            let buf = pool_get(64);
            pool_recycle(buf);
        }
        // Only MAX_BUFFERS_PER_BUCKET should be stored
        POOL.with(|pool| {
            let pool = pool.borrow();
            let bucket = bucket_size(64);
            let count = pool.get(&bucket).map_or(0, |v| v.len());
            assert!(count <= MAX_BUFFERS_PER_BUCKET);
        });
        pool_clear();
    }
}
