use std::sync::atomic::{AtomicU32, Ordering};

pub const MAX_RANKS: usize = 16;
pub const MAX_BLOCKS: usize = 1;

/// Atomic signal structure for inter-process barrier synchronization.
/// Lives in shared memory. Mirrors Penny's `Signal` struct.
///
/// Layout (all arrays indexed by block, then rank):
///   start[block][rank] — peer writes here to signal arrival
///   end[block][rank]   — peer writes here to signal completion
///   flag[block]        — monotonic counter per block
#[repr(C, align(128))]
pub struct Signal {
    pub start: [[AtomicU32; MAX_RANKS]; MAX_BLOCKS],
    pub end: [[AtomicU32; MAX_RANKS]; MAX_BLOCKS],
    pub flag: [AtomicU32; MAX_BLOCKS],
}

/// Size in bytes of the signal region for `world_size` ranks.
/// Each rank gets its own Signal struct in shared memory.
pub fn signal_region_size(world_size: usize) -> usize {
    std::mem::size_of::<Signal>() * world_size
}

/// Get a reference to rank's Signal in the shared memory region.
///
/// # Safety
/// `base` must point to a shared memory region of at least `signal_region_size(world_size)` bytes,
/// properly aligned and zero-initialized.
pub unsafe fn signal_for_rank(base: *mut u8, rank: usize) -> &'static Signal {
    let ptr = base as *const Signal;
    &*ptr.add(rank)
}

impl Signal {
    /// Start barrier: each rank writes its flag to all peers' `start` arrays, then
    /// spins until all peers have written to our `start` array.
    ///
    /// `self_signal` — pointer to this rank's Signal (in shared memory)
    /// `all_signals` — pointers to all ranks' Signals
    /// `rank` — this rank
    /// `world_size` — number of ranks
    /// `block` — block index (always 0 for CPU)
    pub fn barrier_start(
        all_signals: &[&Signal],
        rank: usize,
        world_size: usize,
        block: usize,
    ) {
        let self_sig = all_signals[rank];
        let flag = self_sig.flag[block].load(Ordering::Relaxed) + 1;

        // Write our flag to every peer's start[block][rank]
        for peer in 0..world_size {
            all_signals[peer].start[block][rank].store(flag, Ordering::Release);
        }

        // Spin until all peers have written to our start[block][peer]
        for peer in 0..world_size {
            while self_sig.start[block][peer].load(Ordering::Acquire) != flag {
                std::hint::spin_loop();
            }
        }

        self_sig.flag[block].store(flag, Ordering::Relaxed);
    }

    /// End barrier: same as start barrier but uses `end` arrays.
    pub fn barrier_end(
        all_signals: &[&Signal],
        rank: usize,
        world_size: usize,
        block: usize,
    ) {
        let self_sig = all_signals[rank];
        let flag = self_sig.flag[block].load(Ordering::Relaxed) + 1;

        for peer in 0..world_size {
            all_signals[peer].end[block][rank].store(flag, Ordering::Release);
        }

        for peer in 0..world_size {
            while self_sig.end[block][peer].load(Ordering::Acquire) != flag {
                std::hint::spin_loop();
            }
        }

        self_sig.flag[block].store(flag, Ordering::Relaxed);
    }
}
