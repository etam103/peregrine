//! Heterogeneous GPU + CPU scheduling primitives.
//!
//! Allows overlapping GPU work with CPU/AMX work on a single thread.
//! The key insight: after submitting GPU commands and calling `commit_and_signal()`,
//! the CPU is free to do other work while the GPU executes asynchronously.

use super::context::{gpu_commit_and_signal, gpu_wait_for};

/// Execute GPU and CPU work concurrently on a single thread.
///
/// 1. Calls `gpu_work()` which queues GPU commands (but does NOT sync).
/// 2. Commits the GPU command buffer with a signal event.
/// 3. Calls `cpu_work()` which runs on CPU/AMX concurrently with the GPU.
/// 4. Waits for the GPU signal to ensure GPU work is complete.
///
/// Returns `(gpu_result, cpu_result)`.
///
/// This is single-threaded — no `Send`/`Sync` needed. The overlap comes from
/// the GPU executing asynchronously after commit while the CPU does its own work.
pub fn het_execute<F1, F2, R1, R2>(gpu_work: F1, cpu_work: F2) -> (R1, R2)
where
    F1: FnOnce() -> R1,
    F2: FnOnce() -> R2,
{
    // Phase 1: queue GPU work (dispatch commands but don't wait)
    let gpu_result = gpu_work();

    // Phase 2: commit GPU commands with a signal event (non-blocking)
    let ticket = gpu_commit_and_signal();

    // Phase 3: run CPU work concurrently while GPU executes
    let cpu_result = cpu_work();

    // Phase 4: ensure GPU is done before we use its results
    gpu_wait_for(ticket);

    (gpu_result, cpu_result)
}

/// Calibrate whether heterogeneous execution is beneficial.
///
/// Runs `gpu_fn` and `cpu_fn` independently, measures their wall-clock times,
/// and returns `true` if the overlap ratio exceeds 0.3 (i.e., the faster one
/// takes at least 30% as long as the slower one, so overlap is worthwhile).
pub fn het_calibrate<F1, F2>(gpu_fn: F1, cpu_fn: F2) -> bool
where
    F1: FnOnce(),
    F2: FnOnce(),
{
    use std::time::Instant;

    // Time GPU work (submit + sync)
    let t0 = Instant::now();
    gpu_fn();
    super::context::gpu_sync();
    let gpu_time = t0.elapsed().as_secs_f64();

    // Time CPU work
    let t0 = Instant::now();
    cpu_fn();
    let cpu_time = t0.elapsed().as_secs_f64();

    // Overlap is beneficial if the shorter task takes >= 30% of the longer one
    let (shorter, longer) = if gpu_time < cpu_time {
        (gpu_time, cpu_time)
    } else {
        (cpu_time, gpu_time)
    };

    if longer < 1e-9 {
        return false;
    }

    shorter / longer > 0.3
}
