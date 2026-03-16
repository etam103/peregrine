use crate::comm::signal::Signal;
use std::ptr;

/// Data region layout for ring algorithms.
///
/// For `world_size` ranks with `max_chunk_bytes` per chunk:
///   Each rank owns two zones in the data region:
///     - buffer zone: where the rank's input data lives
///     - destination zone: where peers write data for this rank
///
///   Total per rank: 2 * max_chunk_bytes * world_size
///   Total region: world_size * 2 * max_chunk_bytes * world_size
///
/// Offset calculations:
///   rank_base(rank) = rank * 2 * max_chunk_bytes * world_size
///   buffer_offset(rank, chunk)  = rank_base(rank) + chunk * max_chunk_bytes
///   dest_offset(rank, chunk)    = rank_base(rank) + world_size * max_chunk_bytes + chunk * max_chunk_bytes

/// Size of the data region in bytes.
pub fn data_region_size(world_size: usize, max_chunk_bytes: usize) -> usize {
    world_size * 2 * max_chunk_bytes * world_size
}

fn rank_base(rank: usize, world_size: usize, max_chunk_bytes: usize) -> usize {
    rank * 2 * max_chunk_bytes * world_size
}

fn buffer_offset(rank: usize, chunk: usize, world_size: usize, max_chunk_bytes: usize) -> usize {
    rank_base(rank, world_size, max_chunk_bytes) + chunk * max_chunk_bytes
}

fn dest_offset(rank: usize, chunk: usize, world_size: usize, max_chunk_bytes: usize) -> usize {
    rank_base(rank, world_size, max_chunk_bytes)
        + world_size * max_chunk_bytes
        + chunk * max_chunk_bytes
}

/// Ring AllReduce: reduces `input` across all ranks using sum, producing identical
/// output on every rank.
///
/// Two phases:
///   Phase 1 (Reduce-Scatter): world_size-1 steps, each rank sends a chunk to its
///     right neighbor, receives from left, and accumulates (sum).
///   Phase 2 (AllGather): world_size-1 steps, each rank sends its fully-reduced
///     chunk around the ring so all ranks have the complete result.
///
/// # Safety
/// `data_base` must point to a shared memory region of at least
/// `data_region_size(world_size, chunk_bytes)` where `chunk_bytes >= input.len() * 4 / world_size`.
/// `all_signals` must point to valid Signal structs for all ranks.
pub unsafe fn ring_allreduce(
    input: &[f32],
    output: &mut [f32],
    data_base: *mut u8,
    all_signals: &[&Signal],
    rank: usize,
    world_size: usize,
) {
    let n = input.len();
    assert_eq!(n % world_size, 0, "input length must be divisible by world_size");
    assert_eq!(output.len(), n);

    let chunk_count = n / world_size; // number of f32s per chunk
    let chunk_bytes = chunk_count * std::mem::size_of::<f32>();
    let max_chunk_bytes = chunk_bytes;

    // Copy input into our buffer zone in shared memory
    for chunk in 0..world_size {
        let off = buffer_offset(rank, chunk, world_size, max_chunk_bytes);
        let src = input[chunk * chunk_count..(chunk + 1) * chunk_count].as_ptr();
        ptr::copy_nonoverlapping(src, data_base.add(off) as *mut f32, chunk_count);
    }

    // Barrier: ensure all ranks have written their input
    Signal::barrier_start(all_signals, rank, world_size, 0);

    let send_peer = (rank + 1) % world_size;

    // ---- Phase 1: Reduce-Scatter ----
    // After this phase, each rank holds the fully-reduced chunk for one slice.
    for step in 0..(world_size - 1) {
        // Chunk index this rank sends in this step
        let send_chunk = (rank + world_size - step) % world_size;
        // Chunk index this rank receives (and reduces into)
        let recv_chunk = (rank + world_size - step - 1) % world_size;

        // Copy our send_chunk to peer's destination zone
        let src_off = buffer_offset(rank, send_chunk, world_size, max_chunk_bytes);
        let dst_off = dest_offset(send_peer, send_chunk, world_size, max_chunk_bytes);
        ptr::copy_nonoverlapping(
            data_base.add(src_off),
            data_base.add(dst_off),
            chunk_bytes,
        );

        // Signal and wait
        Signal::barrier_end(all_signals, rank, world_size, 0);

        // Read what recv_peer sent us and reduce (sum) into our buffer
        let recv_dst_off = dest_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        let recv_buf_off = buffer_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        let received = std::slice::from_raw_parts(
            data_base.add(recv_dst_off) as *const f32,
            chunk_count,
        );
        let local = std::slice::from_raw_parts_mut(
            data_base.add(recv_buf_off) as *mut f32,
            chunk_count,
        );
        for i in 0..chunk_count {
            local[i] += received[i];
        }

        Signal::barrier_start(all_signals, rank, world_size, 0);
    }

    // ---- Phase 2: AllGather ----
    // The fully-reduced chunk for slice `rank` is now in our buffer[rank].
    // Broadcast it around the ring.
    for step in 0..(world_size - 1) {
        let send_chunk = (rank + world_size - step + 1) % world_size;

        // Copy our fully-reduced send_chunk to peer's destination zone
        let src_off = buffer_offset(rank, send_chunk, world_size, max_chunk_bytes);
        let dst_off = dest_offset(send_peer, send_chunk, world_size, max_chunk_bytes);
        ptr::copy_nonoverlapping(
            data_base.add(src_off),
            data_base.add(dst_off),
            chunk_bytes,
        );

        Signal::barrier_end(all_signals, rank, world_size, 0);

        // Copy received data into our buffer
        let recv_chunk = (rank + world_size - step) % world_size;
        let recv_dst_off = dest_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        let recv_buf_off = buffer_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        ptr::copy_nonoverlapping(
            data_base.add(recv_dst_off) as *const u8,
            data_base.add(recv_buf_off),
            chunk_bytes,
        );

        Signal::barrier_start(all_signals, rank, world_size, 0);
    }

    // Copy result from buffer to output
    for chunk in 0..world_size {
        let off = buffer_offset(rank, chunk, world_size, max_chunk_bytes);
        let src = data_base.add(off) as *const f32;
        let dst = output[chunk * chunk_count..(chunk + 1) * chunk_count].as_mut_ptr();
        ptr::copy_nonoverlapping(src, dst, chunk_count);
    }
}

/// Ring Reduce-Scatter: reduces `input` and scatters — each rank gets 1/world_size
/// of the fully-reduced result.
///
/// Output length = input.len() / world_size. Rank `r` gets chunk `r`.
pub unsafe fn ring_reduce_scatter(
    input: &[f32],
    output: &mut [f32],
    data_base: *mut u8,
    all_signals: &[&Signal],
    rank: usize,
    world_size: usize,
) {
    let n = input.len();
    assert_eq!(n % world_size, 0);
    let chunk_count = n / world_size;
    let chunk_bytes = chunk_count * std::mem::size_of::<f32>();
    let max_chunk_bytes = chunk_bytes;
    assert_eq!(output.len(), chunk_count);

    // Copy input into buffer zone
    for chunk in 0..world_size {
        let off = buffer_offset(rank, chunk, world_size, max_chunk_bytes);
        let src = input[chunk * chunk_count..(chunk + 1) * chunk_count].as_ptr();
        ptr::copy_nonoverlapping(src, data_base.add(off) as *mut f32, chunk_count);
    }

    Signal::barrier_start(all_signals, rank, world_size, 0);

    let send_peer = (rank + 1) % world_size;

    // Reduce-Scatter phase only
    for step in 0..(world_size - 1) {
        let send_chunk = (rank + world_size - step) % world_size;
        let recv_chunk = (rank + world_size - step - 1) % world_size;

        let src_off = buffer_offset(rank, send_chunk, world_size, max_chunk_bytes);
        let dst_off = dest_offset(send_peer, send_chunk, world_size, max_chunk_bytes);
        ptr::copy_nonoverlapping(
            data_base.add(src_off),
            data_base.add(dst_off),
            chunk_bytes,
        );

        Signal::barrier_end(all_signals, rank, world_size, 0);

        let recv_dst_off = dest_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        let recv_buf_off = buffer_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        let received = std::slice::from_raw_parts(
            data_base.add(recv_dst_off) as *const f32,
            chunk_count,
        );
        let local = std::slice::from_raw_parts_mut(
            data_base.add(recv_buf_off) as *mut f32,
            chunk_count,
        );
        for i in 0..chunk_count {
            local[i] += received[i];
        }

        Signal::barrier_start(all_signals, rank, world_size, 0);
    }

    // After reduce-scatter, the fully reduced chunk at rank r is (r+1) % world_size.
    let my_chunk = (rank + 1) % world_size;
    let off = buffer_offset(rank, my_chunk, world_size, max_chunk_bytes);
    ptr::copy_nonoverlapping(
        data_base.add(off) as *const f32,
        output.as_mut_ptr(),
        chunk_count,
    );
}

/// Ring AllGather: each rank contributes `input` (length N), output is N * world_size
/// with rank r's data at output[r*N .. (r+1)*N].
pub unsafe fn ring_allgather(
    input: &[f32],
    output: &mut [f32],
    data_base: *mut u8,
    all_signals: &[&Signal],
    rank: usize,
    world_size: usize,
) {
    let chunk_count = input.len();
    let chunk_bytes = chunk_count * std::mem::size_of::<f32>();
    let max_chunk_bytes = chunk_bytes;
    assert_eq!(output.len(), chunk_count * world_size);

    // Write our input into our buffer zone at our chunk position
    let off = buffer_offset(rank, rank, world_size, max_chunk_bytes);
    ptr::copy_nonoverlapping(input.as_ptr(), data_base.add(off) as *mut f32, chunk_count);

    Signal::barrier_start(all_signals, rank, world_size, 0);

    let send_peer = (rank + 1) % world_size;

    // AllGather phase
    for step in 0..(world_size - 1) {
        let send_chunk = (rank + world_size - step) % world_size;

        let src_off = buffer_offset(rank, send_chunk, world_size, max_chunk_bytes);
        let dst_off = dest_offset(send_peer, send_chunk, world_size, max_chunk_bytes);
        ptr::copy_nonoverlapping(
            data_base.add(src_off),
            data_base.add(dst_off),
            chunk_bytes,
        );

        Signal::barrier_end(all_signals, rank, world_size, 0);

        let recv_chunk = (rank + world_size - step - 1) % world_size;
        let recv_dst_off = dest_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        let recv_buf_off = buffer_offset(rank, recv_chunk, world_size, max_chunk_bytes);
        ptr::copy_nonoverlapping(
            data_base.add(recv_dst_off) as *const u8,
            data_base.add(recv_buf_off),
            chunk_bytes,
        );

        Signal::barrier_start(all_signals, rank, world_size, 0);
    }

    // Copy all chunks to output
    for chunk in 0..world_size {
        let off = buffer_offset(rank, chunk, world_size, max_chunk_bytes);
        let dst = output[chunk * chunk_count..(chunk + 1) * chunk_count].as_mut_ptr();
        ptr::copy_nonoverlapping(data_base.add(off) as *const f32, dst, chunk_count);
    }
}
