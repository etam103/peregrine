use crate::comm::bootstrap;
use crate::comm::error::CommError;
use crate::comm::ring;
use crate::comm::shm::ShmRegion;
use crate::comm::signal::{self, Signal};
use crate::tensor::Tensor;

/// Maximum bytes per chunk in the data region. 64 MB default — supports allreduce
/// on tensors up to 64MB * world_size (e.g. 256MB for 4 ranks).
const MAX_CHUNK_BYTES: usize = 64 * 1024 * 1024;

pub struct Communicator {
    rank: usize,
    world_size: usize,
    signal_region: ShmRegion,
    data_region: ShmRegion,
}

impl Communicator {
    /// Create a communicator by reading env vars and bootstrapping.
    ///
    /// Required env vars: PEREGRINE_RANK, PEREGRINE_WORLD_SIZE
    /// Optional: PEREGRINE_MASTER_ADDR (default: /tmp/peregrine_comm.sock)
    pub fn from_env() -> Result<Self, CommError> {
        let (rank, world_size, master_addr) = bootstrap::read_env()?;
        Self::new(rank, world_size, &master_addr)
    }

    /// Create a communicator with explicit rank, world_size, and master address.
    pub fn new(rank: usize, world_size: usize, master_addr: &str) -> Result<Self, CommError> {
        if rank >= world_size {
            return Err(CommError::InvalidRank { rank, world_size });
        }

        let session_id = bootstrap::bootstrap(rank, world_size, master_addr)?;

        let sig_name = bootstrap::shm_signal_name(session_id);
        let data_name = bootstrap::shm_data_name(session_id);
        let sig_size = signal::signal_region_size(world_size);
        let data_size = ring::data_region_size(world_size, MAX_CHUNK_BYTES);

        // Rank 0 creates, others open
        let signal_region = if rank == 0 {
            ShmRegion::create(&sig_name, sig_size)?
        } else {
            // Small delay to ensure rank 0 has created the region
            std::thread::sleep(std::time::Duration::from_millis(10));
            ShmRegion::open(&sig_name, sig_size)?
        };

        let data_region = if rank == 0 {
            ShmRegion::create(&data_name, data_size)?
        } else {
            ShmRegion::open(&data_name, data_size)?
        };

        // Barrier to ensure all ranks have mapped shared memory
        let all_signals = Self::get_all_signals(&signal_region, world_size);
        Signal::barrier_start(&all_signals, rank, world_size, 0);

        Ok(Communicator {
            rank,
            world_size,
            signal_region,
            data_region,
        })
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// AllReduce (sum): every rank gets the element-wise sum of all inputs.
    /// Input tensor must have the same shape on all ranks, and total elements
    /// must be divisible by world_size.
    pub fn allreduce(&self, input: &Tensor) -> Result<Tensor, CommError> {
        let data = input.data();
        let shape = input.shape();
        let n = data.len();

        if n % self.world_size != 0 {
            return Err(CommError::NotDivisible {
                total: n,
                world_size: self.world_size,
            });
        }

        let chunk_bytes = (n / self.world_size) * std::mem::size_of::<f32>();
        if chunk_bytes > MAX_CHUNK_BYTES {
            return Err(CommError::SizeMismatch {
                expected: MAX_CHUNK_BYTES,
                got: chunk_bytes,
            });
        }

        let mut output = vec![0.0f32; n];
        let all_signals = self.get_signals();

        unsafe {
            ring::ring_allreduce(
                &data,
                &mut output,
                self.data_region.as_ptr(),
                &all_signals,
                self.rank,
                self.world_size,
            );
        }

        Ok(Tensor::new(output, shape, false))
    }

    /// Reduce-Scatter (sum): each rank gets 1/world_size of the reduced result.
    /// Rank r gets the r-th chunk. Output shape has first dim divided by world_size.
    pub fn reduce_scatter(&self, input: &Tensor) -> Result<Tensor, CommError> {
        let data = input.data();
        let shape = input.shape();
        let n = data.len();

        if n % self.world_size != 0 {
            return Err(CommError::NotDivisible {
                total: n,
                world_size: self.world_size,
            });
        }

        let chunk_count = n / self.world_size;
        let chunk_bytes = chunk_count * std::mem::size_of::<f32>();
        if chunk_bytes > MAX_CHUNK_BYTES {
            return Err(CommError::SizeMismatch {
                expected: MAX_CHUNK_BYTES,
                got: chunk_bytes,
            });
        }

        let mut output = vec![0.0f32; chunk_count];
        let all_signals = self.get_signals();

        unsafe {
            ring::ring_reduce_scatter(
                &data,
                &mut output,
                self.data_region.as_ptr(),
                &all_signals,
                self.rank,
                self.world_size,
            );
        }

        // Adjust shape: divide first dim by world_size
        let mut out_shape = shape;
        out_shape[0] /= self.world_size;
        Ok(Tensor::new(output, out_shape, false))
    }

    /// AllGather: each rank contributes its input, all ranks get the concatenation.
    /// Output has first dim multiplied by world_size.
    pub fn allgather(&self, input: &Tensor) -> Result<Tensor, CommError> {
        let data = input.data();
        let shape = input.shape();
        let chunk_count = data.len();

        let chunk_bytes = chunk_count * std::mem::size_of::<f32>();
        if chunk_bytes > MAX_CHUNK_BYTES {
            return Err(CommError::SizeMismatch {
                expected: MAX_CHUNK_BYTES,
                got: chunk_bytes,
            });
        }

        let mut output = vec![0.0f32; chunk_count * self.world_size];
        let all_signals = self.get_signals();

        unsafe {
            ring::ring_allgather(
                &data,
                &mut output,
                self.data_region.as_ptr(),
                &all_signals,
                self.rank,
                self.world_size,
            );
        }

        let mut out_shape = shape;
        out_shape[0] *= self.world_size;
        Ok(Tensor::new(output, out_shape, false))
    }

    /// Global barrier — blocks until all ranks arrive.
    pub fn barrier(&self) {
        let all_signals = self.get_signals();
        Signal::barrier_start(&all_signals, self.rank, self.world_size, 0);
    }

    fn get_signals(&self) -> Vec<&Signal> {
        Self::get_all_signals(&self.signal_region, self.world_size)
    }

    fn get_all_signals(signal_region: &ShmRegion, world_size: usize) -> Vec<&Signal> {
        (0..world_size)
            .map(|r| unsafe { signal::signal_for_rank(signal_region.as_ptr(), r) })
            .collect()
    }
}
