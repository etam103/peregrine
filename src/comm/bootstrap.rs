use crate::comm::error::CommError;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};

/// Bootstrap info exchanged during process discovery.
pub struct BootstrapInfo {
    pub rank: usize,
    pub world_size: usize,
    pub session_id: u64,
}

/// Read PEREGRINE_RANK, PEREGRINE_WORLD_SIZE, PEREGRINE_MASTER_ADDR from environment.
pub fn read_env() -> Result<(usize, usize, String), CommError> {
    let rank: usize = std::env::var("PEREGRINE_RANK")
        .map_err(|_| CommError::BootstrapFailed("PEREGRINE_RANK not set".into()))?
        .parse()
        .map_err(|_| CommError::BootstrapFailed("PEREGRINE_RANK not a number".into()))?;

    let world_size: usize = std::env::var("PEREGRINE_WORLD_SIZE")
        .map_err(|_| CommError::BootstrapFailed("PEREGRINE_WORLD_SIZE not set".into()))?
        .parse()
        .map_err(|_| CommError::BootstrapFailed("PEREGRINE_WORLD_SIZE not a number".into()))?;

    let master_addr = std::env::var("PEREGRINE_MASTER_ADDR")
        .unwrap_or_else(|_| "/tmp/peregrine_comm.sock".to_string());

    if rank >= world_size {
        return Err(CommError::InvalidRank { rank, world_size });
    }

    Ok((rank, world_size, master_addr))
}

/// Perform bootstrap: rank 0 listens, generates session_id, broadcasts to all ranks.
/// Other ranks connect to rank 0 and receive session_id.
pub fn bootstrap(rank: usize, world_size: usize, master_addr: &str) -> Result<u64, CommError> {
    if rank == 0 {
        bootstrap_rank0(world_size, master_addr)
    } else {
        bootstrap_worker(rank, master_addr)
    }
}

fn bootstrap_rank0(world_size: usize, master_addr: &str) -> Result<u64, CommError> {
    // Clean up stale socket
    let _ = std::fs::remove_file(master_addr);

    let listener = UnixListener::bind(master_addr)
        .map_err(|e| CommError::BootstrapFailed(format!("bind {}: {}", master_addr, e)))?;

    // Generate random session_id
    let session_id = {
        let mut buf = [0u8; 8];
        let fd = unsafe { libc::open(b"/dev/urandom\0".as_ptr() as *const _, libc::O_RDONLY) };
        if fd < 0 {
            return Err(CommError::BootstrapFailed("open /dev/urandom failed".into()));
        }
        unsafe {
            libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, 8);
            libc::close(fd);
        }
        u64::from_le_bytes(buf)
    };

    // Accept connections from all other ranks
    let mut connections: Vec<(usize, UnixStream)> = Vec::with_capacity(world_size - 1);
    for _ in 0..(world_size - 1) {
        let (mut stream, _) = listener
            .accept()
            .map_err(|e| CommError::BootstrapFailed(format!("accept: {}", e)))?;

        // Read the rank from the worker
        let mut rank_buf = [0u8; 8];
        stream
            .read_exact(&mut rank_buf)
            .map_err(|e| CommError::BootstrapFailed(format!("read rank: {}", e)))?;
        let peer_rank = usize::from_le_bytes(rank_buf);
        connections.push((peer_rank, stream));
    }

    // Send session_id to all workers
    let session_bytes = session_id.to_le_bytes();
    for (_, stream) in &mut connections {
        stream
            .write_all(&session_bytes)
            .map_err(|e| CommError::BootstrapFailed(format!("write session_id: {}", e)))?;
    }

    // Clean up socket
    let _ = std::fs::remove_file(master_addr);

    Ok(session_id)
}

fn bootstrap_worker(rank: usize, master_addr: &str) -> Result<u64, CommError> {
    // Retry connecting — rank 0 may not be listening yet
    let mut stream = None;
    for attempt in 0..100 {
        match UnixStream::connect(master_addr) {
            Ok(s) => {
                stream = Some(s);
                break;
            }
            Err(_) if attempt < 99 => {
                std::thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(e) => {
                return Err(CommError::BootstrapFailed(format!(
                    "connect to {}: {}",
                    master_addr, e
                )));
            }
        }
    }
    let mut stream = stream.unwrap();

    // Send our rank
    stream
        .write_all(&rank.to_le_bytes())
        .map_err(|e| CommError::BootstrapFailed(format!("write rank: {}", e)))?;

    // Receive session_id
    let mut session_buf = [0u8; 8];
    stream
        .read_exact(&mut session_buf)
        .map_err(|e| CommError::BootstrapFailed(format!("read session_id: {}", e)))?;

    Ok(u64::from_le_bytes(session_buf))
}

/// Derive shared memory region names from session_id.
/// macOS limits POSIX shm names to 31 chars, so we use the lower 32 bits.
pub fn shm_signal_name(session_id: u64) -> String {
    format!("/pgn_{:08x}_s", session_id as u32)
}

pub fn shm_data_name(session_id: u64) -> String {
    format!("/pgn_{:08x}_d", session_id as u32)
}
