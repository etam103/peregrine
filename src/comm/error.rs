use std::fmt;

#[derive(Debug)]
pub enum CommError {
    ShmOpenFailed(String),
    MmapFailed(String),
    InvalidRank { rank: usize, world_size: usize },
    SizeMismatch { expected: usize, got: usize },
    NotDivisible { total: usize, world_size: usize },
    BootstrapFailed(String),
    Timeout,
}

impl fmt::Display for CommError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CommError::ShmOpenFailed(msg) => write!(f, "shm_open failed: {}", msg),
            CommError::MmapFailed(msg) => write!(f, "mmap failed: {}", msg),
            CommError::InvalidRank { rank, world_size } => {
                write!(f, "invalid rank {} for world_size {}", rank, world_size)
            }
            CommError::SizeMismatch { expected, got } => {
                write!(f, "size mismatch: expected {}, got {}", expected, got)
            }
            CommError::NotDivisible { total, world_size } => {
                write!(
                    f,
                    "tensor size {} not divisible by world_size {}",
                    total, world_size
                )
            }
            CommError::BootstrapFailed(msg) => write!(f, "bootstrap failed: {}", msg),
            CommError::Timeout => write!(f, "operation timed out"),
        }
    }
}

impl std::error::Error for CommError {}
