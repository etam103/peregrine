#![cfg(feature = "comm")]

use std::env;
use std::process::{Command, Stdio};

fn spawn_workers(num_workers: usize, test_name: &str) -> Vec<std::process::Output> {
    let exe = env::current_exe().expect("current_exe");
    let sock_path = format!(
        "/tmp/peregrine_test_{}_{}.sock",
        test_name,
        std::process::id()
    );

    let mut children = Vec::new();
    for rank in 0..num_workers {
        let child = Command::new(&exe)
            .args(["--ignored", &format!("{}_worker", test_name)])
            .env("PEREGRINE_RANK", rank.to_string())
            .env("PEREGRINE_WORLD_SIZE", num_workers.to_string())
            .env("PEREGRINE_MASTER_ADDR", &sock_path)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("failed to spawn worker");
        children.push(child);
    }

    let mut outputs = Vec::new();
    for child in children {
        outputs.push(child.wait_with_output().expect("wait"));
    }
    let _ = std::fs::remove_file(&sock_path);
    outputs
}

fn assert_all_success(outputs: &[std::process::Output]) {
    for (i, output) in outputs.iter().enumerate() {
        if !output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!(
                "Worker {} failed (exit {:?}):\nstdout: {}\nstderr: {}",
                i, output.status.code(), stdout, stderr
            );
        }
    }
}

// ---- Launcher tests (spawn child processes) ----

#[test]
fn test_allreduce_2_ranks() {
    let outputs = spawn_workers(2, "test_allreduce_2_ranks");
    assert_all_success(&outputs);
}

#[test]
fn test_allreduce_4_ranks() {
    let outputs = spawn_workers(4, "test_allreduce_4_ranks");
    assert_all_success(&outputs);
}

#[test]
fn test_reduce_scatter_2_ranks() {
    let outputs = spawn_workers(2, "test_reduce_scatter_2_ranks");
    assert_all_success(&outputs);
}

#[test]
fn test_allgather_2_ranks() {
    let outputs = spawn_workers(2, "test_allgather_2_ranks");
    assert_all_success(&outputs);
}

// ---- Worker tests (run inside spawned child processes) ----

#[test]
#[ignore]
fn test_allreduce_2_ranks_worker() {
    use peregrine::comm::Communicator;
    use peregrine::tensor::Tensor;

    let comm = Communicator::from_env().expect("init");
    let rank = comm.rank();
    let world_size = comm.world_size();
    let n = 1024;

    // Each rank has [rank+1, rank+1, ...] as input
    let val = (rank + 1) as f32;
    let data = vec![val; n];
    let input = Tensor::new(data, vec![n], false);

    let output = comm.allreduce(&input).expect("allreduce");
    let result = output.data();

    // Expected: sum of 1..=world_size for each element
    let expected: f32 = (1..=world_size).map(|r| r as f32).sum();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "rank {} index {}: got {}, expected {}",
            rank,
            i,
            v,
            expected
        );
    }
}

#[test]
#[ignore]
fn test_allreduce_4_ranks_worker() {
    use peregrine::comm::Communicator;
    use peregrine::tensor::Tensor;

    let comm = Communicator::from_env().expect("init");
    let rank = comm.rank();
    let world_size = comm.world_size();
    let n = 2048;

    let val = (rank + 1) as f32;
    let data = vec![val; n];
    let input = Tensor::new(data, vec![n], false);

    let output = comm.allreduce(&input).expect("allreduce");
    let result = output.data();

    let expected: f32 = (1..=world_size).map(|r| r as f32).sum();
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "rank {} index {}: got {}, expected {}",
            rank,
            i,
            v,
            expected
        );
    }
}

#[test]
#[ignore]
fn test_reduce_scatter_2_ranks_worker() {
    use peregrine::comm::Communicator;
    use peregrine::tensor::Tensor;

    let comm = Communicator::from_env().expect("init");
    let rank = comm.rank();
    let world_size = comm.world_size();
    let n = 512; // per rank total, must be divisible by world_size

    // Each rank has [rank+1, rank+1, ...]
    let val = (rank + 1) as f32;
    let data = vec![val; n];
    let input = Tensor::new(data, vec![n], false);

    let output = comm.reduce_scatter(&input).expect("reduce_scatter");
    let result = output.data();

    let expected: f32 = (1..=world_size).map(|r| r as f32).sum();
    let chunk_size = n / world_size;
    assert_eq!(result.len(), chunk_size);
    for (i, &v) in result.iter().enumerate() {
        assert!(
            (v - expected).abs() < 1e-4,
            "rank {} index {}: got {}, expected {}",
            rank,
            i,
            v,
            expected
        );
    }
}

#[test]
#[ignore]
fn test_allgather_2_ranks_worker() {
    use peregrine::comm::Communicator;
    use peregrine::tensor::Tensor;

    let comm = Communicator::from_env().expect("init");
    let rank = comm.rank();
    let world_size = comm.world_size();
    let chunk_size = 256;

    // Each rank contributes [rank+1, rank+1, ...]
    let val = (rank + 1) as f32;
    let data = vec![val; chunk_size];
    let input = Tensor::new(data, vec![chunk_size], false);

    let output = comm.allgather(&input).expect("allgather");
    let result = output.data();

    assert_eq!(result.len(), chunk_size * world_size);
    for r in 0..world_size {
        let expected = (r + 1) as f32;
        for i in 0..chunk_size {
            let idx = r * chunk_size + i;
            assert!(
                (result[idx] - expected).abs() < 1e-4,
                "rank {} gather index {}: got {}, expected {}",
                rank,
                idx,
                result[idx],
                expected
            );
        }
    }
}
