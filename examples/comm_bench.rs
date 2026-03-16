/// Collective communication benchmark.
///
/// Usage: cargo run --features comm --example comm_bench --release -- <num_workers> [size_mb]
///
/// Spawns `num_workers` child processes, each running an allreduce on a tensor
/// of `size_mb` MB (default 16). Reports throughput in GB/s.
use std::env;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    // If PEREGRINE_RANK is set, we're a worker
    if env::var("PEREGRINE_RANK").is_ok() {
        worker_main();
        return;
    }

    // Otherwise, we're the launcher
    let num_workers: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(4);
    let size_mb: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);

    println!("=== Peregrine Comm Benchmark ===");
    println!("Workers: {}", num_workers);
    println!("Tensor size: {} MB ({} floats)", size_mb, size_mb * 1024 * 1024 / 4);
    println!();

    let sock_path = format!("/tmp/peregrine_bench_{}.sock", std::process::id());
    let exe = env::current_exe().expect("current_exe");

    let mut children = Vec::new();
    for rank in 0..num_workers {
        let child = Command::new(&exe)
            .env("PEREGRINE_RANK", rank.to_string())
            .env("PEREGRINE_WORLD_SIZE", num_workers.to_string())
            .env("PEREGRINE_MASTER_ADDR", &sock_path)
            .env("PEREGRINE_SIZE_MB", size_mb.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .expect("failed to spawn worker");
        children.push(child);
    }

    for child in children {
        let output = child.wait_with_output().expect("wait");
        let text = String::from_utf8_lossy(&output.stdout);
        if !text.is_empty() {
            print!("{}", text);
        }
        assert!(output.status.success(), "worker exited with error");
    }

    let _ = std::fs::remove_file(&sock_path);
}

fn worker_main() {
    use peregrine::comm::Communicator;
    use peregrine::tensor::Tensor;

    let comm = Communicator::from_env().expect("communicator init");
    let rank = comm.rank();
    let world_size = comm.world_size();

    let size_mb: usize = env::var("PEREGRINE_SIZE_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let num_floats = size_mb * 1024 * 1024 / 4;
    // Round to multiple of world_size
    let num_floats = (num_floats / world_size) * world_size;

    let data: Vec<f32> = (0..num_floats).map(|i| (i as f32) * 0.001 + rank as f32).collect();
    let input = Tensor::new(data, vec![num_floats], false);

    // Warmup
    let _ = comm.allreduce(&input).expect("warmup allreduce");
    comm.barrier();

    // Benchmark allreduce
    let num_iters = 10;
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = comm.allreduce(&input).expect("allreduce");
    }
    comm.barrier();
    let elapsed = start.elapsed();

    if rank == 0 {
        let bytes_per_iter = num_floats * 4;
        // Ring allreduce transfers 2*(N-1)/N * data_size across the ring
        let algo_bytes = 2.0 * (world_size as f64 - 1.0) / world_size as f64 * bytes_per_iter as f64;
        let total_algo_bytes = algo_bytes * num_iters as f64;
        let bus_bw = total_algo_bytes / elapsed.as_secs_f64() / 1e9;
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / num_iters as f64;

        println!("AllReduce: {} MB x {} iters", size_mb, num_iters);
        println!("  Avg latency: {:.2} ms", avg_ms);
        println!("  Bus bandwidth: {:.2} GB/s", bus_bw);
        println!("  Total time: {:.2} ms", elapsed.as_secs_f64() * 1000.0);
    }

    // Benchmark reduce_scatter
    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = comm.reduce_scatter(&input).expect("reduce_scatter");
    }
    comm.barrier();
    let elapsed = start.elapsed();

    if rank == 0 {
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / num_iters as f64;
        println!("\nReduceScatter: {} MB x {} iters", size_mb, num_iters);
        println!("  Avg latency: {:.2} ms", avg_ms);
    }

    // Benchmark allgather
    let chunk_size = num_floats / world_size;
    let chunk_data: Vec<f32> = (0..chunk_size).map(|i| i as f32 * 0.001).collect();
    let chunk_input = Tensor::new(chunk_data, vec![chunk_size], false);

    let start = Instant::now();
    for _ in 0..num_iters {
        let _ = comm.allgather(&chunk_input).expect("allgather");
    }
    comm.barrier();
    let elapsed = start.elapsed();

    if rank == 0 {
        let avg_ms = elapsed.as_secs_f64() * 1000.0 / num_iters as f64;
        println!("\nAllGather: {} MB x {} iters", size_mb / world_size, num_iters);
        println!("  Avg latency: {:.2} ms", avg_ms);
    }
}
