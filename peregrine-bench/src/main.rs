mod hardware;
mod suite;

use serde::Serialize;

#[derive(Serialize)]
struct BenchOutput {
    schema_version: u32,
    framework: String,
    version: String,
    timestamp: String,
    hardware: hardware::HardwareInfo,
    config: BenchConfig,
    results: Vec<suite::BenchResult>,
}

#[derive(Serialize)]
struct BenchConfig {
    mode: String,
}

fn timestamp() -> String {
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap();
    let secs = d.as_secs();
    // Simple UTC timestamp without chrono
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Days since epoch to Y-M-D (simplified)
    let mut y = 1970i64;
    let mut remaining = days as i64;
    loop {
        let year_days = if y % 4 == 0 && (y % 100 != 0 || y % 400 == 0) { 366 } else { 365 };
        if remaining < year_days { break; }
        remaining -= year_days;
        y += 1;
    }
    let leap = y % 4 == 0 && (y % 100 != 0 || y % 400 == 0);
    let month_days = [31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let mut m = 0;
    for &md in &month_days {
        if remaining < md { break; }
        remaining -= md;
        m += 1;
    }

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, m + 1, remaining + 1, hours, minutes, seconds)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_mode = args.iter().any(|a| a == "--json");
    let quick = args.iter().any(|a| a == "--quick");
    let output_file = args.windows(2)
        .find(|w| w[0] == "--output")
        .map(|w| w[1].clone());

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("peregrine-bench — Reproducible benchmark suite for Peregrine");
        eprintln!();
        eprintln!("Usage: peregrine-bench [OPTIONS]");
        eprintln!();
        eprintln!("Options:");
        eprintln!("  --json          Output JSON to stdout");
        eprintln!("  --quick         Fewer iterations, skip largest benchmarks");
        eprintln!("  --output FILE   Write JSON to file");
        eprintln!("  --help          Show this help");
        return;
    }

    // Detect hardware
    let hw = hardware::detect();
    eprintln!("peregrine-bench v{}", env!("CARGO_PKG_VERSION"));
    eprintln!("Hardware: {} ({}, {}GB RAM, {} cores [{}/{}P/{}E], macOS {})",
        hw.chip, hw.model, hw.memory_gb, hw.cores_total,
        hw.arch, hw.cores_perf, hw.cores_efficiency, hw.os_version);
    eprintln!("Mode: {}", if quick { "quick" } else { "full" });
    eprintln!();

    // Run benchmarks
    let results = suite::run_all_cpu(quick);

    // Build output
    let output = BenchOutput {
        schema_version: 1,
        framework: "peregrine".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp: timestamp(),
        hardware: hw,
        config: BenchConfig {
            mode: if quick { "quick" } else { "full" }.to_string(),
        },
        results: results.clone(),
    };

    // Print summary table
    if !json_mode {
        eprintln!();
        eprintln!("  {:<40} {:>12}", "Operation", "Median (us)");
        eprintln!("  {}", "-".repeat(54));
        for r in &results {
            eprintln!("  {:<40} {:>12.1}", r.op, r.median_us);
        }
        eprintln!("  {}", "-".repeat(54));
        eprintln!("  {} benchmarks completed", results.len());
    }

    let json_str = serde_json::to_string_pretty(&output).unwrap();

    if let Some(path) = output_file {
        std::fs::write(&path, &json_str).expect("write output file");
        eprintln!("\nSaved to {}", path);
    } else if json_mode {
        println!("{}", json_str);
    } else {
        // Default: save next to the binary
        let default_path = "peregrine_bench_results.json";
        std::fs::write(default_path, &json_str).expect("write output file");
        eprintln!("\nSaved to {}", default_path);
    }
}
