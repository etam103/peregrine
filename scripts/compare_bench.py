#!/usr/bin/env python3
"""Compare Peregrine vs multiple ML framework wall-clock benchmark results.

Reads JSON files from target/bench_compare/ and outputs a markdown
comparison table to target/bench_compare/comparison.md.
"""

import json
import math
import sys
from pathlib import Path

BENCH_DIR = Path("target/bench_compare")

# Frameworks to look for, in display order. Peregrine is always the baseline.
FRAMEWORKS = [
    ("peregrine", "peregrine.json", "Peregrine"),
    ("pytorch", "pytorch.json", "PyTorch"),
    ("mlx", "mlx.json", "MLX"),
    ("tensorflow", "tensorflow.json", "TensorFlow"),
    ("tinygrad", "tinygrad.json", "tinygrad"),
    ("jax", "jax.json", "JAX"),
]


def load_results(path):
    """Load benchmark JSON and return dict keyed by op name + raw data."""
    with open(path) as f:
        data = json.load(f)
    return {r["op"]: r for r in data["results"]}, data


def main():
    # Load Peregrine (required)
    pg_path = BENCH_DIR / "peregrine.json"
    if not pg_path.exists():
        print(f"ERROR: {pg_path} not found. Run Peregrine benchmark first.", file=sys.stderr)
        sys.exit(1)

    pg_results, pg_data = load_results(pg_path)
    op_order = [r["op"] for r in pg_data["results"]]

    # Load all available competitor frameworks
    competitors = []  # (key, display_name, results_dict, raw_data)
    for key, filename, display in FRAMEWORKS:
        if key == "peregrine":
            continue
        path = BENCH_DIR / filename
        if path.exists():
            results, raw = load_results(path)
            competitors.append((key, display, results, raw))

    if not competitors:
        print("ERROR: No competitor framework results found.", file=sys.stderr)
        sys.exit(1)

    # Version info
    lines = []
    lines.append("# Peregrine vs ML Frameworks — Wall-Clock Benchmark")
    lines.append("")
    lines.append("All benchmarks run on CPU with `nice -n 10`. Times in microseconds (lower is better).")
    lines.append("")
    lines.append("**Versions:**")
    for key, display, _, raw in competitors:
        version = (
            raw.get("torch_version")
            or raw.get("tf_version")
            or raw.get("mlx_version")
            or raw.get("version")
            or "?"
        )
        lines.append(f"- {display}: {version}")
    lines.append("")

    # --- Build multi-column table ---
    # Header
    header = "| Operation | Peregrine"
    sep = "|-----------|----------:"
    for _, display, _, _ in competitors:
        header += f" | {display}"
        sep += " | " + "-" * max(len(display), 8) + ":"
    header += " | Best |"
    sep += " | ----:|"
    lines.append(header)
    lines.append(sep)

    # Per-framework geometric mean accumulators: log_ratios[key] = [...]
    log_ratios = {key: [] for key, _, _, _ in competitors}

    for op in op_order:
        if op not in pg_results:
            continue

        pg_med = pg_results[op]["median_us"]
        row = f"| {op:<22s} | {pg_med:>9.1f}"

        # Track best time and who has it
        best_time = pg_med
        best_name = "Peregrine"

        for key, display, results, _ in competitors:
            if op in results:
                med = results[op]["median_us"]
                row += f" | {med:>{max(len(display), 8)}.1f}"
                if med < best_time:
                    best_time = med
                    best_name = display
                # Ratio: Peregrine / competitor (< 1 means Peregrine is faster)
                if med > 0:
                    ratio = pg_med / med
                    if not math.isinf(ratio) and ratio > 0:
                        log_ratios[key].append(math.log(ratio))
            else:
                row += f" | {'—':>{max(len(display), 8)}}"

        row += f" | {best_name} |"
        lines.append(row)

    # --- Geometric mean summary ---
    lines.append("")
    lines.append("**Geometric mean ratio (Peregrine / Framework):**")
    lines.append("- < 1.00 = Peregrine is faster")
    lines.append("- \\> 1.00 = Framework is faster")
    lines.append("")
    for key, display, _, _ in competitors:
        if log_ratios[key]:
            geo = math.exp(sum(log_ratios[key]) / len(log_ratios[key]))
            marker = "faster" if geo < 1.0 else "slower"
            lines.append(f"- **Peregrine vs {display}: {geo:.2f}x** (Peregrine is {marker})")

    lines.append("")

    # --- Per-framework wins count ---
    wins = {"Peregrine": 0}
    for _, display, _, _ in competitors:
        wins[display] = 0

    for op in op_order:
        if op not in pg_results:
            continue
        best_time = pg_results[op]["median_us"]
        best_name = "Peregrine"
        for _, display, results, _ in competitors:
            if op in results and results[op]["median_us"] < best_time:
                best_time = results[op]["median_us"]
                best_name = display
        wins[best_name] += 1

    lines.append("**Wins by framework:**")
    for name, count in sorted(wins.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"- {name}: {count}/{len(op_order)} ops")
    lines.append("")
    lines.append("---")
    lines.append("*Median of timed iterations (warmup excluded). Lower is better.*")
    lines.append("")

    md = "\n".join(lines)

    out_path = BENCH_DIR / "comparison.md"
    with open(out_path, "w") as f:
        f.write(md)

    print(md)
    print(f"\nSaved to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
