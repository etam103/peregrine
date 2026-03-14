#!/bin/bash
# Peregrine vs ML Frameworks wall-clock benchmark orchestrator.
#
# Builds Peregrine in release mode, then runs each framework benchmark
# sequentially with nice -n 10 to keep resource usage under 80%.
# Peregrine runs first (cold CPU) for fairest comparison.
# 30-second cooldown between frameworks to reduce thermal variance.
#
# Usage: ./scripts/bench_compare.sh [--rounds N]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OUT_DIR="target/bench_compare"
mkdir -p "$OUT_DIR"

ROUNDS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --rounds)
            ROUNDS="$2"
            shift 2
            ;;
        *)
            echo "Usage: $0 [--rounds N]"
            exit 1
            ;;
    esac
done

echo "=== Peregrine vs ML Frameworks — Wall-Clock Benchmark ==="
if [ "$ROUNDS" -gt 1 ]; then
    echo "Running $ROUNDS rounds per framework (taking minimum times)"
fi
echo ""

# --- Build Peregrine in release mode ---
echo "[1/8] Building Peregrine (release)..."
cargo build --release --bench wallclock 2>&1 | tail -1
echo ""

cooldown() {
    echo "    [cooldown] Waiting 30s for CPU to cool..."
    sleep 30
}

for round in $(seq 1 "$ROUNDS"); do
    if [ "$ROUNDS" -gt 1 ]; then
        echo "========== Round $round/$ROUNDS =========="
    fi

    # --- Run Peregrine benchmark (first — cold CPU) ---
    echo "[2/8] Running Peregrine benchmark..."
    nice -n 10 cargo bench --bench wallclock 2>&1
    echo ""
    cooldown

    # --- Run PyTorch benchmark ---
    echo "[3/8] Running PyTorch benchmark..."
    nice -n 10 .venv/bin/python scripts/bench_pytorch.py
    echo ""
    cooldown

    # --- Run MLX benchmark ---
    echo "[4/8] Running MLX benchmark..."
    nice -n 10 .venv/bin/python scripts/bench_mlx.py
    echo ""
    cooldown

    # --- Run TensorFlow benchmark ---
    echo "[5/8] Running TensorFlow benchmark..."
    nice -n 10 .venv/bin/python scripts/bench_tensorflow.py
    echo ""
    cooldown

    # --- Run tinygrad benchmark ---
    echo "[6/8] Running tinygrad benchmark..."
    nice -n 10 .venv/bin/python scripts/bench_tinygrad.py
    echo ""
    cooldown

    # --- Run JAX benchmark ---
    echo "[7/8] Running JAX benchmark..."
    nice -n 10 .venv/bin/python scripts/bench_jax.py
    echo ""

    if [ "$round" -lt "$ROUNDS" ]; then
        cooldown
    fi
done

# --- Compare results ---
echo "[8/8] Comparing results..."
echo ""
.venv/bin/python scripts/compare_bench.py
echo ""
echo "=== Done ==="
