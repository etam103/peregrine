#!/bin/bash
# Quick single-op benchmark for autoresearch loop.
# Builds in release, runs the wallclock benchmark, and extracts results for target ops.
#
# Usage: ./autoresearch/bench_quick.sh [op_pattern]
# Example: ./autoresearch/bench_quick.sh silu
#          ./autoresearch/bench_quick.sh ""  (all ops)
#
# Outputs tab-separated: op_name  peregrine_us  competitor_us  competitor  ratio

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

OP_PATTERN="${1:-}"

# Build and benchmark (suppress warnings)
cargo bench --bench wallclock 2>/dev/null | tail -1

# Compare against saved framework results and extract target ops
.venv/bin/python3 -c "
import json, sys, os

pattern = '$OP_PATTERN'

# Load Peregrine results
with open('target/bench_compare/peregrine.json') as f:
    data = json.load(f)
    results_list = data.get('results', data) if isinstance(data, dict) else data
    pg = {r['op']: r['median_us'] for r in results_list}

# Load all framework results
frameworks = {}
for name in ['pytorch', 'mlx', 'tensorflow', 'tinygrad', 'jax']:
    path = f'target/bench_compare/{name}.json'
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
            results_list = data.get('results', data) if isinstance(data, dict) else data
            frameworks[name] = {r['op']: r['median_us'] for r in results_list}

# For each Peregrine op, find the best competitor
results = []
for op, pg_us in sorted(pg.items()):
    if pattern and pattern.lower() not in op.lower():
        continue
    best_name = 'none'
    best_us = float('inf')
    for fw_name, fw_data in frameworks.items():
        if op in fw_data and fw_data[op] < best_us:
            best_us = fw_data[op]
            best_name = fw_name
    if best_us == float('inf'):
        best_us = 0
        best_name = 'none'
    ratio = pg_us / best_us if best_us > 0 else 0
    status = 'WIN' if ratio <= 1.0 else 'LOSE'
    results.append((op, pg_us, best_us, best_name, ratio, status))

# Print header and results
print(f'{'op':40s}\t{'peregrine':>10s}\t{'best':>10s}\t{'framework':>10s}\t{'ratio':>8s}\t{'status'}')
print('-' * 100)
for op, pg_us, best_us, best_name, ratio, status in results:
    print(f'{op:40s}\t{pg_us:10.1f}\t{best_us:10.1f}\t{best_name:>10s}\t{ratio:8.3f}\t{status}')

# Summary
wins = sum(1 for r in results if r[5] == 'WIN')
total = len(results)
print(f'\n{wins}/{total} wins')
"
