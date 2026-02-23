#!/bin/bash
# Run benchmarks and print summary.
#
# Usage:
#   ./scripts/bench.sh          # CPU only
#   ./scripts/bench.sh --gpu    # CPU + Metal GPU
#
# Results saved to target/criterion/ with HTML reports.

set -e

FEATURES=""
if [ "$1" = "--gpu" ]; then
    FEATURES="--features metal"
    echo "Running CPU + Metal GPU benchmarks..."
else
    echo "Running CPU benchmarks..."
fi

# Get git metadata
COMMIT=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BRANCH=$(git branch --show-current 2>/dev/null || echo "unknown")

echo "Commit: $COMMIT  Branch: $BRANCH"
echo "---"

cargo bench $FEATURES 2>&1 | grep -E "^[a-z_]+/" | while IFS= read -r line; do
    echo "  $line"
done

echo ""
echo "HTML reports: target/criterion/report/index.html"
