#!/bin/bash
# Compare all frameworks on same questions

NUM_QUESTIONS="${1:-10}"
MODEL_SIZE="${2:-large}"

echo "=========================================="
echo "GAIA Framework Comparison"
echo "=========================================="
echo "Questions: $NUM_QUESTIONS"
echo "Model: $MODEL_IDX"
echo ""

docker-compose run --rm \
    -e TEST_MODE="compare" \
    -e MODEL_IDX="$MODEL_IDX" \
    -e NUM_QUESTIONS="$NUM_QUESTIONS" \
    -e OUTPUT_DIR="/app/output" \
    gaia-agent

echo ""
echo "Comparison complete! Check output/comparison_*.json"
