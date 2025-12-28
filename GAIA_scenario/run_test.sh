#!/bin/bash
# Iterate through different framework/model combinations

OUTPUT_DIR="./output"
mkdir -p "$OUTPUT_DIR"

# Define test configurations
FRAMEWORKS=("crewai")  # Add more: "langgraph" "openai"
NUM_QUESTIONS="${1:-5}"  # Default to 5, or use first argument

echo "=========================================="
echo "GAIA Multi-Framework Test Suite"
echo "=========================================="
echo "Testing $NUM_QUESTIONS questions per combination"
echo ""

# Iterate through all combinations
for FRAMEWORK in "${FRAMEWORKS[@]}"; do
    for MODEL_IDX in range(6); do
        echo ""
        echo "Testing: $FRAMEWORK with $MODEL_IDX model"
        echo "------------------------------------------"
        
        docker-compose run --rm \
            -e FRAMEWORK="$FRAMEWORK" \
            -e MODEL_IDX="$MODEL_IDX" \
            -e NUM_QUESTIONS="$NUM_QUESTIONS" \
            -e OUTPUT_DIR="/app/output" \
            gaia-agent
        
        echo "✓ Completed: $FRAMEWORK - $MODEL_IDX"
        echo ""
        sleep 2  # Brief pause between tests
    done
done

echo ""
echo "=========================================="
echo "All tests complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="

# Show summary
echo ""
echo "Results files:"
ls -lh "$OUTPUT_DIR"/*.json 2>/dev/null || echo "No results yet"
