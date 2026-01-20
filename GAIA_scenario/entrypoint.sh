#!/bin/bash
set -e

echo "======================================"
echo "GAIA Agent Container Starting"
echo "======================================"

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "WARNING: HF_TOKEN not set. Cannot download GAIA dataset."
    echo "Please set HF_TOKEN environment variable."
    exit 1
fi

echo ""
echo "Downloading GAIA dataset (isolated in container)..."
echo "This may take a few minutes on first run..."
python3 data/data_pull.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ GAIA dataset downloaded successfully"
    echo "✓ Data is isolated within this container"
else
    echo ""
    echo "✗ Failed to download GAIA dataset"
    exit 1
fi

echo ""
echo "======================================"
echo "Starting GAIA Test Suite..."
echo "======================================"
echo ""

# Always run the modular test framework
exec python3 gaia_tester.py "$@"
