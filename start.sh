#!/usr/bin/env bash
# ContextCache — one-command startup
# Usage:
#   ./start.sh          # Demo mode (no GPU)
#   ./start.sh --live   # Live mode (requires GPU)

set -e
cd "$(dirname "$0")"

MODE="--demo"
EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --live) MODE="" ;;
        *)      EXTRA_ARGS="$EXTRA_ARGS $arg" ;;
    esac
done

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate
source .venv/bin/activate

# Install dependencies
if [ -n "$MODE" ]; then
    echo "Installing demo dependencies..."
    pip install -q fastapi uvicorn pydantic pyyaml
else
    echo "Installing full dependencies (including PyTorch)..."
    pip install -q -r requirements.txt
fi

echo ""
python scripts/serve/launch.py $MODE $EXTRA_ARGS
