#!/bin/bash
# Start the BeardAR backend server

cd "$(dirname "$0")/.."

# Kill any process running on port 8000
echo "Killing processes on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true

# Kill any existing uvicorn processes
echo "Killing existing uvicorn processes..."
pkill -f "uvicorn main:app" 2>/dev/null || true

# Kill any existing python processes running main.py
echo "Killing existing backend processes..."
pkill -f "python.*main.py" 2>/dev/null || true
pkill -f "python -m uvicorn" 2>/dev/null || true

# Wait a moment for processes to fully terminate
sleep 1

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export SAM_CHECKPOINT="$(pwd)/checkpoints/sam_vit_b_01ec64.pth"
export FINE_TUNED_CHECKPOINT="$(pwd)/checkpoints/sam_beard_best.pth"
export SAM_MODEL_TYPE="vit_b"

# Run the server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload


