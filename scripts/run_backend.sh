#!/bin/bash
# Start the BeardAR backend server

cd "$(dirname "$0")/.."

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export SAM_CHECKPOINT="$(pwd)/checkpoints/sam_vit_b_01ec64.pth"
export FINE_TUNED_CHECKPOINT="$(pwd)/checkpoints/sam_beard_best.pth"
export SAM_MODEL_TYPE="vit_b"

# Run the server
cd backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload

