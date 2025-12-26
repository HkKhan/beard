#!/bin/bash
# Start the BeardAR frontend dev server

cd "$(dirname "$0")/../frontend"

# Kill any process running on port 3000
echo "Killing processes on port 3000..."
lsof -ti:3000 | xargs kill -9 2>/dev/null || true

# Kill any existing vite processes
echo "Killing existing vite processes..."
pkill -f "vite" 2>/dev/null || true

# Kill any existing npm/node processes for this frontend
echo "Killing existing frontend processes..."
pkill -f "npm run dev" 2>/dev/null || true
pkill -f "node.*vite" 2>/dev/null || true

# Wait a moment for processes to fully terminate
sleep 1

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Run dev server
npm run dev


