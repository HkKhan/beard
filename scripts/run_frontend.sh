#!/bin/bash
# Start the BeardAR frontend dev server

cd "$(dirname "$0")/../frontend"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Run dev server
npm run dev

