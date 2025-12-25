#!/bin/bash

echo "========================================"
echo "Tourism Data Monitor Frontend"
echo "========================================"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "[1/2] Installing dependencies..."
    echo ""
    yarn install
    echo ""
else
    echo "[INFO] Dependencies already installed"
    echo ""
fi

echo "[2/2] Starting development server..."
echo ""
echo "Backend should be running on http://localhost:8080"
echo "Frontend will start on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "========================================"

yarn dev
