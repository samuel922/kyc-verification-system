#!/bin/bash

echo "Starting Face Swap Application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed or not in PATH"
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js is not installed or not in PATH"
    exit 1
fi

echo "Python and Node.js found!"
echo ""

# Start AI Service
echo "Starting AI Service..."
cd backend/ai-service
if command -v python3 &> /dev/null; then
    python3 main.py &
else
    python main.py &
fi
AI_PID=$!
cd ../..

# Wait a moment for AI service to start
sleep 3

# Start API Server
echo "Starting API Server..."
cd backend/api-server
npm run dev &
API_PID=$!
cd ../..

echo ""
echo "========================================"
echo "   Face Swap Application Started!"
echo "========================================"
echo ""
echo "AI Service:    http://localhost:8000"
echo "Frontend:      http://localhost:3001"
echo ""
echo "Wait a few seconds for services to fully start,"
echo "then open http://localhost:3001 in your browser"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for interrupt
trap 'echo ""; echo "Stopping services..."; kill $AI_PID $API_PID 2>/dev/null; exit 0' SIGINT
wait