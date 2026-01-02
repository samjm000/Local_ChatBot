#!/bin/bash

# Local Llama Chatbot Startup Script

# Default values - models are stored locally in ./models/
MODEL="${MODEL:-./models/Llama-3.1-8B-Instruct}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9010}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting Local Llama Chatbot${NC}"
echo -e "Model: ${YELLOW}${MODEL}${NC}"
echo -e "Tensor Parallel Size: ${YELLOW}${TENSOR_PARALLEL}${NC}"
echo -e "Server: ${YELLOW}http://${HOST}:${PORT}${NC}"
echo ""

# Run the server
python server.py \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --host "$HOST" \
    --port "$PORT"
