#!/bin/bash

# Local Llama Chatbot Startup Script

# Default values - models are stored in /llm_models/
MODEL="${MODEL:-/llm_models/llama-3-1-70b}"
TENSOR_PARALLEL="${TENSOR_PARALLEL:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-9010}"
KILL_EXISTING="${KILL_EXISTING:-false}"
AUTO_PORT="${AUTO_PORT:-false}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --kill-existing)
            KILL_EXISTING=true
            shift
            ;;
        --auto-port)
            AUTO_PORT=true
            shift
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tensor-parallel-size)
            TENSOR_PARALLEL="$2"
            shift 2
            ;;
        --help)
            echo "Usage: ./run.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model PATH           Path to the model (default: /llm_models/llama-3-1-70b)"
            echo "  --port PORT            Port to bind to (default: 9010)"
            echo "  --tensor-parallel-size N   Number of GPUs (default: 2)"
            echo "  --kill-existing        Kill any process using the port before starting"
            echo "  --auto-port            Auto-select an available port if default is in use"
            echo "  --help                 Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  MODEL                  Same as --model"
            echo "  PORT                   Same as --port"
            echo "  TENSOR_PARALLEL        Same as --tensor-parallel-size"
            echo "  KILL_EXISTING=true     Same as --kill-existing"
            echo "  AUTO_PORT=true         Same as --auto-port"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Starting Local Llama Chatbot${NC}"
echo -e "Model: ${YELLOW}${MODEL}${NC}"
echo -e "Tensor Parallel Size: ${YELLOW}${TENSOR_PARALLEL}${NC}"
echo -e "Server: ${YELLOW}http://${HOST}:${PORT}${NC}"
echo ""

# Build extra arguments
EXTRA_ARGS=""
if [ "$KILL_EXISTING" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --kill-existing"
    echo -e "${YELLOW}Will kill existing process if port is in use${NC}"
fi
if [ "$AUTO_PORT" = "true" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --auto-port"
    echo -e "${YELLOW}Will auto-select port if default is in use${NC}"
fi

# Run the server
python server.py \
    --model "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL" \
    --host "$HOST" \
    --port "$PORT" \
    $EXTRA_ARGS
