# Local Llama Chatbot

A local chatbot powered by Llama 3.1 using vLLM for fast inference, with a Claude Code-style terminal interface.

## Requirements

- Python 3.10+
- NVIDIA GPU(s) with CUDA support
- For Llama 3.1 70B: 2x A100 80GB recommended
- For Llama 3.1 8B: 1x A100 or similar

## Installation

```bash
pip install -r requirements.txt
```

## Model Setup

Place your downloaded Llama 3.1 models in the `models/` directory:

```
models/
├── Llama-3.1-8B-Instruct/
│   ├── config.json
│   ├── tokenizer.json
│   ├── model-00001-of-00004.safetensors
│   └── ...
└── Llama-3.1-70B-Instruct/
    ├── config.json
    ├── tokenizer.json
    ├── model-00001-of-00030.safetensors
    └── ...
```

## Usage

### Quick Start (8B model with 2 GPUs)

```bash
chmod +x run.sh
./run.sh
```

### Using the 70B model

```bash
MODEL=./models/Llama-3.1-70B-Instruct ./run.sh
```

### Custom configuration

```bash
python server.py \
    --model ./models/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --host 0.0.0.0 \
    --port 8000
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL` | `./models/Llama-3.1-8B-Instruct` | Path to local model directory |
| `TENSOR_PARALLEL` | `2` | Number of GPUs for tensor parallelism |
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/info` | GET | Model information |
| `/api/chat` | POST | Send message (streaming SSE) |
| `/api/conversations/{id}` | GET | Get conversation history |
| `/api/conversations/{id}` | DELETE | Delete conversation |
| `/api/conversations/new` | POST | Start new conversation |

## Features

- Streaming responses with Server-Sent Events
- Conversation history management
- Adjustable generation parameters (temperature, max tokens, top-p)
- Dark terminal-style interface
- Markdown formatting support in responses
- Code syntax highlighting

## GPU Memory Requirements

| Model | Min VRAM | Recommended Setup |
|-------|----------|-------------------|
| Llama 3.1 8B | ~20GB | 1x A100 40GB |
| Llama 3.1 70B | ~140GB | 2x A100 80GB |

## Troubleshooting

### Out of memory errors
- Reduce `max_model_len` in `server.py`
- Reduce `gpu_memory_utilization` (default 0.90)
- Use a smaller model

### Slow first response
- First request loads the model into GPU memory
- Subsequent requests will be much faster
