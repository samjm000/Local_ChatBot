"""
Local Llama Chatbot Server
Uses vLLM for fast inference with Llama 3.1 models
"""

import argparse
import asyncio
import atexit
import json
import os
import signal
import sys
import uuid
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine


# Global engine instance
engine: Optional[AsyncLLMEngine] = None
model_name: str = ""
_cleanup_done: bool = False

# Conversation storage (in-memory for simplicity)
conversations: dict[str, list[dict]] = {}


def cleanup_engine() -> None:
    """Safely cleanup the model engine and release GPU resources."""
    global engine, _cleanup_done

    if _cleanup_done:
        return

    _cleanup_done = True

    if engine is not None:
        print("\nCleaning up model engine...")
        try:
            # Shutdown the engine gracefully
            if hasattr(engine, 'shutdown'):
                engine.shutdown()
            elif hasattr(engine, '_engine') and hasattr(engine._engine, 'shutdown'):
                engine._engine.shutdown()

            # Clear any pending requests
            if hasattr(engine, 'abort_all'):
                engine.abort_all()

            # Release the engine reference
            engine = None

            # Force garbage collection to release GPU memory
            import gc
            gc.collect()

            # Try to release CUDA memory if torch is available
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except ImportError:
                pass

            print("Model engine cleaned up successfully.")
        except Exception as e:
            print(f"Warning: Error during engine cleanup: {e}")
            engine = None


def signal_handler(signum: int, frame) -> None:
    """Handle termination signals to ensure clean shutdown."""
    sig_name = signal.Signals(signum).name
    print(f"\nReceived {sig_name}, shutting down gracefully...")
    cleanup_engine()
    sys.exit(0)


def setup_signal_handlers() -> None:
    """Setup signal handlers for graceful shutdown."""
    # Register signal handlers for common termination signals
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # kill command

    # SIGHUP is not available on Windows
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, signal_handler)  # terminal hangup


# Register atexit handler for cleanup on normal exit
atexit.register(cleanup_engine)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    response: str
    conversation_id: str


def get_system_prompt() -> str:
    """System prompt for the assistant."""
    return """You are a helpful AI assistant. You provide clear, accurate, and thoughtful responses.
You format your responses using markdown when appropriate for code, lists, and emphasis.
Be concise but thorough in your explanations."""


def format_prompt(messages: list[dict], model: str) -> str:
    """Format the conversation history into a prompt for Llama 3.1."""
    # Llama 3.1 chat template
    formatted = "<|begin_of_text|>"

    # Add system message
    formatted += "<|start_header_id|>system<|end_header_id|>\n\n"
    formatted += get_system_prompt() + "<|eot_id|>"

    # Add conversation history
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        formatted += f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>"

    # Add assistant header for generation
    formatted += "<|start_header_id|>assistant<|end_header_id|>\n\n"

    return formatted


async def generate_response(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> AsyncGenerator[str, None]:
    """Generate a streaming response using vLLM."""
    global engine

    if engine is None:
        raise RuntimeError("Engine not initialized")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stop=["<|eot_id|>", "<|end_of_text|>"],
    )

    request_id = str(uuid.uuid4())

    results_generator = engine.generate(prompt, sampling_params, request_id)

    previous_text = ""
    async for result in results_generator:
        if result.outputs:
            new_text = result.outputs[0].text
            delta = new_text[len(previous_text):]
            previous_text = new_text
            if delta:
                yield delta


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the vLLM engine on startup."""
    global engine, model_name, _cleanup_done

    # Reset cleanup flag on startup
    _cleanup_done = False

    # Get model from environment or use default
    model_name = os.environ.get("MODEL_NAME", "/llm_models/llama-3-1-70b")
    tensor_parallel_size = int(os.environ.get("TENSOR_PARALLEL_SIZE", "2"))

    print(f"Loading model: {model_name}")
    print(f"Tensor parallel size: {tensor_parallel_size}")

    engine_args = AsyncEngineArgs(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        dtype="auto",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.90,
    )

    try:
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Model loaded successfully!")

        yield

    except Exception as e:
        print(f"Error during model lifecycle: {e}")
        raise
    finally:
        # Cleanup on shutdown (normal or error)
        cleanup_engine()


app = FastAPI(title="Local Llama Chatbot", lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def serve_frontend():
    """Serve the chat frontend."""
    return FileResponse("static/index.html")


@app.get("/api/info")
async def get_info():
    """Get model information."""
    return {
        "model": model_name,
        "status": "ready" if engine is not None else "loading",
    }


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle a chat message and return a streaming response."""
    global conversations

    # Get or create conversation
    if request.conversation_id and request.conversation_id in conversations:
        conv_id = request.conversation_id
    else:
        conv_id = str(uuid.uuid4())
        conversations[conv_id] = []

    # Add user message to history
    conversations[conv_id].append({
        "role": "user",
        "content": request.message
    })

    # Format the prompt
    prompt = format_prompt(conversations[conv_id], model_name)

    async def stream_response():
        full_response = ""
        try:
            async for token in generate_response(
                prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            ):
                full_response += token
                yield f"data: {json.dumps({'token': token, 'conversation_id': conv_id})}\n\n"

            # Store assistant response in history
            conversations[conv_id].append({
                "role": "assistant",
                "content": full_response
            })

            yield f"data: {json.dumps({'done': True, 'conversation_id': conv_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.get("/api/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history."""
    if conversation_id not in conversations:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"messages": conversations[conversation_id]}


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation."""
    if conversation_id in conversations:
        del conversations[conversation_id]
    return {"status": "deleted"}


@app.post("/api/conversations/new")
async def new_conversation():
    """Start a new conversation."""
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = []
    return {"conversation_id": conv_id}


# Mount static files (do this last to not override API routes)
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


if __name__ == "__main__":
    import uvicorn

    # Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    parser = argparse.ArgumentParser(description="Local Llama Chatbot Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=9010, help="Port to bind to")
    parser.add_argument(
        "--model",
        type=str,
        default="/llm_models/llama-3-1-70b",
        help="Path to local model directory (e.g., /llm_models/llama-3-1-70b)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Number of GPUs for tensor parallelism",
    )

    args = parser.parse_args()

    # Set environment variables for the lifespan handler
    os.environ["MODEL_NAME"] = args.model
    os.environ["TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)

    uvicorn.run(app, host=args.host, port=args.port)
