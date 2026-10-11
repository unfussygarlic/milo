"""Dev chat UI — a browser window onto a raw checkpoint.

DEV TOOL ONLY. This is the plain generation path: no tools, no constrained
decoding, no grounding guard, so nothing stops it inventing a price. Customer
traffic goes through milo.agent (and, from Phase 4, milo.serving) instead.

Usage:
    pip install -e ".[serving]"
    python scripts/dev_chat_ui/app.py --checkpoint experiments/sft_20260929_075754/checkpoints/best.ptrom

Then open http://localhost:8000 in a browser.
"""

import argparse
import json
import os
import traceback
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from milo.config import model_config
from milo.model.slm import SLM
from milo.model.tokenizer import ASST_TOKEN, EOS_ID, EOS_TOKEN, USER_TOKEN, decode, encode

# ── Globals ──
model = None
device = None

MODEL_CONFIG = model_config()


def load_model(checkpoint_path):
    """Load the fine-tuned model from a checkpoint."""
    global model, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    model = SLM(
        MODEL_CONFIG["vocab_size"],
        MODEL_CONFIG["d_model"],
        MODEL_CONFIG["n_q_heads"],
        MODEL_CONFIG["n_kv_heads"],
        MODEL_CONFIG["n_layers"],
        MODEL_CONFIG["seq_len"],
        MODEL_CONFIG["dropout"],
        MODEL_CONFIG["share_blocks"],
    )

    # Load weights
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count / 1e6:.1f}M params from {checkpoint_path}")


def build_prompt(messages):
    """
    Build a multi-turn prompt from conversation history.

    messages: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]

    Format:
        <|user|>
        question 1
        <|assistant|>
        answer 1<|endoftext|><|user|>
        question 2
        <|assistant|>
        (model generates from here)
    """
    prompt = ""
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"{USER_TOKEN}\n{msg['content']}\n{ASST_TOKEN}\n"
        elif msg["role"] == "assistant":
            prompt += f"{msg['content']}{EOS_TOKEN}"
    return prompt


@torch.no_grad()
def generate(prompt_text, max_new_tokens=256, temperature=0.7, top_k=50):
    """Generate a response. Returns (text, prompt_tokens, total_tokens, was_truncated)."""
    prompt_ids = encode(prompt_text)

    # Truncate if too long (leave room for generation)
    max_prompt = MODEL_CONFIG["seq_len"] - max_new_tokens
    was_truncated = False
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
        was_truncated = True

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    generated_ids = []
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -MODEL_CONFIG["seq_len"]:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        if next_id.item() == EOS_ID:
            break

        generated_ids.append(next_id.item())
        idx = torch.cat([idx, next_id], dim=1)

    response_text = decode(generated_ids).strip()
    total_tokens = len(prompt_ids) + len(generated_ids)

    return response_text, len(prompt_ids), total_tokens, was_truncated


@torch.no_grad()
def generate_stream(prompt_text, max_new_tokens=256, temperature=0.7, top_k=50):
    """
    Stream tokens one at a time.
    First yields a metadata dict, then token strings.
    """
    prompt_ids = encode(prompt_text)
    original_len = len(prompt_ids)

    max_prompt = MODEL_CONFIG["seq_len"] - max_new_tokens
    was_truncated = False
    if len(prompt_ids) > max_prompt:
        prompt_ids = prompt_ids[-max_prompt:]
        was_truncated = True

    # Yield metadata first
    yield {
        "type": "meta",
        "prompt_tokens": len(prompt_ids),
        "seq_len": MODEL_CONFIG["seq_len"],
        "was_truncated": was_truncated,
        "original_prompt_tokens": original_len,
    }

    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    token_count = 0

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -MODEL_CONFIG["seq_len"]:]

        logits, _ = model(idx_cond)
        logits = logits[:, -1, :]

        if temperature > 0:
            logits = logits / temperature
            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        if next_id.item() == EOS_ID:
            break

        token_text = decode([next_id.item()])
        token_count += 1
        yield {"type": "token", "token": token_text, "total_tokens": len(prompt_ids) + token_count}

        idx = torch.cat([idx, next_id], dim=1)


# ── FastAPI app ──
app = FastAPI(title="Milo Chat")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the chat frontend."""
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_path.read_text())


@app.post("/chat")
async def chat(request: Request):
    """
    Multi-turn chat endpoint.
    Expects JSON: {
        "messages": [{"role": "user", "content": "..."}, ...],
        "temperature": 0.7,
        "max_tokens": 256
    }
    """
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 256)

    if not messages:
        return {"response": "Please enter a message.", "tokens": {}}

    prompt = build_prompt(messages)

    try:
        response, prompt_tokens, total_tokens, was_truncated = generate(
            prompt, max_new_tokens=max_tokens, temperature=temperature
        )
        return {
            "response": response,
            "tokens": {
                "prompt": prompt_tokens,
                "completion": total_tokens - prompt_tokens,
                "total": total_tokens,
                "seq_len": MODEL_CONFIG["seq_len"],
                "was_truncated": was_truncated,
            }
        }
    except Exception as e:
        traceback.print_exc()
        return {"response": f"Error: {e}", "tokens": {}}


@app.post("/chat/stream")
async def chat_stream(request: Request):
    """
    Multi-turn streaming chat endpoint using Server-Sent Events.
    """
    body = await request.json()
    messages = body.get("messages", [])
    temperature = body.get("temperature", 0.7)
    max_tokens = body.get("max_tokens", 256)

    if not messages:
        async def empty():
            yield f"data: {json.dumps({'type': 'token', 'token': 'Please enter a message.'})}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(empty(), media_type="text/event-stream")

    prompt = build_prompt(messages)

    def stream():
        try:
            for item in generate_stream(prompt, max_new_tokens=max_tokens, temperature=temperature):
                yield f"data: {json.dumps(item)}\n\n"
        except Exception as e:
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'token', 'token': f'[Error: {e}]'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "seq_len": MODEL_CONFIG["seq_len"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Milo Chat Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8010, help="Port to bind to")
    args = parser.parse_args()

    load_model(args.checkpoint)

    os.makedirs("static", exist_ok=True)

    print(f"\n  Milo Chat running at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
