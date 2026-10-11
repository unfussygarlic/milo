"""
Milo — ReAct Agent Runtime (v4)

The inference-time loop that lets Milo use tools:
  1. User sends a query
  2. Milo generates until it emits <|tool_result|> (i.e. finished writing TOOL(args))
  3. Runtime parses the call, executes it, injects result<|end_tool|>
  4. Milo continues — another tool call, or the final answer

v4 fixes over v3:
  - Stop on <|tool_result|>, not <|tool_call|>. v3 stopped right after the
    <|tool_call|> tag, BEFORE the model wrote TOOL(args), so the parser never
    matched and no tool ever executed.
  - Args parse to the last ')' — CALCULATE((100 + 50) * 1.18) no longer
    truncates to "(100 + 50".
  - DATE(today) / empty args work; dates use IST, not server-local (EC2 = UTC).
  - Tools come from milo_tools.py, the same code the data generator uses,
    so results at inference match results seen in training.
  - When max_tool_calls is used up, one final no-tool generation produces
    an answer instead of returning nothing.

Usage:
    python react_runtime.py --checkpoint experiments/tools_*/checkpoints/best.pt
    python react_runtime.py --checkpoint best.pt --max-tool-calls 3
    MILO_DISABLE_RUN_CODE=1 python react_runtime.py --checkpoint best.pt
    MILO_MERCHANT_KB=merchant_kbs/kirana.json python react_runtime.py --checkpoint best.pt
"""

import argparse
import time

import torch
import tiktoken

from model import SLM
from milo_tools import (
    TOOLS, TOOL_CALL_TAG, TOOL_RESULT_TAG, END_TOOL_TAG,
    execute, parse_tool_call, ungrounded_numbers, fallback_answer, CopyConstraint,
)

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
MODEL_CONFIG = dict(
    vocab_size=50257,
    d_model=1280,
    n_q_heads=20,
    n_kv_heads=4,
    n_layers=16,
    seq_len=512,
    share_blocks=True,
    dropout=0.0,  # no dropout at inference
)

# ── Tokenizer ──
enc = tiktoken.get_encoding("gpt2")

USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"
EOS_ID = enc.encode(EOS_TOKEN, allowed_special={EOS_TOKEN})[0]


def encode(text):
    return enc.encode(text, allowed_special={
        USER_TOKEN, ASST_TOKEN, EOS_TOKEN,
        TOOL_CALL_TAG, TOOL_RESULT_TAG, END_TOOL_TAG,
    })


def decode(tokens):
    return enc.decode(tokens)


_VOCAB = None


def vocab_strings():
    """Decoded string of every token id (partial UTF-8 bytes become '\ufffd')."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in range(enc.n_vocab)]
    return _VOCAB


CONSTRAINT_STATS = {"steps": 0, "fallbacks": 0}


def apply_constraint(logits, constraint, text_so_far):
    """Mask logits so only tokens allowed by `constraint` remain. Candidates are checked
    best-first (64 → 512 → 4096 → whole vocab), so a confident-but-wrong model still gets
    its best *allowed* token. Only if nothing at all is allowed do we fall back."""
    vocab = vocab_strings()
    CONSTRAINT_STATS["steps"] += 1
    order = None
    for k in (64, 512, 4096, len(vocab)):
        if k < len(vocab):
            cand = torch.topk(logits, k).indices.tolist()
        else:
            order = order or torch.argsort(logits, descending=True).tolist()
            cand = order
        ok = [t for t in cand if (constraint.eos_ok(text_so_far) if t == EOS_ID
                                  else constraint.ok(text_so_far, vocab[t]))]
        if ok:
            masked = torch.full_like(logits, float("-inf"))
            idx = torch.tensor(ok, device=logits.device)
            masked[idx] = logits[idx]
            return masked
    CONSTRAINT_STATS["fallbacks"] += 1
    return logits


def strip_tags(text):
    for tag in (EOS_TOKEN, TOOL_CALL_TAG, TOOL_RESULT_TAG, END_TOOL_TAG,
                USER_TOKEN, ASST_TOKEN):
        text = text.replace(tag, "")
    return text.strip()


# ═══════════════════════════════════════════════════════════════
# MODEL LOADING & GENERATION
# ═══════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load Milo from checkpoint."""
    model = SLM(
        MODEL_CONFIG["vocab_size"], MODEL_CONFIG["d_model"],
        MODEL_CONFIG["n_q_heads"], MODEL_CONFIG["n_kv_heads"],
        MODEL_CONFIG["n_layers"], MODEL_CONFIG["seq_len"],
        MODEL_CONFIG["dropout"], MODEL_CONFIG["share_blocks"],
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    del ckpt

    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def generate(model, token_ids, max_new_tokens=256, temperature=0.7,
             top_k=50, top_p=0.9, stop_strings=None, device="cuda", constraint=None):
    """
    Generate tokens autoregressively. Stops on EOS, any stop_string,
    or when max_new_tokens is reached.

    Returns the full sequence (prompt + generated) as a list of ints.
    """
    seq_len = MODEL_CONFIG["seq_len"]
    generated = list(token_ids)

    for _ in range(max_new_tokens):
        context = generated[-seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        with torch.amp.autocast("cuda"):
            logits, _ = model(x)

        logits = logits[0, -1, :].float()

        if constraint is not None:
            logits = apply_constraint(logits, constraint, decode(generated[len(token_ids):]))

        if temperature > 0:
            logits = logits / temperature

            if top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[-1]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        else:
            next_token = logits.argmax().item()

        generated.append(next_token)

        if next_token == EOS_ID:
            break

        if stop_strings:
            text_so_far = decode(generated[len(token_ids):])
            if any(ss in text_so_far for ss in stop_strings):
                break

    return generated


# ═══════════════════════════════════════════════════════════════
# ReAct LOOP
# ═══════════════════════════════════════════════════════════════

def react_step(model, prompt_tokens, device="cuda", temperature=0.7,
               max_new_tokens=256, allow_tools=True, constraint=None):
    """
    One step of the ReAct loop.

    allow_tools=True:  generate until <|tool_result|> (the model has finished
                       writing TOOL(args)) or EOS. If a call is found, execute
                       it and return the prompt augmented with the result.
    allow_tools=False: generate a final answer; anything from <|tool_call|>
                       onward is discarded.

    Returns (text, tool_called, tokens, tool_name, args, result)
    """
    stop = [TOOL_RESULT_TAG] if allow_tools else [TOOL_CALL_TAG]
    generated = generate(
        model, prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop_strings=stop,
        device=device,
        constraint=constraint,
    )
    new_text = decode(generated[len(prompt_tokens):])

    if allow_tools:
        call = parse_tool_call(new_text)
        if call:
            result = execute(call.name, call.args)
            augmented = (f"{call.before}{TOOL_CALL_TAG}{call.name}({call.args})"
                         f"{TOOL_RESULT_TAG}{result}{END_TOOL_TAG}")
            augmented_tokens = encode(decode(prompt_tokens) + augmented)
            return strip_tags(call.before), True, augmented_tokens, call.name, call.args, result

    # Final response: cut at any (unparseable / disallowed) tool call
    cut = new_text.find(TOOL_CALL_TAG)
    if cut >= 0:
        new_text = new_text[:cut]
    return strip_tags(new_text), False, generated, None, None, None


def build_prompt(user_message, conversation_history=None):
    prompt = ""
    for role, content in (conversation_history or []):
        if role == "user":
            prompt += f"{USER_TOKEN}\n{content}\n"
        else:
            prompt += f"{ASST_TOKEN}\n{content}{EOS_TOKEN}\n"
    prompt += f"{USER_TOKEN}\n{user_message}\n{ASST_TOKEN}\n"
    return prompt


def react_loop(model, user_message: str, conversation_history: list = None,
               device="cuda", max_tool_calls=3, temperature=0.7, verbose=True, guard=True,
               constrain=True):
    """
    Full ReAct loop for a user message.

    guard=True: if tools were called and the final reply contains a number that
    is neither in the user's message nor in any tool result (a mis-copied price,
    an invented total), the reply is replaced by a deterministic one built from
    the tool results. Never let a wrong ₹ amount reach a customer.

    constrain=True: constrained decoding (milo_tools.CopyConstraint) — store /
    search / weather args can only be copied from the customer's message, and
    numbers only from the message or tool results, so the guard rarely fires.

    Returns:
        final_response: The model's complete response text
        tool_calls: List of (tool_name, args, result) tuples
    """
    prompt_tokens = encode(build_prompt(user_message, conversation_history))
    tool_calls = []
    response_parts = []

    for step_i in range(max_tool_calls + 1):
        allow = step_i < max_tool_calls   # last iteration forces a final answer
        constraint = (CopyConstraint(user_message, [user_message] + [r for _, _, r in tool_calls],
                                     numbers_in_reply=bool(tool_calls)) if constrain else None)
        text, tool_called, tokens, tool_name, args, result = react_step(
            model, prompt_tokens, device=device, temperature=temperature,
            allow_tools=allow, constraint=constraint,
        )

        if tool_called:
            if text:
                response_parts.append(text)
            tool_calls.append((tool_name, args, result))
            if verbose:
                print(f"  🔧 {tool_name}({args})")
                print(f"  ⟶  {result}")
            prompt_tokens = tokens
        else:
            response_parts.append(text)
            break

    final_response = " ".join(p for p in response_parts if p).strip()
    if guard and tool_calls:
        bad = ungrounded_numbers(final_response, [user_message] + [r for _, _, r in tool_calls])
        if bad:
            if verbose:
                print(f"  🛡  guard: ungrounded {bad} → answering from tool results")
            final_response = fallback_answer(tool_calls)
    return final_response, tool_calls


# ═══════════════════════════════════════════════════════════════
# INTERACTIVE CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Milo ReAct Agent")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to tool-use checkpoint (best.pt)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tool-calls", type=int, default=3)
    parser.add_argument("--no-verbose", action="store_true")
    parser.add_argument("--no-guard", action="store_true", help="disable the number-grounding guard")
    parser.add_argument("--no-constrain", action="store_true", help="disable constrained copy decoding")
    args = parser.parse_args()

    print(f"\n{'='*50}")
    print(f"  Milo ReAct Agent")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Tools: {', '.join(TOOLS.keys())}")
    print(f"{'='*50}")
    print(f"\nType your message (or 'quit' to exit, 'clear' to reset)\n")

    print("Loading model...", end=" ", flush=True)
    model = load_model(args.checkpoint, args.device)
    print("done!\n")

    conversation = []
    verbose = not args.no_verbose

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            conversation = []
            print("(conversation cleared)\n")
            continue

        t0 = time.time()
        response, tool_calls = react_loop(
            model, user_input,
            conversation_history=conversation,
            device=args.device,
            max_tool_calls=args.max_tool_calls,
            temperature=args.temperature,
            verbose=verbose,
            guard=not args.no_guard,
            constrain=not args.no_constrain,
        )
        elapsed = time.time() - t0

        print(f"\nMilo: {response}")
        if tool_calls and verbose:
            print(f"  ({len(tool_calls)} tool call(s), {elapsed:.1f}s)")
        print()

        conversation.append(("user", user_input))
        conversation.append(("assistant", response))


if __name__ == "__main__":
    main()