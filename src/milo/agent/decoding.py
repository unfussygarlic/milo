"""Autoregressive generation and the constrained-decoding token filter."""

import torch

from milo.agent.constraint import CopyConstraint
from milo.config import model_config
from milo.model.tokenizer import EOS_ID, decode, vocab_strings

CONSTRAINT_STATS = {"steps": 0, "fallbacks": 0}


def apply_constraint(logits: torch.Tensor, constraint: CopyConstraint, text_so_far: str) -> torch.Tensor:
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
        ok = [
            t for t in cand if (constraint.eos_ok(text_so_far) if t == EOS_ID else constraint.ok(text_so_far, vocab[t]))
        ]
        if ok:
            masked = torch.full_like(logits, float("-inf"))
            idx = torch.tensor(ok, device=logits.device)
            masked[idx] = logits[idx]
            return masked
    CONSTRAINT_STATS["fallbacks"] += 1
    return logits


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    token_ids: list[int],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    stop_strings: list[str] | None = None,
    device: str = "cuda",
    constraint: CopyConstraint | None = None,
    use_cache: bool = True,
    amp: bool = True,
) -> list[int]:
    """Generate tokens autoregressively. Stops on EOS, any stop_string, or max_new_tokens.

    Returns the full sequence (prompt + generated) as a list of ints.

    With use_cache (the default) the prompt is encoded once and each new token
    attends to the cached keys and values instead of re-running the whole
    context. The tokens produced are the same either way — tests/test_kv_cache.py
    checks that on real eval prompts with the constraint on, and
    `use_cache=False` keeps the old path available for comparison.

    Once the context passes seq_len the window slides by one token per step, which
    moves every RoPE position, so the cache is rebuilt each step from there on —
    same behaviour and same cost as before for over-long contexts.

    amp=False runs in fp32. Under fp16 autocast, feeding a sequence as one chunk
    or as prefill-plus-steps changes the reduction order and moves logits by
    ~8e-3, which is enough to flip a near-tied greedy choice; fp32 makes the two
    paths agree. It is about 1.6x slower, so the default stays fp16 and
    scripts/verify_kv_cache.py uses the flag to tell ties from bugs.
    """
    seq_len = model_config()["seq_len"]
    generated = list(token_ids)
    cache = None

    for _ in range(max_new_tokens):
        if cache is None or len(generated) > seq_len:
            # (Re)prefill. The window is the most recent seq_len tokens and its
            # positions restart at 0 — the same tokens at the same positions the
            # uncached path would have fed, so the logits are identical.
            x = torch.tensor([generated[-seq_len:]], dtype=torch.long, device=device)
            past = None
        else:
            # Steady state: one token, attending to the cache.
            x = torch.tensor([[generated[-1]]], dtype=torch.long, device=device)
            past = cache

        with torch.amp.autocast("cuda", enabled=amp):
            if use_cache:
                logits, _, cache = model(x, past_key_values=past, use_cache=True)
            else:
                logits, _ = model(x)

        logits = logits[0, -1, :].float()

        if constraint is not None:
            logits = apply_constraint(logits, constraint, decode(generated[len(token_ids) :]))

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
            text_so_far = decode(generated[len(token_ids) :])
            if any(ss in text_so_far for ss in stop_strings):
                break

    return generated
