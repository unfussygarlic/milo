"""Cached vs uncached greedy decoding on every eval prompt, with the constraint ON.

The real-checkpoint half of the KV-cache proof (tests/test_kv_cache.py covers the
mechanics on CPU). Needs a GPU and the tool-use checkpoint; prints the token
mismatch count, the logits delta and tokens/sec both ways.

    python scripts/verify_kv_cache.py [--checkpoint experiments/tools_v51/checkpoints/best.pt]
"""

import argparse
import time

import torch

import milo.tools as T
from milo.agent.constraint import CopyConstraint
from milo.agent.decoding import generate
from milo.eval.cases_dev import CASES
from milo.eval.cases_test import TEST_CASES
from milo.eval.harness import EVAL_NOW, store, stub_search, stub_weather
from milo.model.checkpoint import load_model
from milo.model.tokenizer import decode, encode

ap = argparse.ArgumentParser()
ap.add_argument("--checkpoint", default="experiments/tools_v51/checkpoints/best.pt")
ap.add_argument("--max-new-tokens", type=int, default=120)
args = ap.parse_args()

T.TOOLS["SEARCH"], T.TOOLS["WEATHER"] = stub_search, stub_weather
model = load_model(args.checkpoint, "cuda")
cases = CASES + TEST_CASES
print(f"{len(cases)} prompts (138 dev + 198 held-out test), greedy, constraint ON\n")

mismatch, maxdiff, tok_cached, tok_uncached, t_cached, t_uncached = [], 0.0, 0, 0, 0.0, 0.0
for i, case in enumerate(cases):
    T.set_now(EVAL_NOW)
    T.set_merchant_kb(store(case["kb"] or "kirana"))
    msg = case["q"]
    prompt = encode(f"<|user|>\n{msg}\n<|assistant|>\n")
    kw = dict(max_new_tokens=args.max_new_tokens, temperature=0.0, stop_strings=["<|tool_result|>"])

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    un = generate(model, prompt, constraint=CopyConstraint(msg, [msg], False), use_cache=False, **kw)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    ca = generate(model, prompt, constraint=CopyConstraint(msg, [msg], False), use_cache=True, **kw)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    t_uncached += t1 - t0
    t_cached += t2 - t1
    tok_uncached += len(un) - len(prompt)
    tok_cached += len(ca) - len(prompt)
    if un != ca:
        mismatch.append((msg, decode(un[len(prompt) :]), decode(ca[len(prompt) :])))

    # logits agreement at the same state (prompt + the uncached continuation)
    if i % 12 == 0:
        ctx = un[: len(prompt) + max(0, (len(un) - len(prompt)) // 2)]
        x_full = torch.tensor([ctx[-512:]], device="cuda")
        with torch.no_grad(), torch.amp.autocast("cuda"):
            lg_un, _ = model(x_full)
            _, _, cache = model(x_full[:, :-1], use_cache=True)
            lg_ca, _, _ = model(x_full[:, -1:], past_key_values=cache, use_cache=True)
        maxdiff = max(maxdiff, (lg_un[0, -1].float() - lg_ca[0, -1].float()).abs().max().item())

print(f"token sequences differ: {len(mismatch)} / {len(cases)}")
for m, a, b in mismatch[:5]:
    print(f"  {m!r}\n   uncached: {a[:100]!r}\n   cached:   {b[:100]!r}")
print(f"logits max |Δ| (fp16 autocast, 28 sampled states): {maxdiff:.3e}")
print(f"\ntokens generated: {tok_uncached} uncached / {tok_cached} cached")
print(f"uncached: {t_uncached:6.1f}s → {tok_uncached / t_uncached:6.1f} tok/s")
print(f"cached:   {t_cached:6.1f}s → {tok_cached / t_cached:6.1f} tok/s")
print(f"speedup:  {(tok_cached / t_cached) / (tok_uncached / t_uncached):.2f}x")
T.set_now(None)
T.set_merchant_kb(None)
