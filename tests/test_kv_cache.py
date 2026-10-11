"""Incremental decoding must be indistinguishable from re-running the context.

These run on CPU with a small randomly-initialised model: the point is the cache
mechanics (RoPE offsets, the causal mask, one cache entry per *stack position*
under share_blocks, sliding past seq_len), not the trained weights. The same
check against the real checkpoint on every eval prompt is in
scripts/verify_kv_cache.py.
"""

import pytest
import torch

from milo.agent.constraint import CopyConstraint
from milo.agent.decoding import generate
from milo.model.slm import SLM

VOCAB = 50257  # the constraint's candidate tiers assume the real vocabulary


def tiny_model(share_blocks=True, seq_len=16):
    torch.manual_seed(0)
    return SLM(
        vocab_size=VOCAB,
        d_model=32,
        n_q_heads=4,
        n_kv_heads=2,
        n_layers=4,
        seq_len=seq_len,
        dropout=0.1,
        share_blocks=share_blocks,
    ).eval()


def test_cached_logits_match_full_forward():
    m = tiny_model()
    idx = torch.randint(0, VOCAB, (1, 12))
    with torch.no_grad():
        full, _ = m(idx)
        _, _, cache = m(idx[:, :-1], use_cache=True)
        step, _, _ = m(idx[:, -1:], past_key_values=cache, use_cache=True)
    assert torch.allclose(full[:, -1], step[:, -1], atol=1e-5)
    assert full[:, -1].argmax() == step[:, -1].argmax()


def test_token_by_token_matches_full_forward_at_every_position():
    m = tiny_model()
    idx = torch.randint(0, VOCAB, (1, 12))
    with torch.no_grad():
        full, _ = m(idx)
        cache, one_at_a_time = None, []
        for t in range(idx.shape[1]):
            logits, _, cache = m(idx[:, t : t + 1], past_key_values=cache, use_cache=True)
            one_at_a_time.append(logits[:, -1])
    for t in range(idx.shape[1]):
        assert torch.allclose(full[:, t], one_at_a_time[t], atol=1e-5), f"position {t}"


def test_cache_holds_kv_heads_not_query_heads():
    """GQA: 2 KV heads cached, not 4 query heads — that is the memory saving."""
    m = tiny_model()
    with torch.no_grad():
        _, _, cache = m(torch.randint(0, VOCAB, (1, 7)), use_cache=True)
    assert len(cache) == 4, "one entry per stack position"
    for k, v in cache:
        assert k.shape == (1, 2, 7, 8), k.shape  # [B, n_kv_heads, T, head_dim]
        assert v.shape == (1, 2, 7, 8), v.shape


def test_shared_blocks_get_separate_cache_entries():
    """share_blocks reuses each block object twice; the two uses see different
    activations, so they must not share one cache entry."""
    m = tiny_model(share_blocks=True)
    assert m.blocks[0] is m.blocks[1], "this model is supposed to share blocks"
    with torch.no_grad():
        _, _, cache = m(torch.randint(0, VOCAB, (1, 7)), use_cache=True)
    assert not torch.equal(cache[0][0], cache[1][0])


def test_generate_identical_with_and_without_cache():
    m = tiny_model()
    prompt = torch.randint(0, VOCAB, (8,)).tolist()
    kw = dict(max_new_tokens=12, temperature=0.0, device="cpu")
    assert generate(m, prompt, use_cache=False, **kw) == generate(m, prompt, use_cache=True, **kw)


def test_generate_identical_past_the_context_window():
    """Beyond seq_len the window slides, which moves every RoPE position; the
    cache is rebuilt each step and the tokens stay the same."""
    m = tiny_model(seq_len=16)
    prompt = torch.randint(0, VOCAB, (14,)).tolist()
    kw = dict(max_new_tokens=10, temperature=0.0, device="cpu")
    out_uncached = generate(m, prompt, use_cache=False, **kw)
    out_cached = generate(m, prompt, use_cache=True, **kw)
    assert len(out_uncached) > 16, "this test needs to cross the window"
    assert out_uncached == out_cached


def test_generate_identical_with_the_constraint_on():
    m = tiny_model()
    msg = "basmati rice 5kg ka rate"
    prompt = torch.randint(0, VOCAB, (8,)).tolist()
    kw = dict(max_new_tokens=12, temperature=0.0, device="cpu")
    uncached = generate(m, prompt, constraint=CopyConstraint(msg, [msg], False), use_cache=False, **kw)
    cached = generate(m, prompt, constraint=CopyConstraint(msg, [msg], False), use_cache=True, **kw)
    assert uncached == cached


# ── the real thing: same weights, same prompts, fp32 ──

CHECKED_PROMPTS = [
    # one per category, fixed on purpose so a regression points at a behaviour
    ("store_product", "sona masoori 5 kg ka kya rate chal raha hai", "kirana"),
    ("store_faq", "what time do you shut today", "mobile"),  # the fp16 near-tie case
    ("store_chain", "2 kilo paneer ka total kitna hoga", "dairy"),
    ("calc", "1499 pe 18% gst kitna", "kirana"),
    ("date", "diwali kitne din baad hai", "kirana"),
    ("route", "upi kya hai", "kirana"),
    ("no_tool", "naraz grahak ko kaise shant karu", "kirana"),
    ("weather", "shimla me sardi hai kya", "kirana"),
]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize(("category", "message", "kb"), CHECKED_PROMPTS)
def test_fp32_cached_equals_uncached_on_real_weights(cuda_model, category, message, kb):
    """In fp32 the cache is exact, so greedy decoding must pick the same tokens.

    fp16 autocast is a different story: splitting a sequence into prefill plus
    steps moves logits by ~8e-3 there, which can flip a near-tied choice. That is
    why serving runs fp32 (configs/generation.yaml).
    """
    import milo.tools as T
    from milo.eval.harness import EVAL_NOW, store
    from milo.model.tokenizer import encode

    T.set_now(EVAL_NOW)
    T.set_merchant_kb(store(kb))
    try:
        prompt = encode(f"<|user|>\n{message}\n<|assistant|>\n")
        kw = dict(
            max_new_tokens=100,
            temperature=0.0,
            stop_strings=["<|tool_result|>"],
            device="cuda",
            amp=False,
        )
        uncached = generate(
            cuda_model, prompt, constraint=CopyConstraint(message, [message], False), use_cache=False, **kw
        )
        cached = generate(
            cuda_model, prompt, constraint=CopyConstraint(message, [message], False), use_cache=True, **kw
        )
    finally:
        T.set_now(None)
        T.set_merchant_kb(None)
    assert uncached == cached
