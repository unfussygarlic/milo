# Milo — Roadmap

> 202M parameter SLM built from scratch → WhatsApp Customer Support Agent for Indian MSMEs

---

## Current State (Oct 2026)

- [x] Pretraining (GPT-style, 202M params, block-wise weight sharing)
- [x] SFT (chat fine-tuning)
- [x] Tool-use fine-tuning (CALCULATE, SEARCH, DATE, WEATHER, RUN_CODE)
- [x] ReAct agent runtime (inference loop with tool execution)
- [ ] Fix tool-use training data (v2 — remove no-tool math, rebalance 70/30)
- [ ] Retrain and validate tool-calling accuracy

---

## Tier 1 — Before Launch (MVP)

These are non-negotiable for a working product.

### 1.1 Quantization (INT8 / INT4)
**Why:** Cuts inference cost 2-4x. Essential for serving on cheap GPUs or CPUs.
**How:** Post-training quantization with `bitsandbytes` or GPTQ. No retraining needed.
**Effort:** 1-2 days
**Impact:** Serve on ₹50K/month cloud instances instead of ₹2L/month

### 1.2 RAG Pipeline (Retrieval-Augmented Generation)
**Why:** Milo can't memorize every business's products, policies, hours. RAG lets each customer upload their catalog/FAQ and Milo answers from it.
**How:** FAISS vector store + sentence-transformers for embeddings. Inject retrieved context into prompt before generation.
**Effort:** 1 week
**Impact:** Makes Milo actually useful for real businesses. This is the product differentiator.

### 1.3 WhatsApp Integration
**Why:** 12M+ WhatsApp Business accounts in India. This is where customers already are.
**How:** WhatsApp Business API (via Twilio/Gupshup) → FastAPI backend → Milo inference
**Effort:** 1-2 weeks
**Impact:** Directly sellable product

### 1.4 Business Onboarding Dashboard
**Why:** Non-technical shop owners need to set up Milo without touching code.
**How:** Simple web UI: upload product catalog (CSV/Excel), set business hours, customize greeting, test chatbot
**Effort:** 2 weeks
**Impact:** Self-serve onboarding = scalable GTM

### 1.5 Guardrails & Safety
**Why:** Can't have a customer-facing bot saying wrong things.
**How:** Output filtering, topic boundaries (stay within business context), fallback to human handoff, PII redaction
**Effort:** 1 week
**Impact:** Trust. Businesses won't use a bot that embarrasses them.

---

## Tier 2 — After First Customers (Growth)

These improve quality and expand the market.

### 2.1 DPO (Direct Preference Optimization)
**Why:** SFT teaches format. DPO teaches *quality* — which responses customers actually prefer.
**How:** Collect thumbs-up/down from real conversations → train preference model → DPO fine-tune
**Data needed:** ~5K preference pairs from real usage
**Effort:** 2 weeks (after data collection)
**Impact:** Measurably better responses. Lower human escalation rate.

### 2.2 Hindi & Hinglish Support
**Why:** 600M+ Hindi speakers. Most WhatsApp conversations in India are Hinglish.
**How:** Collect/generate Hindi+Hinglish SFT data → fine-tune. May need to expand tokenizer vocab for Devanagari efficiency.
**Effort:** 3-4 weeks
**Impact:** 5x larger addressable market

### 2.3 RoPE Context Extension (512 → 2048 tokens)
**Why:** Customer conversations can get long. 512 tokens is about 2-3 exchanges — too short for complex support queries.
**How:** YaRN or NTK-aware RoPE scaling. Fine-tune on longer sequences.
**Effort:** 1 week + retraining
**Impact:** Handle multi-turn conversations without forgetting context

### 2.4 LoRA Adapter System
**Why:** Every business is different. A restaurant bot ≠ a clinic bot ≠ a clothing store bot.
**How:** Train domain-specific LoRA adapters (4-8MB each) that plug into the base model. One base model, many adapters.
**Effort:** 2 weeks for the system, ongoing for adapters
**Impact:** Personalization without maintaining separate models

### 2.5 Conversation Analytics Dashboard
**Why:** Businesses need to see what customers ask, where the bot fails, peak hours, resolution rates.
**How:** Log all conversations → aggregate stats → web dashboard
**Effort:** 2-3 weeks
**Impact:** Retention. Businesses that see ROI stay subscribed.

---

## Tier 3 — Milo v2 (Competitive Moat)

Bigger bets that require more resources.

### 3.1 MoE (Mixture of Experts)
**Why:** Get 500M-param quality from 202M-param inference cost. Each token activates only 2 of 8 expert FFN blocks.
**How:** Replace SwiGLU FFN with MoE layer + router. Pretrain from scratch.
**Effort:** 2-3 months
**Impact:** Step-change in quality. Compete with larger models on capability while staying cheap.

### 3.2 RETRO (Retrieval-Enhanced Transformer)
**Why:** Bake retrieval INTO the architecture instead of bolting RAG on top. More natural grounding.
**How:** Add chunked cross-attention layers, build retrieval database, pretrain from scratch.
**Effort:** 3-4 months
**Impact:** Better factual accuracy, less hallucination, but requires full retraining.

### 3.3 Knowledge Distillation
**Why:** Use a large teacher model (7B+) to make Milo punch above its weight.
**How:** Generate high-quality training data from teacher → train Milo on soft labels
**Effort:** 1-2 months
**Impact:** 202M model with 1B-level response quality

### 3.4 Speculative Decoding
**Why:** Faster inference. Use a tiny draft model (50M) to propose tokens, Milo verifies in batch.
**How:** Train a small draft model, implement spec-decode in inference loop.
**Effort:** 2-3 weeks
**Impact:** 2-3x faster response time

### 3.5 Multi-Modal (Image Input)
**Why:** Customers send product photos, screenshots of errors, pictures of items they want to buy.
**How:** Add a vision encoder (SigLIP-small) + projection layer. Fine-tune on image-text pairs.
**Effort:** 2-3 months
**Impact:** Handle "send me a photo of the issue" workflows

---

## Product Roadmap

### Phase 1 — MVP (Month 1-2)
- Quantized Milo + RAG + WhatsApp = working demo
- Target: 5 pilot businesses (restaurants, clinics, shops in one city)
- Pricing: Free pilot → ₹999/month

### Phase 2 — Product-Market Fit (Month 3-6)
- Onboarding dashboard + analytics
- Hindi/Hinglish support
- DPO from pilot feedback
- Target: 50 paying businesses
- Pricing: ₹999-2999/month based on volume

### Phase 3 — Scale (Month 6-12)
- LoRA adapters for vertical specialization
- API for developers (SDE1 target audience)
- Context extension for longer conversations
- Target: 500+ businesses
- Pricing: ₹999-9999/month tiered

### Phase 4 — Milo v2 (Year 2)
- MoE architecture
- Multi-modal support
- Knowledge distillation
- Target: Enterprise deals, API marketplace

---

## Architecture

```
Customer (WhatsApp)
    ↓
WhatsApp Business API (Gupshup/Twilio)
    ↓
FastAPI Backend
    ├── Auth & Rate Limiting
    ├── Conversation Manager
    ├── RAG Pipeline (FAISS + embeddings)
    │     └── Business Knowledge Base
    ├── Milo Inference (quantized, ReAct loop)
    │     └── Tools: CALCULATE, SEARCH, DATE, WEATHER, RUN_CODE
    ├── Guardrails & Safety Filter
    └── Analytics Logger
    ↓
Business Dashboard (React)
    ├── Onboarding wizard
    ├── Conversation logs
    ├── Analytics & metrics
    └── Settings & customization
```

---

## Key Metrics to Track

| Metric | Target (MVP) | Target (PMF) |
|--------|-------------|--------------|
| Response accuracy | >85% | >92% |
| Tool-call accuracy | >90% | >95% |
| Avg response time | <3s | <1.5s |
| Human escalation rate | <30% | <15% |
| Customer satisfaction | >3.5/5 | >4.2/5 |
| Monthly churn | <15% | <8% |
| Conversations/day/business | 20+ | 50+ |

---

*Last updated: October 2026*
*Model: Milo 202M (SLM, GPT-style, block-wise weight sharing)*
---

## Phase 4 Docker check (do not drop)

The image build must run, inside the container, against the Phase 0 baseline:

- `pytest -m ""` — 12 tests / 154 assertions, replay 17240/17240, fuzz 0 dead ends
- `milo-generate-data --seed 42 --scale 0.2` — sha256 must be
  `bf93414c0087b15e3d8ebfc51249ed2b792f59fcbf41db4e928f970e6014bdca` (train) and
  `7207d3145bca10ac2b18ddb3118a1eef8d1456d91efb501e008fac9595cbd2e6` (val)
- `milo-eval --rescore eval_results/<run>.jsonl` — tables identical to the baseline

A dependency version that changes the model's behaviour should fail the build, not ship.

---

## Decoding notes carried into Phase 4

**Trim, don't slide.** The KV cache is rebuilt from scratch on every step once the
context passes 512 tokens, because a sliding window moves every RoPE position. We
deliberately did *not* build a sliding cache (it would need keys stored pre-rotation
and would no longer be bit-exact against the plain path). Instead, Phase 4 must trim
conversation history so that *prompt + max_new_tokens* always fits in 512.
`milo.agent.decoding.generate` logs a warning the first time a generation crosses the
window, so if trimming is ever wrong it says so in the serving logs rather than just
getting slow.

**Re-benchmark precision on the serving GPU.** `configs/generation.yaml` ships
`amp: false` (fp32) because on an RTX 4060 Laptop at batch 1 we are kernel-launch
bound: fp16 autocast's per-op casts cost more than the math saves, and fp32 also makes
cached and uncached decoding numerically identical. Measured there, 100 new tokens,
greedy, no constraint:

| prompt tokens | fp16 uncached | fp16 cached | fp32 uncached | fp32 cached |
|---|---|---|---|---|
| 35 | 76.4 t/s | 88.9 | 52.9 | **111.0** |
| 140 | 59.9 t/s | 88.3 | 32.8 | **106.5** |
| 280 | 43.2 t/s | 84.8 | 19.7 | **91.2** |

That ordering is hardware- and batch-size-specific. On the EC2 GPU (bigger, possibly
batching several merchants' traffic) fp16 may well win. Re-run the sweep there before
choosing `amp`, and if fp16 wins, expect the one near-tie case to resolve differently —
it is a ±0.01 logit tie, not a correctness problem.
