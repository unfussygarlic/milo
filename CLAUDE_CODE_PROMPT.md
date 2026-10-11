# Milo — restructure + product wrap

You're working in the repo for **Milo**, a 202M-parameter transformer trained from scratch
(pretraining → SFT → tool-use fine-tuning, all done). It's becoming a **WhatsApp customer-support
agent for Indian MSMEs** (kirana, mobile shops, tailors, bakeries…), answering in Hinglish/English.
The model is frozen: **do not retrain, do not change training data formats.** Your job is to
(1) turn this research folder into a clean, production-grade codebase without changing behaviour,
then (2) build the serving layer around it.

Work in phases. **Stop after Phase 0 and wait for my approval of the plan.** After that, finish each
phase, run the checks, commit, give me a short summary, and continue.

---

## What exists (read these first)

| File | What it is |
|---|---|
| `model.py` | `SLM` transformer: d_model 1280, 20 query heads / 4 KV heads (GQA), 16 layers, `share_blocks=True`, seq_len 512, GPT-2 tiktoken vocab |
| `milo_tools.py` | **Single source of truth for tools**: CALCULATE (AST-safe, helpers `emi()`, `pct_change()`, comparisons), DATE (IST, festival calendar), MQ_SEARCH (merchant catalog/FAQ retrieval with Hinglish synonyms), SEARCH (Wikipedia), WEATHER (wttr.in), RUN_CODE (subprocess, *not* a sandbox). Also: tool-call parser, grounding utilities (`numbers_in`, `ungrounded_numbers`, `fallback_answer`) and `CopyConstraint` (constrained decoding rules) |
| `react_runtime.py` | ReAct loop: generate → stop at `<\|tool_result\|>` → run tool → inject result → continue. Constrained decoding (`apply_constraint`) + number-grounding guard. CLI chat |
| `merchants.py` | 16 synthetic merchant types → randomised store knowledge bases (catalog + FAQ JSON) |
| `generate_tool_data_v4.py` | Training-data generator (actually v5; name is historical). Uses real tools so results are byte-identical to runtime |
| `paraphrase_banks.py` | Hand-written customer-message templates merged into the generator |
| `finetune_tools.py` | Tool-use fine-tuning (loss masked on tool results, dynamic padding) |
| `eval_tools.py` | Behavioural eval: 138 dev + **198 held-out test** prompts, deterministic stores/clock, `--constrain`, `--rescore` |
| `test_milo.py` | 154 unit tests (tools, parser, constraint, loss mask) |
| `check_constraint_vs_data.py` | Invariant: constraint must accept 100% of training targets |
| `fuzz_constraint.py` | Invariant: constraint must have 0 dead-end states |
| `generate_tool_data.py` | Old v3 generator — obsolete |
| also | pretraining / SFT scripts, `roadmap.md`, `data/`, `experiments/`, `merchant_kbs/`, `eval_results/` |

Current production checkpoint: `experiments/tools_v51/checkpoints/best.pt`
(SFT base: `experiments/sft_20260929_075754/checkpoints/best.ptrom` — yes, `.ptrom`, typo in the filename).
Held-out test score: 87% with constraint on. Hardware: RTX 4060 8 GB locally; deploy target EC2.

### Tool-call format (never change)
```
<|user|>\n{msg}\n<|assistant|>\n<|tool_call|>TOOL(args)<|tool_result|>{result}<|end_tool|>\n{answer}<|endoftext|>
```

### Invariants — break none of these
1. Tool result strings are byte-identical between the generator and the runtime (both use the same tool code).
2. `check_constraint_vs_data.py` → all training targets accepted. `fuzz_constraint.py` → 0 dead ends.
3. `eval_tools.TEST_CASES` is never copied into templates or training data (the generator has a leak filter that imports it — keep that working).
4. Customer-facing replies never contain a number that isn't in the user message or a tool result (constraint + guard). Never turn these off in serving.
5. RUN_CODE is **always disabled** for WhatsApp traffic (`MILO_DISABLE_RUN_CODE=1`). It is not a sandbox.

---

## Phase 0 — analyse and propose (then stop)

Read the whole repo (including scripts I didn't list) and propose a production layout. Something like the
sketch below, but adapt it to what's actually there; don't create empty placeholder modules:

```
milo/
├── pyproject.toml            # package + deps + entry points; ruff + pytest config
├── README.md                 # what Milo is, setup, train/eval/serve commands
├── .env.example
├── configs/                  # model / finetune / serving config (YAML or TOML) instead of constants in code
├── src/milo/
│   ├── model/                # SLM, tokenizer helpers, checkpoint loading
│   ├── tools/                # one module per tool + registry, parser, formatting
│   ├── agent/                # react loop, generation, constraint, guard, KV-cache decoding
│   ├── data/                 # generator, merchants, paraphrase banks
│   ├── training/             # pretrain, sft, finetune_tools
│   ├── eval/                 # eval harness + cases (dev/test split kept separate)
│   ├── merchant/             # KB schema, loading, onboarding (Phase 3)
│   └── serving/              # WhatsApp webhook app, conversations, handoff (Phases 2, 4)
├── scripts/                  # thin CLI wrappers if entry points aren't enough
├── tests/                    # pytest: split test_milo.py by module + invariant checks
├── merchant_kbs/             # example store JSONs (committed)
└── data/ experiments/ eval_results/   # git-ignored
```

Give me: the proposed tree, a file-by-file move/split map (old → new), what's deleted
(e.g. the v3 generator — ask first), how imports change, and the dependency list. **Wait for my OK.**

## Phase 1 — restructure (behaviour must not change)

- Use `git mv` so history survives. Update all imports; no compatibility shims left behind.
- Move magic constants (model config, paths, seq_len, tool limits) into `configs/` with one loader.
- Rename `generate_tool_data_v4.py` to something version-free; keep the CLI flags.
- Convert `test_milo.py` into pytest modules. Add the two invariant scripts as tests (mark the fuzz test slow).
- Add ruff (lint + format) and type hints on public functions. Don't do unrelated refactors.

**Proof of no behaviour change** (do all of these, and show me the output):
1. Before moving anything, generate a dataset: `--seed 42 --scale 0.2` → save the files. After the
   restructure, regenerate with the same args → **must be byte-identical** (`cmp` / sha256).
2. `eval_tools.py --rescore eval_results/*.jsonl` → identical tables before and after.
3. All tests pass; constraint replay = 100%; fuzz = 0 dead ends.

## Phase 2 — KV-cache decoding

Right now `generate()` re-runs the full context for every new token. Add incremental decoding:
- Add `past_key_values` / cache support to the attention in the model (mind GQA: cache the 4 KV heads, not 20; and `share_blocks=True`).
- Read how positions are encoded (RoPE vs learned) and handle offsets correctly.
- When a tool result is injected, the current code re-encodes the whole text. Keep the token sequence
  identical to what the model saw in training: simplest correct option is to re-prefill after each
  injection (only 1–3 per reply) and use the cache within each generation run. Don't change tokenization.
- Context longer than 512: keep current behaviour (most recent 512) by re-prefilling the window.
- Verify: with greedy decoding, cached and uncached generation produce **identical tokens** on all
  eval prompts; logits max abs diff small (fp16 tolerance). Report tokens/sec before vs after on the 4060.
- Also profile `apply_constraint` (it decodes candidates in Python each step); speed it up if it dominates,
  without changing which tokens are allowed (re-run replay + fuzz).

## Phase 3 — merchant onboarding

A shop owner should be able to fill a spreadsheet and get a store KB:
- CSV template(s): products (name, variant/size, price incl. GST, stock: in stock / out of stock / only N left,
  optional offer, extra keywords incl. Hinglish names) and store info/FAQ (hours, address, payment modes,
  delivery terms, returns, bulk discount, free-text extra FAQs).
- `milo-onboard products.csv store.csv --out merchant_kbs/<shop>.json` with validation and clear error
  messages (row number + problem). Output must match the KB schema `milo_tools.mq_search` reads
  (same as `merchants.build_kb`). Define the schema once (pydantic or dataclass) and use it in both places.
- After building, run a retrieval self-check: search every product name and FAQ topic and report anything
  that doesn't come back first.

## Phase 4 — WhatsApp serving

FastAPI app for the **WhatsApp Business Cloud API** (check Meta's current docs for the Graph API version,
webhook payload shape and send-message endpoint; don't trust memory, and keep the version in config):
- `GET /webhook` verification (verify token), `POST /webhook` receiver with **`X-Hub-Signature-256`
  verification** using the app secret. Return 200 fast; process in a background worker.
- Dedupe by WhatsApp message id (Meta retries). Ignore status updates. Non-text messages
  (image/voice/sticker) → polite fixed reply asking for text.
- Map `phone_number_id` → merchant KB, so one deployment can serve several shops.
- Per-customer conversation history (SQLite to start), trimmed so prompt + reply fits the 512-token window.
- One model instance on one GPU with a single inference queue. Constraint + guard always on, RUN_CODE disabled,
  sampling settings from config (greedy or low temperature for production).
- Log every turn (message, tool calls, results, reply, latency, guard/fallback events) to SQLite/JSONL.
  This is the future real-data eval set. Keep secrets in env, and don't log access tokens.
- Basic per-number rate limiting; graceful reply if the model or a tool errors.
- `Dockerfile` + `docker-compose.yml` (app + volume for DB/checkpoints), `.env.example`, health endpoint.
- Tests with mocked Graph API calls (no network in tests), including signature verification and dedupe.

## Phase 5 — owner handoff

Orders, complaints, order-status questions, payments and "talk to a human" requests must reach the shop owner:
- Detect these with simple, testable rules on the customer message (keywords in Hinglish/English) plus the
  model's reply. No new model. Put the rules in one module with unit tests.
- Send the owner a short summary (customer number, message, what Milo replied).
- **Flag to me before building the delivery channel:** WhatsApp only allows free-form business-initiated
  messages inside the 24-hour customer-service window; outside it you need approved message templates.
  Propose options (template message, email, Telegram bot, a simple owner inbox page) with trade-offs and let me choose.

---

## Rules
- Ask before deleting files, changing dependencies' major versions, or anything that touches `data/`, `experiments/` or checkpoints.
- Commit per phase with a clear message. Keep `README.md` current: setup, train, eval, onboard, serve.
- No frontend in this task. The customer UI is WhatsApp; owners use spreadsheets for now.
- If something in these instructions conflicts with what the code actually does, tell me instead of guessing.