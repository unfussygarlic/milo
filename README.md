# Milo

A 202M-parameter transformer, trained from scratch, that answers customer messages for Indian
MSMEs (kirana shops, mobile shops, tailors, bakeries…) in Hinglish or English — on WhatsApp.

It is small enough to serve on one cheap GPU, so it is built to **look things up instead of
remembering them**: prices, stock, store policies and dates come from tools, and a decoding
constraint plus a grounding guard make sure no number reaches a customer that did not come from
the customer's own message or a tool result.

| | |
|---|---|
| Parameters | 202M total (101M unique — 16 layers from 8 shared blocks) |
| d_model / heads | 1280 / 20 query, 4 KV (GQA) |
| Context | 512 tokens |
| Vocab | 50,257 (GPT-2 tiktoken) |
| Architecture | RMSNorm, SwiGLU, RoPE, block-wise weight sharing (MobileLLM-LS) |
| Pipeline | pretrain 3.6B tokens → SFT 90k instructions → tool-use fine-tune |
| Held-out score | **87%** on 198 behavioural test cases (constrained decoding on) |

## Tools

One module per tool in [src/milo/tools/](src/milo/tools/), used by **both** the data generator and
the runtime — so a tool result in training is byte-identical to the same result at inference.

| Tool | What it does |
|---|---|
| `MQ_SEARCH` | the merchant's catalog / FAQ / store info, with Hinglish synonyms and fuzzy matching |
| `CALCULATE` | arithmetic over an AST allow-list (never `eval`), plus `emi()` and `pct_change()` |
| `DATE` | IST clock, lenient date parsing, Indian festival calendar, working-day counts |
| `SEARCH` | general knowledge (Wikipedia) |
| `WEATHER` | current conditions (wttr.in) |
| `RUN_CODE` | Python in a child process — **not a sandbox**, disabled for customer traffic |

The format the model emits, which never changes:

```
<|user|>\n{msg}\n<|assistant|>\n<|tool_call|>TOOL(args)<|tool_result|>{result}<|end_tool|>\n{answer}<|endoftext|>
```

## Layout

```
configs/              deployment settings (model dims, sampling, training hyperparameters)
src/milo/
  config.py           the one config loader
  special_tokens.py   every special-token string, in one place
  model/              SLM architecture, tokenizer, checkpoint loading
  tools/              one module per tool + registry, parser, grounding
  agent/              generation, constrained decoding, the ReAct loop, chat CLI
  data/               merchants, paraphrase banks, the training-data generator
  eval/               behavioural eval + dev/test cases + the constraint invariants
  training/           pretraining, SFT, tool-use fine-tuning, loss masking
scripts/              corpus collection, EC2 setup, the dev chat UI
tests/                pytest
docs/research/        roadmap and notes
data/ experiments/ eval_results/    generated artifacts (gitignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cu130   # match your CUDA
pip install -e ".[dev]"
```

`tiktoken` is a **required** dependency, not an optional one: the GPT-2 BPE files decide
tokenization and the generator's `--max_tokens` filter, so a missing encoding would silently
change the dataset. For offline or Docker use, pre-cache those files and point
`TIKTOKEN_CACHE_DIR` at them (see [.env.example](.env.example)) — otherwise the first run tries to
download them.

Copy `.env.example` to `.env` for local settings. Keep `MILO_DISABLE_RUN_CODE=1` anywhere
customers can reach.

## Use

```bash
# chat with the agent in a terminal (tools, constraint and guard all on)
milo-chat --checkpoint experiments/tools_v51/checkpoints/best.pt
MILO_MERCHANT_KB=merchant_kbs/kirana.json milo-chat --checkpoint …/best.pt

# generate tool-use training data (deterministic for a given seed)
milo-generate-data --seed 42 --scale 1.0 --output data/tool_use_train.jsonl

# behavioural eval: dev + held-out test, with and without the constraint
milo-eval --checkpoint experiments/tools_v51/checkpoints/best.pt --constrain both
milo-eval --rescore eval_results/tools_v51+constrain.jsonl      # re-score saved traces, no GPU

# training (in pipeline order)
milo-pretrain
milo-sft
milo-finetune-tools --checkpoint experiments/sft_20260929_075754/checkpoints/best.ptrom
```

Hyperparameters for all three training stages live in [configs/training.yaml](configs/training.yaml);
inference settings in [configs/model.yaml](configs/model.yaml) and
[configs/generation.yaml](configs/generation.yaml).

A browser chat UI for local poking lives in [scripts/dev_chat_ui/](scripts/dev_chat_ui/). It is the
plain no-tool generation path — **dev only**, no constraint and no guard.

## Tests and invariants

```bash
pytest                      # unit tests (fast)
pytest -m slow              # + the constraint dead-end fuzzer (~90s)
milo-check-constraint data/tool_use_v51_train.jsonl   # replay: constraint accepts every target
milo-fuzz-constraint  data/tool_use_v51_train.jsonl   # fuzz: zero dead ends
```

The invariant tests need a generated dataset (`data/` is gitignored); they skip when there isn't
one. Point them elsewhere with `MILO_INVARIANT_DATASET`.

## The five invariants

1. Tool result strings are byte-identical between the generator and the runtime — both import
   [src/milo/tools/](src/milo/tools/).
2. The decoding constraint accepts 100% of training targets and has zero dead ends
   (`tests/test_constraint_replay.py`, `tests/test_constraint_fuzz.py`).
3. `milo.eval.cases_test` is never copied into templates or training data — the generator's leak
   filter drops any template sharing a four-word run with a test case.
4. A customer-facing reply never contains a number that is not in the customer's message or a tool
   result (constrained decoding + the grounding guard). Never turned off in serving.
5. `RUN_CODE` is always disabled for WhatsApp traffic (`MILO_DISABLE_RUN_CODE=1`). It is not a
   sandbox.

Constants that decide tool output or what the constraint allows (`MAX_RESULT_CHARS`, `MQ_MAX_HITS`,
`WORKING_WEEKDAYS`, the festival tables, `ALWAYS_OK_NUMBERS`, the special tokens) live in code, not
in YAML, each with a comment saying it is part of the training-data contract. YAML is for
deployment settings only.

## Status

Done: pretraining, SFT, tool-use fine-tuning, the ReAct runtime, constrained decoding, the
behavioural eval, this restructure.
Next: KV-cache decoding, merchant onboarding from spreadsheets, the WhatsApp serving layer, owner
handoff.
