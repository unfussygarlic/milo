"""
Instruction fine-tuning data collector for Milo v2 (202M)
Streams from 5 open-source HuggingFace datasets, filters for quality,
and saves as JSONL ready for fine-tuning.

Sources:
  1. SlimOrca        → general instruction following (GPT-4 quality)
  2. MetaMathQA      → step-by-step math reasoning
  3. CodeAlpaca-20k  → coding instructions
  4. SciQ            → science QA with explanations
  5. OpenHermes-2.5  → diverse multi-domain

Usage:
    cd ~/milo
    pip install datasets tiktoken --break-system-packages -q
    python collect_sft_data.py

Output:
    data/sft/train.jsonl   (~13.5K examples)
    data/sft/val.jsonl     (~1.5K examples)
"""

import hashlib
import json
import os
import random
import sys
import time
from collections import defaultdict

try:
    import tiktoken
except ImportError:
    os.system("pip install tiktoken --break-system-packages -q")
    import tiktoken

try:
    from datasets import load_dataset
except ImportError:
    os.system("pip install datasets --break-system-packages -q")
    from datasets import load_dataset

# ─── Config ───────────────────────────────────────────────

TARGETS = {
    "general": 30000,  # SlimOrca
    "math": 25000,  # MetaMathQA
    "code": 15000,  # CodeAlpaca
    "science": 10000,  # SciQ
    "diverse": 20000,  # OpenHermes 2.5
}

TOTAL_TARGET = sum(TARGETS.values())  # 15,000

MAX_SEQ_TOKENS = 480  # full formatted example must fit in 512 seq_len
MIN_RESPONSE_TOKENS = 15
MIN_RESPONSE_CHARS = 40
VAL_RATIO = 0.10
OUTPUT_DIR = "data/sft"
SEED = 42

enc = tiktoken.get_encoding("gpt2")

# ─── Helpers ──────────────────────────────────────────────


def count_tokens(text):
    return len(enc.encode(text, allowed_special={"<|endoftext|>"}))


def format_example(user_msg, assistant_msg):
    """Format as Milo's chat template."""
    return f"<|user|>\n{user_msg.strip()}\n<|assistant|>\n{assistant_msg.strip()}<|endoftext|>"


def is_valid(user_msg, assistant_msg):
    """Quality + length filter."""
    if not user_msg or not assistant_msg:
        return False
    user_msg = user_msg.strip()
    assistant_msg = assistant_msg.strip()

    if len(user_msg) < 10 or len(assistant_msg) < MIN_RESPONSE_CHARS:
        return False

    # Token length check
    formatted = format_example(user_msg, assistant_msg)
    n = count_tokens(formatted)
    if n > MAX_SEQ_TOKENS or n < 20:
        return False
    if count_tokens(assistant_msg) < MIN_RESPONSE_TOKENS:
        return False

    # Basic English heuristic
    ascii_ratio = sum(1 for c in assistant_msg if ord(c) < 128) / max(len(assistant_msg), 1)
    if ascii_ratio < 0.80:
        return False

    # Skip if response is mostly code-fence with no explanation (low quality)
    if assistant_msg.count("```") >= 4 and len(assistant_msg) < 100:
        return False

    return True


seen = set()


def is_dup(text):
    h = hashlib.md5(text.strip().lower()[:300].encode()).hexdigest()
    if h in seen:
        return True
    seen.add(h)
    return False


def collect(name, ds_iter, extract_fn, target, category, max_scan=None):
    """
    Generic collector: streams dataset, extracts (user, assistant) pairs,
    filters, deduplicates, and returns a list of examples.
    """
    collected = []
    scanned = 0
    skip_dup = 0
    skip_bad = 0
    skip_err = 0
    t0 = time.time()

    if max_scan is None:
        max_scan = target * 20  # scan up to 20x target before giving up

    print(f"\n{'─' * 60}")
    print(f"  {category.upper()}: {name}  (target {target:,})")
    print(f"{'─' * 60}")

    for item in ds_iter:
        if len(collected) >= target:
            break
        if scanned >= max_scan:
            print(f"  ⚠ Hit scan limit ({max_scan:,}), stopping with {len(collected):,}")
            break

        scanned += 1

        try:
            user_msg, assistant_msg = extract_fn(item)
        except Exception:
            skip_err += 1
            continue

        if is_dup(user_msg):
            skip_dup += 1
            continue

        if not is_valid(user_msg, assistant_msg):
            skip_bad += 1
            continue

        collected.append(
            {
                "messages": [
                    {"role": "user", "content": user_msg.strip()},
                    {"role": "assistant", "content": assistant_msg.strip()},
                ],
                "category": category,
                "source": name,
            }
        )

        if len(collected) % 500 == 0:
            elapsed = time.time() - t0
            rate = len(collected) / elapsed
            print(
                f"  {len(collected):>5,}/{target:,}  "
                f"(scanned {scanned:,} | dup {skip_dup:,} | bad {skip_bad:,} | "
                f"{rate:.0f} ex/s)"
            )

    elapsed = time.time() - t0
    print(
        f"  ✓ {len(collected):,}/{target:,} collected in {elapsed:.1f}s  "
        f"(scanned {scanned:,}, dup {skip_dup:,}, bad {skip_bad:,}, err {skip_err:,})"
    )
    return collected


# ═══════════════════════════════════════════════════════════
#  Dataset-specific collectors
# ═══════════════════════════════════════════════════════════


def collect_general():
    """
    SlimOrca — curated subset of OpenOrca, GPT-4 generated.
    High quality general instruction following.
    Fields: system_prompt, question, response
    """
    print("\n  Loading SlimOrca (streaming)...")
    ds = load_dataset("Open-Orca/SlimOrca", split="train", streaming=True)

    def extract(item):
        convos = item["conversations"]
        user_msg = None
        assistant_msg = None
        for turn in convos:
            role = turn.get("from", "")
            if role == "human" and user_msg is None:
                user_msg = turn["value"]
            elif role == "gpt" and user_msg is not None and assistant_msg is None:
                assistant_msg = turn["value"]
                break
        if not user_msg or not assistant_msg:
            raise ValueError("incomplete")
        return user_msg, assistant_msg

    return collect("SlimOrca", ds, extract, TARGETS["general"], "general", max_scan=300_000)


def collect_math():
    """
    MetaMathQA — math problems with step-by-step solutions.
    Fields: query, response, type
    """
    print("\n  Loading MetaMathQA (streaming)...")
    ds = load_dataset("meta-math/MetaMathQA", split="train", streaming=True)

    def extract(item):
        return item["query"], item["response"]

    return collect("MetaMathQA", ds, extract, TARGETS["math"], "math", max_scan=400_000)


def collect_code():
    """
    CodeAlpaca-20k — coding instruction/output pairs.
    Fields: instruction, input, output
    """
    print("\n  Loading CodeAlpaca-20k (streaming)...")
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train", streaming=True)

    def extract(item):
        instruction = item["instruction"]
        inp = (item.get("input") or "").strip()
        if inp:
            instruction = f"{instruction}\n\nInput:\n{inp}"
        return instruction, item["output"]

    return collect("CodeAlpaca-20k", ds, extract, TARGETS["code"], "code", max_scan=20_000)


def collect_science():
    """
    SciQ — science exam questions with support paragraphs.
    Fields: question, correct_answer, support, distractor1/2/3
    We combine the short answer with the support text for a proper explanation.
    """
    print("\n  Loading SciQ (streaming)...")
    ds = load_dataset("allenai/sciq", split="train", streaming=True)

    def extract(item):
        question = item["question"]
        answer = item["correct_answer"].strip()
        support = (item.get("support") or "").strip()

        # Build a full response: answer + explanation
        if support and len(support) > 30:
            response = f"{answer}\n\n{support}"
        else:
            response = answer
        return question, response

    return collect("SciQ", ds, extract, TARGETS["science"], "science", max_scan=15_000)


def collect_diverse():
    """
    OpenHermes 2.5 — very diverse multi-source instruction data.
    Fields: conversations (list of {from: human/gpt, value: str})
    We take the first user→assistant turn only.
    """
    print("\n  Loading OpenHermes-2.5 (streaming)...")
    ds = load_dataset("teknium/OpenHermes-2.5", split="train", streaming=True)

    def extract(item):
        convos = item["conversations"]
        user_msg = None
        assistant_msg = None
        for turn in convos:
            role = turn.get("from", "")
            if role == "human" and user_msg is None:
                user_msg = turn["value"]
            elif role == "gpt" and user_msg is not None and assistant_msg is None:
                assistant_msg = turn["value"]
                break
        if not user_msg or not assistant_msg:
            raise ValueError("incomplete")
        return user_msg, assistant_msg

    return collect("OpenHermes-2.5", ds, extract, TARGETS["diverse"], "diverse", max_scan=300_000)


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════


def main():
    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    category_counts = defaultdict(int)
    t_start = time.time()

    # Collect from each source in order
    collectors = [
        collect_general,
        collect_math,
        collect_code,
        collect_science,
        collect_diverse,
    ]

    for fn in collectors:
        try:
            examples = fn()
            all_data.extend(examples)
            for ex in examples:
                category_counts[ex["category"]] += 1
        except Exception as e:
            print(f"\n  ✗ FAILED: {e}")
            import traceback

            traceback.print_exc()
            continue

    if not all_data:
        print("\n✗ No data collected!")
        sys.exit(1)

    # Shuffle
    random.shuffle(all_data)

    # Train / val split
    val_size = max(1, int(len(all_data) * VAL_RATIO))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]

    # Save
    train_path = os.path.join(OUTPUT_DIR, "train.jsonl")
    val_path = os.path.join(OUTPUT_DIR, "val.jsonl")

    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(val_path, "w") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # ─── Token stats ──────────────────────────────────────
    total_tokens = 0
    token_counts = []
    for item in all_data:
        formatted = format_example(
            item["messages"][0]["content"],
            item["messages"][1]["content"],
        )
        n = count_tokens(formatted)
        total_tokens += n
        token_counts.append(n)

    token_counts.sort()
    avg_tokens = total_tokens / len(all_data)
    p50 = token_counts[len(token_counts) // 2]
    p90 = token_counts[int(len(token_counts) * 0.9)]
    p99 = token_counts[int(len(token_counts) * 0.99)]

    elapsed = time.time() - t_start

    # ─── Summary ──────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  COLLECTION COMPLETE  ({elapsed:.0f}s)")
    print(f"{'=' * 60}")
    print(f"  Total examples : {len(all_data):,}")
    print(f"    Train        : {len(train_data):,}")
    print(f"    Val          : {len(val_data):,}")
    print()
    print("  By category:")
    for cat in ["general", "math", "code", "science", "diverse"]:
        n = category_counts.get(cat, 0)
        target = TARGETS.get(cat, 0)
        pct = n / len(all_data) * 100
        bar = "█" * int(pct / 2)
        print(f"    {cat:12s}  {n:>5,} / {target:,}  ({pct:4.1f}%)  {bar}")
    print()
    print("  Token stats:")
    print(f"    Total        : {total_tokens:>10,}")
    print(f"    Avg/example  : {avg_tokens:>10.0f}")
    print(f"    Median (p50) : {p50:>10,}")
    print(f"    p90          : {p90:>10,}")
    print(f"    p99          : {p99:>10,}")
    print()
    print("  Saved to:")
    print(f"    {os.path.abspath(train_path)}")
    print(f"    {os.path.abspath(val_path)}")
    print()

    # ─── Preview ──────────────────────────────────────────
    print("  Sample examples:")
    for cat in ["general", "math", "code", "science"]:
        for item in train_data:
            if item["category"] == cat:
                u = item["messages"][0]["content"][:80].replace("\n", " ")
                a = item["messages"][1]["content"][:80].replace("\n", " ")
                print(f"\n    [{cat}]")
                print(f"    User: {u}...")
                print(f"    Asst: {a}...")
                break

    print(f"\n{'=' * 60}")
    print("  Ready for fine-tuning!")
    print(f"  Next: python finetune.py --data {train_path} --val {val_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
