"""``milo-generate-data`` - write a tool-use training set.

    milo-generate-data                                  # -> data/tool_use_v4_train.jsonl + _val.jsonl
    milo-generate-data --scale 0.1 --show 20            # quick look
    milo-generate-data --seed 42 --scale 0.2 --output /tmp/t.jsonl

Results are produced by the real tools in :mod:`milo.tools`, so training bytes
equal runtime bytes (invariant 1). For a given seed the output is reproducible
down to the byte.
"""

import argparse
import collections
import json
import os
import random
import re

from milo.data.generator.build import generate, token_len_fn, validate
from milo.data.generator.core import LEAK_DROPPED


def main():
    ap = argparse.ArgumentParser(description="Milo v4 tool-use data")
    ap.add_argument("--output", default="data/tool_use_v4_train.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all family sizes")
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--max_tokens", type=int, default=500)
    ap.add_argument("--show", type=int, default=6, help="print N random samples")
    args = ap.parse_args()

    print("=" * 64)
    print("  Milo — tool-use data v4")
    print("=" * 64)
    train, counts, skips = generate(args.seed, args.scale)
    val, _, _ = generate(args.seed + 100003, args.scale * args.val_frac, quiet=True)
    train_texts = {e["text"] for e in train}
    val = [e for e in val if e["text"] not in train_texts]

    tok, how = token_len_fn()
    for name, data in (("train", train), ("val", val)):
        before = len(data)
        data[:] = [e for e in data if tok(e["text"]) <= args.max_tokens]
        if before != len(data):
            print(f"  dropped {before - len(data)} {name} examples over {args.max_tokens} tokens")

    problems = validate(train) + validate(val)
    if problems:
        print("  VALIDATION PROBLEMS: " + ", ".join(f"{k} ×{v}" for k, v in problems.items()))
    else:
        print("  validation: all examples passed (structure, parser round-trip, CALCULATE + RUN_CODE re-run, grounding)")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    val_path = args.output.replace("train", "val") if "train" in args.output else args.output + ".val"
    for path, data in ((args.output, train), (val_path, val)):
        with open(path, "w", encoding="utf-8") as f:
            for e in data:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")

    n = len(train)
    tool_n = sum(1 for e in train if e["tools"])
    multi = sum(1 for e in train if len(e["tools"]) > 1)
    by_tool = collections.Counter(t for e in train for t in e["tools"])
    lens = [tok(e["text"]) for e in train]
    hi_share = sum(1 for e in train if re.search(r"\b(hai|kya|kitne|kitna|batao|chahiye)\b", e["text"].split("<|assistant|>")[0])) / n
    print(f"\n  train {n}  |  val {len(val)}  (val de-duplicated against train)")
    print(f"  tool {tool_n} ({tool_n / n:.0%})  no-tool {n - tool_n} ({(n - tool_n) / n:.0%})  multi-tool {multi} ({multi / n:.0%})")
    print(f"  unique texts {len({e['text'] for e in train}) / n:.1%}   Hinglish user msgs ≈{hi_share:.0%}")
    print(f"  tokens ({how}): avg {sum(lens) / n:.0f}, max {max(lens)}")
    print("  tool calls: " + ", ".join(f"{k} {v}" for k, v in by_tool.most_common()))
    print("  families: " + ", ".join(f"{k} {v}" for k, v in counts.most_common()))
    top_skips = skips.most_common(6)
    if top_skips:
        print("  most common rejections: " + "; ".join(f"{k} ×{v}" for k, v in top_skips))
    print(f"  leak filter dropped {len(set(LEAK_DROPPED))} bank templates that overlapped the test split")
    print(f"\n  → {args.output}\n  → {val_path}")

    if args.show:
        print("\n" + "=" * 64 + "\n  Samples\n" + "=" * 64)
        for e in random.Random(1).sample(train, min(args.show, n)):
            print(f"[{e['family']}]\n{e['text']}\n" + "─" * 50)


if __name__ == "__main__":
    main()
