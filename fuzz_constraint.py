"""
Dead-end fuzzer for milo_tools.CopyConstraint.

A dead end = a state where no character and no end-of-reply is allowed. At runtime that
forces a fallback to unconstrained decoding (the eval's "fallbacks" counter). We start from
random prefixes of real training targets and random-walk through allowed single characters
(GPT-2 has a token for every byte, so single chars are always available).

    python fuzz_constraint.py data/tool_use_v51_train.jsonl
"""
import json, random, string, sys, collections
import milo_tools as T
from check_constraint_vs_data import steps

CHARS = list(string.printable) + ["₹", "×", "—", "�", "é"]


def walk(rng, user, results, prefix, n=40):
    c = T.CopyConstraint(user, [user] + results, numbers_in_reply=bool(results))
    text = prefix
    for _ in range(n):
        if text.endswith(T.TOOL_RESULT_TAG):
            return None                                   # step would stop here — fine
        ok = [ch for ch in CHARS if c.ok(text, ch)]
        if not ok:
            return None if c.eos_ok(text) else text       # dead end
        text += rng.choice(ok)
    return None


def main(path, n_examples=3000, walks=4, seed=0):
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    rng.shuffle(rows)
    deads = collections.Counter(); samples = {}
    for r in rows[:n_examples]:
        user, body = r["text"][len("<|user|>\n"):].split("\n<|assistant|>\n", 1)
        for step_text, results in steps(body.replace("<|endoftext|>", "")):
            for _ in range(walks):
                cut = rng.randrange(len(step_text) + 1)
                d = walk(rng, user, results, step_text[:cut])
                if d is not None:
                    key = d[d.rfind("<|tool_call|>"):][:28] if "<|tool_call|>" in d[-80:] else "reply"
                    deads[key.split("(")[0]] += 1
                    samples.setdefault(key.split("(")[0], (user, d[-70:]))
    total = sum(deads.values())
    print(f"dead ends: {total}")
    for k, v in deads.most_common():
        print(f"  {k!r}: {v}   e.g. user={samples[k][0][:40]!r} …{samples[k][1]!r}")
    return total


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1] if len(sys.argv) > 1 else "data/tool_use_v51_train.jsonl") else 0)