"""
Invariant: the decoding constraint must never block anything the model was trained to write.
Replays every assistant turn in a dataset character by character through milo_tools.CopyConstraint,
exactly as the runtime builds it (sources = user msg + results so far), plus the end-of-reply check.

    python check_constraint_vs_data.py data/tool_use_v43_train.jsonl
"""
import json, re, sys, collections
import milo_tools as T

TC, TR, ET = T.TOOL_CALL_TAG, T.TOOL_RESULT_TAG, T.END_TOOL_TAG


def steps(body):
    """Split the assistant body into generation steps, as the runtime sees them."""
    out, results = [], []
    rest = body
    while True:
        j = rest.find(TR)
        if j < 0:
            out.append((rest, list(results))); return out
        k = rest.find(ET, j)
        out.append((rest[:j + len(TR)], list(results)))
        results.append(rest[j + len(TR):k])
        rest = rest[k + len(ET):]


def check(text_record):
    user, body = text_record[len("<|user|>\n"):].split("\n<|assistant|>\n", 1)
    body = body.replace("<|endoftext|>", "")
    for step_text, results in steps(body):
        c = T.CopyConstraint(user, [user] + results, numbers_in_reply=bool(results))
        for i, ch in enumerate(step_text):
            if not c.ok(step_text[:i], ch):
                return f"blocked {ch!r} after …{step_text[max(0, i - 40):i]!r}"
        if not step_text.endswith(TR) and not c.eos_ok(step_text):
            return f"end-of-reply blocked after …{step_text[-40:]!r}"
    return None


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/tool_use_v43_train.jsonl"
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    bad = collections.Counter(); examples = {}
    for r in rows:
        why = check(r["text"])
        if why:
            bad[r["family"]] += 1; examples.setdefault(r["family"], (why, r["text"][:300]))
    n = len(rows)
    print(f"{n - sum(bad.values())}/{n} training targets fully accepted by the constraint")
    for fam, k in bad.most_common():
        why, txt = examples[fam]
        print(f"  {fam}: {k} blocked — e.g. {why}")
    sys.exit(1 if bad else 0)