"""Which tokens the tool-use loss is computed on.

The model must learn to produce the tool CALL and the final answer, and never the
tool RESULT - that comes from the runtime. Train on the result and you teach Milo
to make prices up.
"""

import tiktoken

from milo.training.masking import loss_mask

TEXT = (
    "<|user|>\n₹1499 pe 18% GST?\n<|assistant|>\n"
    "<|tool_call|>CALCULATE(1499 * 18 / 100)<|tool_result|>269.82<|end_tool|>\n"
    "GST: ₹269.82<|endoftext|>"
)


def _offsets(text):
    enc = tiktoken.get_encoding("gpt2")
    return enc.decode_with_offsets(enc.encode(text, allowed_special="all"))[1]


def test_loss_mask_spans():
    offs = _offsets(TEXT)
    m = loss_mask(TEXT, offs)
    res_start = TEXT.index("269.82")
    res_end = TEXT.index("<|end_tool|>") + len("<|end_tool|>")

    def trained(lo, hi):
        return any(k for o, k in zip(offs, m) if lo <= o < hi)

    assert not trained(0, TEXT.index("<|assistant|>\n") + 14), "no loss on the prompt"
    assert not trained(res_start, res_end), "no loss inside the tool result"
    assert trained(TEXT.index("CALCULATE"), TEXT.index("<|tool_result|>")), "loss on the call args"
    assert trained(TEXT.index("<|tool_result|>"), res_start), "loss on the result tag"
    assert trained(res_end, len(TEXT)), "loss on the answer"
