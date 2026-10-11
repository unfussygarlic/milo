"""Which tokens the tool-use loss is computed on.

Part of the training-data contract: the model is trained to produce the tool
CALL and the final answer, and never to produce the tool RESULT (that comes from
the runtime). Getting this wrong teaches Milo to hallucinate prices.

Deliberately free of torch and tiktoken so the unit tests can exercise it
directly.
"""

import re

from milo.special_tokens import ASST_TOKEN, END_TOOL_TAG, TOOL_RESULT_TAG

ASST_MARKER = ASST_TOKEN + "\n"
TOOL_RESULT_SPAN = re.compile(re.escape(TOOL_RESULT_TAG) + r"(.*?)" + re.escape(END_TOOL_TAG), re.DOTALL)


def loss_mask(text: str, offsets: list[int]) -> list[int]:
    """
    offsets[i] = char index where token i starts (from enc.decode_with_offsets).
    Returns a list of bools: True = this token is a training target.
    """
    a = text.rfind(ASST_MARKER)
    train_start = a + len(ASST_MARKER) if a >= 0 else 0
    no_loss = [(m.start(1), m.end()) for m in TOOL_RESULT_SPAN.finditer(text)]
    return [off >= train_start and not any(s <= off < e for s, e in no_loss) for off in offsets]
