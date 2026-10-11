"""Every special token string Milo's prompt format uses.

Part of the training-data contract, and the narrowest module in the package: no
torch, no tiktoken, no imports at all, so the loss-mask code and the tests can
use it freely.

The one format the model was fine-tuned on::

    <|user|>\\n{msg}\\n<|assistant|>\\n<|tool_call|>TOOL(args)<|tool_result|>{result}<|end_tool|>\\n{answer}<|endoftext|>

Changing any of these strings invalidates the checkpoint, the decoding
constraint and every dataset under data/.
"""

USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"

TOOL_CALL_TAG = "<|tool_call|>"
TOOL_RESULT_TAG = "<|tool_result|>"
END_TOOL_TAG = "<|end_tool|>"
