"""GPT-2 tiktoken encoding plus Milo's special tokens.

Part of the training-data contract: the tokenizer, the special-token strings and
the way they are encoded (``allowed_special``) must match what the model was
trained on, or the prompt it sees at inference is not the prompt it learned.
"""

import tiktoken

from milo.special_tokens import (
    ASST_TOKEN,
    END_TOOL_TAG,
    EOS_TOKEN,
    TOOL_CALL_TAG,
    TOOL_RESULT_TAG,
    USER_TOKEN,
)

enc = tiktoken.get_encoding("gpt2")
EOS_ID = enc.encode(EOS_TOKEN, allowed_special={EOS_TOKEN})[0]


def encode(text: str) -> list[int]:
    return enc.encode(text, allowed_special={
        USER_TOKEN, ASST_TOKEN, EOS_TOKEN,
        TOOL_CALL_TAG, TOOL_RESULT_TAG, END_TOOL_TAG,
    })


def decode(tokens: list[int]) -> str:
    return enc.decode(tokens)


_VOCAB = None


def vocab_strings() -> list[str]:
    """Decoded string of every token id (partial UTF-8 bytes become '\ufffd')."""
    global _VOCAB
    if _VOCAB is None:
        _VOCAB = [enc.decode_single_token_bytes(i).decode("utf-8", errors="replace") for i in range(enc.n_vocab)]
    return _VOCAB


def strip_tags(text: str) -> str:
    for tag in (EOS_TOKEN, TOOL_CALL_TAG, TOOL_RESULT_TAG, END_TOOL_TAG,
                USER_TOKEN, ASST_TOKEN):
        text = text.replace(tag, "")
    return text.strip()
