"""The tool-use dataset: JSONL in, padded batches out, loss masked to the answer."""

import json

import numpy as np
import torch

from milo.model.tokenizer import enc, encode
from milo.training.masking import loss_mask


class ToolUseDataset:
    """
    Loads JSONL tool-use examples: {"text": "<|user|>\n...\n<|assistant|>\n...<|endoftext|>"}

    Loss masking (v4):
      - no loss on the prompt (everything up to and including the last "<|assistant|>\n")
      - no loss on tool-result content and <|end_tool|> — the runtime injects these,
        so training on them only teaches the model to HALLUCINATE tool results
      - loss stays on TOOL(args) and on the <|tool_result|> tag itself, because the
        runtime stops generation when it sees <|tool_result|>
    Masks are computed once at load time from token character offsets.
    """

    def __init__(self, path, seq_len):
        self.seq_len = seq_len
        self.examples = []  # list of (tokens, loss_mask)
        dropped_long = dropped_decode = 0
        trained = total = 0

        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line)["text"]
                tokens = encode(text)
                if len(tokens) > seq_len:
                    dropped_long += 1
                    continue
                decoded, offsets = enc.decode_with_offsets(tokens)
                if decoded != text:
                    dropped_decode += 1
                    continue
                mask = loss_mask(text, offsets)
                self.examples.append((tokens, mask))
                trained += sum(mask[1:])
                total += len(tokens) - 1

        print(f"  Loaded {len(self.examples)} examples from {path}")
        if dropped_long or dropped_decode:
            print(f"  Dropped: {dropped_long} over {seq_len} tokens, {dropped_decode} decode mismatches")
        lengths = [len(t) for t, _ in self.examples]
        print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths) / len(lengths):.0f}")
        print(f"  Loss on {trained / max(total, 1) * 100:.1f}% of tokens")

    def __len__(self):
        return len(self.examples)

    def get_batch(self, batch_size, device):
        """
        Returns (input_ids, labels); labels=-100 where no loss is taken.
        Dynamic padding: pad only to the longest example in THIS batch, not to
        seq_len. Examples average ~76 tokens, so padding to 511 wasted ~85% of
        compute. (The runtime already feeds variable-length inputs to the model.)
        """
        indices = np.random.randint(0, len(self.examples), size=batch_size)
        max_len = max(len(self.examples[i][0]) for i in indices) - 1
        batch_x, batch_y = [], []
        for idx in indices:
            tokens, mask = self.examples[idx]
            x = tokens[:-1]
            y = [t if m else -100 for t, m in zip(tokens[1:], mask[1:])]
            pad = max_len - len(x)
            batch_x.append(x + [0] * pad)
            batch_y.append(y + [-100] * pad)
        x = torch.tensor(batch_x, dtype=torch.long, device=device)
        y = torch.tensor(batch_y, dtype=torch.long, device=device)
        return x, y
