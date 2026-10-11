"""
Milo — Tool-Use Fine-Tuning

Fine-tunes the SFT checkpoint to learn when and how to call tools.
Uses the same patterns as finetune.py (assistant-only loss masking,
same checkpoint format, same Experiment logging).

Usage:
    python generate_tool_data.py --output data/tool_use_train.jsonl
    python finetune_tools.py --checkpoint experiments/sft_20260929_075754/checkpoints/best.pt

Tool call format in training data:
    <|tool_call|>TOOL_NAME(args)<|tool_result|>result<|end_tool|>
"""

import argparse
import glob
import json
import math
import os
import re
import sys
import time

import numpy as np
import torch
import tiktoken

from model import SLM

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
CONFIG = dict(
    # Model (must match)
    vocab_size=50257,
    d_model=1280,
    n_q_heads=20,
    n_kv_heads=4,
    n_layers=16,
    seq_len=512,
    share_blocks=True,
    dropout=0.1,

    # Training
    batch_size=2,
    grad_accum_steps=16,   # effective batch = 32 (same, fits 8GB VRAM)
    lr=2e-5,               # same as SFT — low since we're fine-tuning
    min_lr=2e-6,
    warmup_steps=100,
    weight_decay=0.1,
    max_grad_norm=1.0,
    epochs=3,

    # Checkpointing
    eval_every=200,
    log_every=10,

    # Data
    train_data="data/tool_use_train.jsonl",
    val_data="data/tool_use_val.jsonl",
)

# ── Tokenizer ──
enc = tiktoken.get_encoding("gpt2")

USER_TOKEN = "<|user|>"
ASST_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"


def encode(text):
    return enc.encode(text, allowed_special={USER_TOKEN, ASST_TOKEN, EOS_TOKEN,
                                              "<|tool_call|>", "<|tool_result|>", "<|end_tool|>"})


ASST_MARKER = ASST_TOKEN + "\n"
TOOL_RESULT_SPAN = re.compile(
    re.escape("<|tool_result|>") + r"(.*?)" + re.escape("<|end_tool|>"), re.DOTALL)


def loss_mask(text, offsets):
    """
    offsets[i] = char index where token i starts (from enc.decode_with_offsets).
    Returns a list of bools: True = this token is a training target.
    """
    a = text.rfind(ASST_MARKER)
    train_start = a + len(ASST_MARKER) if a >= 0 else 0
    no_loss = [(m.start(1), m.end()) for m in TOOL_RESULT_SPAN.finditer(text)]
    return [off >= train_start and not any(s <= off < e for s, e in no_loss)
            for off in offsets]


# ═══════════════════════════════════════════════════════════════
# LOGGING (same pattern as original train.py)
# ═══════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a", buffering=1)

    def write(self, msg):
        self.terminal.write(msg)
        self.log.write(msg)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ═══════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════
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
        self.examples = []   # list of (tokens, loss_mask)
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
        print(f"  Token lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.0f}")
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

# ═══════════════════════════════════════════════════════════════
# LR SCHEDULE
# ═══════════════════════════════════════════════════════════════
def get_lr(step, total_steps, warmup, lr, min_lr):
    if step < warmup:
        return lr * step / warmup
    if step >= total_steps:
        return min_lr
    progress = (step - warmup) / (total_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * progress))


# ═══════════════════════════════════════════════════════════════
# EVALUATION
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def evaluate(model, val_data, device, n_batches=50):
    model.eval()
    total_loss = 0.0
    n = 0
    for _ in range(n_batches):
        x, y = val_data.get_batch(CONFIG["batch_size"], device)
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        if loss is not None:
            total_loss += loss.item()
            n += 1
    model.train()
    return total_loss / max(n, 1)


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Milo — Tool-Use Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to SFT checkpoint (best.pt)")
    parser.add_argument("--train_data", type=str, default=CONFIG["train_data"])
    parser.add_argument("--val_data", type=str, default=CONFIG["val_data"])
    parser.add_argument("--run_name", type=str, default=None)
    args = parser.parse_args()

    # ── Experiment directory ──
    if args.run_name:
        exp_name = args.run_name
    else:
        exp_name = f"tools_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = os.path.join("experiments", exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    logger = Logger(os.path.join(exp_dir, "train.log"))
    sys.stdout = logger
    sys.stderr = logger

    print(f"\n{'='*60}")
    print(f"  Milo — Tool-Use Fine-Tuning")
    print(f"  Experiment: {exp_name}")
    print(f"  Base checkpoint: {args.checkpoint}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("\nLoading data...")
    train_data = ToolUseDataset(args.train_data, CONFIG["seq_len"])
    val_data = ToolUseDataset(args.val_data, CONFIG["seq_len"])

    # ── Build and load model ──
    model = SLM(
        CONFIG["vocab_size"], CONFIG["d_model"], CONFIG["n_q_heads"],
        CONFIG["n_kv_heads"], CONFIG["n_layers"], CONFIG["seq_len"],
        CONFIG["dropout"], CONFIG["share_blocks"],
    )

    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    del ckpt

    model.to(device)
    model.train()
    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")

    # ── Training setup ──
    steps_per_epoch = len(train_data) // (CONFIG["batch_size"] * CONFIG["grad_accum_steps"])
    total_steps = steps_per_epoch * CONFIG["epochs"]
    print(f"\nSteps per epoch: {steps_per_epoch}")
    print(f"Total steps: {total_steps} ({CONFIG['epochs']} epochs)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"],
        betas=(0.9, 0.95), weight_decay=CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({**CONFIG, "checkpoint": args.checkpoint,
                    "total_steps": total_steps}, f, indent=2)

    # ── Training loop ──
    print(f"\n{'='*60}")
    print(f"  Training starts")
    print(f"  LR: {CONFIG['lr']} → {CONFIG['min_lr']}")
    print(f"  Batch: {CONFIG['batch_size']} × {CONFIG['grad_accum_steps']} = {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}")
    print(f"{'='*60}\n")

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    loss_accum = 0.0

    for epoch in range(CONFIG["epochs"]):
        print(f"\n── Epoch {epoch + 1}/{CONFIG['epochs']} ──")

        for _ in range(steps_per_epoch):
            # Set LR
            lr = get_lr(step, total_steps, CONFIG["warmup_steps"],
                        CONFIG["lr"], CONFIG["min_lr"])
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation
            optimizer.zero_grad()
            step_loss = 0.0

            for _ in range(CONFIG["grad_accum_steps"]):
                x, y = train_data.get_batch(CONFIG["batch_size"], device)
                with torch.amp.autocast("cuda"):
                    _, loss = model(x, y)
                    loss = loss / CONFIG["grad_accum_steps"]
                scaler.scale(loss).backward()
                step_loss += loss.item()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])
            scaler.step(optimizer)
            scaler.update()

            loss_accum += step_loss
            step += 1

            # ── Log ──
            if step % CONFIG["log_every"] == 0:
                elapsed = time.time() - t0
                avg_loss = loss_accum / CONFIG["log_every"]
                pct = step / total_steps * 100
                print(f"  step {step:>5d}/{total_steps} ({pct:5.1f}%) | "
                      f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                      f"{elapsed:.1f}s")
                t0 = time.time()
                loss_accum = 0.0

            # ── Eval ──
            if step % CONFIG["eval_every"] == 0:
                val_loss = evaluate(model, val_data, device)
                improved = val_loss < best_val_loss
                print(f"  ► val loss: {val_loss:.4f}"
                      f"{' ★ new best!' if improved else f' (best: {best_val_loss:.4f})'}")

                if improved:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(),
                               os.path.join(ckpt_dir, "best.pt"))

                # Save latest (full state for resume)
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "step": step,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "config": CONFIG,
                }, os.path.join(ckpt_dir, "latest.pt"))

    # ── Final ──
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total steps: {step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {ckpt_dir}/")
    print(f"{'='*60}")

    # Final save
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))


if __name__ == "__main__":
    main()