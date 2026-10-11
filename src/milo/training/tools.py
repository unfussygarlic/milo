"""Tool-use fine-tuning: teach the SFT checkpoint when and how to call tools.

Loss is masked so nothing inside a tool result is trained on
(:mod:`milo.training.masking`), batches are padded dynamically
(:mod:`milo.training.tool_dataset`), and hyperparameters come from the ``tools``
section of ``configs/training.yaml``.

Usage:
    milo-generate-data --output data/tool_use_train.jsonl
    milo-finetune-tools --checkpoint experiments/sft_20260929_075754/checkpoints/best.ptrom
"""

import argparse
import json
import math
import os
import sys
import time

import torch

from milo.config import training_config
from milo.model.slm import SLM
from milo.training.tool_dataset import ToolUseDataset

CONFIG = training_config("tools")


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
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to SFT checkpoint (best.pt)")
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

    print(f"\n{'=' * 60}")
    print("  Milo — Tool-Use Fine-Tuning")
    print(f"  Experiment: {exp_name}")
    print(f"  Base checkpoint: {args.checkpoint}")
    print(f"{'=' * 60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("\nLoading data...")
    train_data = ToolUseDataset(args.train_data, CONFIG["seq_len"])
    val_data = ToolUseDataset(args.val_data, CONFIG["seq_len"])

    # ── Build and load model ──
    model = SLM(
        CONFIG["vocab_size"],
        CONFIG["d_model"],
        CONFIG["n_q_heads"],
        CONFIG["n_kv_heads"],
        CONFIG["n_layers"],
        CONFIG["seq_len"],
        CONFIG["dropout"],
        CONFIG["share_blocks"],
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
        model.parameters(),
        lr=CONFIG["lr"],
        betas=(0.9, 0.95),
        weight_decay=CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")

    # Save config
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump({**CONFIG, "checkpoint": args.checkpoint, "total_steps": total_steps}, f, indent=2)

    # ── Training loop ──
    print(f"\n{'=' * 60}")
    print("  Training starts")
    print(f"  LR: {CONFIG['lr']} → {CONFIG['min_lr']}")
    print(
        f"  Batch: {CONFIG['batch_size']} × {CONFIG['grad_accum_steps']} = {CONFIG['batch_size'] * CONFIG['grad_accum_steps']}"
    )
    print(f"{'=' * 60}\n")

    best_val_loss = float("inf")
    step = 0
    t0 = time.time()
    loss_accum = 0.0

    for epoch in range(CONFIG["epochs"]):
        print(f"\n── Epoch {epoch + 1}/{CONFIG['epochs']} ──")

        for _ in range(steps_per_epoch):
            # Set LR
            lr = get_lr(step, total_steps, CONFIG["warmup_steps"], CONFIG["lr"], CONFIG["min_lr"])
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
                print(
                    f"  step {step:>5d}/{total_steps} ({pct:5.1f}%) | "
                    f"loss {avg_loss:.4f} | lr {lr:.2e} | "
                    f"{elapsed:.1f}s"
                )
                t0 = time.time()
                loss_accum = 0.0

            # ── Eval ──
            if step % CONFIG["eval_every"] == 0:
                val_loss = evaluate(model, val_data, device)
                improved = val_loss < best_val_loss
                print(f"  ► val loss: {val_loss:.4f}{' ★ new best!' if improved else f' (best: {best_val_loss:.4f})'}")

                if improved:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

                # Save latest (full state for resume)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(),
                        "step": step,
                        "epoch": epoch,
                        "best_val_loss": best_val_loss,
                        "config": CONFIG,
                    },
                    os.path.join(ckpt_dir, "latest.pt"),
                )

    # ── Final ──
    print(f"\n{'=' * 60}")
    print("  Training complete!")
    print(f"  Total steps: {step}")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints: {ckpt_dir}/")
    print(f"{'=' * 60}")

    # Final save
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))


if __name__ == "__main__":
    main()
