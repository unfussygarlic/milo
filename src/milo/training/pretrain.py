import argparse
import json
import math
import os
import sys
import time
from datetime import datetime

import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

from milo.config import training_config
from milo.model.slm import SLM


# --- Experiment tracking ---
class Experiment:
    """Manages experiment directory, logging, and artifacts."""

    def __init__(self, base_dir="experiments"):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(base_dir, self.timestamp)
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(os.path.join(self.dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "samples"), exist_ok=True)

        # Logger
        self.log_path = os.path.join(self.dir, "train.log")
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        self._setup_logger()

    def _setup_logger(self):
        class Logger:
            def __init__(self, filepath):
                self.terminal = sys.__stdout__
                self.log = open(filepath, "a")

            def write(self, msg):
                self.terminal.write(msg)
                self.log.write(msg)
                self.log.flush()

            def flush(self):
                self.terminal.flush()
                self.log.flush()

        sys.stdout = Logger(self.log_path)
        sys.stderr = Logger(self.log_path)

    def save_config(self, config):
        path = os.path.join(self.dir, "config.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)

    def log_metrics(self, step, **kwargs):
        """Append one metrics row to JSONL file."""
        row = {"step": step, "timestamp": time.time(), **kwargs}
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_checkpoint(self, model, optimizer, scaler, config, step, val_loss, best=False):
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
            "timestamp": self.timestamp,
        }
        # Always save latest (for resume)
        path = os.path.join(self.dir, "checkpoints", "latest.pt")
        torch.save(state, path)

        # Periodic checkpoint
        path = os.path.join(self.dir, "checkpoints", f"step_{step}.pt")
        torch.save(state, path)

        # Best model (lighter — no optimizer)
        if best:
            best_state = {
                "model": model.state_dict(),
                "config": config,
                "step": step,
                "val_loss": val_loss,
            }
            path = os.path.join(self.dir, "checkpoints", "best.pt")
            torch.save(best_state, path)

        # Cleanup: keep only last 3 periodic checkpoints + best + latest
        self._cleanup_checkpoints()

    def _cleanup_checkpoints(self):
        ckpt_dir = os.path.join(self.dir, "checkpoints")
        step_files = sorted(
            [f for f in os.listdir(ckpt_dir) if f.startswith("step_")], key=lambda f: int(f.split("_")[1].split(".")[0])
        )
        # Keep last 3
        for f in step_files[:-3]:
            os.remove(os.path.join(ckpt_dir, f))

    def save_samples(self, step, samples):
        """Save generation samples to a text file."""
        path = os.path.join(self.dir, "samples", f"step_{step}.txt")
        with open(path, "w") as f:
            for prompt, output in samples:
                f.write(f"PROMPT: {prompt}\n")
                f.write(f"OUTPUT: {output}\n")
                f.write("-" * 60 + "\n")

    def save_summary(self, config, total_steps, best_val_loss, total_time):
        """Save final experiment summary."""
        summary = {
            "experiment_id": self.timestamp,
            "config": config,
            "total_steps": total_steps,
            "best_val_loss": best_val_loss,
            "total_time_hours": total_time / 3600,
            "total_tokens_processed": total_steps
            * config["batch_size"]
            * config["grad_accum_steps"]
            * config["seq_len"],
        }
        path = os.path.join(self.dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# --- Config ---
# Hyperparameters live in the [pretrain] section of configs/training.yaml.
CONFIG = training_config("pretrain")

enc = tiktoken.get_encoding("gpt2")


def encode(s):
    return enc.encode(s, allowed_special={"<|endoftext|>"})


def decode(tokens):
    return enc.decode(tokens)


# --- Data loading (memory-mapped for large datasets) ---
class CurriculumDataset:
    def __init__(self, data_dir, seq_len, device):
        self.seq_len = seq_len
        self.device = device
        self.stages = {}

        # Try curriculum .npy stages first
        curriculum_dir = os.path.join(data_dir, "curriculum")
        if os.path.exists(curriculum_dir):
            npy_files = sorted([f for f in os.listdir(curriculum_dir) if f.endswith(".npy")])
            if npy_files:
                for f in npy_files:
                    path = os.path.join(curriculum_dir, f)
                    mmap = np.load(path, mmap_mode="r")
                    name = f.replace(".npy", "")
                    self.stages[name] = mmap
                    print(f"  {name}: {len(mmap):,} tokens (mmap)")

        # Fallback: raw chunk files
        if not self.stages:
            all_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".npy")])
            if all_files:
                cosmo = sorted([f for f in all_files if f.startswith("cosmo")])
                fineweb = sorted([f for f in all_files if f.startswith("fineweb")])
                n_c, n_f = len(cosmo), len(fineweb)

                stage_map = {
                    "stage_1": cosmo[: n_c // 5],
                    "stage_2": cosmo[n_c // 5 : n_c * 3 // 5] + fineweb[: n_f // 2],
                    "stage_3": cosmo[n_c * 3 // 5 :] + fineweb[n_f // 2 :],
                }

                for stage_name, files in stage_map.items():
                    mmaps = [np.load(os.path.join(data_dir, f), mmap_mode="r") for f in files]
                    sizes = [len(m) for m in mmaps]
                    cumsum = np.cumsum([0] + sizes)
                    self.stages[stage_name] = (mmaps, sizes, cumsum, sum(sizes))
                    print(f"  {stage_name}: {sum(sizes):,} tokens ({len(files)} files, mmap)")

        # Fallback: JSON
        if not self.stages:
            json_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".json") and "stage" in f])
            for f in json_files:
                with open(os.path.join(data_dir, f)) as fh:
                    examples = json.load(fh)
                tokens = []
                for ex in examples:
                    tokens.extend(encode(ex["text"]))
                name = f.replace(".json", "")
                self.stages[name] = np.array(tokens, dtype=np.uint16)
                print(f"  {name}: {len(tokens):,} tokens (json)")

        if not self.stages:
            raise RuntimeError(f"No data found in {data_dir}")

        # Build validation set
        self._build_val()

    def _build_val(self):
        last = list(self.stages.values())[-1]
        if isinstance(last, tuple):
            last_mmap = last[0][-1]
            n_val = min(len(last_mmap), 200_000)
            val = np.array(last_mmap[-n_val:], dtype=np.int64)
        else:
            n_val = min(len(last), 200_000)
            val = np.array(last[-n_val:], dtype=np.int64)
        self.val_data = torch.from_numpy(val).long().to(self.device)
        print(f"  validation: {len(self.val_data):,} tokens")

    def get_batch(self, stage_name, batch_size):
        stage = self.stages[stage_name]

        if isinstance(stage, tuple):
            # Multi-file stage
            mmaps, sizes, cumsum, total = stage
            positions = np.random.randint(0, total - self.seq_len - 1, size=batch_size)
            chunks = []
            for pos in positions:
                # Binary search for which file
                file_idx = np.searchsorted(cumsum[1:], pos, side="right")
                file_idx = min(file_idx, len(mmaps) - 1)
                local_pos = pos - cumsum[file_idx]
                local_pos = min(local_pos, len(mmaps[file_idx]) - self.seq_len - 1)
                local_pos = max(local_pos, 0)
                chunk = np.array(mmaps[file_idx][local_pos : local_pos + self.seq_len + 1], dtype=np.int64)
                if len(chunk) < self.seq_len + 1:
                    pad = np.zeros(self.seq_len + 1 - len(chunk), dtype=np.int64)
                    chunk = np.concatenate([chunk, pad])
                chunks.append(chunk)
            batch = np.stack(chunks)
        else:
            # Single array
            ix = np.random.randint(0, len(stage) - self.seq_len - 1, size=batch_size)
            batch = np.stack([np.array(stage[i : i + self.seq_len + 1], dtype=np.int64) for i in ix])

        batch = torch.from_numpy(batch).long().to(self.device)
        return batch[:, :-1], batch[:, 1:]

    def get_val_batch(self, batch_size):
        ix = torch.randint(len(self.val_data) - self.seq_len - 1, (batch_size,))
        x = torch.stack([self.val_data[i : i + self.seq_len] for i in ix])
        y = torch.stack([self.val_data[i + 1 : i + self.seq_len + 1] for i in ix])
        return x, y


# --- LR Schedule ---
def get_lr(step, total_steps):
    if step < CONFIG["warmup_steps"]:
        return CONFIG["lr"] * (step + 1) / CONFIG["warmup_steps"]
    progress = (step - CONFIG["warmup_steps"]) / (total_steps - CONFIG["warmup_steps"])
    return CONFIG["min_lr"] + 0.5 * (CONFIG["lr"] - CONFIG["min_lr"]) * (1 + math.cos(math.pi * progress))


# --- Generation ---
def generate(model, prompt, max_tokens=150, temperature=0.7):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -CONFIG["seq_len"] :]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            if next_token.item() == encode("<|endoftext|>")[0]:
                break
    result = decode(idx[0].tolist())
    if "<|endoftext|>" in result:
        result = result[: result.index("<|endoftext|>")]
    return result


# --- Evaluation ---
@torch.no_grad()
def evaluate(model, dataset, n_batches=20):
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = dataset.get_val_batch(CONFIG["batch_size"])
        with torch.amp.autocast("cuda"):
            _, loss = model(x, y)
        losses.append(loss.item())
    return sum(losses) / len(losses)


# --- Main ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data/smollm")
    parser.add_argument("--exp_dir", type=str, default="experiments")
    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Device: {device}\n")

    # Experiment
    exp = Experiment(base_dir=args.exp_dir)
    exp.save_config(CONFIG)
    print(f"Experiment: {exp.dir}\n")

    # Model
    model = SLM(
        CONFIG["vocab_size"],
        CONFIG["d_model"],
        CONFIG["n_q_heads"],
        CONFIG["n_kv_heads"],
        CONFIG["n_layers"],
        CONFIG["seq_len"],
        CONFIG["dropout"],
        CONFIG["share_blocks"],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params ({total_params / 1e6:.1f}M)")
    print(
        f"  d={CONFIG['d_model']}, layers={CONFIG['n_layers']}, "
        f"q_heads={CONFIG['n_q_heads']}, kv_heads={CONFIG['n_kv_heads']}, "
        f"shared={CONFIG['share_blocks']}"
    )

    # Data
    print(f"\nLoading data from {args.data_dir}:")
    dataset = CurriculumDataset(args.data_dir, CONFIG["seq_len"], device)

    # Curriculum
    total_steps = CONFIG["stage_1_steps"] + CONFIG["stage_2_steps"] + CONFIG["stage_3_steps"]
    stage_names = sorted(dataset.stages.keys())
    stage_boundaries = [
        CONFIG["stage_1_steps"],
        CONFIG["stage_1_steps"] + CONFIG["stage_2_steps"],
        total_steps,
    ]

    def get_stage(step):
        for i, boundary in enumerate(stage_boundaries):
            if step < boundary:
                return stage_names[min(i, len(stage_names) - 1)]
        return stage_names[-1]

    effective_batch = CONFIG["batch_size"] * CONFIG["grad_accum_steps"]
    tokens_per_step = effective_batch * CONFIG["seq_len"]

    print("\nCurriculum:")
    for i, name in enumerate(stage_names):
        s = stage_boundaries[i - 1] if i > 0 else 0
        e = stage_boundaries[i] if i < len(stage_boundaries) else total_steps
        print(f"  Steps {s:,}-{e:,}: {name}")
    print(f"  Total: {total_steps:,} steps")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Tokens/step: {tokens_per_step:,}")
    print(f"  Total tokens: {tokens_per_step * total_steps:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["lr"],
        weight_decay=CONFIG["weight_decay"],
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    # Resume
    start_step = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("val_loss", float("inf"))
        print(f"  Resumed at step {start_step}")

    # Sample prompts
    sample_prompts = [
        "The most important thing about",
        "Gravity is a fundamental force that",
        "<|user|>\nWhat causes earthquakes?\n<|assistant|>\n",
        "<|user|>\nExplain how the internet works.\n<|assistant|>\n",
        "<|user|>\nWhat is photosynthesis?\n<|assistant|>\n",
    ]

    # Training
    print(f"\n{'=' * 60}")
    print("TRAINING START")
    print(f"{'=' * 60}\n")

    model.train()
    t0 = time.time()
    train_start = time.time()
    step = start_step

    while step < total_steps:
        stage = get_stage(step)
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(CONFIG["grad_accum_steps"]):
            x, y = dataset.get_batch(stage, CONFIG["batch_size"])
            with torch.amp.autocast("cuda"):
                _, loss = model(x, y)
                loss = loss / CONFIG["grad_accum_steps"]
            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["max_grad_norm"])

        lr = get_lr(step, total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        # --- Logging ---
        if step % CONFIG["log_interval"] == 0:
            elapsed = time.time() - t0
            tok_s = tokens_per_step * CONFIG["log_interval"] / max(elapsed, 1e-6) if step > start_step else 0
            total_elapsed = time.time() - train_start
            steps_done = step - start_step
            eta = (total_steps - step) / (steps_done / total_elapsed) if steps_done > 0 else 0

            print(
                f"Step {step:5d}/{total_steps} | {stage} | "
                f"Loss: {accum_loss:.4f} | LR: {lr:.2e} | "
                f"Tok/s: {tok_s:,.0f} | "
                f"ETA: {eta / 3600:.1f}h"
            )

            exp.log_metrics(step, train_loss=accum_loss, lr=lr, tok_s=tok_s, stage=stage)
            t0 = time.time()

        # --- Evaluation ---
        if step > 0 and step % CONFIG["eval_interval"] == 0:
            val_loss = evaluate(model, dataset)
            is_best = val_loss < best_val_loss
            print(f"\n  >>> Val loss: {val_loss:.4f} {'(new best!)' if is_best else ''}")
            exp.log_metrics(step, val_loss=val_loss, is_best=is_best)

            if is_best:
                best_val_loss = val_loss
                exp.save_checkpoint(model, optimizer, scaler, CONFIG, step, val_loss, best=True)

            model.train()

        # --- Samples ---
        if step > 0 and step % CONFIG["sample_interval"] == 0:
            print(f"\n  --- Samples at step {step} ---")
            samples = []
            for prompt in sample_prompts:
                output = generate(model, prompt, max_tokens=100)
                display = output.split("<|assistant|>")[-1].strip() if "<|assistant|>" in output else output
                short_prompt = prompt[:40].replace("\n", " ")
                print(f"  '{short_prompt}...'")
                print(f"    {display[:200]}\n")
                samples.append((prompt, output))

            exp.save_samples(step, samples)
            model.train()

        # --- Checkpoint ---
        if step > 0 and step % CONFIG["checkpoint_interval"] == 0:
            exp.save_checkpoint(model, optimizer, scaler, CONFIG, step, best_val_loss, best=False)
            print(f"  >>> Checkpoint saved at step {step}")

        step += 1

    # --- Final ---
    total_time = time.time() - train_start
    exp.save_checkpoint(model, optimizer, scaler, CONFIG, step, best_val_loss, best=False)
    exp.save_summary(CONFIG, step, best_val_loss, total_time)

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {total_time / 3600:.1f} hours")
    print(f"  Experiment: {exp.dir}")
    print(f"{'=' * 60}")

    # Final samples
    print("\n=== Final Generations ===")
    for prompt in sample_prompts:
        output = generate(model, prompt, max_tokens=150)
        display = output.split("<|assistant|>")[-1].strip() if "<|assistant|>" in output else output
        print(f"\nPrompt: '{prompt.strip()}'")
        print(f"Output: {display[:300]}")
    print()


if __name__ == "__main__":
    train()
