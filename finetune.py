"""
Supervised Fine-Tuning (SFT) for Milo v2 (202M)

Loads the pretrained checkpoint and fine-tunes on instruction data
with assistant-only loss masking — the model only learns to generate
responses, not to predict the user's question.

Usage:
    cd ~/milo
    python finetune.py \
        --checkpoint experiments/20260926_220527/checkpoints/best.pt \
        --train_data data/sft/train.jsonl \
        --val_data data/sft/val.jsonl

    # Resume interrupted run:
    python finetune.py --resume experiments/sft_YYYYMMDD_HHMMSS/checkpoints/latest.pt
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import tiktoken
import json
import math
import time
import os
import sys
import random
import argparse
import glob
from datetime import datetime

from model import SLM


# --- Experiment tracking (same as train.py) ---
class Experiment:
    """Manages experiment directory, logging, and artifacts."""
    def __init__(self, base_dir="experiments", resume_dir=None):
        if resume_dir:
            self.dir = resume_dir
            self.timestamp = os.path.basename(resume_dir)
        else:
            self.timestamp = "sft_" + datetime.now().strftime("%Y%m%d_%H%M%S")
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
                self.log = open(filepath, 'a')
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
            [f for f in os.listdir(ckpt_dir) if f.startswith("step_")],
            key=lambda f: int(f.split("_")[1].split(".")[0])
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
            "total_tokens_processed": total_steps * config['batch_size'] * config['grad_accum_steps'] * config['seq_len'],
        }
        path = os.path.join(self.dir, "summary.json")
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)


# --- Config ---
CONFIG = {
    # Model (must match pretraining)
    'vocab_size': 50257,
    'd_model': 1280,
    'n_q_heads': 20,
    'n_kv_heads': 4,
    'n_layers': 16,
    'seq_len': 512,
    'dropout': 0.1,
    'share_blocks': True,

    # SFT training
    'batch_size': 8,
    'grad_accum_steps': 4,       # effective batch = 32
    'epochs': 3,
    'lr': 2e-5,
    'min_lr': 2e-6,
    'warmup_steps': 200,
    'weight_decay': 0.05,
    'max_grad_norm': 1.0,

    # Logging
    'log_interval': 50,
    'eval_interval': 500,
    'sample_interval': 500,
    'checkpoint_interval': 1000,
}

enc = tiktoken.get_encoding("gpt2")
def encode(s): return enc.encode(s, allowed_special={'<|endoftext|>'})
def decode(tokens): return enc.decode(tokens)


# --- SFT Dataset ---
class SFTDataset:
    """
    Loads instruction data from JSONL and tokenizes with assistant-only masking.

    Format:  <|user|>\n{question}\n<|assistant|>\n{answer}<|endoftext|>
    Labels:  [-100, -100, ..., -100, answer_tok_1, answer_tok_2, ..., eot]
              ^^^^^^^^^^^^^^^^^^^^^^^^
              prefix (masked — no loss)
    """

    def __init__(self, jsonl_path, seq_len, shuffle=True):
        self.seq_len = seq_len
        self.eot = enc.eot_token  # 50256

        print(f"  Loading {jsonl_path}...")
        raw = []
        with open(jsonl_path) as f:
            for line in f:
                raw.append(json.loads(line))

        if shuffle:
            random.shuffle(raw)

        # Tokenize all examples upfront
        self.examples = []
        skipped = 0
        for item in raw:
            result = self._tokenize(item['messages'])
            if result is not None:
                self.examples.append(result)
            else:
                skipped += 1

        print(f"  {len(self.examples):,} examples loaded ({skipped} skipped)")

    def _tokenize(self, messages):
        user_msg = messages[0]['content'].strip()
        assistant_msg = messages[1]['content'].strip()

        prefix = f"<|user|>\n{user_msg}\n<|assistant|>\n"
        response = assistant_msg
        full_text = prefix + response

        # Tokenize separately to find the split point
        prefix_tokens = encode(prefix)
        full_tokens = encode(full_text)
        full_tokens.append(self.eot)  # add <|endoftext|>

        # Must fit in seq_len + 1 (for input/target shift)
        if len(full_tokens) > self.seq_len + 1:
            full_tokens = full_tokens[:self.seq_len + 1]

        if len(full_tokens) < 10:
            return None

        # Split into input and target (shifted by 1)
        input_ids = full_tokens[:-1]
        labels = full_tokens[1:]

        # Mask prefix in labels
        prefix_len = len(prefix_tokens)
        mask_len = min(prefix_len - 1, len(labels))
        for i in range(mask_len):
            labels[i] = -100

        # Pad to seq_len
        pad_len = self.seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.eot] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# --- LR Schedule ---
def get_lr(step, total_steps):
    if step < CONFIG['warmup_steps']:
        return CONFIG['lr'] * (step + 1) / CONFIG['warmup_steps']
    progress = (step - CONFIG['warmup_steps']) / (total_steps - CONFIG['warmup_steps'])
    return CONFIG['min_lr'] + 0.5 * (CONFIG['lr'] - CONFIG['min_lr']) * (1 + math.cos(math.pi * progress))


# --- Generation ---
def generate(model, prompt, max_tokens=150, temperature=0.7):
    model.eval()
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=next(model.parameters()).device)
    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -CONFIG['seq_len']:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
            if next_token.item() == encode("<|endoftext|>")[0]:
                break
    result = decode(idx[0].tolist())
    if "<|endoftext|>" in result:
        result = result[:result.index("<|endoftext|>")]
    return result


# --- Evaluation ---
@torch.no_grad()
def evaluate(model, val_dataset, device, n_batches=50):
    """Compute average val loss (assistant tokens only, via ignore_index=-100)."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    indices = list(range(len(val_dataset)))
    random.shuffle(indices)

    for i in range(0, min(len(indices), n_batches * CONFIG['batch_size']), CONFIG['batch_size']):
        batch_idx = indices[i:i + CONFIG['batch_size']]
        if len(batch_idx) < CONFIG['batch_size']:
            break

        input_ids = torch.stack([val_dataset[j][0] for j in batch_idx]).to(device)
        labels = torch.stack([val_dataset[j][1] for j in batch_idx]).to(device)

        with torch.amp.autocast('cuda'):
            logits, _ = model(input_ids)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        n_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * n_tokens
        total_tokens += n_tokens

    return total_loss / max(total_tokens, 1)


# --- Sample prompts ---
SAMPLE_PROMPTS = [
    "<|user|>\nWhat causes earthquakes?\n<|assistant|>\n",
    "<|user|>\nA store sells apples for $3 per kg. If I buy 4.5 kg, how much do I pay?\n<|assistant|>\n",
    "<|user|>\nExplain how the internet works in simple terms.\n<|assistant|>\n",
    "<|user|>\nWrite a Python function to check if a number is prime.\n<|assistant|>\n",
    "<|user|>\nWhat is photosynthesis?\n<|assistant|>\n",
    "<|user|>\nWhy do seasons change throughout the year?\n<|assistant|>\n",
]


# --- Main ---
def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to pretrained best.pt')
    parser.add_argument('--train_data', type=str, default='data/sft/train.jsonl')
    parser.add_argument('--val_data', type=str, default='data/sft/val.jsonl')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to latest.pt to resume from')
    parser.add_argument('--exp_dir', type=str, default='experiments')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Device: {device}\n")

    # Auto-find checkpoint if not specified
    if not args.checkpoint and not args.resume:
        exp_dirs = sorted(glob.glob(os.path.join(args.exp_dir, '202*')))
        for d in reversed(exp_dirs):
            best = os.path.join(d, 'checkpoints', 'best.pt')
            if os.path.exists(best) and 'sft' not in d:
                args.checkpoint = best
                break
        if not args.checkpoint:
            print("No checkpoint found. Specify --checkpoint path/to/best.pt")
            sys.exit(1)
        print(f"Auto-detected checkpoint: {args.checkpoint}")

    if not args.resume and not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Experiment
    resume_exp_dir = None
    if args.resume:
        # resume path is like experiments/sft_.../checkpoints/latest.pt
        resume_exp_dir = os.path.dirname(os.path.dirname(args.resume))
    exp = Experiment(base_dir=args.exp_dir, resume_dir=resume_exp_dir)
    exp.save_config(CONFIG)
    print(f"Experiment: {exp.dir}\n")

    # Model
    model = SLM(
        CONFIG['vocab_size'], CONFIG['d_model'], CONFIG['n_q_heads'],
        CONFIG['n_kv_heads'], CONFIG['n_layers'], CONFIG['seq_len'],
        CONFIG['dropout'], CONFIG['share_blocks'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params:,} params ({total_params/1e6:.1f}M)")
    print(f"  d={CONFIG['d_model']}, layers={CONFIG['n_layers']}, "
          f"q_heads={CONFIG['n_q_heads']}, kv_heads={CONFIG['n_kv_heads']}, "
          f"shared={CONFIG['share_blocks']}")

    # Load pretrained weights
    if not args.resume:
        print(f"\nLoading pretrained checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model.load_state_dict(state_dict)
        print(f"  Loaded (pretrained val loss: {ckpt.get('val_loss', '?')})")

    # Data
    print(f"\nLoading SFT data:")
    train_dataset = SFTDataset(args.train_data, CONFIG['seq_len'], shuffle=True)
    val_dataset = SFTDataset(args.val_data, CONFIG['seq_len'], shuffle=False)

    steps_per_epoch = len(train_dataset) // (CONFIG['batch_size'] * CONFIG['grad_accum_steps'])
    total_steps = steps_per_epoch * CONFIG['epochs']
    effective_batch = CONFIG['batch_size'] * CONFIG['grad_accum_steps']
    tokens_per_step = effective_batch * CONFIG['seq_len']

    print(f"\nSFT Training Plan:")
    print(f"  Train examples : {len(train_dataset):,}")
    print(f"  Val examples   : {len(val_dataset):,}")
    print(f"  Epochs         : {CONFIG['epochs']}")
    print(f"  Steps/epoch    : {steps_per_epoch:,}")
    print(f"  Total steps    : {total_steps:,}")
    print(f"  Effective batch: {effective_batch}")
    print(f"  Tokens/step    : {tokens_per_step:,}")
    print(f"  Total tokens   : {tokens_per_step * total_steps:,}")
    print(f"  LR             : {CONFIG['lr']} -> {CONFIG['min_lr']}")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay'], betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler('cuda')

    # Resume
    start_step = 0
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scaler.load_state_dict(ckpt['scaler'])
        start_step = ckpt['step']
        best_val_loss = ckpt.get('val_loss', float('inf'))
        # Figure out which epoch we're in
        if steps_per_epoch > 0:
            start_epoch = start_step // steps_per_epoch
        print(f"  Resumed at step {start_step}, epoch {start_epoch}")

    # Initial eval
    print(f"\nInitial evaluation...")
    val_loss = evaluate(model, val_dataset, device)
    print(f"  Initial val loss: {val_loss:.4f}")
    exp.log_metrics(0, val_loss=val_loss, is_best=True)
    best_val_loss = min(best_val_loss, val_loss)

    # Training
    print(f"\n{'='*60}")
    print(f"SFT TRAINING START")
    print(f"{'='*60}\n")

    model.train()
    t0 = time.time()
    train_start = time.time()
    step = start_step

    for epoch in range(start_epoch, CONFIG['epochs']):
        # Shuffle each epoch
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        micro_step = 0

        for i in range(0, len(indices) - CONFIG['batch_size'] + 1, CONFIG['batch_size']):
            batch_idx = indices[i:i + CONFIG['batch_size']]

            input_ids = torch.stack([train_dataset[j][0] for j in batch_idx]).to(device)
            labels = torch.stack([train_dataset[j][1] for j in batch_idx]).to(device)

            # Forward — can't use model's built-in loss (no ignore_index support)
            with torch.amp.autocast('cuda'):
                logits, _ = model(input_ids)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                )
                loss = loss / CONFIG['grad_accum_steps']

            scaler.scale(loss).backward()
            micro_step += 1

            # Optimizer step after accumulation
            if micro_step % CONFIG['grad_accum_steps'] == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])

                lr = get_lr(step, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # --- Logging ---
                if step % CONFIG['log_interval'] == 0:
                    elapsed = time.time() - t0
                    tok_s = tokens_per_step * CONFIG['log_interval'] / max(elapsed, 1e-6) if step > start_step else 0
                    total_elapsed = time.time() - train_start
                    steps_done = step - start_step
                    eta = (total_steps - step) / (steps_done / total_elapsed) if steps_done > 0 else 0

                    accum_loss = loss.item() * CONFIG['grad_accum_steps']
                    print(f"Step {step:5d}/{total_steps} | Epoch {epoch+1}/{CONFIG['epochs']} | "
                          f"Loss: {accum_loss:.4f} | LR: {lr:.2e} | "
                          f"Tok/s: {tok_s:,.0f} | "
                          f"ETA: {eta/3600:.1f}h")

                    exp.log_metrics(step, train_loss=accum_loss, lr=lr, tok_s=tok_s,
                                    epoch=epoch+1)
                    t0 = time.time()

                # --- Evaluation ---
                if step > 0 and step % CONFIG['eval_interval'] == 0:
                    val_loss = evaluate(model, val_dataset, device)
                    is_best = val_loss < best_val_loss
                    print(f"\n  >>> Val loss: {val_loss:.4f} {'(new best!)' if is_best else ''}")
                    exp.log_metrics(step, val_loss=val_loss, is_best=is_best)

                    if is_best:
                        best_val_loss = val_loss
                        exp.save_checkpoint(model, optimizer, scaler, CONFIG,
                                            step, val_loss, best=True)

                    model.train()

                # --- Samples ---
                if step > 0 and step % CONFIG['sample_interval'] == 0:
                    print(f"\n  --- Samples at step {step} ---")
                    samples = []
                    for prompt in SAMPLE_PROMPTS:
                        output = generate(model, prompt, max_tokens=150)
                        # Show only the assistant response
                        display = output.split("<|assistant|>")[-1].strip() if "<|assistant|>" in output else output
                        short_prompt = prompt.replace('<|user|>\n', '').replace('\n<|assistant|>\n', '')
                        print(f"  Q: {short_prompt}")
                        print(f"  A: {display[:200]}\n")
                        samples.append((prompt, output))

                    exp.save_samples(step, samples)
                    model.train()

                # --- Checkpoint ---
                if step > 0 and step % CONFIG['checkpoint_interval'] == 0:
                    exp.save_checkpoint(model, optimizer, scaler, CONFIG,
                                        step, best_val_loss, best=False)
                    print(f"  >>> Checkpoint saved at step {step}")

                step += 1

                if step >= total_steps:
                    break

        if step >= total_steps:
            break

        print(f"\n  === Epoch {epoch+1} complete ===\n")

    # --- Final ---
    total_time = time.time() - train_start

    # Final eval
    val_loss = evaluate(model, val_dataset, device)
    is_best = val_loss < best_val_loss
    if is_best:
        best_val_loss = val_loss

    exp.save_checkpoint(model, optimizer, scaler, CONFIG, step, best_val_loss, best=is_best)
    exp.save_summary(CONFIG, step, best_val_loss, total_time)

    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Total time: {total_time/3600:.1f} hours")
    print(f"  Experiment: {exp.dir}")
    print(f"{'='*60}")

    # Final samples
    print("\n=== Final Generations ===")
    for prompt in SAMPLE_PROMPTS:
        output = generate(model, prompt, max_tokens=150)
        display = output.split("<|assistant|>")[-1].strip() if "<|assistant|>" in output else output
        short_prompt = prompt.replace('<|user|>\n', '').replace('\n<|assistant|>\n', '')
        print(f"\nQ: {short_prompt}")
        print(f"A: {display[:300]}")
    print()


if __name__ == "__main__":
    train()