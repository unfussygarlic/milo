import tiktoken
import numpy as np
import os
import time
import argparse
from datasets import load_dataset

enc = tiktoken.get_encoding("gpt2")
EOT = enc.encode("<|endoftext|>", allowed_special={'<|endoftext|>'})

def stream_and_save(dataset_name, subset, target_tokens, output_dir, prefix):
    print(f"\n=== {prefix} (target: {target_tokens:,} tokens) ===")

    ds = load_dataset(dataset_name, subset, split="train", streaming=True)

    chunk_size = 20_000_000  # 20M tokens per file (~40MB) — flush often
    buffer = []
    total_saved = 0
    chunk_num = 0
    doc_count = 0
    t0 = time.time()

    for example in ds:
        text = example.get('text', '')
        if len(text) < 100:
            continue

        tokens = enc.encode(text, allowed_special={'<|endoftext|>'}) + EOT
        buffer.extend(tokens)
        doc_count += 1

        # Flush to disk frequently to keep RAM low
        if len(buffer) >= chunk_size:
            chunk_num += 1
            arr = np.array(buffer[:chunk_size], dtype=np.uint16)
            np.save(os.path.join(output_dir, f"{prefix}_{chunk_num:04d}.npy"), arr)
            total_saved += chunk_size
            buffer = buffer[chunk_size:]

            elapsed = time.time() - t0
            rate = total_saved / max(elapsed, 1)
            eta = (target_tokens - total_saved) / rate if rate > 0 else 0
            print(f"  [{prefix}] Chunk {chunk_num} | "
                  f"{total_saved:,}/{target_tokens:,} | "
                  f"Docs: {doc_count:,} | "
                  f"{rate/1e6:.1f}M tok/s | "
                  f"ETA: {eta/60:.0f} min")

        if total_saved >= target_tokens:
            break

    # Save remainder
    if buffer and total_saved < target_tokens:
        chunk_num += 1
        arr = np.array(buffer, dtype=np.uint16)
        np.save(os.path.join(output_dir, f"{prefix}_{chunk_num:04d}.npy"), arr)
        total_saved += len(buffer)

    elapsed = time.time() - t0
    print(f"  {prefix} done: {total_saved:,} tokens, {chunk_num} files, {elapsed/60:.1f} min")
    return total_saved


def build_curriculum(output_dir):
    print("\n=== Building curriculum ===")

    all_files = sorted([f for f in os.listdir(output_dir)
                        if f.endswith('.npy') and not f.startswith('stage')])

    if not all_files:
        print("No data files found!")
        return

    # Count total tokens without loading everything
    total_tokens = 0
    file_sizes = []
    for f in all_files:
        path = os.path.join(output_dir, f)
        size = os.path.getsize(path) // 2  # uint16 = 2 bytes per token
        file_sizes.append((f, size))
        total_tokens += size

    print(f"  Total: {total_tokens:,} tokens across {len(all_files)} files")

    # Split files into stages (don't load into RAM — just move/symlink)
    curriculum_dir = os.path.join(output_dir, "curriculum")
    os.makedirs(curriculum_dir, exist_ok=True)

    s1_target = total_tokens * 15 // 100
    s2_target = total_tokens * 55 // 100
    # s3 gets the rest (30%)

    stage_files = {"stage_1": [], "stage_2": [], "stage_3": []}
    accumulated = 0

    for f, size in file_sizes:
        if accumulated < s1_target:
            stage_files["stage_1"].append(f)
        elif accumulated < s1_target + s2_target:
            stage_files["stage_2"].append(f)
        else:
            stage_files["stage_3"].append(f)
        accumulated += size

    # Concatenate files per stage
    for stage_name, files in stage_files.items():
        arrays = []
        for f in files:
            arrays.append(np.load(os.path.join(output_dir, f)))
        combined = np.concatenate(arrays)

        path = os.path.join(curriculum_dir, f"{stage_name}.npy")
        np.save(path, combined)
        print(f"  {stage_name}: {len(combined):,} tokens ({len(combined)*2/1e9:.2f} GB)")

        # Free memory
        del arrays, combined

    print(f"  Saved to: {curriculum_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens', type=int, default=4_000_000_000)
    parser.add_argument('--output', type=str, default='data/smollm')
    parser.add_argument('--skip-curriculum', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    cosmo_target = int(args.tokens * 0.6)
    fineweb_target = int(args.tokens * 0.4)

    print(f"Target: {args.tokens:,} tokens")
    print(f"  Cosmopedia: {cosmo_target:,}")
    print(f"  FineWeb-Edu: {fineweb_target:,}")
    print(f"  Disk: ~{args.tokens * 2 / 1e9:.1f} GB")

    t_start = time.time()

    # Check for existing progress
    existing_cosmo = sum(
        os.path.getsize(os.path.join(args.output, f)) // 2
        for f in os.listdir(args.output)
        if f.startswith('cosmo') and f.endswith('.npy')
    ) if os.path.exists(args.output) else 0

    existing_fineweb = sum(
        os.path.getsize(os.path.join(args.output, f)) // 2
        for f in os.listdir(args.output)
        if f.startswith('fineweb') and f.endswith('.npy')
    ) if os.path.exists(args.output) else 0

    if existing_cosmo > 0:
        print(f"\n  Found existing cosmo: {existing_cosmo:,} tokens")
    if existing_fineweb > 0:
        print(f"  Found existing fineweb: {existing_fineweb:,} tokens")

    if existing_cosmo < cosmo_target:
        if existing_cosmo > 0:
            print(f"  Resuming cosmo from {existing_cosmo:,} tokens...")
            remaining = cosmo_target - existing_cosmo
        else:
            remaining = cosmo_target
        stream_and_save(
            "HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
            remaining, args.output, "cosmo"
        )
    else:
        print("  Cosmopedia already complete, skipping.")

    if existing_fineweb < fineweb_target:
        if existing_fineweb > 0:
            print(f"  Resuming fineweb from {existing_fineweb:,} tokens...")
            remaining = fineweb_target - existing_fineweb
        else:
            remaining = fineweb_target
        stream_and_save(
            "HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup",
            remaining, args.output, "fineweb"
        )
    else:
        print("  FineWeb-Edu already complete, skipping.")

    if not args.skip_curriculum:
        build_curriculum(args.output)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {total_time/60:.0f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()