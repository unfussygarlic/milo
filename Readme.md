# Milo

A 202M parameter transformer language model, built from scratch.

Decoder-only GPT-style architecture with RMSNorm, GQA (grouped query attention), SwiGLU FFN, RoPE, and block-wise weight sharing (16 layers from 8 unique blocks).

## Architecture

| Parameter | Value |
|-----------|-------|
| Parameters | 202M total (101M unique) |
| d_model | 1280 |
| Attention | 20 query heads, 4 KV heads (GQA) |
| Layers | 16 (8 unique, shared) |
| Context | 512 tokens |
| Vocab | 50,257 (GPT-2 tokenizer) |
| Normalization | RMSNorm |
| FFN | SwiGLU |
| Positional | RoPE |

## Project Structure

```
milo/
├── model.py                 # Model architecture (SLM class)
├── train.py                 # Pretraining script
├── finetune.py              # Supervised fine-tuning (SFT)
├── finetune_tools.py        # Tool-use fine-tuning
├── serve.py                 # FastAPI chat server
├── react_runtime.py         # ReAct agent with tool execution
│
├── data/                    # Data generation & streaming
│   ├── collect_sft_data.py  # Collect SFT instruction data
│   ├── generate_tool_data.py# Generate tool-use training examples
│   └── data_v2.py           # Stream pretraining data (SmolLM-Corpus)
│
├── static/
│   └── index.html           # Chat UI frontend
│
├── infra/                   # Deployment & cloud training
│   ├── setup_ec2.sh         # EC2 instance setup
│   ├── continue_pretrain.py # Continued pretraining (spot-resilient)
│   └── spot_training_guide.sh
│
├── research/
│   └── potential_improvements.md
│
└── experiments/             # Training runs (gitignored)
    ├── 20260926_220527/     # Pretraining — val loss 3.30
    └── sft_20260929_075754/ # SFT — val loss 1.4956
```

## Training Pipeline

```
Pretraining (3.6B tokens)
    → SFT (90K instruction examples)
        → Tool-use fine-tuning (10K examples)
            → ReAct agent runtime
```

### 1. Pretraining

```bash
python train.py
```

Trained on SmolLM-Corpus (Cosmopedia v2 + FineWeb-Edu). 55K steps, val loss 3.30.

### 2. Supervised Fine-Tuning

```bash
python finetune.py
```

90K instruction-following examples. 8,436 steps, val loss 1.4956.

### 3. Tool-Use Fine-Tuning

```bash
# Generate training data
cd data && python generate_tool_data.py && cd ..

# Fine-tune
python finetune_tools.py --checkpoint experiments/sft_20260929_075754/checkpoints/best.ptrom
```

Teaches the model 5 tools: `CALCULATE`, `SEARCH`, `DATE`, `WEATHER`, `RUN_CODE`.

### 4. Inference

**Chat server:**
```bash
python serve.py --checkpoint <path_to_best.ptrom>
# Open http://localhost:8000
```

**ReAct agent (tool-calling CLI):**
```bash
python react_runtime.py --checkpoint <path_to_best.pt>
```

## Tool-Call Format

```
<|user|>
What's 15% tip on $85?
<|assistant|>
<|tool_call|>CALCULATE(85 * 0.15)<|tool_result|>12.75<|end_tool|>
A 15% tip on $85 would be $12.75, making your total $97.75.
<|endoftext|>
```

## Chat Format

```
<|user|>
Your question here
<|assistant|>
Model response here<|endoftext|>
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (CUDA)
- tiktoken
- numpy
- FastAPI + uvicorn (for serve.py)