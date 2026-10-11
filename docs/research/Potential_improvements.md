# Improving Milo: Research on Methods Beyond More Data

**Context:** Milo is a 202M parameter decoder-only transformer (GPT-style) trained on 3.6B tokens with SFT completed. The model follows instructions but has weak factual accuracy, arithmetic, and reasoning — inherent to the small parameter count. Instead of just throwing more data at it, these are smarter approaches to make it punch above its weight.

---

## 1. Tool-Augmented Generation (Toolformer Approach)

**Paper:** [Toolformer: Language Models Can Teach Themselves to Use Tools](https://ar5iv.labs.arxiv.org/html/2302.04761) — Meta, NeurIPS 2023

**Key idea:** A 6.7B model with tools **outperformed GPT-3 at 175B** on several benchmarks. The model learns *when* to call a tool purely from self-supervised signal — no human annotation needed.

**How it works:**
1. Start with plain text from a large corpus
2. Use few-shot prompting to have the model annotate text with candidate API calls at various positions
3. Execute every candidate call against the real tool
4. **Filter by usefulness:** Compare the model's loss on upcoming tokens *with* and *without* the tool result. Keep only calls where the result actually helped predict the next tokens (L⁻ − L⁺ > threshold)
5. Finetune the model on this self-annotated dataset with standard language modeling objective

**Tools they implemented:** Calculator, Wikipedia search, Question answering (Atlas), Machine translation (NLLB-600M), Calendar

**Why it matters for Milo:** Instead of trying to memorize math or facts in 202M weights, teach Milo to emit `CALCULATE(3 × 4.5)` and get the right answer back. The self-supervised approach means you don't need to hand-label thousands of tool-use examples — the model generates its own training data.

**Feasibility:** HIGH — this is directly on the roadmap (tool-use fine-tuning → ReAct loop). The Toolformer approach can be adapted to generate training data using Milo itself or a slightly larger teacher model.

---

## 2. RETRO — Retrieval-Enhanced Transformer

**Paper:** [Improving Language Models by Retrieving from Trillions of Tokens](https://arxiv.org/pdf/2112.04426) — DeepMind, ICML 2022

**Key idea:** Instead of stuffing knowledge into model weights, give the model access to a **retrieval database** (up to 2 trillion tokens) that it queries during generation. A RETRO-7.5B beat a 175B Jurassic-1 model on 10/16 benchmarks.

**How it works:**
- Input text is chunked into passages
- For each passage, a nearest-neighbor search (using frozen BERT embeddings) finds similar passages in an external database, along with their continuations
- The architecture interleaves two types of attention:
  - **Regular self-attention** — operates on the input document (standard transformer)
  - **Cross-attention to retrieved neighbors** — retrieved passages are processed by a separate encoder, then cross-attended to by the main model
- The retrieval database can be updated without retraining the model

**Tested sizes:** 172M, 425M, 1.5B, and 7.5B parameters

**Why it matters for Milo:** They tested at **172M parameters** — almost exactly Milo's size. This means a small model can "know" very little in its weights but look things up at inference time, like an open-book exam. Performance keeps improving as the database grows. The retrieval also makes outputs more interpretable — you can see which passages the model relied on.

**What it would require:**
- Modify `model.py` to add cross-attention layers that attend to retrieved passages
- Build a vector database (FAISS) of text chunks with BERT embeddings
- Retrain from scratch (the cross-attention layers need to be trained with retrieval)
- At inference, each forward pass includes a retrieval step

**Feasibility:** MEDIUM — requires architecture changes and retraining, but proven at Milo's scale. Best as a Milo v2 project. Open-source implementation exists: [RETRO-pytorch](https://github.com/lucidrains/RETRO-pytorch)

---

## 3. RAG (Retrieval-Augmented Generation) — No Architecture Change

**Key idea:** Unlike RETRO which bakes retrieval into the architecture, RAG works as a **wrapper around the existing model**. Retrieve relevant documents, prepend them to the prompt, then generate. No retraining needed.

**How it works:**
1. User asks a question
2. Encode the question with a retrieval model (e.g., `all-MiniLM-L6-v2`)
3. Search a vector database (FAISS/ChromaDB) for relevant passages
4. Prepend the top-k passages to the prompt: `Context: {passages}\n<|user|>\n{question}\n<|assistant|>\n`
5. Model generates using the provided context

**Limitation for Milo:** With seq_len=512, context window is tiny. After stuffing in retrieved passages, there's very little room left for the question and answer. This makes RAG much less effective than it is for models with 2K-8K context.

**A paper worth noting:** [Can Small Language Models Use What They Retrieve?](https://arxiv.org/pdf/2603.11513) — found that retrieval utilization degrades significantly at smaller model scales. Small models struggle to identify and use the relevant parts of retrieved context.

**Feasibility:** HIGH for implementation, LOW-MEDIUM for actual impact due to the 512-token context window and small model's difficulty with retrieval utilization.

---

## 4. MemoryLLM — Self-Updatable Memory Pool

**Paper:** [MemoryLLM: Towards Self-Updatable Large Language Models](https://arxiv.org/html/2402.04624v2)

**Key idea:** Add a fixed-size pool of **learnable memory vectors** inside the transformer. The memory updates without backpropagation — you can feed the model new text and it absorbs it into the memory pool, gradually forgetting old knowledge.

**How it works:**
- Each transformer layer has a set of memory tokens (hidden vectors)
- During generation, each layer's hidden states attend to that layer's memory tokens
- To add new knowledge: feed text through the model, and the last k memory tokens get updated with compressed representations
- Old knowledge gradually fades as new knowledge is written
- Tested: ~1B memory pool added to a 7B base model
- Survived nearly 1 million updates without performance degradation

**Why it's interesting:** It's like giving the model a working memory that persists across conversations. The model can "learn" new facts at inference time without fine-tuning.

**Feasibility:** LOW — research-stage, no off-the-shelf implementation for small models, tested only at 7B scale. Interesting conceptually but not practical for Milo right now.

---

## 5. Hybrid Architecture (Mamba + Attention)

**Models:** Hymba-1.5B, Nemotron-H

**Key idea:** Combine Mamba (state-space model) layers with attention layers. Mamba handles long-range dependencies efficiently (linear complexity), while attention handles local patterns. These hybrids get better throughput and handle longer contexts than pure transformers at the same parameter count.

**Feasibility:** LOW for current Milo — requires retraining from scratch with a new architecture. Interesting for Milo v2 if the goal is to build a better base model.

---

## 6. Specialist Agent System (Modular SLM)

**Paper:** [Small Language Models are the Future of Agentic AI](https://arxiv.org/pdf/2506.02153)

**Key idea:** Instead of one model doing everything, use multiple specialized small models:
- A **router** directs queries to the right specialist
- **Task-specific LoRA adapters** on the same base model (math, code, conversation, tool-calling)
- An orchestrator stitches results together
- Data flywheel: log agent interactions → curate → retrain specialists

**The paper's argument:** ~60-70% of agent subtasks can be handled by SLMs. Use a large model only as the orchestrator when general reasoning is needed.

**For Milo:** Train several LoRA adapters on the same 202M base:
- `milo-tools` — tool calling specialist
- `milo-code` — code generation
- `milo-chat` — conversation
- Router logic determines which adapter to load

**Feasibility:** MEDIUM — LoRA fine-tuning is cheap, but building the routing and orchestration system is engineering work.

---

## 7. Knowledge Distillation (Response-Level)

**Key idea:** Use a stronger model (Qwen-0.5B, Phi-3-mini, or even GPT-4) to generate high-quality training data. Then fine-tune Milo on it. This isn't logit-level distillation (vocabulary mismatch prevents that) — it's using teacher-generated text as training data.

**Approaches:**
- Generate synthetic textbooks explaining concepts step-by-step
- Generate chain-of-thought reasoning traces for math/logic problems
- Generate tool-use examples showing when and how to call APIs
- Generate code with detailed comments

**Why it works:** 500M tokens of synthetic, curated, step-by-step explanations can teach more than 8B tokens of raw web crawl. Quality over quantity.

**Feasibility:** HIGH — can generate data locally with Qwen-0.5B or use API credits for a stronger model. Directly compatible with existing training pipeline.

---

## Recommended Strategy for Milo

Given: 202M params, $100 budget, roadmap → tool-use → ReAct agent

### Phase 1: Light Continued Pretraining (~$10-15)
- 2B tokens of curated, high-quality data
- Focus on: textbook-style explanations, code, step-by-step reasoning
- Use the `continue_pretrain.py` script already written
- Goal: solidify the base, not radically change it

### Phase 2: Tool-Use Fine-Tuning (minimal compute cost)
- Adapt Toolformer's self-supervised approach to generate tool-call training data
- Tools: CALCULATE(), SEARCH(), LOOKUP(), WEATHER()
- Fine-tune from SFT checkpoint using existing `finetune.py` patterns
- This is where 202M params stops being a limitation — the model offloads hard tasks to tools

### Phase 3: RAG Pipeline (engineering, no GPU cost)
- Build a FAISS vector database of knowledge chunks
- Retrieval wrapper around the model at inference time
- Compensates for limited parametric knowledge
- Works within the 512-token context window by retrieving small, targeted chunks

### Phase 4: ReAct Agent Loop (engineering, no GPU cost)
- Python runtime: query → model reasons → calls tool → gets result → reasons again → responds
- Ties together tool use + retrieval + conversation
- The actual deliverable: Milo as a functioning agent

### Future (Milo v2):
- RETRO-style architecture with built-in retrieval cross-attention
- Larger model (400-500M) with hybrid Mamba+Attention
- Full Toolformer self-supervised tool training at scale

---

## Key Papers & Resources

| Paper | Year | Key Contribution | Link |
|-------|------|-----------------|------|
| Toolformer | 2023 | Self-supervised tool-use training | [arXiv](https://ar5iv.labs.arxiv.org/html/2302.04761) |
| RETRO | 2022 | Retrieval-enhanced transformer, tested at 172M | [DeepMind](https://deepmind.google/discover/blog/improving-language-models-by-retrieving-from-trillions-of-tokens/) |
| MemoryLLM | 2024 | Self-updatable memory pool in transformer | [arXiv](https://arxiv.org/html/2402.04624v2) |
| SLMs for Agentic AI | 2025 | Small models as specialist agents | [arXiv](https://arxiv.org/pdf/2506.02153) |
| SLMs Can Pack a Punch | 2025 | Comprehensive SLM techniques survey | [arXiv](https://arxiv.org/html/2501.05465v2) |
| Can SLMs Use Retrieval? | 2026 | Retrieval utilization across model scale | [arXiv](https://arxiv.org/pdf/2603.11513) |
| Memory-Augmented LLMs | — | Overview of memory approaches | [EmergentMind](https://www.emergentmind.com/topics/memory-augmented-llm-systems) |
| RETRO-pytorch | — | Open-source RETRO implementation | [GitHub](https://github.com/lucidrains/RETRO-pytorch) |