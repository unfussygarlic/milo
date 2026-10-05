import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- RoPE (same as before, proven to work) ---
def precompute_rope_frequencies(head_dim, max_seq_len=4096, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(max_seq_len).float()
    angles = torch.outer(positions, freqs)
    return angles.cos(), angles.sin()

def apply_rope(x, cos, sin):
    B, n_heads, T, head_dim = x.shape
    x_pairs = x.view(B, n_heads, T, head_dim // 2, 2)
    x_even, x_odd = x_pairs[..., 0], x_pairs[..., 1]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x_even_new = x_even * cos - x_odd * sin
    x_odd_new  = x_even * sin + x_odd * cos
    return torch.stack([x_even_new, x_odd_new], dim=-1).view(B, n_heads, T, head_dim)

# --- Change 1: RMSNorm instead of LayerNorm ---
# One pass instead of two. No mean subtraction.
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# --- Change 2: GQA (Grouped Query Attention) ---
# More Q heads than KV heads. KV heads are shared across groups.
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_q_heads // n_kv_heads  # Q heads per KV group
        self.head_dim = d_model // n_q_heads

        # Q projects to all heads, K and V project to fewer heads
        self.W_q = nn.Linear(d_model, n_q_heads * self.head_dim, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, rope_cos, rope_sin):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_q_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # RoPE on Q and K
        Q = apply_rope(Q, rope_cos, rope_sin)
        K = apply_rope(K, rope_cos, rope_sin)

        # Expand K, V to match Q heads by repeating
        # [B, n_kv_heads, T, head_dim] -> [B, n_q_heads, T, head_dim]
        K = K.repeat_interleave(self.n_groups, dim=1)
        V = V.repeat_interleave(self.n_groups, dim=1)

        # Standard attention from here
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))
        attn_weights = self.attn_dropout(F.softmax(scores, dim=-1))

        out = (attn_weights @ V).transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)

# --- Change 3: SwiGLU FFN ---
# Gate mechanism: SiLU(W1(x)) * W3(x), then W2 projects back
class SwiGLUFeedForward(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        # Hidden dim: 8/3 * d_model, rounded to multiple of 64
        hidden = int(8 / 3 * d_model)
        hidden = ((hidden + 63) // 64) * 64

        self.w1 = nn.Linear(d_model, hidden, bias=False)  # gate
        self.w2 = nn.Linear(hidden, d_model, bias=False)  # down
        self.w3 = nn.Linear(d_model, hidden, bias=False)  # up
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads, dropout=0.1):
        super().__init__()
        self.ln1 = RMSNorm(d_model)
        self.attn = GroupedQueryAttention(d_model, n_q_heads, n_kv_heads, dropout)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLUFeedForward(d_model, dropout)

    def forward(self, x, rope_cos, rope_sin):
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ln2(x))
        return x

# --- Full Model ---
class SLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_q_heads, n_kv_heads,
                 n_layers, seq_len, dropout=0.1, share_blocks=True):
        super().__init__()
        self.seq_len = seq_len
        self.head_dim = d_model // n_q_heads
        self.n_layers = n_layers

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # --- Change 4: Block-wise weight sharing (MobileLLM-LS) ---
        # Instead of n_layers unique blocks, create n_layers/2 unique blocks
        # and reuse each one twice. Same param count, more effective depth.
        if share_blocks:
            n_unique = n_layers // 2
            unique_blocks = [
                TransformerBlock(d_model, n_q_heads, n_kv_heads, dropout)
                for _ in range(n_unique)
            ]
            # Each unique block is used twice: [0,0,1,1,2,2,...]
            self.blocks = nn.ModuleList([
                unique_blocks[i // 2] for i in range(n_layers)
            ])
            # Store unique blocks so their params are registered
            self.unique_blocks = nn.ModuleList(unique_blocks)
        else:
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, n_q_heads, n_kv_heads, dropout)
                for _ in range(n_layers)
            ])
            self.unique_blocks = self.blocks

        self.ln_final = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # RoPE
        rope_cos, rope_sin = precompute_rope_frequencies(
            self.head_dim, max_seq_len=seq_len * 2
        )
        self.register_buffer('rope_cos', rope_cos)
        self.register_buffer('rope_sin', rope_sin)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx)

        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.reshape(B * T))
        return logits, loss


if __name__ == "__main__":
    import time
    device = torch.device('cuda')

    model = SLM(
        vocab_size=50257, d_model=1024, n_q_heads=16, n_kv_heads=4,
        n_layers=16, seq_len=512, dropout=0.1, share_blocks=True,
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,} ({params/1e6:.1f}M)")

    # Simulate actual training batch
    x = torch.randint(0, 50257, (16, 512), device=device)
    y = torch.randint(0, 50257, (16, 512), device=device)

    # Test with mixed precision (like actual training)
    scaler = torch.amp.GradScaler('cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    with torch.amp.autocast('cuda'):
        logits, loss = model(x, y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)

    print(f"Forward + backward OK")
    print(f"Logits: {logits.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"VRAM reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")