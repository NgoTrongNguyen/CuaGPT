# embedding + positional encoding + vài lớp self-attention (causal) + MLP +
# head dự đoán token tiếp theo.

import math
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3*d_model)
        self.out = nn.Linear(d_model, d_model)
        #Chance to make some to 0
        self.dropout = nn.Dropout(dropout)
        # causal mask: không nhìn quá tương lai

        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).unsqueeze(0).unsqueeze(0))

    def forward(self,x):
        #B: batch size, T: seq length, C: d_model
        B,T,C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv[:,:,0], qkv[:,:,1], qkv[:,:,2]
        #Hoán đổi T và heads để tính attention
        q = q.transpose(1,2)  # B, heads, T, head_dim
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        att = att.masked_fill(self.mask[:,:, :T, :T] == 0, float("-inf"))
        #Chuẩn hóa ma trận attention
        att = torch.softmax(att, dim=-1)
        att = self.dropout(att)
        y = att @ v  # B, heads, T, head_dim
        y = y.transpose(1,2).contiguous().view(B, T, C)
        return self.out(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout, block_size)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self,x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    def __init__(self, char_size, d_model, n_layers, n_heads, dropout, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_emb = nn.Embedding(char_size, d_model)
        self.tok_emb = nn.Embedding(char_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout, block_size) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, char_size)

    def forward(self, x):
        B, T = x.shape
        tok_emb = self.token_emb(x)
        pos = torch.arange(0, T, device=x.device).unsqueeze(0) # [0,1,...,T-1]
        pos_emb = self.pos_emb(pos) # (T, d_model)
        x = tok_emb + pos_emb # B, T, d_model
        for block in self.blocks:
            x = block(x) # B, T, d_model
        x = self.ln_f(x) # B, T, d_model
        logits = self.head(x) # B, T, char_size
        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature, top_k):
        for i in range(max_new_tokens):
            # Cắt chuỗi khi vượt size
            idx_cond = idx[:, -self.block_size:]
            logits = self.forward(idx_cond)
            # Lấy token cuối
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            # Sampling chọn token mới
            next_id = torch.multinomial(probs, num_samples=1)
            # Ghép token mới vào chuỗi
            idx = torch.cat((idx, next_id), dim=1)
        return idx