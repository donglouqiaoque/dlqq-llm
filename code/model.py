# -*- coding: utf-8 -*-
r"""模型定义（唯一一份）
================================================================
训练（code\train.py）、对话（code\chat.py）、桥接服务（code\http_server.py）
都从这里 import，**不要再各自复制一份**。

改动结构时只需要改这个文件，但必须保证旧权重还能 `load_state_dict`：
参数名和形状一旦变化，所有已发布的 `.pt` 都会加载失败。

接口：
    GPT(vocab, block_size, n_layer, n_head, n_embd, dropout)
    forward(idx)             -> logits            （推理）
    forward(idx, targets)    -> (logits, loss)    （训练）
    sample(idx, n, ...)      -> 逐 token 产出 id   （对话/服务流式）
    generate(idx, n, ...)    -> 拼好的完整序列      （兼容旧的整段调用）
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.drop if self.training else 0.0)
        return self.proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                 nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, vocab, block_size, n_layer, n_head, n_embd, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)
        self.head.weight = self.tok_emb.weight        # 权重共享，与训练时一致

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        logits = self.head(self.ln_f(self.blocks(x)))
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def sample(self, idx, n, temperature=1.0, top_k=None):
        """逐 token 产出 id：对话与服务用它做流式输出。"""
        for _ in range(n):
            logits = self(idx[:, -self.block_size:])[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
            yield int(idx[0, -1].item())

    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None):
        """整段生成（老接口），内部复用 sample。"""
        for _ in self.sample(idx, n, temperature, top_k):
            pass
        return idx
