# -*- coding: utf-8 -*-
"""
跟你自己训练的模型对话
=====================
用法（在工程根目录执行）：
    python code/chat.py model/chat_demo.pt                      # 交互式对话
    python code/chat.py model=model/chat_demo.pt                # 同上，key=value 写法
    python code/chat.py model/chat_demo.pt --once "用户：你好"    # 只生成一次就退出

单独一个不带 "=" 的参数会被当成模型路径；其它可覆盖的配置键有
model / temperature / top_k / max_new / chat_mode / device。

交互式里可以用的命令：
    :q            退出
    :t 0.7        改温度（越小越保守，越大越野）
    :n 200        改生成长度
    :raw          切换「续写模式 / 对话模式」
    :p            看看当前的设置
"""
import sys, os

# Windows 控制台默认是 GBK，遇到 → 这类字符会抛 UnicodeEncodeError
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 参数解析 ----------------
cfg = dict(model="mymodel.pt", temperature=0.8, top_k=40, max_new=180,
           chat_mode=True, device=None)
once = None
args = sys.argv[1:]
i = 0
while i < len(args):
    a = args[i]
    if a == "--once":
        i += 1; once = args[i] if i < len(args) else "用户：你好"
    elif a in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)
    elif a.startswith("-"):
        print(f"[用法] 不认识的参数：{a}（参数是 key=value 形式，例如 model=model/chat_demo.pt）")
        sys.exit(2)
    elif "=" in a:
        k, v = a.split("=", 1)
        if k in cfg:
            cur = cfg[k]
            try:
                cfg[k] = int(v) if isinstance(cur, int) else float(v) if isinstance(cur, float) else v
            except ValueError:
                cfg[k] = v
    else:
        # 位置参数：直接当成模型路径（python chat.py model\mymodel.pt）
        cfg["model"] = a
    i += 1

# 交互输入：只有在「管道/重定向」进来时才强制 UTF-8（否则中文会变乱码）。
# 真实控制台必须保持系统编码（Windows 中文环境是 GBK），不能改成 UTF-8 ——
# 那样读键盘输入的中文反而会变成乱码。
try:
    if not sys.stdin.isatty():
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

DEVICE = cfg["device"] or ("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- 模型定义（跟 train_llm.py 保持一致）----------------
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
        self.net = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(),
                                 nn.Linear(4*n_embd, n_embd), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn, self.mlp = MultiHeadAttention(n_embd, n_head, dropout), MLP(n_embd, dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x));  return x + self.mlp(self.ln2(x))

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
        self.head.weight = self.tok_emb.weight
    def forward(self, idx):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        return self.head(self.ln_f(self.blocks(x)))
    @torch.no_grad()
    def generate(self, idx, n, temperature=1.0, top_k=None):
        for _ in range(n):
            logits = self(idx[:, -self.block_size:])[:, -1, :] / max(temperature, 1e-6)
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], dim=1)
        return idx

# ---------------- 加载 ----------------
path = cfg["model"]
if not os.path.isfile(path):
    print(f"[错误] 找不到模型文件：{path}")
    print(f"       当前目录：{os.getcwd()}")
    print( "       先训练一个：python code/train_llm.py data_path=src/chat_demo.jsonl out_path=model/chat_demo.pt")
    print( "       或指定别的模型：python code/chat.py model/你的模型.pt")
    sys.exit(1)

ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
C, chars = ckpt["config"], ckpt["chars"]
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
model = GPT(len(chars), C["block_size"], C["n_layer"], C["n_head"], C["n_embd"], 0.0).to(DEVICE)
model.load_state_dict(ckpt["model"]); model.eval()
print(f"[模型] {path} | 参数量 {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
      f"| 词表 {len(chars)} | 设备 {DEVICE}")

def unknown(s):
    return [c for c in s if c not in stoi]

def reply(text):
    """把用户输入拼成提示词，让模型往下续写"""
    if cfg["chat_mode"]:
        prompt = f"用户：{text}\n助手："
    else:
        prompt = text
    bad = unknown(prompt)
    if bad:
        if cfg["chat_mode"]:
            print("[说明] 这个模型的词表里没有「用户：」「助手：」这些字 —— "
                  "说明它不是用对话语料训的。")
            print("       已自动切到【续写模式】：把你说的话当成一段文本，让它往下写。")
            cfg["chat_mode"] = False
            prompt = text
            bad = unknown(prompt)
        if bad:
            print(f"[提示] 这些字不在词表里，已跳过：{''.join(bad)}")
            prompt = "".join(c for c in prompt if c in stoi)
    if not prompt:
        return ""
    idx = torch.tensor([[stoi[c] for c in prompt]], device=DEVICE)
    out = model.generate(idx, cfg["max_new"], cfg["temperature"], cfg["top_k"])
    full = "".join(itos[i] for i in out[0].tolist())
    # 只截取「助手：」之后的部分
    if cfg["chat_mode"] and "助手：" in full:
        full = full.split("助手：", 1)[1]
    return full.split("用户：")[0].strip()      # 碰到下一个「用户：」就停

# ---------------- 跑 ----------------
if once is not None:
    print("\n你  >", once); print("它  >", reply(once)); sys.exit(0)

print("\n输入内容后回车，跟它对话。:q 退出，:p 看设置\n")
while True:
    try:
        s = input("你  > ").strip()
    except (EOFError, KeyboardInterrupt):
        print(); break
    if not s: continue
    if s in (":q", ":quit", ":exit"): break
    if s == ":p":
        print(f"     温度={cfg['temperature']} top_k={cfg['top_k']} 长度={cfg['max_new']} "
              f"模式={'对话' if cfg['chat_mode'] else '续写'}"); continue
    if s == ":raw":
        cfg["chat_mode"] = not cfg["chat_mode"]; print(f"     模式 → {'对话' if cfg['chat_mode'] else '续写'}"); continue
    if s.startswith(":t"):
        cfg["temperature"] = float(s.split()[1]); print(f"     温度 → {cfg['temperature']}"); continue
    if s.startswith(":n"):
        cfg["max_new"] = int(s.split()[1]); print(f"     长度 → {cfg['max_new']}"); continue
    print("它  >", reply(s), "\n")
