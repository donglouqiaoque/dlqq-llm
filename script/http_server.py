# -*- coding: utf-8 -*-
r"""把训练好的模型起成一个 OpenAI 兼容的 HTTP 服务（桥接服务）
================================================================
本文件独立自足：只依赖 torch 和 Python 标准库（http.server / json / threading），
不引用工程里任何其它模块，也不依赖 ai_res（那是临时目录）。

供 DSH（或任何 OpenAI 兼容客户端）接入：DSH 侧只需要在界面里
「设置 → 模型 → 添加提供方 → 自定义提供方」，填上本服务的地址即可，不用改任何配置文件。

接口：
    GET  /health                 健康检查
    GET  /v1/models              模型清单（DSH 点「获取可用模型」时调它）
    POST /v1/chat/completions    对话（支持流式与非流式）

行为约定 —— 桥接层只做两件事，内容判断一律交给模型：
    · system / developer 角色的消息**直接忽略**（这个模型从没用系统提示训过）；
    · 请求里的 tools / tool_choice / functions **一律忽略**（DSH 界面里没有关闭工具的
      开关，所以不能报错）；服务只返回普通文本，**永远不返回 tool_calls**；
      想让带 tools 的请求直接报错，就加 --reject-tools；
    · 多轮对话会拼成「用户：/助手：」的提示词，超出模型上下文时丢弃较早的轮次，
      只剩一轮还太长就按字符保留尾部；
    · 除此之外**不做任何过滤、打分或关键词匹配** —— 答成什么样是模型的事；
    · 只做基本对话，不支持图片。

用法（在工程根目录执行）：
    python script\http_server.py
    python script\http_server.py --model model\chat_demo.pt --port 49128
    python script\http_server.py --served-name my-model --device cpu
或者直接双击 script\http_server.bat。
"""
import os
import re
import sys
import json
import time
import uuid
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))


# ==================================================================
# 启动它的 python 没装 torch 时，换一个装了 torch 的重新执行自己
# ==================================================================
def _torch_ok(exe):
    """问 pip 的清单最快；pip 不可用再真的 import 一次"""
    if not exe or not os.path.isfile(exe):
        return False
    try:
        import subprocess
        r = subprocess.run([exe, "-m", "pip", "list"],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and r.stdout:
            return re.search(r"(?m)^torch\s", r.stdout) is not None
    except Exception:
        pass
    try:
        import subprocess
        r = subprocess.run([exe, "-c", "import torch"], capture_output=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


def _relaunch_if_no_torch():
    try:
        import torch  # noqa: F401
        return
    except Exception:
        pass

    import shutil
    import subprocess
    # 只认 PATH（和显式指定），不去猜安装目录
    cands = [os.environ.get("PYTHON") or os.environ.get("TLLM_PYTHON")]
    for name in ("python", "python3", "py"):
        exe = shutil.which(name)
        if exe:
            cands.append(exe)
    cands.append(sys.executable)

    me = os.path.abspath(__file__)
    print(f"  当前解释器没有 torch：{sys.executable}")
    print("  正在查找其它装了 torch 的 python …")
    seen = set()
    for exe in cands:
        if not exe or exe in seen:
            continue
        seen.add(exe)
        if os.path.abspath(exe).lower() == os.path.abspath(sys.executable).lower():
            continue
        if _torch_ok(exe):
            print(f"  找到：{exe}")
            print("  用它重新启动服务 …\n")
            sys.stdout.flush()
            try:
                sys.exit(subprocess.call([exe, me] + sys.argv[1:], cwd=os.getcwd()))
            except KeyboardInterrupt:
                sys.exit(130)
    print("  [错误] 没找到任何装了 torch 的 python。")
    print("         装一个：python -m pip install torch")
    sys.exit(1)


_relaunch_if_no_torch()


def find_root(start):
    """往上找到含 code\train_llm.py 的那一级，作为工程根目录"""
    cur = start
    for _ in range(6):
        if os.path.isfile(os.path.join(cur, "code", "train_llm.py")):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(start)


ROOT = find_root(HERE)
DEFAULT_MODEL = os.path.join(ROOT, "model", "chat_demo.pt")

USER_TAG, ASSIST_TAG = "用户：", "助手："


# ==================================================================
# 模型：结构与 code/chat.py 完全一致，加载 .pt 里的 config / chars
# ==================================================================
def load_model(path, device=None):
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
            self.head.weight = self.tok_emb.weight

        def forward(self, idx):
            B, T = idx.shape
            x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
            return self.head(self.ln_f(self.blocks(x)))

    if not os.path.isfile(path):
        print(f"[错误] 找不到模型：{path}")
        print("       先训练一个：python code/train_llm.py data_path=src/chat_demo.txt out_path=model/chat_demo.pt")
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    C, chars = ckpt["config"], ckpt["chars"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPT(len(chars), C["block_size"], C["n_layer"], C["n_head"], C["n_embd"], 0.0)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, chars, C, device


# ==================================================================
# 生成：把 messages 拼成提示词，逐字符采样
# ==================================================================
class Engine:
    def __init__(self, path, device=None):
        self.model, chars, self.cfg, self.device = load_model(path, device)
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.block = self.cfg["block_size"]
        self.lock = threading.Lock()      # 同一时刻只跑一个生成，避免并发互相踩
        self.loaded = time.strftime("%Y-%m-%d %H:%M:%S")
        self.warned_tools = False         # 只在第一次遇到 tools 时提示一次

    # ---- 提示词 ----
    def build_prompt(self, messages):
        """messages → 提示词。
        只做两件事，别的都交给模型：
          1. 丢掉 system / developer 角色（这个模型从没用系统提示训过）；
          2. 历史太长时从最早的一轮开始丢，保证塞得进 block_size。
        不做任何「像不像人话」的内容判断 —— 那是模型的事，不是桥接层的事。"""
        convo = []                       # [(role, text)]
        for m in messages or []:
            role = (m.get("role") or "").lower()
            if role in ("system", "developer"):
                continue                 # 系统提示直接忽略
            if role not in ("user", "assistant"):
                continue                 # 工具结果之类的角色也忽略
            text = m.get("content")
            if isinstance(text, list):   # 兼容 {"type":"text","text":...} 形式
                text = "".join(p.get("text", "") for p in text
                               if isinstance(p, dict) and p.get("type") == "text")
            text = self.clean(text or "")
            if text:
                convo.append((role, text))

        # 最后一条必须是用户说的话，否则没法接着答
        while convo and convo[-1][0] != "user":
            convo.pop()
        if not convo:
            return "", 0

        # 历史塞不下就从最早的一轮开始丢；只剩一轮还太长，就按字符保留尾部
        budget = max(self.block - 8, 8)
        while convo:
            prompt = "".join(f"{USER_TAG if r == 'user' else ASSIST_TAG}{t}\n"
                             for r, t in convo) + ASSIST_TAG
            if len(prompt) <= budget:
                return prompt, len(convo)
            if len(convo) == 1:
                return prompt[-budget:], 1
            convo.pop(0)
        return "", 0

    def clean(self, text):
        """去掉不在词表里的字符"""
        return "".join(c for c in str(text) if c in self.stoi)

    # ---- 采样 ----
    def generate(self, prompt, max_chars, temperature=0.8, top_k=40, stop_on_user=True):
        import torch
        import torch.nn.functional as F

        ids = [self.stoi[c] for c in prompt if c in self.stoi]
        if not ids:
            return ""
        tags = (USER_TAG, ASSIST_TAG)
        # 回看窗口要比最长的标记多一格：判断「刚生成的字符会不会凑成标记」时，
        # 需要看到包括它在内的完整后缀，否则生成到「助手：」的冒号那一刻还看不到标记。
        lookback = max(len(t) for t in tags) + 1
        with self.lock:
            idx = torch.tensor([ids], dtype=torch.long, device=self.device)
            out_chars = []
            tail = ""
            for _ in range(max(1, int(max_chars))):
                cond = idx[:, -self.block:]
                with torch.no_grad():
                    logits = self.model(cond)[:, -1, :] / max(float(temperature), 1e-6)
                if top_k:
                    v, _ = torch.topk(logits, min(int(top_k), logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, 1)
                idx = torch.cat([idx, nxt], dim=1)
                ch = self.itos[int(nxt.item())]
                # 一旦凑出完整标记（含冒号）就停：模型开始自问自答了
                if stop_on_user and (tail + ch).endswith(tags):
                    break
                tail = (tail + ch)[-lookback:]
                out_chars.append(ch)
        return _clean_reply("".join(out_chars))


def _clean_reply(text):
    """收尾清理：模型常在结尾自问自答地写上「助手」，把残留标记去掉。
    它在很多样本里只写「助手」不带冒号，所以停止条件抓不到，必须在这里补一刀。"""
    t = text.strip()
    # 带冒号的标记是模型「开始自问自答」的信号，从它出现的位置直接截断
    cut = [i for i in (t.find(ASSIST_TAG), t.find(USER_TAG)) if i > 0]
    if cut:
        t = t[:min(cut)]
    while True:
        before = t
        t = t.rstrip()
        for tag in (ASSIST_TAG, USER_TAG):          # 结尾的完整标记
            if t.endswith(tag):
                t = t[:-len(tag)]
        t = re.sub(r"(?:\r?\n)*\s*(?:助手|用户)[：:]?\s*$", "", t)   # 结尾的裸标记
        t = t.strip()
        if t == before:
            return t


# ==================================================================
# HTTP
# ==================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "dsh-local-gpt/1.0"
    protocol_version = "HTTP/1.1"

    def _send(self, code, payload, ctype="application/json"):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, message, err_type="invalid_request_error", param=None):
        self._send(code, {"error": {"message": message, "type": err_type,
                                    "param": param, "code": None}})

    def log_message(self, fmt, *args):
        print(f"  [{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/") or "/"
        if path in ("/health", "/"):
            self._send(200, {"status": "ok", "model": ENG.args.served_name,
                             "device": ENG.device, "loaded": ENG.loaded})
        elif path == "/v1/models":
            self._send(200, {"object": "list", "data": [{
                "id": ENG.args.served_name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "local",
            }]})
        else:
            self._err(404, f"unknown path: {self.path}", "not_found")

    def do_POST(self):
        path = self.path.split("?")[0].rstrip("/")
        if path != "/v1/chat/completions":
            self._err(404, f"unknown path: {self.path}（本服务只实现 /v1/chat/completions）",
                      "not_found")
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
            req = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:
            self._err(400, f"请求体不是合法 JSON：{e}")
            return
        self.handle_chat(req)

    def handle_chat(self, req):
        # DSH 是 agent 框架，请求里默认会带 tools；而界面里没有「关掉工具」的开关。
        # 所以这里**不能报错**，只能忽略 tools 字段，回一段普通文本、不返回 tool_calls。
        # 想恢复「直接拒绝」的行为，启动时加 --reject-tools。
        if req.get("tools") or req.get("tool_choice") or req.get("functions"):
            if ENG.args.reject_tools:
                self._err(400,
                          "本模型不支持工具调用（tools / tool_choice / functions）",
                          "unsupported_feature", "tools")
                return
            if not ENG.warned_tools:
                ENG.warned_tools = True
                n_tools = len(req.get("tools") or [])
                print(f"  [提示] 本请求带了 {n_tools} 个工具定义，已忽略 —— "
                      f"服务只会返回普通文本，不会返回 tool_calls", flush=True)

        model_id = req.get("model") or ENG.args.served_name
        prompt, turns = ENG.build_prompt(req.get("messages"))
        if not prompt:
            self._err(400, "messages 里没有可用的用户消息", param="messages")
            return

        if ENG.args.debug:
            self._dump_debug(req, prompt, turns)

        temperature = req.get("temperature")
        temperature = ENG.args.temperature if temperature is None else float(temperature)
        top_k = req.get("top_k")
        top_k = ENG.args.top_k if top_k is None else int(top_k)
        max_chars = req.get("max_tokens") or req.get("max_completion_tokens") or ENG.args.max_chars
        # 字符级模型：把「token 数」当成字符数，并夹在合理区间里
        max_chars = max(1, min(int(max_chars), ENG.args.max_chars_cap))

        t0 = time.time()
        text = ENG.generate(prompt, max_chars, temperature, top_k,
                            stop_on_user=not ENG.args.no_stop_on_user)
        dt = time.time() - t0
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        print(f"  → 轮数 {turns} | 提示 {len(prompt)} 字 | 生成 {len(text)} 字 | {dt:.2f}s", flush=True)

        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model_id,
                       "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]})
            for i in range(0, len(text), 8):      # 按小块推，让客户端一点点显示
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                           "model": model_id,
                           "choices": [{"index": 0, "delta": {"content": text[i:i + 8]},
                                        "finish_reason": None}]})
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created,
                       "model": model_id,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
            self.wfile.write(b"data: [DONE]\n\n")
            try:
                self.wfile.flush()
            except Exception:
                pass
            return

        self._send(200, {
            "id": cid,
            "object": "chat.completion",
            "created": created,
            "model": model_id,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(prompt),
                "completion_tokens": len(text),
                "total_tokens": len(prompt) + len(text),
            },
        })

    def _sse(self, obj):
        try:
            self.wfile.write(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8"))
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _dump_debug(self, req, prompt, turns):
        """--debug 时把客户端发来的消息与我拼出的提示词打出来，便于排查"""
        print("  ┌─ 调试 ─────────────────────────────────────────", flush=True)
        for m in req.get("messages") or []:
            role = m.get("role")
            text = m.get("content")
            if isinstance(text, list):
                text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
            text = (text or "").replace("\n", "⏎")
            print(f"  │ {role:<10} {len(text):>5} 字 | {text[:160]}", flush=True)
        print(f"  │ 工具: {len(req.get('tools') or [])} 个 | stream={bool(req.get('stream'))} "
              f"| max_tokens={req.get('max_tokens') or req.get('max_completion_tokens')}", flush=True)
        print(f"  │ 拼出提示词（{turns} 轮 / {len(prompt)} 字）：{prompt.replace(chr(10), '⏎')}", flush=True)
        print("  └────────────────────────────────────────────────", flush=True)


ENG = None


def main():
    global ENG
    ap = argparse.ArgumentParser(description="把训练好的模型起成 OpenAI 兼容服务")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="模型 .pt 路径")
    ap.add_argument("--served-name", default=None,
                    help="对外暴露的模型名，默认用 .pt 文件名（不含扩展名）")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只绑本机")
    ap.add_argument("--port", type=int, default=49128, help="监听端口，默认 49128")
    ap.add_argument("--device", default=None, choices=["cpu", "cuda"],
                    help="推理设备，默认自动（有 CUDA 就用 CUDA）")
    ap.add_argument("--temperature", type=float, default=0.8, help="默认温度")
    ap.add_argument("--top-k", type=int, default=40, help="默认 top_k")
    ap.add_argument("--max-chars", type=int, default=180, help="默认单次生成多少字符")
    ap.add_argument("--max-chars-cap", type=int, default=600,
                    help="单次生成字符数的硬上限（客户端要再多也不给）")
    ap.add_argument("--no-stop-on-user", action="store_true",
                    help="不在生成出「用户：」时停止")
    ap.add_argument("--reject-tools", action="store_true",
                    help="带 tools 的请求直接返回 400（默认是忽略 tools 并正常回答）")
    ap.add_argument("--debug", action="store_true",
                    help="把客户端发来的消息与拼出的提示词打印出来（排查用）")
    a = ap.parse_args()
    a.model = os.path.abspath(a.model)
    if not a.served_name:
        a.served_name = os.path.splitext(os.path.basename(a.model))[0]

    print("=" * 62)
    print("  桥接服务：本地模型（OpenAI 兼容）")
    print("=" * 62)
    t0 = time.time()
    ENG = Engine(a.model, a.device)
    ENG.args = a
    print(f"[模型] {a.model}")
    print(f"[名称] {a.served_name} | 词表 {len(ENG.chars)} | 上下文 {ENG.block} 字 | 设备 {ENG.device}")
    print(f"[加载] 用时 {time.time() - t0:.1f}s")
    print(f"[模式] {'GPU' if ENG.device == 'cuda' else 'CPU'} —— "
          f"{'单次生成独占推理锁' if ENG.device == 'cuda' else 'CPU 推理，速度较慢'}")
    print("[工具] " + ("带 tools 的请求直接返回 400（--reject-tools）"
                       if a.reject_tools else
                       "忽略请求里的 tools，只回普通文本，不返回 tool_calls"))
    print()
    print(f"  接口     http://{a.host}:{a.port}/v1")
    print(f"  模型名   {a.served_name}（DSH 里填这个名字）")
    print(f"  健康检查 http://{a.host}:{a.port}/health")
    print("  停止服务：在这个窗口按 Ctrl+C")
    print("=" * 62)

    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    srv.daemon_threads = True
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[停止] 服务已退出")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
