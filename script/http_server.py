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
    python script\http_server.py --model model\chat_demo\chat_demo.pt --port 49128
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
import urllib.parse
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
        raise RuntimeError(f"找不到模型文件：{path}")

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
def read_model_props(path):
    r"""读 properties\<模型名>.properties 里的对话设置（改完不用重训就生效）"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stem = os.path.splitext(os.path.basename(path))[0]
    p = os.path.join(root, "properties", stem + ".properties")
    out = {}
    if not os.path.isfile(p):
        return out
    try:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError:
        return {}
    return out


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

        # 对话设置优先级：properties\<模型名>.properties > 模型文件里存的 > 服务启动参数
        cfg = self.cfg
        props = read_model_props(path)
        self.chat_mode = bool(cfg.get("chat_mode", True))
        if "chat_mode" in props:
            self.chat_mode = props["chat_mode"].strip().lower() in ("1", "true", "yes", "y", "on")
        self.defaults = {}
        if "chat_temperature" in cfg:
            try:
                self.defaults["temperature"] = float(cfg["chat_temperature"])
            except (TypeError, ValueError):
                pass
        if "chat_top_k" in cfg:
            try:
                self.defaults["top_k"] = int(cfg["chat_top_k"])
            except (TypeError, ValueError):
                pass
        if "chat_max_new" in cfg:
            try:
                self.defaults["max_chars"] = int(cfg["chat_max_new"])
            except (TypeError, ValueError):
                pass
        for prop_key, name, cast in (("chat_temperature", "temperature", float),
                                     ("chat_top_k", "top_k", int),
                                     ("chat_max_new", "max_chars", int)):
            if prop_key in props:
                try:
                    self.defaults[name] = cast(props[prop_key])
                except (TypeError, ValueError):
                    pass

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

        # 纯续写模型（诗词）：把用户输入当一段文本的开头，不加任何标记
        if not self.chat_mode:
            text = convo[-1][1]
            return text[-self.block:], 1

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
# 模型清单 + 懒加载池
# ==================================================================
def scan_models(dirs):
    r"""扫目录里的 .pt 权重，模型 id 用目录名（不含扩展名）。
    模型放在 model\<名字>\<名字>.pt；模型 id 优先用目录名，
    目录名和文件名不一致时才写成 目录/文件名，重名再加 @父目录 区分。"""
    found = {}
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for root, subdirs, names in os.walk(d):
            subdirs[:] = sorted(s for s in subdirs if not s.startswith("."))
            for fn in sorted(names):
                if not fn.lower().endswith(".pt"):
                    continue
                path = os.path.join(root, fn)
                if not os.path.isfile(path):
                    continue
                base = os.path.splitext(fn)[0]
                parent = os.path.basename(root.rstrip(os.sep))
                mid = base if base == parent else f"{parent}/{base}"
                if mid in found:
                    mid = f"{mid}@{parent}"
                k = 2
                while mid in found:
                    mid = f"{mid}@{parent}-{k}"
                    k += 1
                found[mid] = path
    return found


class ModelPool:
    """按需加载：/v1/models 只列文件名，不动显存；
    真正有对话请求进来才 torch.load，并做 LRU 淘汰（默认只留一个在显存里）。"""

    def __init__(self, registry, device, max_loaded=1, extra_args=None):
        self.registry = dict(registry)
        self.device = device
        self.max_loaded = max(1, int(max_loaded))
        self.args = extra_args
        self.lock = threading.Lock()
        self.loaded = {}          # id -> Engine
        self.order = []           # LRU，最近用的在末尾
        self.hits = {}            # id -> 调用次数

    # ---- 清单 ----
    def ids(self):
        return sorted(self.registry)

    def path_of(self, mid):
        return self.registry.get(mid)

    def resolve(self, mid):
        """把请求里的 model 字段解析成 registry 里的 id。
        支持：完整 id、去扩展名的文件名、文件路径；大小写不敏感。"""
        if not mid:
            return None
        mid = str(mid).strip()
        if mid in self.registry:
            return mid
        base = os.path.splitext(os.path.basename(mid))[0]
        low = base.lower()

        def leaf(k):
            # id 可能是 "名字" 或 "目录/名字"，也可能带 @后缀
            return k.split("@")[0].split("/")[-1]

        for k in self.registry:
            if k == base or leaf(k) == base:
                return k
        for k in self.registry:
            if k.lower() == base.lower() or leaf(k).lower() == low:
                return k
        return None

    # ---- 加载 / 淘汰 ----
    def get(self, mid):
        with self.lock:
            eng = self.loaded.get(mid)
            if eng is not None:
                self.order.remove(mid)
                self.order.append(mid)
                self.hits[mid] = self.hits.get(mid, 0) + 1
                return eng

            path = self.registry.get(mid)
            if not path:
                raise KeyError(mid)

            # 先腾地方：超出上限就把最久没用的卸掉
            while len(self.loaded) >= self.max_loaded and self.order:
                old = self.order.pop(0)
                print(f"[卸载] {old}（显存/内存留给 {mid}）", flush=True)
                self.loaded.pop(old, None)
                _free_memory()

            print(f"[加载] {mid} ← {path}", flush=True)
            t0 = time.time()
            eng = Engine(path, self.device)
            print(f"[加载] {mid} 完成：词表 {len(eng.chars)} | 上下文 {eng.block} 字 | "
                  f"{time.time() - t0:.1f}s", flush=True)
            self.loaded[mid] = eng
            self.order.append(mid)
            self.hits[mid] = self.hits.get(mid, 0) + 1
            return eng

    def status(self):
        return {
            "loaded": [{"id": m, "loaded_at": self.loaded[m].loaded,
                        "calls": self.hits.get(m, 0)} for m in self.order],
            "available": len(self.registry),
        }


def _free_memory():
    """卸模型后把缓存还给系统，避免下一个模型 OOM"""
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


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
            st = POOL.status()
            self._send(200, {"status": "ok", "models": st["available"],
                             "loaded": [x["id"] for x in st["loaded"]],
                             "device": POOL.device,
                             "default": ARGS.default_model})
        elif path == "/v1/models":
            data = []
            for mid in POOL.ids():
                p = POOL.path_of(mid)
                try:
                    created = int(os.path.getmtime(p))
                except OSError:
                    created = int(time.time())
                data.append({"id": mid, "object": "model", "created": created,
                             "owned_by": "local"})
            self._send(200, {"object": "list", "data": data})
        elif path.startswith("/v1/models/"):
            mid = path[len("/v1/models/"):]
            real = POOL.resolve(urllib.parse.unquote(mid))
            if not real:
                self._err(404, f"没有这个模型：{mid}", "model_not_found", "model")
                return
            p = POOL.path_of(real)
            self._send(200, {"id": real, "object": "model",
                             "created": int(os.path.getmtime(p)), "owned_by": "local"})
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
        global WARNED_TOOLS
        if req.get("tools") or req.get("tool_choice") or req.get("functions"):
            if ARGS.reject_tools:
                self._err(400,
                          "本模型不支持工具调用（tools / tool_choice / functions）",
                          "unsupported_feature", "tools")
                return
            if not WARNED_TOOLS:
                WARNED_TOOLS = True
                n_tools = len(req.get("tools") or [])
                print(f"  [提示] 本请求带了 {n_tools} 个工具定义，已忽略 —— "
                      f"服务只会返回普通文本，不会返回 tool_calls", flush=True)

        # ---- 按请求里的 model 选模型；第一次用到才加载 ----
        want = req.get("model") or ARGS.default_model
        mid = POOL.resolve(want)
        if not mid:
            avail = ", ".join(POOL.ids()) or "(没有任何 .pt)"
            self._err(404, f"没有这个模型：{want}（可用：{avail}）",
                      "model_not_found", "model")
            return
        try:
            eng = POOL.get(mid)
        except Exception as e:
            self._err(500, f"加载模型 {mid} 失败：{e}", "model_load_error", "model")
            return

        model_id = mid
        prompt, turns = eng.build_prompt(req.get("messages"))
        if not prompt:
            self._err(400, "messages 里没有可用的用户消息", param="messages")
            return

        if ARGS.debug:
            self._dump_debug(req, prompt, turns)

        temperature = req.get("temperature")
        if temperature is None:
            temperature = eng.defaults.get("temperature", ARGS.temperature)
        else:
            temperature = float(temperature)
        top_k = req.get("top_k")
        if top_k is None:
            top_k = eng.defaults.get("top_k", ARGS.top_k)
        else:
            top_k = int(top_k)
        max_chars = req.get("max_tokens") or req.get("max_completion_tokens")
        if max_chars is None:
            max_chars = eng.defaults.get("max_chars", ARGS.max_chars)
        # 字符级模型：把「token 数」当成字符数，并夹在合理区间里
        max_chars = max(1, min(int(max_chars), ARGS.max_chars_cap))

        t0 = time.time()
        text = eng.generate(prompt, max_chars, temperature, top_k,
                            stop_on_user=(eng.chat_mode and not ARGS.no_stop_on_user))
        dt = time.time() - t0
        cid = "chatcmpl-" + uuid.uuid4().hex[:24]
        created = int(time.time())
        print(f"  → [{model_id}] 轮数 {turns} | 提示 {len(prompt)} 字 | 生成 {len(text)} 字 | {dt:.2f}s", flush=True)

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


ARGS = None
POOL = None
WARNED_TOOLS = False


def main():
    global ARGS, POOL
    ap = argparse.ArgumentParser(description="把训练好的模型起成 OpenAI 兼容服务（支持多模型）")
    ap.add_argument("--model", action="append", default=[],
                    help="额外指定 .pt 文件（可写多次）；默认自动扫 model\\ 和 release\\")
    ap.add_argument("--model-dir", action="append", default=[],
                    help="额外的模型目录（可写多次）")
    ap.add_argument("--default-model", default=None,
                    help="请求没带 model 时用哪个；默认清单里第一个")
    ap.add_argument("--served-name", default=None,
                    help="给 --model 指定的那个文件起个别名（兼容旧用法）")
    ap.add_argument("--max-loaded", type=int, default=1,
                    help="同时留在显存里的模型数，默认 1（切模型就卸载上一个）")
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

    # ---- 收集模型清单 ----
    dirs = [os.path.join(ROOT, "model"), os.path.join(ROOT, "release")]
    dirs += [os.path.abspath(d) for d in (a.model_dir or [])]
    reg = scan_models(dirs)
    for p in (a.model or []):
        p = os.path.abspath(p)
        if not os.path.isfile(p):
            print(f"[错误] --model 指的文件不存在：{p}")
            sys.exit(1)
        base = os.path.splitext(os.path.basename(p))[0]
        mid = a.served_name if (a.served_name and len(a.model) == 1) else base
        k = 2
        while mid in reg and reg[mid] != p:
            mid = f"{base}-{k}"
            k += 1
        reg[mid] = p

    if not reg:
        print("[错误] 一个 .pt 都没找到。")
        print(f"       扫过的目录：{', '.join(dirs)}")
        print("       先训练一个：script\\train.bat，或把权重放进 model\\")
        sys.exit(1)

    ids = sorted(reg)
    default = a.default_model or ids[0]
    if default not in reg:
        base = os.path.splitext(os.path.basename(default))[0]
        hit = [k for k in ids if k == base or k.split("@")[0] == base]
        if not hit:
            print(f"[错误] --default-model 不在清单里：{default}（可用：{', '.join(ids)}）")
            sys.exit(1)
        default = hit[0]
    a.default_model = default

    if a.device is None:
        try:
            import torch
            a.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            a.device = "cpu"

    ARGS = a
    POOL = ModelPool(reg, a.device, a.max_loaded, a)

    print("=" * 68)
    print("  桥接服务：本地模型（OpenAI 兼容，多模型 + 懒加载）")
    print("=" * 68)
    print(f"[清单] 共 {len(ids)} 个模型（现在不加载，谁被调用才加载谁）")
    for mid in ids:
        p = reg[mid]
        try:
            mb = os.path.getsize(p) / 1e6
            mt = time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p)))
        except OSError:
            mb, mt = 0, "?"
        mark = "  ← 默认" if mid == default else ""
        print(f"       {mid:<24} {mb:>8.1f} MB  {mt}{mark}")
    print(f"[设备] {a.device} | 同时最多留 {a.max_loaded} 个模型在显存里")
    print(f"[工具] " + ("带 tools 的请求直接返回 400（--reject-tools）"
                       if a.reject_tools else
                       "忽略请求里的 tools，只回普通文本，不返回 tool_calls"))
    print()
    print(f"  接口     http://{a.host}:{a.port}/v1")
    print(f"  模型清单 http://{a.host}:{a.port}/v1/models")
    print(f"  健康检查 http://{a.host}:{a.port}/health")
    print("  停止服务：在这个窗口按 Ctrl+C")
    print("=" * 68)

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
