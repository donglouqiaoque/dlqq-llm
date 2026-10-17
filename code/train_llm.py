# -*- coding: utf-8 -*-
"""
从零训练一个字符级 GPT
=======================
这是 nanoGPT 那一量级的迷你模型，但包含了完整训练流程：
语料 → 字符级词表 → Transformer → 训练循环 → 采样生成。

支持任意大小的语料（实测 1 GB 级）：数据分块编码成二进制，训练时内存映射，
**内存占用与语料大小无关**，只跟 block_size / batch_size 有关。

用法（在工程根目录执行）：
    python code\\train_llm.py                                  # 用默认配置，自动下载 TinyShakespeare
    python code\\train_llm.py max_iters=500 n_layer=2 n_embd=64  # 命令行覆盖任意配置

用工程自带的演示语料（这是 script\\train.bat 实际执行的命令）：
    python code\\train_llm.py data_path=src\\chat_demo.txt out_path=model\\chat_demo.pt max_iters=3000

吃一整份大语料（txt 或 jsonl 都行，第一次运行会先花些时间切词，之后自动复用 cache）：
    python code\\train_llm.py data_path=ai_res\\doc\\firefly-train-1.1M.jsonl out_path=model\\firefly.pt ^
        max_vocab=8000 n_layer=8 n_head=8 n_embd=512 block_size=256 batch_size=32 max_iters=60000

jsonl 里会自动找问答字段（input/target、Q/A、question/answer、src/tgt …），
也可以用 q_field= 和 a_field= 显式指定。切词缓存放 ai_res\\cache\\。
"""
import os, sys, math, time, json, hashlib, urllib.request

try:
    import numpy as np
except ImportError:
    print("[错误] 缺少 numpy。装一个：python -m pip install numpy")
    sys.exit(1)

# Windows 控制台默认是 GBK，打印 ⚠️ / ← 这类字符会直接抛 UnicodeEncodeError。
# 把标准输出切成 UTF-8 并允许降级替换，避免训练跑完却在最后一屏崩掉。
# line_buffering：输出被重定向到文件 / IDE 时也逐行刷新，否则看起来像卡死。
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True, write_through=True)
    except Exception:
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================================================================
# 配置（直接改这里，或者命令行 key=value 覆盖）
# ==================================================================
CONFIG = dict(
    data_path   = None,      # None = 自动下载 TinyShakespeare；也可填自己的 txt
    block_size  = 128,       # 上下文长度（一次能看多少个字）
    n_layer     = 4,         # 层数
    n_head      = 4,         # 注意力头数
    n_embd      = 128,       # 隐藏维度（决定宽度）
    dropout     = 0.1,
    batch_size  = 32,
    max_iters   = 2000,      # 训练步数
    eval_interval = 200,
    eval_iters  = 50,
    lr          = 3e-4,
    weight_decay= 0.1,
    grad_clip   = 1.0,
    seed        = 1337,
    out_path    = "model/mymodel.pt",
    # ---- 大语料相关 ----
    max_vocab   = 8000,      # 词表上限：只保留最常出现的前 N 个字符
    val_frac    = 0.05,      # 从语料尾部切这个比例做验证集
    cache_dir   = "ai_res/cache",   # 切好的二进制缓存放这里（临时文件统一放 ai_res）
    prompt      = "",        # 训练结束采样用的开头；留空则自动挑
    # ---- 语料格式 ----
    data_format = "auto",    # auto / txt / jsonl；auto 按扩展名和文件开头判断
    q_field     = "",        # jsonl 里「问」的字段名；留空自动推断
    a_field     = "",        # jsonl 里「答」的字段名；留空自动推断
)

# ------------------------------------------------------------------
# 显存能开到多大 —— 参考值，不是硬限制（8 GB 笔记本显卡的实测档位）
# ------------------------------------------------------------------
#   档位       n_layer  n_head  n_embd  block  batch   参数量   显存占用   说明
#   tiny          2       2       64     64     16     ~0.2M    < 0.5G   几秒钟一个 epoch
#   small         4       4      128    128     32     ~0.8M    < 1G     默认，能看出效果
#   medium        6       6      384    256     64     ~10M     ~2-3G    语料够大时明显更好
#   large         8       8      512    256     32     ~25M     ~6-7G    8G 的上限附近，别再大
# ==================================================================

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def apply_cli_overrides():
    for arg in sys.argv[1:]:
        if "=" not in arg:
            continue
        k, v = arg.split("=", 1)
        if k not in CONFIG:
            print(f"[警告] 未知配置项 {k}，忽略"); continue
        cur = CONFIG[k]
        try:
            v2 = int(v) if isinstance(cur, int) else float(v) if isinstance(cur, float) else v
        except ValueError:
            v2 = v
        CONFIG[k] = v2
        print(f"[覆盖] {k} = {v2}")

apply_cli_overrides()      # ← 必须调用，否则命令行覆盖不生效

# ==================================================================
# 1) 数据：字符级词表 + 一次性切成二进制（内存占用与语料大小无关）
# ==================================================================
# 1 GB 中文大约 3.5 亿字。如果像以前那样先 read() 再 encode() 成 Python 列表，
# 光这一份就要吃掉十几 GB 内存，必然 OOM。所以这里改成：
#     流式统计字频 → 分块编码 → 写 uint16 二进制 → 训练时 np.memmap 随机读
# 峰值内存只取决于单个分块（几十 MB），跟语料是 1 MB 还是 1 GB 无关。
CHUNK_CHARS = 4_000_000        # 每次读多少字符（4M 字 ≈ 16 MB 的中间数组）

USER_TAG, ASSIST_TAG = "用户：", "助手："      # 和 code/chat.py 的对话模式保持一致

# jsonl 里常见的「问 / 答」字段名，按顺序尝试
FIELD_CANDIDATES = [
    ("input", "target"), ("q", "a"), ("question", "answer"),
    ("src", "tgt"), ("instruction", "output"), ("prompt", "response"),
    ("query", "response"), ("ask", "answer"), ("title", "content"),
]


def detect_encoding(path):
    """自动识别编码 —— Windows 记事本存出来的 txt 通常是 GBK，不是 UTF-8"""
    for enc in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            with open(path, encoding=enc) as f:
                f.read(2_000_000)
            return enc
        except UnicodeDecodeError:
            continue
    return None


def iter_chunks(path, enc, size=CHUNK_CHARS):
    """按字符数分块读，永远不会把整个文件读进内存"""
    with open(path, encoding=enc, errors="replace") as f:
        while True:
            s = f.read(size)
            if not s:
                return
            yield s


def detect_format(path):
    """auto 模式：看扩展名，再看文件开头是不是 { 或 ["""
    if CONFIG["data_format"] in ("txt", "jsonl"):
        return CONFIG["data_format"]
    if path.lower().endswith((".jsonl", ".ndjson", ".json")):
        return "jsonl"
    with open(path, "rb") as f:
        head = f.read(4096)
    if head.startswith(b"\xef\xbb\xbf"):        # 去掉 UTF-8 BOM 再判断
        head = head[3:]
    return "jsonl" if head.lstrip()[:1] in (b"{", b"[") else "txt"


def detect_fields(path, enc):
    """从 jsonl 的前几行推断问答字段名；q_field / a_field 显式给了就校验后直接用"""
    qf, af = CONFIG["q_field"], CONFIG["a_field"]

    with open(path, encoding=enc, errors="replace") as f:
        for _ in range(50):
            line = f.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            if line[:1] == "[":
                print("[数据] x 这是 JSON 数组，不是 JSONL（一行一条）")
                print("[数据]   用脚本转成一行一条再喂，或者 data_format=txt 自己拼成纯文本")
                sys.exit(1)
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            keys = list(obj.keys())
            if qf and af:
                # 显式指定了字段就校验一下，免得写错名字后默默丢掉整个语料
                if qf not in obj or af not in obj:
                    print(f"[数据] x 指定的字段 q_field={qf} a_field={af} 不在数据里")
                    print(f"[数据]   这一行的字段是：{keys}")
                    sys.exit(1)
                return qf, af
            lower = {k.lower(): k for k in keys}
            for cq, ca in FIELD_CANDIDATES:
                if cq in lower and ca in lower:
                    return lower[cq], lower[ca]
            strs = [k for k in keys if isinstance(obj.get(k), str)]
            if len(strs) >= 2:
                return strs[0], strs[1]
            print(f"[数据] x 这一行的字段是 {keys}，找不到可用的问答字段")
            print("[数据]   用 q_field=字段名 a_field=字段名 显式指定")
            sys.exit(1)

    print("[数据] x 前 50 行都解析不出 JSON 对象，确认这是 JSONL 文件")
    print("[数据]   如果其实是纯文本，加 data_format=txt")
    sys.exit(1)


def iter_text_chunks(path, enc, fmt, qf, af, stats=None):
    """按块产出纯文本。
    txt  : 原样分块读。
    jsonl: 逐行解析，拼成「用户：…／助手：…」—— 直接喂 jsonl 会把括号引号学进文法里。
    stats: 可选 dict，用来回传 (处理条数, 坏行数)。
    """
    if fmt == "txt":
        yield from iter_chunks(path, enc)
        return

    buf, buf_len = [], 0
    n_ok = n_bad = 0
    with open(path, encoding=enc, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                q = str(obj.get(qf, "") or "").strip()
                a = str(obj.get(af, "") or "").strip()
            except Exception:
                n_bad += 1
                continue
            if not q or not a:
                n_bad += 1
                continue
            buf.append(f"{USER_TAG}{q}\n{ASSIST_TAG}{a}\n")
            buf_len += len(q) + len(a) + 8
            n_ok += 1
            if buf_len >= CHUNK_CHARS:
                if stats is not None:
                    stats["ok"], stats["bad"] = n_ok, n_bad
                yield "".join(buf)
                buf, buf_len = [], 0
    if buf:
        yield "".join(buf)
    if stats is not None:
        stats["ok"], stats["bad"] = n_ok, n_bad


def resolve_corpus_path():
    """确定语料文件：data_path 优先；没指定就下载 TinyShakespeare 存到 cache"""
    p = CONFIG["data_path"]
    if p:
        if os.path.isfile(p):
            return p
        print(f"[数据] x 找不到文件：{p}")
        print(f"[数据]   当前工作目录：{os.getcwd()}")
        print('[数据]   路径带反斜杠的话，用引号包起来试试，例如 data_path="E:/我的文本.txt"')
        print("=" * 62)
        print("  没有语料，训练无法开始。")
        print("=" * 62)
        sys.exit(1)

    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    dst = os.path.join(CONFIG["cache_dir"], "tinyshakespeare.txt")
    if os.path.isfile(dst):
        print(f"[数据] 用已下载的 {dst}")
        return dst
    try:
        print(f"[数据] 下载 {URL}")
        raw = urllib.request.urlopen(URL, timeout=20).read()
        with open(dst, "wb") as f:
            f.write(raw)
        print(f"[数据] 下载完成 {len(raw):,} 字节 → {dst}")
        return dst
    except Exception as e:
        print(f"[数据] 下载失败（{e}）")
        print("=" * 62)
        print("  没有语料，训练无法开始。二选一：")
        print("    1) 检查网络后重跑（要下 TinyShakespeare，约 1.1 MB）")
        print("    2) 用本地语料：  data_path=你的语料.txt")
        print("=" * 62)
        sys.exit(1)


def count_char_freq(path, enc, fmt, qf, af):
    """流式统计每个字符出现多少次 —— 用 numpy 一次算一整块，比 Python 循环快几百倍"""
    counts = np.zeros(0x110000, dtype=np.int64)
    done, t0 = 0, time.time()
    for s in iter_text_chunks(path, enc, fmt, qf, af):
        cps = np.frombuffer(s.encode("utf-32-le", errors="ignore"), dtype="<u4")
        counts += np.bincount(cps, minlength=0x110000)
        done += len(s)
        print(f"\r[词表] 已扫描 {done:,} 字符 | {time.time() - t0:5.1f}s", end="", flush=True)
    print()
    return counts


def build_vocab(counts, max_vocab):
    """按出现次数取前 max_vocab 个字符；换行/制表保留，其它控制字符丢掉。
    限制词表大小是必须的：GB 级语料里会混进几万个生僻字和符号，
    词表一大，输出层参数就爆炸，而且那些字在训练里根本学不好。"""
    idx = np.nonzero(counts)[0]
    keep = idx[(idx >= 32) | (idx == 10) | (idx == 9)]
    if len(keep) == 0:
        return []
    order = keep[np.argsort(counts[keep])[::-1]]
    if max_vocab and len(order) > max_vocab:
        order = order[:max_vocab]
    return [chr(int(i)) for i in order]


def tokenize_corpus(path, enc, fmt, qf, af, chars, bin_path):
    """把语料编码成 uint16 序列写进二进制文件；词表外的字直接丢掉。
    返回 (写出的 token 数, 被丢掉的字符数, 跳过的坏行数)。"""
    codes = np.array([ord(c) for c in chars], dtype=np.uint32)
    order = np.argsort(codes)
    codes_sorted = codes[order]                 # 升序，供 searchsorted 用
    ids_of_sorted = order.astype(np.uint16)     # 排序位置 → chars 里的原 id

    stats = {"ok": 0, "bad": 0}
    total = dropped = 0
    t0 = time.time()
    with open(bin_path, "wb") as out:
        for s in iter_text_chunks(path, enc, fmt, qf, af, stats):
            cps = np.frombuffer(s.encode("utf-32-le", errors="ignore"), dtype="<u4")
            pos = np.searchsorted(codes_sorted, cps)
            np.clip(pos, 0, len(codes_sorted) - 1, out=pos)
            hit = codes_sorted[pos] == cps      # 真的在词表里（不是被 clip 挤到一起的）
            ids = ids_of_sorted[pos[hit]]
            out.write(ids.tobytes())
            total += len(ids)
            dropped += len(cps) - len(ids)
            msg = f"\r[编码] 已写出 {total:,} token"
            if fmt == "jsonl":
                msg += f" | 已处理 {stats['ok']:,} 条"
            print(f"{msg} | {time.time() - t0:5.1f}s", end="", flush=True)
    print()
    return total, dropped, stats["bad"], stats["ok"]


def prepare_data():
    """准备训练数据：优先复用 cache；语料或词表设置变了就重建。
    返回 (chars, bin 路径, 训练 token 数, 总 token 数)。"""
    path = resolve_corpus_path()
    enc = detect_encoding(path)
    if enc is None:
        print("[数据] x utf-8 / gbk / big5 都解不出来，可能不是文本文件")
        sys.exit(1)

    fmt = detect_format(path)
    if fmt == "jsonl":
        qf, af = detect_fields(path, enc)
    else:
        qf = af = ""

    size, mtime = os.path.getsize(path), int(os.path.getmtime(path))
    os.makedirs(CONFIG["cache_dir"], exist_ok=True)
    want = dict(source=os.path.abspath(path), size=size, mtime=mtime,
                max_vocab=CONFIG["max_vocab"], val_frac=CONFIG["val_frac"],
                format=fmt, q_field=qf, a_field=af)

    # 缓存文件名带上「语料 + 词表设置」的指纹：
    # 这样 demo 的 txt 和 GB 级 jsonl 可以各存一份，来回切换不用重切词。
    tag = hashlib.md5(json.dumps(want, sort_keys=True).encode("utf-8")).hexdigest()[:10]
    bin_path = os.path.join(CONFIG["cache_dir"], f"data-{tag}.bin")
    meta_path = os.path.join(CONFIG["cache_dir"], f"meta-{tag}.json")

    meta = None
    if os.path.isfile(bin_path) and os.path.isfile(meta_path):
        try:
            old = json.load(open(meta_path, encoding="utf-8"))
            if all(old.get(k) == v for k, v in want.items()) and old.get("total"):
                meta = old
        except Exception:
            meta = None

    if meta:
        print(f"[数据] 复用缓存 {bin_path}（这份语料切过词了，跳过）")
    else:
        print(f"[数据] 语料 {path}")
        print(f"[数据] 格式 {fmt} | 编码 {enc} | {size / 1e6:.1f} MB")
        if fmt == "jsonl":
            print(f"[数据] 问答字段 {qf} → {af}，会拼成「{USER_TAG}…／{ASSIST_TAG}…」")
        counts = count_char_freq(path, enc, fmt, qf, af)
        chars = build_vocab(counts, CONFIG["max_vocab"])
        if not chars:
            print("[数据] x 语料里没有任何可用字符")
            sys.exit(1)
        print(f"[词表] 保留 {len(chars)} 个字符（上限 {CONFIG['max_vocab']}）")
        total, dropped, bad, ok = tokenize_corpus(path, enc, fmt, qf, af, chars, bin_path)
        if fmt == "jsonl":
            print(f"[数据] 共解析 {ok:,} 条问答" + (f"，跳过 {bad:,} 行（空/坏/字段缺）" if bad else ""))
        if total < 1000:
            print(f"[数据] x 编码后只剩 {total} 个 token，太少，检查语料或字段设置")
            sys.exit(1)
        n_train = int(total * (1 - CONFIG["val_frac"]))
        meta = dict(want, chars=chars, total=total, n_train=n_train,
                    dropped=dropped, built=time.strftime("%Y-%m-%d %H:%M:%S"))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        print(f"[数据] 编码完成 → {bin_path}"
              f"（{os.path.getsize(bin_path) / 1e6:.1f} MB，以后直接用这份）")

    chars = meta["chars"]
    total, n_train = meta["total"], meta["n_train"]
    dropped = meta.get("dropped", 0)
    print(f"[数据] 词表 {len(chars)} | 总量 {total:,} token | 训练 {n_train:,}"
          f" | 验证 {total - n_train:,}"
          + (f" | 词表外丢弃 {dropped:,} 字" if dropped else ""))
    if total < 100_000:
        print(f"[警告] 语料只有 {total:,} token，偏小 —— 模型会很快背下来（过拟合）。")
        print("       建议至少 10 万 token；想看出泛化，100 万以上更好。")
    return chars, bin_path, n_train, total


chars, DATA_BIN, N_TRAIN, N_TOTAL = prepare_data()
VOCAB = len(chars)
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s if c in stoi]      # 词表外的字跳过
decode = lambda ids: "".join(itos[i] for i in ids)

# 内存映射：不加载进内存，操作系统按需把用到的页换进来
train_data = np.memmap(DATA_BIN, dtype=np.uint16, mode="r",
                       offset=0, shape=(N_TRAIN,))
val_data = np.memmap(DATA_BIN, dtype=np.uint16, mode="r",
                     offset=N_TRAIN * 2, shape=(N_TOTAL - N_TRAIN,))
print(f"[数据] 已内存映射 {DATA_BIN}（不会占用等量内存）")

# 语料太短的话 block_size 会自动缩小 —— 否则取随机起点时会拿到负数范围
_safe_T = min(len(train_data), len(val_data)) - 2
if CONFIG["block_size"] > _safe_T:
    print(f"[警告] 语料太小（训练 {len(train_data)} / 验证 {len(val_data)} token），"
          f"block_size 从 {CONFIG['block_size']} 自动降到 {max(8, _safe_T)}")
    CONFIG["block_size"] = max(8, _safe_T)

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[设备] {DEVICE}" + (f" — {torch.cuda.get_device_name(0)}" if DEVICE == "cuda" else ""))
USE_AMP = DEVICE == "cuda"          # GPU 上用 bf16 混合精度；CPU 上用会很慢（见之前的实测）

def get_batch(split):
    """从内存映射里随机取一批片段 —— 一次只读几 KB，语料多大都一样快"""
    d = train_data if split == "train" else val_data
    B, T = CONFIG["batch_size"], CONFIG["block_size"]
    if len(d) <= T + 1:
        raise RuntimeError(
            f"[{split}] 可用 token 只剩 {len(d)} 个，装不下 block_size={T} 的一段。"
            f"请把 block_size 调小，或者换更大的语料。")
    ix = np.random.randint(0, len(d) - T - 1, size=B)
    x = np.stack([d[i:i + T] for i in ix]).astype(np.int64)
    y = np.stack([d[i + 1:i + T + 1] for i in ix]).astype(np.int64)   # 目标是「下一个字符」
    return (torch.from_numpy(x).to(DEVICE),
            torch.from_numpy(y).to(DEVICE))

# ==================================================================
# 2) 模型：一个精简的 GPT（Pre-Norm Transformer Decoder）
# ==================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head, self.hs = n_head, n_embd // n_head
        self.qkv  = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = dropout

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B,T,C) -> (B, n_head, T, head_size)
        q = q.view(B, T, self.n_head, self.hs).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.hs).transpose(1, 2)
        # is_causal=True 自动做因果掩码（不能看未来）；GPU 上会走 Flash Attention
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.drop if self.training else 0.0)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
            nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Pre-Norm + 残差"""
    def __init__(self, n_embd, n_head, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.mlp  = MLP(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

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
        self.head.weight = self.tok_emb.weight        # 权重共享，省参数
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.drop(self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device)))
        x = self.blocks(x)
        logits = self.head(self.ln_f(x))              # (B,T,V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B*T, -1), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]          # 超出上下文就截断
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx

model = GPT(VOCAB, CONFIG["block_size"], CONFIG["n_layer"],
            CONFIG["n_head"], CONFIG["n_embd"], CONFIG["dropout"]).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"[模型] 参数量 {n_params/1e6:.2f} M")

# ==================================================================
# 3) 优化器：2 维以上参数才加 weight decay（标准做法）
# ==================================================================
decay, no_decay = [], []
for p in model.parameters():
    if not p.requires_grad: continue
    (decay if p.dim() >= 2 else no_decay).append(p)
opt = torch.optim.AdamW([{"params": decay,    "weight_decay": CONFIG["weight_decay"]},
                         {"params": no_decay, "weight_decay": 0.0}], lr=CONFIG["lr"])

@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(CONFIG["eval_iters"])
        for i in range(CONFIG["eval_iters"]):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                _, loss = model(*get_batch(split))
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

# ==================================================================
# 4) 训练循环
# ==================================================================
print("[训练] 开始")
# 输出目录不存在就先建出来，否则 torch.save 会在中途报错
_out_dir = os.path.dirname(os.path.abspath(CONFIG["out_path"]))
if _out_dir:
    os.makedirs(_out_dir, exist_ok=True)

best_val = float("inf")
saved = False
t0 = time.time()
for it in range(CONFIG["max_iters"] + 1):
    if it % CONFIG["eval_interval"] == 0:
        L = estimate_loss()
        mark = ""
        if L["val"] < best_val:
            best_val = L["val"]
            torch.save({"model": model.state_dict(), "config": CONFIG,
                        "chars": chars}, CONFIG["out_path"])
            saved = True
            mark = "  ← 已保存"
        print(f"  step {it:>5} | train {L['train']:.4f} | val {L['val']:.4f}"
              f" | 困惑度 {math.exp(L['val']):>7.1f} | {time.time()-t0:>6.1f}s{mark}")
    xb, yb = get_batch("train")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
        _, loss = model(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
    opt.step()

print(f"[训练] 完成，用时 {time.time()-t0:.1f}s，最好验证损失 {best_val:.4f}")

# ==================================================================
# 5) 采样：让它自己往下写
# ==================================================================
model.eval()
prompt = CONFIG["prompt"] or ("用户：" if all(c in stoi for c in "用户：") else chars[0])
ids = encode(prompt) or [0]                        # 万一开头全是不认识的字，退到 id 0
idx = torch.tensor([ids], dtype=torch.long, device=DEVICE)
out = model.generate(idx, max_new_tokens=300, temperature=0.8, top_k=40)
print("\n===== 模型生成 =====")
if N_TOTAL < 5000:
    print("（注意：语料不到 5000 token，所以下面这些它基本是【原样背出来】的）")
print()
print(decode(out[0].tolist()))
print("\n====================")
if saved:
    print(f"模型已保存到 {CONFIG['out_path']}")
else:
    print("训练过程中验证损失一次都没有改善，没有保存模型。")
