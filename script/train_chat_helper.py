# -*- coding: utf-8 -*-
r"""交互式入口（供 script\train.bat 与 script\chat.bat 调用）
================================================================
本文件放在 script\ 下，与几个 bat 同级；工程根目录靠向上查找
code\train_llm.py 得到，所以脚本换位置也不会失效。

职责：
    · 挑一个装了 torch 的 python；依赖缺失就直接终止，并让用户去跑自检脚本；
    · 打印运行模式（CPU / GPU 单精度 / GPU 混合精度）与占用的显存或内存；
    · 训练：列出 src\ 下的语料让用户选（默认 src\chat_demo.jsonl），模型名称由用户现场输入；
    · 对话：模型默认 model\chat_demo.pt（缺失即终止），也可输编号或名称换一个。

模型名称只存在于本次运行的变量里，不写进任何文件，也不回显。
"""
import os
import re
import sys
import glob
import time
import shutil
import argparse
import subprocess

for _s in ("stdout", "stderr"):
    try:
        # line_buffering：输出重定向到文件 / IDE 时也逐行刷新，
        # 否则训练日志会憋在缓冲区里，看起来像卡死
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True, write_through=True)
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))


def find_root(start):
    """往上找到含 code\\train_llm.py 的那一级，作为工程根目录（本脚本搬家也不会失效）"""
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
CODE = os.path.join(ROOT, "code")

_DEMO_JSONL = os.path.join(ROOT, "src", "chat_demo.jsonl")
# demo 语料只有考题库这一份（由 script\fetch_exam_data.bat 生成）
DEFAULT_CORPUS = _DEMO_JSONL
DEFAULT_MODEL = os.path.join(ROOT, "model", "chat_demo.pt")

# 配置文件所在目录（可放多份，train.bat 会让用户按编号选）
CONFIG_DIR = os.path.join(ROOT, "properties")

# 这些键会原样传给 code/train_llm.py（它的 CONFIG 里都有同名项）
TRAIN_KEYS = ("n_layer", "n_head", "n_embd", "block_size", "batch_size",
              "max_iters", "eval_interval", "eval_iters", "lr", "dropout",
              "weight_decay", "grad_clip", "seed", "max_vocab", "val_frac",
              "cache_dir", "grad_accum", "warmup_iters", "lr_decay",
              "min_lr_frac", "grad_checkpoint", "resume", "save_every",
              "save_dtype", "save_optimizer")

# 训练档位：命令行 -p 指定（配置文件里一般直接写满各项，不用档位）
PROFILES = {
    "tiny":   dict(n_layer=2, n_head=2, n_embd=64,  block_size=64,  batch_size=16, max_iters=2000),
    "small":  dict(n_layer=4, n_head=4, n_embd=128, block_size=128, batch_size=32, max_iters=3000),
    "medium": dict(n_layer=6, n_head=6, n_embd=384, block_size=256, batch_size=64, max_iters=8000),
    "large":  dict(n_layer=8, n_head=8, n_embd=512, block_size=256, batch_size=32, max_iters=8000),
}

# 档位说明（挑档位时显示给用户看）
PROFILE_HINT = {
    "tiny":   "2 层 / 64 维    约 0.2 M 参数   最快，用来验证流程",
    "small":  "4 层 / 128 维   约 0.9 M 参数   默认",
    "medium": "6 层 / 384 维   约 10 M 参数    语料够大时明显更好",
    "large":  "8 层 / 512 维   约 33 M 参数    8 GB 显存的上限附近",
}

CHECK_HINT = "请先运行 script\\torch_check.bat 检查运行环境"


# ==================================================================
# 找一个装了 torch 的 python（库缺失就终止，不做自检）
# ==================================================================
REQUIRED = ("torch", "torch.nn", "torch.nn.functional", "numpy")


def missing_modules(exe):
    """返回缺失/装不起来的模块名；空列表 = 依赖齐全。
    必须真的 import 一次：find_spec 对 torch.nn 这类子模块判定不可靠。"""
    code = ("import importlib\n"
            "bad = []\n"
            "for m in %r:\n"
            "    try:\n"
            "        importlib.import_module(m)\n"
            "    except Exception as e:\n"
            "        bad.append('%%s(%%s)' %% (m, type(e).__name__))\n"
            "print('MISSING:' + ','.join(bad))\n" % (REQUIRED,))
    try:
        r = subprocess.run([exe, "-c", code], capture_output=True, text=True, timeout=600)
    except Exception as e:
        return [f"探测失败({type(e).__name__})"]
    if r.returncode != 0:
        return ["torch(导入失败)"]
    for line in (r.stdout or "").splitlines():
        if line.startswith("MISSING:"):
            return [m for m in line[8:].split(",") if m]
    return ["torch"]


def fmt_missing(miss):
    """把 ['torch(ModuleNotFoundError)', 'torch.nn(...)'] 收拾成 'torch'"""
    roots = []
    for m in miss or []:
        root = m.split("(")[0].split(".")[0].strip()
        if root and root not in roots:
            roots.append(root)
    return "、".join(roots) or "torch"


def resolve_python(explicit):
    """决定用哪个解释器，返回 (exe, 缺失模块)；失败返回 (None, 缺失模块)。

    · explicit 非空（配置文件 / 环境变量指定）→ 只认它：
        不存在、是目录、缺 torch/numpy 都直接报错，**绝不去找别的解释器**。
    · explicit 为空 → 用 PATH 上的 python；它缺依赖也直接报错，不换别的。
    """
    if explicit:
        exe = path_from_cfg(explicit)
        if not os.path.isfile(exe):
            print(f"[错误] 配置里指定的 python 不存在：{exe}")
            print("       改 properties 里的 python= 那一行（要指向 python.exe），或者留空用 PATH 上的 python。")
            return None, ["python 路径无效"]
        miss = missing_modules(exe)
        if miss:
            print(f"[错误] 配置里指定的 python 缺少依赖：{fmt_missing(miss)}")
            print(f"       {exe}")
            print(f"       装一下：\"{exe}\" -m pip install torch numpy")
            return None, miss
        print(f"[环境] 用配置指定的 python：{exe}")
        return exe, []

    exe = None
    for name in ("python", "python3", "py"):
        exe = shutil.which(name)
        if exe:
            break
    if not exe:
        print("[错误] PATH 上找不到 python（python / python3 / py 都没有）。")
        print("       装一个，或者在 properties 里用 python= 指定完整路径。")
        return None, ["python"]
    miss = missing_modules(exe)
    if miss:
        print(f"[错误] PATH 上的 python 缺少依赖：{fmt_missing(miss)}")
        print(f"       {exe}")
        print(f"       装一下：\"{exe}\" -m pip install torch numpy")
        return None, miss
    return exe, []


def find_python():
    """训练/对话用：不带显式指定时等价于 resolve_python(None)"""
    return resolve_python(None)


def abort_missing_libs(miss):
    print("[错误] 运行环境缺少依赖：" + "、".join(miss or ["torch"]))
    print(f"       {CHECK_HINT}。")
    return 1


# ==================================================================
# 运行模式：CPU / GPU 单精度 / GPU 混合精度（bf16），并报占用
# ==================================================================
MODE_PROBE = r'''
import ctypes, sys

def system_ram():
    """返回 (总量GB, 已用GB)；取不到就返回 (None, None)"""
    if sys.platform == "win32":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        st = MEMORYSTATUSEX(); st.dwLength = ctypes.sizeof(st)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            t = st.ullTotalPhys / 1024**3
            u = (st.ullTotalPhys - st.ullAvailPhys) / 1024**3
            return t, u
    else:
        try:
            info = {}
            for line in open("/proc/meminfo"):
                k, _, v = line.partition(":")
                info[k.strip()] = int(v.split()[0])
            t = info["MemTotal"] / 1024**2
            a = info.get("MemAvailable", info["MemFree"]) / 1024**2
            return t, t - a
        except Exception:
            pass
    return None, None

lines = []
try:
    import torch
except Exception as e:
    print("NO_TORCH=" + type(e).__name__)
    raise SystemExit(0)

if torch.cuda.is_available():
    import subprocess, shutil
    p = torch.cuda.get_device_properties(0)
    total = (getattr(p, "total_memory", 0) or getattr(p, "total_mem", 0)) / 1024**3
    used = None
    exe = shutil.which("nvidia-smi")
    if exe:
        try:
            q = subprocess.run([exe, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=20)
            used = int(q.stdout.strip().splitlines()[0]) / 1024
        except Exception:
            used = None
    if used is None:
        used = torch.cuda.memory_allocated(0) / 1024**3
    lines.append("DEVICE=GPU")
    lines.append("MODE=" + "混合精度 bf16（推荐：显存占用更低、张量核更快）")
    lines.append("GPU=" + ("%s | sm_%d%d" % (p.name, p.major, p.minor)))
    lines.append("MEMKIND=显存")
    lines.append("MEM=%.2f|%.2f" % (total, used))
else:
    lines.append("DEVICE=CPU")
    lines.append("MODE=" + "单精度 fp32（没用上显卡）")
    lines.append("GPU=" + "无可用 CUDA 设备")
    t, u = system_ram()
    if t:
        lines.append("MEMKIND=内存")
        lines.append("MEM=%.2f|%.2f" % (t, u))
print("\n".join(lines))
'''


def probe_device(exe):
    try:
        r = subprocess.run([exe, "-c", MODE_PROBE], capture_output=True, text=True, timeout=600)
    except Exception:
        return None
    out = (r.stdout or "").strip()
    if not out or out.startswith("NO_TORCH"):
        return None
    d = {}
    for line in out.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            d[k.strip()] = v.strip()
    return d


def fmt_mem(pair):
    try:
        total, used = (float(x) for x in pair.split("|")[:2])
    except Exception:
        return None
    return "总 %.1f GB，已用 %.1f GB（可用 %.1f GB）" % (total, used, max(total - used, 0.0))


def print_mode(exe):
    """打印运行模式与占用；返回 True 表示用 GPU"""
    d = probe_device(exe)
    if not d:
        print("[模式] 读不到设备信息（torch 探测失败）")
        return False
    print(f"[模式] {d.get('DEVICE', '?')} —— {d.get('MODE', '?')}")
    print(f"[设备] {d.get('GPU', '?')}")
    mem = fmt_mem(d.get("MEM", ""))
    if mem:
        print(f"[{d.get('MEMKIND', '内存')}] {mem}")
    return d.get("DEVICE") == "GPU"


# ==================================================================
# 输入辅助
# ==================================================================
_NOECHO_STATE = {}


def _echo_off():
    """关掉控制台回显 —— 让输入的模型名称不出现在屏幕上（失败就算了）"""
    if os.name != "nt":
        return False
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-10)                 # STD_INPUT_HANDLE
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False                       # 不是真控制台（管道/重定向）
        if not k.SetConsoleMode(h, mode.value & ~0x0004):   # 清掉 ENABLE_ECHO_INPUT
            return False
        _NOECHO_STATE["handle"] = (k, h, mode.value)
        return True
    except Exception:
        return False


def _echo_on():
    st = _NOECHO_STATE.pop("handle", None)
    if not st:
        return
    try:
        k, h, mode = st
        k.SetConsoleMode(h, mode)
    except Exception:
        pass


def ask(prompt, default=None, quiet=False):
    """quiet=True 时不回显输入内容（用于模型名称这类不想留在屏幕上的输入）
    返回值：用户输入的字符串；直接回车 → default；输入流结束（EOF）→ None。"""
    tip = f"{prompt}" + (f"（直接回车 = {default}）" if default else "") + "："
    muted = quiet and _echo_off()
    try:
        s = input(tip).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return None                      # EOF 必须和「直接回车」区分开，否则会死循环
    finally:
        if muted:
            _echo_on()
    if not s:
        return default
    return s


def safe_name(name):
    """模型名只允许普通文件名，防止 ..\\ 之类跑出目录"""
    name = (name or "").strip().strip('"').strip("'")
    if not name:
        return None
    for ch in '\\/:*?"<>|':
        if ch in name:
            return None
    if name in (".", ".."):
        return None
    return name


def short(path):
    """尽量显示成工程内的相对路径；解释器在别的盘上时退回绝对路径"""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return path


def run(argv):
    """直接跑，共用同一个控制台（stdio 不接管，中文输入输出都正常）"""
    shown = [short(argv[0])]
    for a in argv[1:]:
        # 输出路径只显示目录，不回显用户输入的模型名称
        shown.append(os.path.dirname(a) + "\\<...>.pt" if a.endswith(".pt") else a)
    print("> " + " ".join(shown))
    try:
        return subprocess.call(argv, cwd=ROOT)
    except KeyboardInterrupt:
        print("\n[中断]")
        return 130


# ==================================================================
# 训练
# ==================================================================
def list_corpora():
    r"""列出 src\ 下可用于训练的语料（txt / jsonl），按文件名排序"""
    listing = []
    for pat in ("*.txt", "*.jsonl", "*.json"):
        listing += glob.glob(os.path.join(ROOT, "src", pat))
    return sorted(set(os.path.abspath(p) for p in listing))


def pick_corpus(given):
    """决定用哪份语料：命令行给了就用它，否则列出来让用户选。
    返回 (路径, 错误信息)。"""
    # ---- 命令行直接指定 ----
    if given:
        cand = given
        if not os.path.isabs(cand):
            cand = os.path.join(ROOT, cand)
        if not os.path.isfile(cand):
            # 允许只写文件名不带目录
            alt = os.path.join(ROOT, "src", given)
            if os.path.isfile(alt):
                return alt, None
            print(f"[错误] 找不到语料：{given}")
            print(f"       试过的路径：{short(cand)}")
            return None, "missing"

    listing = list_corpora()
    if not listing:
        print(f"[错误] src\\ 下没有任何语料（.txt / .jsonl）")
        print(f"       把语料放进 {short(os.path.join(ROOT, 'src'))} 再重跑。")
        return None, "empty"

    # ---- 交互选择：直接回车 = chat_demo.jsonl（没有就第一个）----
    default = next((p for p in listing
                    if os.path.basename(p) == os.path.basename(DEFAULT_CORPUS)), listing[0])
    print("可用语料：")
    for i, p in enumerate(listing, 1):
        size = os.path.getsize(p) / 1e6
        mark = "  ← 默认" if p == default else ""
        print(f"  [{i}] {os.path.basename(p):<28} {size:>9.1f} MB{mark}")
    raw = ask("请输入编号或文件名（直接回车用默认语料）", default="")
    if raw is None:
        print("[取消] 输入已结束")
        return None, "eof"
    if not raw:
        return default, None
    if raw.isdigit() and 1 <= int(raw) <= len(listing):
        return listing[int(raw) - 1], None
    hit = [p for p in listing if os.path.basename(p).lower() == raw.lower()
           or os.path.basename(p).lower() == raw.lower() + ".txt"]
    if hit:
        return hit[0], None
    cand = raw if os.path.isabs(raw) else os.path.join(ROOT, raw)
    if os.path.isfile(cand):
        return os.path.abspath(cand), None
    print(f"[错误] 找不到这个语料：{raw}")
    return None, "missing"


# ==================================================================
# 配置文件（properties/key=value）
# ==================================================================
def list_configs():
    """properties\\ 下的配置文件（另加 script\\settings.properties，若存在）"""
    files = sorted(glob.glob(os.path.join(CONFIG_DIR, "*.properties")))
    extra = os.path.join(HERE, "settings.properties")
    if os.path.isfile(extra):
        files = [extra] + [f for f in files if os.path.abspath(f) != os.path.abspath(extra)]
    return files


def pick_config_interactive():
    """列出配置让用户按编号选，返回绝对路径。
    列表和提示都走 stdout（同一路输出，顺序才是对的）；结果另由 --pick-out 落文件。"""
    files = list_configs()
    if not files:
        print(f"[错误] properties\\ 下没有配置文件：{CONFIG_DIR}")
        print("       放一个 *.properties 进去，例如 properties\\chat_demo.properties")
        return None

    print("可用配置：")
    for i, p in enumerate(files, 1):
        size = os.path.getsize(p)
        print(f"  [{i}] {os.path.basename(p):<24} {size:>6} 字节")
    sys.stdout.flush()

    for _ in range(3):
        raw = ask("请输入编号（直接回车 = 1）", default="")
        if raw is None:
            print("\n[错误] 读不到键盘输入（当前终端没有可交互的 stdin）。")
            print("       两个办法：")
            print("         1) 直接双击 script\\train.bat 运行（不要用 cmd /c 或 IDE 的运行按钮）")
            print("         2) 直接指定配置：script\\train.bat chat_demo")
            return None
        if not raw:
            return files[0]
        if raw.isdigit() and 1 <= int(raw) <= len(files):
            return files[int(raw) - 1]
        print(f"[提示] 请输入 1 到 {len(files)} 之间的编号")
    print("[提示] 连续三次没选对，用第一个")
    return files[0]


def load_properties(path):
    """读 .properties：UTF-8，支持 # 注释和空行，key=value，键名不区分大小写。"""
    cfg = {}
    with open(path, encoding="utf-8-sig") as f:
        for raw in f:
            line = raw.strip()
            if not line or line[0] in "#;":
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if v[:1] in "\"'":                      # 允许值用引号包起来
                v = v.strip("\"'")
            if k:
                cfg[k] = v
    return cfg


def path_from_cfg(value, default_rel=""):
    """把配置里的路径解析成绝对路径（相对路径按工程根目录算）"""
    v = (value or default_rel).strip()
    if not v:
        return ""
    v = v.replace("\\", "/")
    return v if os.path.isabs(v) else os.path.join(ROOT, v)


def do_train_config(cfg, cfg_path):
    """按配置文件训练：所有参数都来自配置，不向用户提问（除非 confirm=yes）"""
    name = os.path.basename(cfg_path)
    print("=" * 62)
    print(f"  训练（配置文件：{name}）")
    print("=" * 62)

    # ---- 语料 ----
    data = path_from_cfg(cfg.get("data_path"))
    if not data or not os.path.isfile(data):
        print(f"[错误] 配置里的 data_path 找不到：{cfg.get('data_path') or '(空)'}")
        print(f"       解析后路径：{data or '(空)'}")
        if data and os.path.basename(data).startswith("chat_demo"):
            print("       demo 语料还没生成，先跑：script\\fetch_exam_data.bat")
        return 1
    head = open(data, "rb").read(2)
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        print("[错误] 这个语料是 UTF-16 编码，训练脚本认不出它。")
        print("       用记事本打开 → 另存为 → 编码选「UTF-8」→ 覆盖原文件，再重跑。")
        return 1

    # ---- 模型名（配置里只写名字，统一落到 model\ 目录）----
    raw_model = (cfg.get("model") or "model").strip()
    if os.path.splitext(raw_model)[1].lower() != ".pt":
        raw_model += ".pt"
    if os.path.isabs(raw_model):
        out = raw_model
    elif os.sep in raw_model or "/" in raw_model:
        out = path_from_cfg(raw_model)
    else:
        out = os.path.join(ROOT, "model", raw_model)

    # ---- 解释器（配置指定就只认它，不可用直接失败）----
    py, miss = resolve_python(cfg.get("python"))
    if not py:
        print("\n[终止] 解释器不可用，未开始训练。")
        return 1

    # ---- 组装 train_llm.py 的参数 ----
    args = [os.path.join(CODE, "train_llm.py"),
            f"data_path={data}", f"out_path={out}"]
    for k in TRAIN_KEYS:
        if k in cfg and cfg[k] != "":
            args.append(f"{k}={cfg[k]}")

    print()
    print(f"[语料] {short(data)}")
    print(f"[输出] {short(out)}")
    shown = " ".join(f"{a.split('=', 1)[0]}={a.split('=', 1)[1]}"
                     for a in args[1:] if "=" in a and not a.startswith(("data_path", "out_path")))
    print(f"[参数] {shown or '(全部用 train_llm.py 的默认值)'}")
    print(f"[环境] {py}")

    if (cfg.get("confirm", "yes").strip().lower() in ("y", "yes", "1", "true")):
        ans = ask("\n开始训练？(Y/n)", default="y")
        if ans is None:
            print("[取消] 输入已结束")
            return 1
        if (ans or "y").lower() not in ("y", "yes"):
            print("[取消]")
            return 0

    rc = run([py] + args)
    if rc == 0:
        print("\n[完成] 模型已保存")
        print("        想聊天就跑：script\\chat.bat")
    else:
        print(f"\n[失败] 训练脚本退出码 {rc}")
    return rc


def pick_profile(given):
    """选训练档位：命令行给了就用它，否则列出来让用户按编号选。
    返回档位名；输入结束（EOF）返回 None。"""
    if given:
        low = str(given).strip().lower()
        if low in PROFILES:
            return low
        print(f"[提示] 不认识的档位 {given!r}，改用 small")
        return "small"

    names = list(PROFILES)
    print("训练档位：")
    for i, n in enumerate(names, 1):
        mark = "  ← 默认" if n == "small" else ""
        print(f"  [{i}] {n:<8}{PROFILE_HINT[n]}{mark}")
    for _ in range(3):
        # default="" 让「空行」返回 ""，与「输入结束（None）」区分开
        raw = ask("请输入编号或名称（直接回车 = small）", default="")
        if raw is None:
            return None
        if not raw:
            return "small"
        if raw.isdigit() and 1 <= int(raw) <= len(names):
            return names[int(raw) - 1]
        if raw.strip().lower() in PROFILES:
            return raw.strip().lower()
        print("[提示] 只支持 " + " / ".join(names) + "，或者输编号")
    print("[提示] 连续三次没选对，按默认 small 继续")
    return "small"


def pick_iters(given, profile_name):
    """选训练步数：命令行给了就用它，否则问一句（回车=档位默认值）。
    返回步数（int）；输入结束返回 None。"""
    if given:
        try:
            return int(given)
        except ValueError:
            print(f"[提示] 步数 {given!r} 不是整数，改用档位默认值")
    default = PROFILES[profile_name]["max_iters"]
    raw = ask(f"训练多少步（直接回车 = {default}）", default="")
    if raw is None:
        return None
    if not raw:
        return default
    try:
        n = int(raw)
        if n <= 0:
            raise ValueError
        return n
    except ValueError:
        print(f"[提示] {raw!r} 不是正整数，用默认的 {default}")
        return default


def do_train(profile_name=None, iters=None, corpus=None):
    print("=" * 62)
    print("  训练一个新模型")
    print("=" * 62)

    # ---- 档位：命令行指定 > 交互选择 ----
    print()
    profile_name = pick_profile(profile_name)
    if profile_name is None:
        print("[取消] 输入已结束")
        return 1

    steps = pick_iters(iters, profile_name)
    if steps is None:
        print("[取消] 输入已结束")
        return 1

    profile = dict(PROFILES[profile_name])
    profile["max_iters"] = steps
    profile["eval_interval"] = max(1, min(200, steps // 4 or 1))

    # ---- 语料：命令行指定 > 交互选择 ----
    print()
    corpus, err = pick_corpus(corpus)
    if not corpus:
        return 1
    head = open(corpus, "rb").read(2)
    if head[:2] in (b"\xff\xfe", b"\xfe\xff"):
        print("[错误] 这个语料是 UTF-16 编码，训练脚本认不出它。")
        print("       用记事本打开 → 另存为 → 编码选「UTF-8」→ 覆盖原文件，再重跑。")
        return 1

    # ---- 模型名称（现场输入，不写死）----
    print()
    tries = 0
    while True:
        raw = ask("请输入新模型的名称（不用写 .pt）", quiet=True)
        if raw is None:
            print("[取消] 没有读到模型名称（输入已结束）")
            return 1
        tries += 1
        name = safe_name(raw or "")
        if not name:
            print("[提示] 名称不能为空，也不能包含 \\ / : * ? \" < > | 这些字符")
            if tries >= 5:
                print("[取消] 连续 5 次没拿到合法名称")
                return 1
            continue
        if not name.lower().endswith(".pt"):
            name += ".pt"
        out = os.path.join(ROOT, "model", name)
        if os.path.exists(out):
            ans = ask("这个名称已经存在，要覆盖吗？(y/N)", default="n")
            if ans is None:
                print("[取消] 输入已结束")
                return 1
            if (ans or "n").lower() not in ("y", "yes"):
                print("[提示] 换一个名称吧")
                continue
        break

    # ---- 依赖与运行模式 ----
    py, miss = find_python()
    if not py:
        return abort_missing_libs(miss)
    print()
    print_mode(py)

    args = [os.path.join(CODE, "train_llm.py"),
            f"data_path={corpus}", f"out_path={out}"]
    args += [f"{k}={v}" for k, v in profile.items()]

    print(f"\n[语料] {short(corpus)}")
    print("[输出] model\\<你输入的名称>.pt")
    print(f"[档位] {profile_name}  " + " ".join(f"{k}={v}" for k, v in profile.items()))
    print(f"[解释器] {py}")
    if (ask("\n开始训练？(Y/n)", default="y") or "y").lower() not in ("y", "yes"):
        print("[取消]")
        return 1

    rc = run([py] + args)
    if rc == 0:
        print("\n[完成] 模型已保存到 model\\ 目录")
        print("        想聊天就跑：script\\chat.bat")
    else:
        print(f"\n[失败] 训练脚本退出码 {rc}")
    return rc


# ==================================================================
# 对话
# ==================================================================
def list_models():
    """可用模型：model\\ 下的（含子目录）、release\\ 下的、工程根目录的。
    不扫 ai_res\\ —— 那是临时目录，里面的半成品不该出现在选择列表里。"""
    models = []
    for d in ("model", "release"):
        base = os.path.join(ROOT, d)
        models += glob.glob(os.path.join(base, "*.pt"))
        models += glob.glob(os.path.join(base, "**", "*.pt"), recursive=True)
    models += glob.glob(os.path.join(ROOT, "*.pt"))
    seen, listing = set(), []
    for m in sorted(set(models)):
        rp = os.path.abspath(m)
        if rp in seen:
            continue
        seen.add(rp)
        listing.append(rp)
    return listing


def pick_model_interactive():
    """列出现有模型让用户按编号选，返回绝对路径（stderr 打列表，路径写 --pick-out）。
    给 http_server.bat 用：选完直接把路径交给服务器。"""
    listing = list_models()
    if not listing:
        print(f"[错误] 没有找到任何模型：{os.path.join(ROOT, 'model')}\\*.pt")
        print("       先训练一个：script\\train.bat")
        return None

    default = next((p for p in listing
                    if os.path.normcase(p) == os.path.normcase(DEFAULT_MODEL)), None)
    print("可用模型：")
    now = time.time()
    for i, p in enumerate(listing, 1):
        size = os.path.getsize(p) / 1e6
        mark = "  ← 默认" if default and p == default else ""
        busy = ""
        try:
            if now - os.path.getmtime(p) < 120:
                busy = "  ← 刚被修改过，可能正在训练"
        except OSError:
            pass
        print(f"  [{i}] {short(p):<40} {size:>8.1f} MB{mark}{busy}")
    sys.stdout.flush()

    for _ in range(3):
        raw = ask("请输入编号或模型名（直接回车用默认模型）", default="")
        if raw is None:
            print("\n[错误] 读不到键盘输入（当前终端没有可交互的 stdin）。")
            print("       直接双击 bat 运行，或者用参数指定模型名。")
            return None
        if not raw:
            return default or listing[0]
        if raw.isdigit() and 1 <= int(raw) <= len(listing):
            return listing[int(raw) - 1]
        want = raw if raw.lower().endswith(".pt") else raw + ".pt"
        hit = [p for p in listing if os.path.basename(p).lower() == want.lower()]
        if hit:
            return hit[0]
        cand = os.path.join(ROOT, "model", want)
        if os.path.isfile(cand):
            return os.path.abspath(cand)
        print(f"[提示] 没有这个模型：{raw}")
    print("[提示] 连续三次没选对，用第一个")
    return default or listing[0]


def do_chat(name):
    listing = list_models()
    default_exists = os.path.isfile(DEFAULT_MODEL)

    given = safe_name(name or "")
    if given:
        if not given.lower().endswith(".pt"):
            given += ".pt"
        target = os.path.join(ROOT, "model", given)
        if not os.path.isfile(target):
            hit = [p for p in listing if os.path.basename(p).lower() == given.lower()]
            if hit:
                target = hit[0]
            else:
                print(f"[错误] 找不到模型：model\\{given}")
                if listing:
                    print("       可用模型：")
                    for i, p in enumerate(listing, 1):
                        print(f"  [{i}] {short(p)}")
                else:
                    print("       当前没有任何模型，请先准备模型或先训练一个。")
                return 1
    else:
        if not default_exists and not listing:
            print(f"[错误] 找不到默认模型：{short(DEFAULT_MODEL)}")
            print("       当前也没有任何其它 .pt。请先准备语料和模型（或先训练一个），再重新运行。")
            return 1

        # 全部模型一起编号，默认那个也列出来（只是标注一下），
        # 免得只看到「其它模型」里没有 demo，误以为模型丢了
        print("\n可用模型：")
        for i, p in enumerate(listing, 1):
            mark = "  ← 默认" if os.path.abspath(p) == os.path.abspath(DEFAULT_MODEL) else ""
            print(f"  [{i}] {short(p)}{mark}")
        if not default_exists:
            print(f"       （默认模型 {short(DEFAULT_MODEL)} 还不存在，先训练或选别的）")

        raw = ask("请输入模型名称或编号（直接回车用默认模型）")
        if raw is None:
            print("\n[取消] 读不到键盘输入。")
            return 1
        if not raw:
            raw = DEFAULT_MODEL
        if os.path.basename(raw).lower() == os.path.basename(DEFAULT_MODEL).lower():
            target = DEFAULT_MODEL
            if not default_exists:
                print(f"[错误] 找不到默认模型：{short(DEFAULT_MODEL)}")
                print("       请先准备语料和模型（src\\chat_demo.jsonl 与 model\\chat_demo.pt），再重新运行。")
                return 1
        elif raw.isdigit() and listing and 1 <= int(raw) <= len(listing):
            target = listing[int(raw) - 1]
        else:
            cand = safe_name(raw)
            if not cand:
                print("[错误] 名称不合法")
                return 1
            if not cand.lower().endswith(".pt"):
                cand += ".pt"
            hit = [p for p in listing if os.path.basename(p).lower() == cand.lower()]
            target = hit[0] if hit else os.path.join(ROOT, "model", cand)
            if not os.path.isfile(target):
                print(f"[错误] 找不到模型：{short(target)}")
                return 1

    py, miss = find_python()
    if not py:
        return abort_missing_libs(miss)

    print(f"\n[模型] {short(target)}")
    print_mode(py)
    print(f"[解释器] {py}")
    print("[提示] 对话中输入 :q 退出，:p 看设置，:t 0.7 改温度\n")
    return run([py, os.path.join(CODE, "chat.py"), target])


def main():
    ap = argparse.ArgumentParser(description="交互式训练 / 对话入口")
    ap.add_argument("--mode", choices=["train", "chat"],
                    help="train = 训练；chat = 对话")
    ap.add_argument("--find-python", action="store_true",
                    help="只打印一个装了 torch 的 python 路径（供 bat 调用）")
    ap.add_argument("--cwd", default=None,
                    help="配合 --find-python：从这一级开始找工程根目录")
    ap.add_argument("-p", "--profile", default=None,
                    help="训练档位：tiny / small / medium / large（默认 small）")
    ap.add_argument("-m", "--model", default=None,
                    help="对话用：模型名称，给了就不再问（仍然会校验存在性）")
    ap.add_argument("-i", "--iters", default=None,
                    help="只训这么多步（调试用；给了会覆盖档位里的 max_iters）")
    ap.add_argument("-c", "--corpus", default=None,
                    help="训练用哪份语料（文件名或路径）；不给就列出来让你选")
    ap.add_argument("--config", default=None,
                    help="用哪个配置文件（properties/*.properties）；给了就不问任何问题")
    ap.add_argument("--pick-config", action="store_true",
                    help="列出配置文件让用户选，把选中的路径打到 stdout（供 bat 调用）")
    ap.add_argument("--pick-out", default=None,
                    help="配合 --pick-config / --pick-model：把选中的路径写进这个文件（供 bat 读取）")
    ap.add_argument("--pick-model", action="store_true",
                    help="列出模型让用户选，把选中的路径打到 stdout（供 bat 调用）")
    a = ap.parse_args()

    if a.find_python:
        global ROOT, CODE, DEFAULT_CORPUS, DEFAULT_MODEL
        if a.cwd:
            ROOT = find_root(os.path.abspath(a.cwd))
            CODE = os.path.join(ROOT, "code")
            DEFAULT_CORPUS = os.path.join(ROOT, "src", "chat_demo.jsonl")
            DEFAULT_MODEL = os.path.join(ROOT, "model", "chat_demo.pt")
        exe, _ = find_python()
        if exe:
            print(exe)
            return 0
        return 1

    # ---- 供 bat 用：只输出一个配置路径 ----
    if a.pick_config:
        chosen = pick_config_interactive()
        if not chosen:
            return 1
        chosen = os.path.abspath(chosen)
        if a.pick_out:
            try:
                with open(a.pick_out, "w", encoding="utf-8") as f:
                    f.write(chosen + "\n")
            except OSError as e:
                print(f"[错误] 写不了 {a.pick_out}：{e}")
                return 1
        print(chosen)                            # stdout 只留这一行
        return 0

    # ---- 供 bat 用：只输出一个模型路径（http_server.bat 用）----
    if a.pick_model:
        chosen = pick_model_interactive()
        if not chosen:
            return 1
        chosen = os.path.abspath(chosen)
        if a.pick_out:
            try:
                with open(a.pick_out, "w", encoding="utf-8") as f:
                    f.write(chosen + "\n")
            except OSError as e:
                print(f"[错误] 写不了 {a.pick_out}：{e}")
                return 1
        print(chosen)
        return 0

    if not a.mode:
        ap.error("需要 --mode train 或 --mode chat")

    # ---- 配置文件优先：给了就完全不提问 ----
    if a.config:
        cfg_path = path_from_cfg(a.config)
        if not os.path.isfile(cfg_path):
            print(f"[错误] 找不到配置文件：{a.config}")
            print(f"       解析后路径：{cfg_path}")
            return 1
        cfg = load_properties(cfg_path)
        if a.mode == "train":
            return do_train_config(cfg, cfg_path)
        model = cfg.get("model") or ""
        if model:
            if os.path.splitext(model)[1].lower() != ".pt":
                model += ".pt"
            model = model if os.path.isabs(model) or os.sep in model else os.path.join(ROOT, "model", model)
        return do_chat(model or a.model)

    # 命令行 / 环境变量给了就用，没给就让 do_train 现场问（完全不传参也能用）
    profile = (a.profile or os.environ.get("PROFILE") or "").strip() or None
    return do_train(profile, a.iters, a.corpus) if a.mode == "train" else do_chat(a.model)


if __name__ == "__main__":
    sys.exit(main() or 0)
