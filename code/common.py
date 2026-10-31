# -*- coding: utf-8 -*-
r"""工程级公共工具
================================================================
给 code\ 下的各个入口共用：定位工程根、properties 读写、路径解析、
父进程看门狗（WebUI 关闭时子进程自杀）、隐藏目录。

约定：
    · 只依赖标准库；
    · 不在这里 import torch（谁用谁 import，避免 WebUI 起得慢）。
"""
import ctypes
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT_MARKER = os.path.join("code", "train.py")


def find_root(start=None):
    """向上找到含 code\train.py 的那一级，作为工程根。"""
    cur = os.path.abspath(start or os.path.dirname(os.path.abspath(__file__)))
    for _ in range(8):
        if os.path.isfile(os.path.join(cur, ROOT_MARKER)):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.abspath(__file__))


def load_properties(path):
    """读 .properties：UTF-8，支持 # 注释与空行，value 按 split("=", 1)。"""
    out = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                out[k.strip()] = v.strip()
    except OSError as e:
        print(f"[错误] 读不了配置文件 {path}：{e}")
        return {}
    return out


def save_properties(path, updates):
    """只改给定键的 value，注释、空行、键顺序全部保留；新键追加到末尾。

    返回 (changed, added) 两个列表，供界面提示用。
    """
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    changed, added = [], []
    pending = dict(updates)
    for i, line in enumerate(lines):
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, old = s.split("=", 1)
        k = k.strip()
        if k in pending:
            new = str(pending.pop(k))
            if new != old.strip():
                lines[i] = f"{k}={new}"
                changed.append(k)
    for k, v in pending.items():
        lines.append(f"{k}={v}")
        added.append(k)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp, path)
    return changed, added


def coerce_like(value, old):
    """按 CONFIG 里现有值的类型把字符串转过去（bool 要单独识别）。"""
    if isinstance(old, bool):
        return str(value).strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(old, int):
        return int(float(value))
    if isinstance(old, float):
        return float(value)
    return str(value)


# properties 里有、但 CONFIG 里没有的键：不是错误，只是不参与训练
IGNORED_KEYS = ("python", "confirm")


def apply_properties(config, props, root=None):
    """把 properties 的键值套到 CONFIG 上，返回被忽略的键列表。

    特例：model=dlqq_llm → out_path=model/dlqq_llm/dlqq_llm.pt（与老 entry.py 行为一致）。
    """
    root = root or find_root()
    ignored = []
    for k, v in props.items():
        if k == "model":
            name = (v or "").strip()
            if not name:
                continue
            if os.path.splitext(name)[1].lower() != ".pt":
                name += ".pt"
            config["out_path"] = os.path.join("model", os.path.splitext(name)[0], name)
            continue
        if k in config:
            try:
                config[k] = coerce_like(v, config[k])
            except (TypeError, ValueError):
                print(f"[警告] 配置项 {k}={v!r} 类型不对，已忽略")
                ignored.append(k)
        elif k not in IGNORED_KEYS:
            ignored.append(k)
    return ignored


def resolve_path(root, p):
    """相对路径按工程根解析；空值原样返回。"""
    if not p:
        return p
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(root, p))


def config_stem(name):
    """配置名统一成不带 .properties 的形式。

    界面下拉给的是文件名（chat_demo.properties），内部一律用 chat_demo。
    """
    name = (name or "").strip()
    if name.lower().endswith(".properties"):
        name = name[:-len(".properties")]
    return name


def config_model(root, name):
    """配置名 → (模型绝对路径 | None, props)。

    `model=` 里只写名字（不带 .pt），按 model\\<名字>\\<名字>.pt 解析；
    权重不存在时返回 (None, props)，由调用方决定是跳过还是报错。
    """
    name = config_stem(name)
    props = load_properties(os.path.join(root, "properties", name + ".properties"))
    m = (props.get("model") or "").strip()
    if not m:
        return None, props
    if os.path.splitext(m)[1].lower() != ".pt":
        m += ".pt"
    path = os.path.join(root, "model", os.path.splitext(m)[0], m)
    return (path if os.path.isfile(path) else None), props


def chat_configs(root):
    """所有能对应到现成权重的配置：[{name, model, path}]，按名字排序。"""
    out = []
    d = os.path.join(root, "properties")
    try:
        names = sorted(f[:-len(".properties")] for f in os.listdir(d)
                       if f.endswith(".properties"))
    except OSError:
        return out
    for name in names:
        path, props = config_model(root, name)
        if path:
            out.append({"name": name,
                        "model": (props.get("model") or name).strip(),
                        "path": path})
    return out


def resolve_python(root, config_name=""):
    """训练 / 对话要用的解释器。

    返回 (exe, source)：source 是 "config" / "python" / "python3" / "py"；
    解析失败返回 (None, 原因)，由调用方原样展示给用户。
    """
    config_name = config_stem(config_name)
    props = {}
    if config_name:
        props = load_properties(os.path.join(root, "properties", config_name + ".properties"))
    raw = (props.get("python") or "").strip()
    if raw:
        p = os.path.expandvars(os.path.expanduser(raw))
        if not os.path.isabs(p):
            p = os.path.normpath(os.path.join(root, p))
        if os.path.isdir(p):
            p = os.path.join(p, "python.exe")
        if os.path.basename(p).lower() in ("pythonw.exe", "pythonw"):
            p = os.path.join(os.path.dirname(p), "python.exe")
        if os.path.isfile(p):
            return p, "config"
        return None, f"配置 {config_name} 里的 python= 指向的文件不存在：{p}"
    for name in ("python", "python3", "py"):
        exe = shutil.which(name)
        if not exe:
            continue
        if name == "py":
            # py 只是启动器，解析它真正会用的解释器，避免“写的是 py、跑的是别的版本”
            try:
                r = subprocess.run([exe, "-c", "import sys;print(sys.executable)"],
                                   capture_output=True, text=True, encoding="utf-8",
                                   errors="replace", timeout=20)
                real = (r.stdout or "").strip().splitlines()
                real = real[-1] if real else ""
            except (OSError, subprocess.SubprocessError):
                real = ""
            if real and os.path.isfile(real):
                return real, name
        return exe, name
    return None, "没有指定 python=，PATH 上也找不到 python / python3 / py"


def hide_dir(path):
    """把目录设为隐藏（Windows 资源管理器默认不显示）；失败不影响功能。"""
    if os.name != "nt" or not os.path.isdir(path):
        return False
    try:
        subprocess.run(["attrib", "+h", path], shell=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=5, check=False)
        return True
    except (OSError, subprocess.SubprocessError):
        return False


def _parent_alive_by_handle(pid):
    """用父进程句柄判断存活：句柄一有信号说明父进程已退出。"""
    SYNCHRONIZE = 0x00100000
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
    if not handle:
        return False
    try:
        return kernel32.WaitForSingleObject(handle, 0) == 0x102   # WAIT_TIMEOUT = 还活着
    finally:
        kernel32.CloseHandle(handle)


def _pid_alive_fallback(pid):
    """非 Windows 或拿不到句柄时的兜底：能 os.kill(pid, 0) 就算活。"""
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def start_parent_watchdog(parent_pid, interval=3.0, on_exit=None):
    """父进程（WebUI）一退出就让当前进程跟着退出。

    参数：
        parent_pid  父进程 pid，来自 --parent-pid；0/None 表示不启用
        interval    轮询间隔（秒）
        on_exit     退出前回调（例如保存检查点），异常会被吞掉并打印
    """
    if not parent_pid:
        return None
    pid = int(parent_pid)
    if pid <= 0:
        return None

    def loop():
        while True:
            alive = (_parent_alive_by_handle(pid) if os.name == "nt"
                     else _pid_alive_fallback(pid))
            if not alive:
                print(f"[退出] 父进程 {pid} 已结束，本进程跟着退出", flush=True)
                if on_exit:
                    try:
                        on_exit()
                    except Exception:
                        traceback.print_exc()
                sys.stdout.flush()
                os._exit(1)
            time.sleep(interval)

    t = threading.Thread(target=loop, name="parent-watchdog", daemon=True)
    t.start()
    return t


def setup_stdio():
    """所有 .py 入口统一调用：UTF-8 输出 + 逐行刷新。"""
    for s in ("stdout", "stderr"):
        try:
            getattr(sys, s).reconfigure(encoding="utf-8", errors="replace",
                                        line_buffering=True, write_through=True)
        except (AttributeError, ValueError):
            pass
