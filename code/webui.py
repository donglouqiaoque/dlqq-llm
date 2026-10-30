# -*- coding: utf-8 -*-
r"""本地 WebUI 服务
================================================================
一个进程同时提供：静态页面（web\）、JSON API、SSE 事件流，并管理所有子进程。

启动（一般由 script\start.bat 调用）：
    python code\webui.py              # 随机端口，自动开浏览器
    python code\webui.py --port 8080  # 指定端口
    python code\webui.py --no-browser # 不开浏览器

生命周期：
    · 关掉 start.bat 的 cmd 窗口 = 退出，所有子进程一起结束；
    · 关浏览器标签不影响任何进程，重新访问控制台打印的地址即可。

安全：只绑 127.0.0.1；API 需要启动时生成的一次性 token（地址里带 ?t=）。
"""
import argparse
import ctypes
import json
import os
import queue
import secrets
import socket
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common                                                     # noqa: E402

common.setup_stdio()
ROOT = common.find_root()
WEB_DIR = os.path.join(ROOT, "web")
LOG_DIR = os.path.join(ROOT, ".ai_res", "cache", "train-logs")
STATE = {"token": "", "started": time.time(), "train": None, "server": None}


def tail_train_log(n=400):
    """取最新一份训练日志的末尾 n 行。

    内存队列被读过就出队了，刷新页面只能补一次；日志同时落盘在 train-logs\\，
    这里直接读文件，刷新多少次都能补回历史。
    """
    try:
        names = [f for f in os.listdir(LOG_DIR)
                 if f.endswith(".log") and not f.startswith("server-")]
    except OSError:
        return []
    if not names:
        return []
    paths = sorted((os.path.join(LOG_DIR, f) for f in names), key=os.path.getmtime)
    try:
        with open(paths[-1], encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return []
    return lines[-n:]


# ==================================================================
# 子进程生命周期：Job Object 兜底
# ==================================================================
# 关掉 start.bat 的 cmd 窗口时，WebUI 可能来不及执行清理代码就被系统结束。
# 把子进程放进一个 KILL_ON_JOB_CLOSE 的 Job：WebUI 进程一消失，内核自动
# 关掉 job 句柄，job 里的训练/服务/对话进程一起被杀。--parent-pid 看门狗
# 仍然保留，两层兜底。
JOB_HANDLE = None


def _make_job_object():
    """建一个“句柄关掉就杀光里面进程”的 Job；失败返回 None（只靠看门狗）。"""
    if os.name != "nt":
        return None
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.CreateJobObjectW.restype = ctypes.c_void_p
        job = kernel32.CreateJobObjectW(None, None)
        if not job:
            return None

        class _Basic(ctypes.Structure):
            _fields_ = [
                ("PerProcessUserTimeLimit", ctypes.c_int64),
                ("PerJobUserTimeLimit", ctypes.c_int64),
                ("LimitFlags", ctypes.c_uint32),
                ("MinimumWorkingSetSize", ctypes.c_size_t),
                ("MaximumWorkingSetSize", ctypes.c_size_t),
                ("ActiveProcessLimit", ctypes.c_uint32),
                ("Affinity", ctypes.c_size_t),
                ("PriorityClass", ctypes.c_uint32),
                ("SchedulingClass", ctypes.c_uint32),
            ]

        class _Io(ctypes.Structure):
            _fields_ = [
                ("ReadOperationCount", ctypes.c_uint64),
                ("WriteOperationCount", ctypes.c_uint64),
                ("OtherOperationCount", ctypes.c_uint64),
                ("ReadTransferCount", ctypes.c_uint64),
                ("WriteTransferCount", ctypes.c_uint64),
                ("OtherTransferCount", ctypes.c_uint64),
            ]

        class _Extended(ctypes.Structure):
            _fields_ = [
                ("BasicLimitInformation", _Basic),
                ("IoInfo", _Io),
                ("ProcessMemoryLimit", ctypes.c_size_t),
                ("JobMemoryLimit", ctypes.c_size_t),
                ("PeakProcessMemoryUsed", ctypes.c_size_t),
                ("PeakJobMemoryUsed", ctypes.c_size_t),
            ]

        info = _Extended()
        info.BasicLimitInformation.LimitFlags = 0x2000    # KILL_ON_JOB_CLOSE
        if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info),
                                                ctypes.sizeof(info)):
            kernel32.CloseHandle(job)
            return None
        return job
    except (AttributeError, OSError):
        return None


def _join_job(proc):
    """把刚起的子进程放进 JOB_HANDLE；失败不影响功能（看门狗仍然有效）。"""
    if not JOB_HANDLE:
        return
    try:
        ctypes.windll.kernel32.AssignProcessToJobObject(JOB_HANDLE, int(proc._handle))
    except (AttributeError, OSError):
        pass


# ==================================================================
# 子进程管理
# ==================================================================
class Child:
    """一个子进程 + 读线程；退出时统一清理。"""

    def __init__(self, args, cwd=None, log_path=None, on_line=None):
        self.args = args
        self.log = None
        if log_path:
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            self.log = open(log_path, "a", encoding="utf-8", buffering=1)
        self.proc = subprocess.Popen(
            args, cwd=cwd or ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        _join_job(self.proc)
        self.lines = queue.Queue(maxsize=20000)
        self.on_line = on_line
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\n")
                if self.log:
                    try:
                        self.log.write(line + "\n")
                    except OSError:
                        pass
                if self.on_line:
                    try:
                        self.on_line(line)
                    except Exception:
                        traceback.print_exc()
                try:
                    self.lines.put_nowait(line)
                except queue.Full:
                    pass
        except (OSError, ValueError):
            pass
        finally:
            try:
                self.proc.stdout.close()
            except (OSError, ValueError):
                pass

    def alive(self):
        return self.proc.poll() is None

    def stop(self, timeout=5):
        if not self.alive():
            return
        try:
            self.proc.terminate()
            self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(self.proc.pid)],
                           capture_output=True)
        except OSError:
            pass
        if self.log:
            try:
                self.log.close()
            except OSError:
                pass


CHILDREN = []


def stop_all():
    for c in list(CHILDREN):
        try:
            c.stop()
        except Exception:
            traceback.print_exc()
    CHILDREN.clear()


def spawn(args, log_name=None, on_line=None):
    log_path = os.path.join(LOG_DIR, log_name) if log_name else None
    c = Child(args, ROOT, log_path, on_line)
    CHILDREN.append(c)
    return c


# ==================================================================
# SSE 广播
# ==================================================================
class Hub:
    def __init__(self):
        self.clients = []
        self.lock = threading.Lock()

    def add(self):
        q = queue.Queue(maxsize=2000)
        with self.lock:
            self.clients.append(q)
        return q

    def drop(self, q):
        with self.lock:
            if q in self.clients:
                self.clients.remove(q)

    def push(self, event, data):
        payload = (event, data)
        with self.lock:
            for q in list(self.clients):
                try:
                    q.put_nowait(payload)
                except queue.Full:
                    pass


HUB = Hub()


# ==================================================================
# 业务：配置 / 模型 / 语料
# ==================================================================
def list_configs():
    out = []
    d = os.path.join(ROOT, "properties")
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if not name.endswith(".properties"):
            continue
        p = os.path.join(d, name)
        props = common.load_properties(p)
        desc = ""
        try:
            with open(p, encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s.startswith("#") and len(s) > 2:
                        desc = s.lstrip("# ").strip()
                        break
        except OSError:
            pass
        out.append({"name": name, "desc": desc, "props": props,
                    "mtime": os.path.getmtime(p)})
    return out


def list_models():
    out = []
    for dirpath, _dirs, files in os.walk(os.path.join(ROOT, "model")):
        for fn in sorted(files):
            if not fn.endswith(".pt"):
                continue
            p = os.path.join(dirpath, fn)
            size = os.path.getsize(p)
            sig = p + ".sha256"
            out.append({
                "id": os.path.splitext(fn)[0],
                "path": os.path.relpath(p, ROOT).replace("\\", "/"),
                "size_mb": round(size / 1e6, 1),
                "mtime": time.strftime("%m-%d %H:%M", time.localtime(os.path.getmtime(p))),
                "signed": os.path.isfile(sig),
            })
    return sorted(out, key=lambda m: m["path"])


def list_corpora():
    out = []
    base = os.path.join(ROOT, "src")
    if not os.path.isdir(base):
        return out
    for name in sorted(os.listdir(base)):
        d = os.path.join(base, name)
        if not os.path.isdir(d):
            continue
        total = 0
        n = 0
        for fn in os.listdir(d):
            fp = os.path.join(d, fn)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
                n += 1
        out.append({"dir": f"src/{name}", "files": n, "gb": round(total / 1e9, 2)})
    return out


# ==================================================================
# HTTP
# ==================================================================
class Handler(BaseHTTPRequestHandler):
    server_version = "dlqq-webui/1.0"

    def log_message(self, fmt, *args):        # 静音默认访问日志
        return

    # ---- 工具 ----
    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _err(self, code, msg):
        self._json({"ok": False, "error": msg}, code)

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

    def _sse(self, event, data):
        payload = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(payload.encode("utf-8"))
        self.wfile.flush()

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    def _auth(self, qs):
        tok = (qs.get("t") or [""])[0]
        if tok != STATE["token"]:
            self._err(403, "token 无效，请用控制台打印的完整地址打开")
            return False
        return True

    # ---- 静态 ----
    def _static(self, path):
        rel = path.lstrip("/") or "index.html"
        if ".." in rel:
            self._err(400, "bad path")
            return
        if rel == "logo.ico":                      # 站点图标直接用仓库里那张
            fp = os.path.join(ROOT, "script", "logo.ico")
        else:
            fp = os.path.join(WEB_DIR, rel)
        if not os.path.isfile(fp):
            self._err(404, f"没有文件：{rel}")
            return
        ext = os.path.splitext(fp)[1].lower()
        ctype = {".html": "text/html; charset=utf-8", ".js": "application/javascript; charset=utf-8",
                 ".css": "text/css; charset=utf-8", ".png": "image/png", ".svg": "image/svg+xml",
                 ".ico": "image/x-icon"}.get(ext, "application/octet-stream")
        with open(fp, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        path = u.path
        if not path.startswith("/api/"):
            self._static(path)
            return
        if not self._auth(qs):
            return

        if path == "/api/configs":
            self._json({"ok": True, "configs": list_configs()})
        elif path == "/api/models":
            self._json({"ok": True, "models": list_models()})
        elif path == "/api/corpora":
            self._json({"ok": True, "corpora": list_corpora()})
        elif path == "/api/train/status":
            c = STATE["train"]
            self._json({"ok": True, "running": bool(c and c.alive()),
                        "config": STATE.get("train_config")})
        elif path == "/api/train/log":
            c = STATE["train"]
            if c:                       # 顺手清空内存队列，避免长时间训练把内存吃满
                while True:
                    try:
                        c.lines.get_nowait()
                    except queue.Empty:
                        break
            self._json({"ok": True, "lines": tail_train_log(400)})
        elif path == "/api/server/status":
            c = STATE["server"]
            self._json({"ok": True, "running": bool(c and c.alive()),
                        "port": STATE.get("server_port")})
        elif path == "/api/env":
            py = (qs.get("py") or [sys.executable])[0]
            self._json(probe_python(py))
        elif path == "/api/events":
            self._events()
        else:
            self._err(404, f"未知接口：{path}")

    def do_POST(self):
        u = urlparse(self.path)
        qs = parse_qs(u.query)
        path = u.path
        if not self._auth(qs):
            return
        body = self._body()

        try:
            if path == "/api/config/save":
                self._config_save(body)
            elif path == "/api/train/start":
                self._train_start(body)
            elif path == "/api/train/stop":
                self._stop("train")
            elif path == "/api/chat":
                self._chat(body, qs)
            elif path == "/api/server/start":
                self._server_start(body)
            elif path == "/api/server/stop":
                self._stop("server")
            elif path == "/api/sign":
                self._sign(body)
            else:
                self._err(404, f"未知接口：{path}")
        except Exception as e:
            traceback.print_exc()
            self._err(500, f"{type(e).__name__}: {e}")

    # ---- 业务处理 ----
    def _config_save(self, body):
        name = body.get("name") or ""
        updates = body.get("updates") or {}
        if not name.endswith(".properties") or "/" in name or "\\" in name:
            return self._err(400, "配置名不合法")
        p = os.path.join(ROOT, "properties", name)
        if not os.path.isfile(p):
            return self._err(404, f"没有这个配置：{name}")
        backup = os.path.join(ROOT, ".ai_res", "cache", "cfg-backup")
        os.makedirs(backup, exist_ok=True)
        import shutil
        shutil.copy2(p, os.path.join(backup, f"{name}.{int(time.time())}.bak"))
        changed, added = common.save_properties(p, updates)
        self._json({"ok": True, "changed": changed, "added": added})

    def _train_start(self, body):
        cfg = body.get("config") or ""
        if not cfg.endswith(".properties"):
            return self._err(400, "请选择一个 .properties 配置")
        p = os.path.join(ROOT, "properties", os.path.basename(cfg))
        if not os.path.isfile(p):
            return self._err(404, f"没有这个配置：{cfg}")
        c = STATE["train"]
        if c and c.alive():
            return self._err(409, "已有训练在跑")
        name = os.path.splitext(os.path.basename(p))[0]
        log_name = f"{name}-{time.strftime('%Y%m%d-%H%M%S')}.log"
        args = [sys.executable, os.path.join(ROOT, "code", "train.py"),
                "--config", os.path.relpath(p, ROOT), "--parent-pid", str(os.getpid())]
        # resume 由界面按钮决定：开始训练=从头，继续训练=接着存档（没有存档时训练脚本会自动从头）
        args.append("resume=true" if body.get("resume") else "resume=false")
        STATE["train"] = spawn(args, log_name,
                               on_line=lambda line: HUB.push("train-log", {"line": line}))
        STATE["train_config"] = os.path.basename(p)
        self._json({"ok": True, "log": log_name})

    def _stop(self, which):
        c = STATE.get(which)
        if c and c.alive():
            c.stop()
        STATE[which] = None
        self._json({"ok": True})

    def _chat(self, body, qs):
        model = body.get("model") or ""
        # 前端传 messages（多轮）；也兼容旧的单条 text
        msgs = body.get("messages")
        if isinstance(msgs, list) and msgs:
            parts = []
            for i, m in enumerate(msgs):
                c = m.get("content") or ""
                if m.get("role") == "user":
                    parts.append(c if i == 0 else "用户：" + c)
                else:
                    parts.append("助手：" + c)
            text = "\n".join(parts)
        else:
            text = (body.get("text") or "").strip()
        if not text:
            return self._err(400, "消息为空")
        path = os.path.join(ROOT, model.replace("/", os.sep)) if model else ""
        if not path or not os.path.isfile(path):
            return self._err(404, f"模型不存在：{model}")
        # 训练中且同名模型 → 拒绝（与方案一致）
        running = STATE.get("train") and STATE["train"].alive()
        training_model = ""
        if running and STATE.get("train_config"):
            props = common.load_properties(os.path.join(ROOT, "properties", STATE["train_config"]))
            training_model = (props.get("model") or "").strip()
        mid = os.path.splitext(os.path.basename(path))[0]
        if running and training_model and mid == training_model:
            return self._err(409, f"{mid} 正在训练，换一个模型对话")

        import chat as chat_mod
        try:
            s = get_chat_session(path)
            self._sse_start()
            self._sse("meta", {"model": mid, "info": s.info()})
            pieces = []
            for piece in s.reply_stream(text):
                pieces.append(piece)
                self._sse("token", {"text": piece})
            self._sse("done", {"text": chat_mod._clean_final("".join(pieces))})
        except (OSError, RuntimeError) as e:
            traceback.print_exc()
            try:
                self._sse("error", {"error": f"{type(e).__name__}: {e}"})
            except OSError:
                pass

    def _server_start(self, body):
        c = STATE["server"]
        if c and c.alive():
            return self._err(409, "服务已在运行")
        port = int(body.get("port") or 49128)
        if port_in_use(port):
            return self._err(409, f"端口 {port} 已被占用（可能有上次残留的服务进程），"
                                  "请先结束它，或换一个端口")
        args = [sys.executable, os.path.join(ROOT, "code", "http_server.py"),
                "--port", str(port), "--events", "--parent-pid", str(os.getpid())]

        def on_line(line):
            if line.startswith("@@EV@@ "):
                try:
                    HUB.push("server-chat", json.loads(line[7:]))
                except ValueError:
                    pass
            HUB.push("server-log", {"line": line})

        STATE["server"] = spawn(args, f"server-{port}.log", on_line=on_line)
        STATE["server_port"] = port
        self._json({"ok": True, "port": port})

    def _sign(self, body):
        mode = "verify" if (body.get("mode") == "verify") else "compute"
        args = [sys.executable, os.path.join(ROOT, "code", "model_signature.py"),
                "--mode", mode, "--parent-pid", str(os.getpid())]
        one = (body.get("file") or "").strip()
        if one:
            args += ["--file", one]
        try:
            r = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=300)
            self._json({"ok": r.returncode == 0, "mode": mode,
                        "out": (r.stdout or "")[-8000:], "err": (r.stderr or "")[-4000:]})
        except subprocess.TimeoutExpired:
            self._err(500, "签名任务超时")

    def _events(self):
        q = HUB.add()
        self._sse_start()
        try:
            self._sse("hello", {"ok": True})
            while True:
                try:
                    event, data = q.get(timeout=15)
                    self._sse(event, data)
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            HUB.drop(q)


class LANHandler(Handler):
    """局域网只读对话页：只放行页面、模型清单和对话接口。

    训练 / 配置 / 签名 / 服务管理 / SSE 一概 403，避免把控制台暴露到内网。
    """

    server_version = "dlqq-webui-lan/1.0"
    GET_WHITELIST = ("/", "/chat.html", "/logo.ico", "/api/models")
    POST_WHITELIST = ("/api/chat",)

    def _auth(self, qs):
        return True                              # 内网对话页不带 token

    def do_GET(self):
        path = urlparse(self.path).path
        if path not in self.GET_WHITELIST:
            self._err(403, "局域网只开放对话页")
            return
        if path in ("/", "/chat.html"):
            self._send_file(os.path.join(WEB_DIR, "chat.html"),
                            "text/html; charset=utf-8")
        elif path == "/logo.ico":
            self._send_file(os.path.join(ROOT, "script", "logo.ico"),
                            "image/x-icon")
        else:
            self._json({"ok": True, "models": list_models()})

    def do_POST(self):
        if urlparse(self.path).path not in self.POST_WHITELIST:
            self._err(403, "局域网只开放对话页")
            return
        body = self._body()
        allowed = {m["path"] for m in list_models()}
        if (body.get("model") or "").replace("\\", "/") not in allowed:
            self._err(400, "模型不在清单里")
            return
        try:
            self._chat(body, {})
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        except Exception as e:
            traceback.print_exc()
            try:
                self._err(500, f"{type(e).__name__}: {e}")
            except OSError:
                pass

    def _send_file(self, fp, ctype):
        try:
            with open(fp, "rb") as f:
                data = f.read()
        except OSError:
            self._err(404, "文件不存在")
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)


CHAT_LOCK = threading.Lock()
CHAT_SESSION = {"path": "", "obj": None}


def get_chat_session(path):
    """模型常驻内存；换模型时释放旧的（含显存）。"""
    import chat as chat_mod
    with CHAT_LOCK:
        if CHAT_SESSION["path"] == path and CHAT_SESSION["obj"] is not None:
            return CHAT_SESSION["obj"]
        CHAT_SESSION["obj"] = None
        CHAT_SESSION["path"] = ""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        s = chat_mod.ChatSession(path)
        CHAT_SESSION["obj"] = s
        CHAT_SESSION["path"] = path
        return s


def probe_python(py):
    code = ("import json,sys;"
            "d={'version':sys.version.split()[0],'exe':sys.executable};"
            "import importlib;"
            "d['torch']=importlib.util.find_spec('torch') is not None;"
            "d['numpy']=importlib.util.find_spec('numpy') is not None;"
            "d['tk']=importlib.util.find_spec('tkinter') is not None;"
            "print(json.dumps(d))")
    try:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=20)
        info = json.loads((r.stdout or "{}").strip().splitlines()[-1])
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    if info.get("torch"):
        try:
            r2 = subprocess.run(
                [py, "-c", "import torch;print(int(torch.cuda.is_available()));"
                           "print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
            lines = (r2.stdout or "").strip().splitlines()
            info["cuda"] = bool(int(lines[0])) if lines else False
            info["cuda_name"] = lines[1] if len(lines) > 1 else ""
        except (OSError, ValueError, subprocess.TimeoutExpired):
            info["cuda"], info["cuda_name"] = False, ""
    return {"ok": True, **info}


def port_in_use(port, host="127.0.0.1", timeout=0.3):
    """端口上已经有人在听？用来在启动服务前挡住重复实例。"""
    s = socket.socket()
    s.settimeout(timeout)
    try:
        s.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def find_free_port(preferred):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", preferred))
        return preferred
    except OSError:
        pass
    finally:
        s.close()
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _lan_ipv4_from_adapters():
    """用 GetAdaptersInfo 找“像样的”局域网 IPv4：192.168.x.x、有网关、非虚拟网卡。"""
    if os.name != "nt":
        return ""
    from ctypes import wintypes

    class IP_ADDR_STRING(ctypes.Structure):
        pass

    IP_ADDR_STRING._fields_ = [
        ("Next", ctypes.POINTER(IP_ADDR_STRING)),
        ("IpAddress", ctypes.c_char * 16),
        ("IpMask", ctypes.c_char * 16),
        ("Context", wintypes.DWORD),
    ]

    class IP_ADAPTER_INFO(ctypes.Structure):
        pass

    IP_ADAPTER_INFO._fields_ = [
        ("Next", ctypes.POINTER(IP_ADAPTER_INFO)),
        ("ComboIndex", wintypes.DWORD),
        ("AdapterName", ctypes.c_char * 260),
        ("Description", ctypes.c_char * 132),
        ("AddressLength", wintypes.UINT),
        ("Address", ctypes.c_ubyte * 8),
        ("Index", wintypes.DWORD),
        ("Type", wintypes.UINT),
        ("DhcpEnabled", wintypes.UINT),
        ("CurrentIpAddress", ctypes.POINTER(IP_ADDR_STRING)),
        ("IpAddressList", IP_ADDR_STRING),
        ("GatewayList", IP_ADDR_STRING),
        ("DhcpServer", IP_ADDR_STRING),
        ("HaveWins", wintypes.BOOL),
        ("PrimaryWinsServer", IP_ADDR_STRING),
        ("SecondaryWinsServer", IP_ADDR_STRING),
        ("LeaseObtained", wintypes.DWORD),
        ("LeaseExpires", wintypes.DWORD),
    ]

    iphlpapi = ctypes.windll.iphlpapi
    size = wintypes.ULONG(0)
    iphlpapi.GetAdaptersInfo(None, ctypes.byref(size))      # 先问需要多大
    buf = ctypes.create_string_buffer(size.value)
    if iphlpapi.GetAdaptersInfo(buf, ctypes.byref(size)) != 0:
        return ""

    skip = ("vmware", "virtual", "hyper-v", "wsl", "loopback", "tap", "npcap", "vpn", "ppp")
    found = []
    node = ctypes.cast(buf, ctypes.POINTER(IP_ADAPTER_INFO))
    while node:
        a = node.contents
        desc = a.Description.decode("mbcs", "replace").lower()
        ip = a.IpAddressList.IpAddress.decode("ascii", "replace").rstrip("\x00")
        gw = a.GatewayList.IpAddress.decode("ascii", "replace").rstrip("\x00")
        if (ip.startswith("192.168.") and gw not in ("", "0.0.0.0")
                and not any(k in desc for k in skip)):
            found.append(ip)
        node = a.Next
    return found[0] if found else ""


def local_ip():
    """本机在局域网里的 IPv4。

    优先物理网卡的 192.168.x.x（虚拟网卡、VPN 会被跳过）；实在找不到
    才退回默认路由的出口地址。拿不到就返回 127.0.0.1。
    """
    try:
        ip = _lan_ipv4_from_adapters()
        if ip:
            return ip
    except (AttributeError, OSError, ValueError):
        pass
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))          # UDP connect 不发包，只为让系统选出口网卡
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    global JOB_HANDLE
    ap = argparse.ArgumentParser(description="dlqq-llm 本地 WebUI")
    ap.add_argument("--port", type=int, default=0, help="端口，0 = 自动挑一个空闲的")
    ap.add_argument("--host", default="127.0.0.1", help="监听地址，默认只绑本机")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    ap.add_argument("--lan-port", type=int, default=49135,
                    help="局域网对话页端口，默认 49135")
    ap.add_argument("--lan-host", default="0.0.0.0",
                    help="局域网对话页监听地址，默认所有网卡")
    ap.add_argument("--no-lan", action="store_true", help="不启动局域网对话页")
    a = ap.parse_args()

    JOB_HANDLE = _make_job_object()
    STATE["token"] = secrets.token_urlsafe(12)
    want = a.port or 49134
    port = find_free_port(want)
    common.hide_dir(os.path.join(ROOT, ".ai_res"))
    os.makedirs(LOG_DIR, exist_ok=True)

    url = f"http://{a.host}:{port}/?t={STATE['token']}"
    print("=" * 66)
    print("  dlqq-llm WebUI")
    print("=" * 66)
    print(f"[地址] {url}")
    if port != want:
        print(f"[提示] 端口 {want} 被占用，已改用 {port}")

    lan = None
    if not a.no_lan:
        try:
            lan = ThreadingHTTPServer((a.lan_host, a.lan_port), LANHandler)
        except OSError as e:
            print(f"[警告] 局域网对话页没起来（{a.lan_host}:{a.lan_port}）：{e}")
        else:
            lan.daemon_threads = True
            threading.Thread(target=lan.serve_forever, daemon=True,
                             name="lan-chat").start()
            print(f"[内网] http://{local_ip()}:{a.lan_port}/   （只有对话页）")
            print("[提示] 首次可能需要在 Windows 防火墙里允许 Python 访问专用网络")

    print("[注意] 关掉本窗口 = 退出，所有子进程（训练/服务）一起结束")
    print("[提示] 关浏览器不影响运行，重新访问上面的地址即可")
    print("=" * 66, flush=True)

    if not a.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    httpd = ThreadingHTTPServer((a.host, port), Handler)
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[退出] 收到 Ctrl+C")
    finally:
        stop_all()
        if lan:
            try:
                lan.server_close()
            except OSError:
                pass
        print("[退出] 子进程已清理")


if __name__ == "__main__":
    main()
