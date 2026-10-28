# -*- coding: utf-8 -*-
r"""对话核心（WebUI 与命令行共用）
================================================================
本文件既是模块也是脚本：
    · 被 WebUI 调用时用 ChatSession —— 加载一次、反复对话、可流式取字；
    · 直接运行时就是原来的命令行对话（保留给调试用，不是给用户的入口）。

用法（在工程根目录执行）：
    python code\chat.py model\chat_demo\chat_demo.pt
    python code\chat.py model=model\chat_demo\chat_demo.pt
    python code\chat.py model/chat_demo.pt --once "你好"

可覆盖的配置键：model / temperature / top_k / max_new / chat_mode / device。
优先级：命令行显式 > properties\<模型名>.properties > 模型文件里存的 > 默认值。

交互命令：:q 退出、:p 看设置、:t 0.7 改温度、:n 200 改长度、:raw 切换续写/对话模式。
"""
import os
import re
import sys

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True, write_through=True)
    except (AttributeError, ValueError):
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import common                                                   # noqa: E402

ROOT = common.find_root()
DEFAULT_MODEL = os.path.join(ROOT, "model", "chat_demo", "chat_demo.pt")


# ==================================================================
# 模型定义：统一在 code\model.py，这里只导入（不要再复制一份）
# ==================================================================
from model import GPT                                          # noqa: E402


def load_chat_props(model_path):
    r"""读 properties\<模型名>.properties 里的对话设置（改完不用重训就生效）"""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    p = os.path.join(ROOT, "properties", stem + ".properties")
    if not os.path.isfile(p):
        return {}, None
    return common.load_properties(p), p


def _display(full, prompt, chat_mode):
    """把「到目前为止生成的全部字符」裁成应该展示给用户的部分。"""
    s = full
    if chat_mode:
        if "助手：" not in s:
            return ""
        s = s.split("助手：", 1)[1]
        s = s.split("用户：", 1)[0]
    else:
        if s.startswith(prompt):
            s = s[len(prompt):]
    return s


RE_TAIL_MARK = re.compile(r"\n[ \t]*(用|用户|助手)[：:]?[ \t]*$")


def _clean_final(s):
    """收尾清理：切掉下一轮的「用户：」以及被截断的残留标记（常见的是单独一行的「用」）。"""
    s = s.rstrip()
    s = RE_TAIL_MARK.sub("", s)
    for mark in ("\n用户：", "\n用户", "\n助手："):
        i = s.find(mark)
        if i >= 0:
            s = s[:i]
    return s.rstrip()


class ChatSession:
    """一次加载、反复对话；reply_stream 逐字产出，供 WebUI 流式显示。"""

    def __init__(self, path=None, device=None):
        self.path = os.path.abspath(path or DEFAULT_MODEL)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(f"找不到模型文件：{self.path}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.config = ckpt.get("config", {})
        self.chars = ckpt["chars"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.model = GPT(len(self.chars), self.config["block_size"],
                         self.config["n_layer"], self.config["n_head"],
                         self.config["n_embd"], 0.0).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.params = dict(temperature=0.8, top_k=40, max_new=180,
                           chat_mode=bool(self.config.get("chat_mode", True)))
        # 模型文件里存的采样参数
        for key, ck in (("temperature", "chat_temperature"), ("top_k", "chat_top_k"),
                        ("max_new", "chat_max_new")):
            if ck in self.config:
                self.params[key] = self.config[ck]
        # properties 覆盖模型里存的
        self.props, self.props_path = load_chat_props(self.path)
        if "chat_mode" in self.props:
            self.params["chat_mode"] = self.props["chat_mode"].strip().lower() in (
                "1", "true", "yes", "y", "on")
        for key, prop in (("temperature", "chat_temperature"), ("top_k", "chat_top_k"),
                          ("max_new", "chat_max_new")):
            if prop in self.props:
                try:
                    self.params[key] = type(self.params[key])(self.props[prop])
                except (TypeError, ValueError):
                    pass

    # ---- 信息与参数 ----
    def info(self):
        return {
            "path": self.path,
            "name": os.path.splitext(os.path.basename(self.path))[0],
            "device": self.device,
            "params_m": round(sum(p.numel() for p in self.model.parameters()) / 1e6, 2),
            "vocab": len(self.chars),
            "block_size": self.config.get("block_size"),
            "chat_mode": self.params["chat_mode"],
            "temperature": self.params["temperature"],
            "top_k": self.params["top_k"],
            "max_new": self.params["max_new"],
            "props": os.path.basename(self.props_path) if self.props_path else None,
        }

    def set_params(self, **kw):
        for k, v in kw.items():
            if k in self.params and v is not None:
                self.params[k] = type(self.params[k])(v)
        return dict(self.params)

    # ---- 生成 ----
    def reply_stream(self, text, temperature=None, top_k=None, max_new=None):
        """逐段产出回复文本；调用方可以随时停止迭代来中断生成。"""
        p = dict(self.params)
        if temperature is not None:
            p["temperature"] = float(temperature)
        if top_k is not None:
            p["top_k"] = int(top_k)
        if max_new is not None:
            p["max_new"] = int(max_new)
        chat_mode = bool(p["chat_mode"])

        prompt = f"用户：{text}\n助手：" if chat_mode else text
        bad = [c for c in prompt if c not in self.stoi]
        if bad and chat_mode:
            # 词表里没有「用户：」「助手：」→ 这个模型不是对话语料训的，自动切续写
            chat_mode = False
            prompt = text
            bad = [c for c in prompt if c not in self.stoi]
        if bad:
            prompt = "".join(c for c in prompt if c in self.stoi)
        if not prompt:
            return

        idx = torch.tensor([[self.stoi[c] for c in prompt]], device=self.device)
        full = prompt          # 从提示词开始累计，才能正确切掉「助手：」之前的部分
        shown = ""
        for tid in self.model.sample(idx, p["max_new"], p["temperature"], p["top_k"]):
            full += self.itos[tid]
            if chat_mode:
                seg = full.split("助手：", 1)[-1]
                cut = seg.find("用户")          # 不带冒号也要停，否则结尾会挂着「用户」两个字
                if cut >= 0:
                    full = full[: len(full) - (len(seg) - cut)]
                    piece = _display(full, prompt, chat_mode)
                    if piece != shown:
                        yield piece[len(shown):]
                    return
            piece = _display(full, prompt, chat_mode)
            if piece != shown:
                yield piece[len(shown):]
                shown = piece

    def reply(self, text, **kw):
        return _clean_final("".join(self.reply_stream(text, **kw)))


# ==================================================================
# 命令行（调试用）
# ==================================================================
def _main(argv):
    cfg = dict(model=None, temperature=None, top_k=None, max_new=None,
               chat_mode=None, device=None)
    once = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--once":
            i += 1
            once = argv[i] if i < len(argv) else "你好"
        elif a in ("-h", "--help"):
            print(__doc__)
            return 0
        elif a == "--device":
            i += 1
            cfg["device"] = argv[i]
        elif a.startswith("-"):
            print(f"[用法] 不认识的参数：{a}（参数是 key=value 形式，例如 model=model/chat_demo.pt）")
            return 2
        elif "=" in a:
            k, v = a.split("=", 1)
            if k in cfg:
                cur = cfg[k]
                if cur is None:
                    cfg[k] = v
                else:
                    try:
                        cfg[k] = type(cur)(v)
                    except ValueError:
                        cfg[k] = v
        else:
            cfg["model"] = a
        i += 1

    path = cfg["model"] or DEFAULT_MODEL
    path = common.resolve_path(ROOT, path)
    if not os.path.isfile(path):
        print(f"[错误] 找不到模型文件：{path}")
        print("       先训练一个：python code\\train.py --config properties\\chat_demo.properties")
        return 1

    s = ChatSession(path, device=cfg["device"])
    over = {k: v for k, v in cfg.items() if k in ("temperature", "top_k", "max_new", "chat_mode")}
    s.set_params(**over)
    info = s.info()
    print(f"[模型] {info['path']} | 参数量 {info['params_m']}M | 词表 {info['vocab']} | 设备 {info['device']}"
          + ("" if info["chat_mode"] else " | 续写模式"))
    print(f"[采样] 温度={info['temperature']} top_k={info['top_k']} 长度={info['max_new']}"
          + (f"（来自 {info['props']}）" if info["props"] else "（模型配置/默认值）"))

    if once is not None:
        print("\n你  >", once)
        print("它  >", s.reply(once))
        return 0

    print("\n输入内容后回车，跟它对话。:q 退出，:p 看设置\n")
    while True:
        try:
            line = input("你  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":q", ":quit", ":exit"):
            break
        if line == ":p":
            info = s.info()
            print(f"     温度={info['temperature']} top_k={info['top_k']} "
                  f"长度={info['max_new']} 模式={'对话' if info['chat_mode'] else '续写'}")
            continue
        if line == ":raw":
            print(f"     模式 → {'续写' if s.params['chat_mode'] else '对话'}")
            s.set_params(chat_mode=not s.params["chat_mode"])
            continue
        if line.startswith(":t"):
            s.set_params(temperature=float(line.split()[1]))
            print(f"     温度 → {s.params['temperature']}")
            continue
        if line.startswith(":n"):
            s.set_params(max_new=int(line.split()[1]))
            print(f"     长度 → {s.params['max_new']}")
            continue
        print("它  >", s.reply(line), "\n")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
