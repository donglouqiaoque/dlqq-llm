# -*- coding: utf-8 -*-
r"""对话子进程：由 WebUI 用「用户 Python」启动
================================================================
WebUI 自己跑在只带标准库的运行环境里，不能 import torch；网页对话
（本机 + 局域网）都把活交给这个子进程，父子之间用 stdin/stdout 的
JSON 行通信。

用法（WebUI 内部调用，不面向用户）：
    python code\chat_worker.py --config dlqq_llm --parent-pid 1234

协议：
    请求  {"id":1,"op":"chat","text":"你好"}
          {"op":"info"} / {"op":"quit"}
    事件  {"event":"ready","info":{...}}         启动完成
          {"id":1,"event":"meta","info":{...}}
          {"id":1,"event":"token","text":"你"}
          {"id":1,"event":"done","text":"你好呀"}
          {"id":1,"event":"error","error":"..."}
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import common                                                     # noqa: E402


def send(obj):
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def main():
    common.setup_stdio()
    # 父子之间固定用 UTF-8 的 JSON 行；不跟着控制台代码页走，
    # 否则中文在 cp936 下会解成乱码，模型收到空提示就只能回空串。
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    ap = argparse.ArgumentParser(description="WebUI 的对话子进程")
    ap.add_argument("--config", required=True, help="properties 里的配置名")
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="父进程 pid；父进程退出时本进程跟着退出")
    a = ap.parse_args()

    if a.parent_pid:
        common.start_parent_watchdog(a.parent_pid)

    try:
        import chat
    except Exception as e:                    # 用户 Python 没装 torch 时走这里
        send({"event": "error", "error": f"import chat 失败：{type(e).__name__}: {e}"})
        return 1

    try:
        sess = chat.ChatSession(config=a.config)
    except Exception as e:
        send({"event": "error", "error": f"{type(e).__name__}: {e}"})
        return 1

    send({"event": "ready", "info": sess.info()})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError:
            continue
        op = req.get("op")
        rid = req.get("id")
        if op == "quit":
            break
        if op == "info":
            send({"id": rid, "event": "meta", "info": sess.info()})
            continue
        if op != "chat":
            send({"id": rid, "event": "error", "error": f"未知 op：{op}"})
            continue
        text = (req.get("text") or "").strip()
        if not text:
            send({"id": rid, "event": "error", "error": "消息为空"})
            continue
        try:
            send({"id": rid, "event": "meta", "info": sess.info()})
            pieces = []
            for piece in sess.reply_stream(text):
                pieces.append(piece)
                send({"id": rid, "event": "token", "text": piece})
            send({"id": rid, "event": "done",
                  "text": chat._clean_final("".join(pieces))})
        except Exception as e:
            send({"id": rid, "event": "error", "error": f"{type(e).__name__}: {e}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
