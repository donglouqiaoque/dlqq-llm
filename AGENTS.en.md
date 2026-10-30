# AGENTS.md — dlqq-llm-train engineering conventions

> For AI assistants (DeepSeek Harness / Qwen Code / Claude Code / Cursor …): hard rules,
> directory layout, startup chain, API contract and runtime behaviour.
> User guide: [README.md](README.md) (Chinese) / [README.en.md](README.en.md).
> This file is the English twin of [AGENTS.md](AGENTS.md); **change both together**.
>
> Companion repositories:
> - `dlqq-llm-model`: published `.pt` weights only, no scripts at all;
> - `dlqq-llm-dataset`: private corpus backup, unrelated to the code path.

## 1. Hard rules

1. **Fixed stack**: Python 3.10+ / PyTorch 2.x. Backend may use only `torch` and `numpy`
   as third-party packages; everything else must be stdlib. Frontend: plain HTML/CSS/JS,
   **no framework, no CDN** (must work offline).
2. **No tokenizer / HuggingFace / transformers**: this is a character-level model; the vocabulary
   is built by `code/train.py` itself.
3. **Never run git commands** (not even `status` / `diff` / `log`) and **never read or write
   the `.git` directory**. Ask the user about commit/staging state.
4. **No scripted bulk rewrites of source files**: edit file by file, confirm each change;
   for large refactors, list the plan first.
5. **AI must not run side-effecting actions**: training, corpus download/cleaning, deleting files,
   starting the WebUI or the bridge server. Read-only checks, syntax/static checks and doc edits
   are fine.
6. **`.bat` files must stay pure ASCII**; `.py` / `.properties` / `.md` / `.html` / `.js` / `.css`
   are UTF-8 without BOM; text files use CRLF on Windows.
7. **Temp files live in `.ai_res\`** (git-ignored): caches, logs, backups. Never the system temp dir.
8. **No hard-coded local paths** in docs or comments; examples must stay generic.
9. **Weights and corpora stay out of the repo**: `model/*` and `src/*` are ignored. Weights are
   published in `dlqq-llm-model` as `model\<name>\<name>.pt` plus the same-named `.pt.sha256`;
   corpora are backed up in `dlqq-llm-dataset`.
10. **README / AGENTS must not describe ignored paths in detail**: only hints such as
    "caches may live under `.ai_res`" are allowed; the details go to `.ai_res\AIRES.md`.

## 2. Layout (current architecture)

```
Everything except start.bat lives under code\ and web\; script\ holds the launcher only.
script/                 Double-click entry point (the only one)
  start.bat             find python → run code\webui.py in the foreground → open browser
  logo.ico              shortcut icon
code/                   all Python
  webui.py              local web service: static files + JSON API + SSE + child processes
  train.py              training (data pipeline + model + loop + sampling)
  chat.py               chat core: ChatSession (load / reply_stream / set_params)
  http_server.py        OpenAI-compatible bridge (ThreadingHTTPServer, lazy multi-model)
  common.py             find_root / order-preserving properties I/O / paths / watchdog / hide dir
  model_signature.py    weight SHA-256 signatures (compute / verify, stdlib only)
web/                    frontend, served by webui.py (no separate web server)
  index.html            five tabs: train / chat / server / config / signature
  chat.html             LAN read-only chat page (served by LANHandler)
properties/             four training configs (open-sourced, comments written for outsiders)
model/                  local weights (not in git; model\<name>\<name>.pt)
src/                    corpora (not in git)
.ai_res/                scratch area (caches / logs / backups / local scripts, not in git)
```

- The project root is found by walking up to the directory containing `code\train.py`
  (`common.find_root`), so moving scripts around does not break anything.
- Model id = directory name (`model\chat_demo\chat_demo.pt` → `chat_demo`); when directory and
  file name differ, write `dir/file`.

## 3. Startup chain and process model

```
script\start.bat
  └─ python code\webui.py            ← foreground; the console window is the lifecycle anchor
       ├─ child python code\train.py --config <cfg> --parent-pid <webui_pid>
       ├─ child python code\http_server.py --port <port> --events --parent-pid <webui_pid>
       ├─ child python code\model_signature.py --mode compute|verify --parent-pid <webui_pid>
       └─ in-process ChatSession (torch + model stay in the webui process)
```

- **No user-facing CLI**: docs must not advertise command-line usage. `--config` / `--parent-pid` /
  `--events` are an **internal protocol** between the WebUI and its children; no compatibility promise.
- **Lifecycle**: closing the cmd window of `start.bat` means exit — children are `terminate()`d,
  and `taskkill /F /T` is the fallback after 5 seconds. **Closing the browser tab is not an exit**;
  training keeps running and you can simply reopen the URL.
- **Watchdog**: children poll `--parent-pid` every 3 s via `ctypes` `WaitForSingleObject` on the
  parent handle and call `os._exit(1)` when the parent is gone (protects against a hard-killed webui).
- **Job Object**: children also join a `KILL_ON_JOB_CLOSE` job, so when the webui process dies the
  kernel kills every process in the job — even if the cmd window is force-closed mid-run.
- **Ports**: the console defaults to `127.0.0.1:49134` (falls back to a random free port with a
  notice if busy); the LAN read-only chat page defaults to `0.0.0.0:49135`
  (`--lan-port` changes it, `--no-lan` disables it).
- **Security**: the console binds `127.0.0.1` only, generates a one-shot token per start and checks
  it on every `/api/*` call. The LAN port has no token but only serves the chat page,
  `/api/models` and `POST /api/chat`.

## 4. WebUI API contract (changing it requires updating `web/index.html` too)

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/env` | interpreter + dependency probe (version / torch / CUDA) |
| GET | `/api/configs` | config list with parsed key/values |
| GET | `/api/models` | `model\**\*.pt` list (with signed flag) |
| GET | `/api/chat/configs` | configs that map to an existing weight (name + model) |
| GET | `/api/corpora` | corpus directories under `src\` (file count, size) |
| GET | `/api/train/status` `train/log` | training state and incremental log |
| GET | `/api/server/status` | bridge state and port |
| GET | `/api/events` | SSE: `train-log` / `server-log` / `server-chat` |
| POST | `/api/config/save` | line-level value update (keeps comments/order), backs up first |
| POST | `/api/train/start` `train/stop` | start / stop the training child |
| POST | `/api/chat` | chat, returns SSE: `meta` / `token` / `done` / `error` |
| POST | `/api/server/start` `server/stop` | start / stop the bridge child |
| POST | `/api/sign` | `mode=compute|verify`, runs the signature tool |

- The bridge prints one `@@EV@@ {json}` line per completed chat (`--events`); the webui reader
  turns it into an SSE `server-chat` event, which the "server" tab renders as a Q&A list.
- SSE frames are `data: <json>\n\n`; one thread per connection, so new long-lived endpoints
  must consider the thread budget.
- **LAN port** (`LANHandler`) only allows `GET /`, `/chat.html`, `/logo.ico`, `/api/models`,
  `/api/chat/configs` and `POST /api/chat`; everything else returns 403. `POST /api/chat` only
  accepts a `config` that appears in the config list. Changing LAN endpoints means updating
  `web/chat.html` too.

## 5. Config (properties) conventions

- **Syntax**: one `key=value` per line; `#` starts a whole-line comment.
- **No trailing comments**: the parser uses `split("=", 1)`, so `key=1  # note` breaks the value.
- **Keep the four configs isomorphic**: `chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem`
  must share the same key set, order and sections.
- **Only keys that exist in `CONFIG`**: `code/train.py`'s `CONFIG` is the single source of truth.
- **Saving rules**: always go through `common.save_properties` (value-only edits, comments and
  order preserved) and back up to `.ai_res\cache\cfg-backup\` first.
- **`python=`**: empty = use python from PATH; non-empty = use exactly that interpreter, abort if
  it is missing or lacks torch/numpy.
- **Resume is a UI action**: the "Resume training" button sends `resume=true`; the config files no
  longer carry a `resume` key.

| Key | Meaning |
|---|---|
| `data_path` | corpus **directory** (relative to the project root) |
| `model` | model name → `model\<name>\<name>.pt` |
| `cache_dir` | tokenisation cache dir (default `.ai_res/cache`) |
| `chat_mode` / `chat_temperature` / `chat_top_k` / `chat_max_new` | chat-time settings |
| `n_layer` / `n_head` / `n_embd` / `block_size` / `batch_size` | architecture and tokens per step |
| `max_iters` / `eval_interval` / `eval_iters` | step count and eval cadence |
| `lr` / `warmup_iters` / `lr_decay` / `min_lr_frac` / `weight_decay` / `grad_clip` / `dropout` / `seed` | optimisation |
| `grad_accum` / `grad_checkpoint` | effective batch / memory saving |
| `save_every` / `save_dtype` / `save_optimizer` | periodic checkpoint / dtype / keep optimizer |
| `max_vocab` / `val_frac` | vocabulary cap / tail validation split |
| `python` / `confirm` | interpreter / confirmation |

## 6. Runtime behaviour (understand before editing)

**Data pipeline**: read in 4M-char chunks → `np.bincount` frequency count → keep top `max_vocab`
chars → `searchsorted` encode to uint16 cache → train via `np.memmap` random reads. Memory does not
grow with corpus size. Cache fingerprint = source absolute path + per-file (rel path, size, mtime,
format, Q/A fields) + `max_vocab` + `val_frac`; **changing any of these invalidates the cache**
(≈5 minutes to re-tokenise 15 GB).

**Validation** is split from the **tail** of the corpus; shuffle category-sorted corpora first.

**Context auto-shrink**: `_safe_T = min(train tokens, val tokens) - 2`; a larger `block_size`
is reduced automatically with a warning.

**Checkpoint policy**: save when validation loss improves; additionally every `save_every` steps.

**Chat parameter precedence**: explicit args > `properties\<config>.properties` > values stored in
the `.pt` > defaults (used by both the web chat and the bridge; implemented twice in `chat.py` and
`http_server.py` — keep them in sync).

**Stopping**: prompts use `用户：` / `助手：`; generation stops at the next `用户：`. If the vocabulary
lacks those markers, the session falls back to plain continuation mode.

**Chatting while training**: only a **different** model is allowed; the model named by the running
training config is rejected (it would slow training and may read a half-written file).

## 7. Code style

- **File header**: `# -*- coding: utf-8 -*-` plus an `r"""..."""` block (purpose, usage, caveats).
- **Encoding guard**: call `common.setup_stdio()` (or reconfigure stdout/stderr to UTF-8 with
  `errors="replace"`) in every entry point.
- **Comments in Chinese**, explaining *why*, not restating the code; record the measurements that
  justify defaults.
- **Fail fast**: invalid args/env/data → `sys.exit(1)` with an actionable hint; no guessing fallbacks.
- **Never swallow exceptions**: catch concrete types; print or re-raise.
- **Print format**: `[tag] message` with `[数据]` / `[训练]` / `[设备]` / `[错误]` / `[警告]`;
  use `flush=True` inside training loops.
- **Naming**: `snake_case` functions/variables, `UPPER_SNAKE_CASE` constants, `CamelCase` classes;
  config keys match `CONFIG`.
- **No type annotations**: stay consistent with the existing code.
- **No logging framework**: `print(..., flush=True)` or the webui log queue.
- **Known duplication**: the model is defined in `code/train.py`, `code/chat.py` and
  `code/http_server.py` (training variant has loss, inference variants are slimmer).
  **Any architecture change must be applied to all three**, otherwise `load_state_dict` fails.
  Extracting `code/model.py` is planned.

## 8. Known limitations

- **Character level + short context**: with `block_size` 256 only the tail of a long message is seen.
- **Small models memorise**: they replay training answers; do not benchmark on the training set.
- **The bridge is not an agent**: no tool calling, no planning; it only validates the pipeline.
- **The WebUI console is local-only**: it binds `127.0.0.1` with a one-shot token; the LAN side
  exposes only the read-only chat page (`0.0.0.0:49135`, no auth — anyone on the LAN can chat).
- **CPU training** is very slow and has no bf16; use it only to smoke-test tiny models.

## 9. References

- Weights: [ModelScope · donglouqiaoque/dlqq-llm](https://www.modelscope.cn/models/donglouqiaoque/dlqq-llm)
- Source mirrors: [GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch docs: <https://pytorch.org/docs/stable/index.html>
- nanoGPT: <https://github.com/karpathy/nanoGPT>
