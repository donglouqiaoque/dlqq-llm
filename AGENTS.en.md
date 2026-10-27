# AGENTS.md — dlqq-llm-train engineering conventions

> For AI assistants (DeepSeek Harness / Qwen Code / Claude Code / Cursor …):
> hard rules, directory layout, config syntax and runtime behaviour.
> User documentation: [README.md](README.md) (English: [README.en.md](README.en.md)).
> Chinese version of this file: [AGENTS.md](AGENTS.md).
> README and AGENTS are the **only** documentation, and each exists in two languages —
> changing one means changing the other.

## 1. Hard rules (must follow)

1. **Fixed stack**: Python 3.10+ / PyTorch 2.x. The training and chat paths may use only
   `torch` and `numpy` as third-party libraries; everything else is the standard library.
2. **No tokenizer / HuggingFace / transformers**: this is a character-level model; the
   vocabulary is counted and built by `code/train_llm.py` itself.
3. **Never run any git command**: including read-only `git status` / `diff` / `log` /
   `show`, and `git add` / `commit` / `push` / `restore`. Version control belongs to the
   user; the AI does not touch it at all.
4. **Never rewrite source in bulk with scripts**: no PowerShell / Python / sed / awk batch
   find-and-replace or automatic rewriting of source files. Every change must be made
   file by file, location by location, with an editing tool; for large refactors, produce a
   list first and get the user's approval, then apply the items one by one.
5. **Never run scripts with side effects**: training (`script\train.bat`,
   `code\train_llm.py`), corpus downloads (`script\fetch_chat_demo.bat`,
   `.ai_res\script\fetch_*.bat`), deletions and bulk cleanups are for the user to run by
   hand. The AI only does read-only work: reading files, listing directories, syntax and
   static checks.
6. **`.bat` files are ASCII only**: comments, `echo`, prompts — all English; Chinese text
   lives in Python. Reason: cmd's default code page is GBK, so UTF-8 Chinese is garbled and
   can even break batch parsing.
7. **Encoding and line endings**: `.py` / `.properties` / `.md` are UTF-8 without BOM;
   `.bat` is pure ASCII; on Windows text files use CRLF.
8. **Temp files go to `.ai_res\` only**: caches, downloads and throwaway artifacts belong in
   `.ai_res\` (git-ignored); never the system temp directory. Helper scripts go in
   `.ai_res\script\`, not at the root of `.ai_res\`.
9. **Never hard-code machine-specific paths**: `python=` in `properties` is empty by default
   (use PATH); examples in comments must be generic (e.g. `C:\Python312\python.exe`) and must
   not contain someone's `D:\Soft\...`, user name or conda environment path.
10. **Weights and corpora never enter the repository**: `model/*` and `src/*` are git-ignored
    (`chat_demo*` excepted); a corpus is always a **directory**, never a single file. Released
    weights live in the `dlqq-llm-model` repo and must be byte-identical to the local `model\`,
    each with a `.pt.sha256` alongside.
11. **Never read or write the `.git` directory**: that includes `.git\index`,
    `.git\COMMIT_EDITMSG`, `.git\logs\*` and every other file — **even read-only inspection**
    (checking mtimes counts too). Committed-vs-staged state can only come from the user.

## 2. Config files (`properties`) conventions

- **Syntax**: one `key=value` per line; a line starting with `#` is a comment.
- **No trailing comments**: the parser uses `split("=", 1)`, so `block_size=128  # note`
  swallows `# note` into the value and fails (or silently misbehaves). Comments must be on
  their own line.
- **Keep the four configs isomorphic**: `chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem`
  share the same key set, order and section layout; restructure one and the other three must
  follow.
- **Only keys that exist in `CONFIG`**: `CONFIG` in `code/train_llm.py` is the single source
  of truth; unknown keys print "unknown option" and are ignored.
- **`python=` semantics**: empty = use `python` / `python3` / `py` from PATH; non-empty =
  **that interpreter only**, and a missing path, a directory, or missing `torch`/`numpy`
  aborts immediately. **No fallback search.**
- **`resume=true`** only continues when the checkpoint has `iter` / `optimizer`; without a
  checkpoint training simply starts from scratch.

| Key | Meaning |
|---|---|
| `data_path` | Corpus **directory** (relative to the repo root); a file path errors with a migration hint |
| `model` | Model name; lands at `model\<name>\<name>.pt` |
| `cache_dir` | Tokenised-cache directory; keep it `.ai_res/cache` |
| `chat_mode` / `chat_temperature` / `chat_top_k` / `chat_max_new` | Dialogue parameters, read at load time |
| `n_layer` / `n_head` / `n_embd` / `block_size` / `batch_size` | Model shape and tokens per step |
| `max_iters` / `eval_interval` / `eval_iters` | Steps and evaluation cadence |
| `lr` / `warmup_iters` / `lr_decay` / `min_lr_frac` / `weight_decay` / `grad_clip` / `dropout` / `seed` | Optimisation |
| `grad_accum` / `grad_checkpoint` | Effective batch when VRAM is short / checkpointing |
| `resume` / `save_every` / `save_dtype` / `save_optimizer` | Resume and checkpoint size |
| `max_vocab` / `val_frac` | Vocabulary cap / tail fraction used for validation |
| `python` / `confirm` | Interpreter / ask once more |

## 3. Directory layout

```
code/                 Training and inference (works without properties; CLI overrides allowed)
  train_llm.py        Data pipeline + model + training loop + sampling, 840 lines
  chat.py             Loads a .pt and chats; model definition mirrors train_llm.py (copy, not import)
script/               Entry points (double-click)
  train.bat           pick a config → helper → train_llm.py
  chat.bat            pick a model → helper → chat.py
  http_server.bat     start the OpenAI-compatible bridge
  torch_check.bat     9-item environment self-check
  fetch_chat_demo.bat download exam data → src\dlqq_exam\dlqq_exam.jsonl
  train_chat_helper.py  interactive logic behind the .bat files (config pick, interpreter check, device probe)
  http_server.py      the bridge itself (self-contained, torch + stdlib only)
  torch_check.py      the individual checks
  model_signature.py  weight SHA-256 signatures (compute / verify, stdlib only, no torch)
  compute_signature.bat / verify_signature.bat  double-click entry points
  fetch_chat_demo.py  exam download and format conversion
properties/           four training configs (open-sourced; comments are written for outsiders)
src/                  corpora (not in the repo)
model/                models (not in the repo; model\<name>\<name>.pt)
.ai_res/              scratch (not in the repo): cache / doc / script / git_log
```

- The repo root is found by walking up until a directory contains `code\train_llm.py`, so
  scripts keep working if moved.
- Model id = directory name (`model\chat_demo\chat_demo.pt` → `chat_demo`); `dir/file` is used
  only when the directory and file names differ.

## 4. Runtime behaviour (understand before changing code)

**Data pipeline**: read in 4M-character chunks → `np.bincount` for character frequency → keep
the top `max_vocab` → `searchsorted` into uint16 written to `.ai_res\cache\data-<fp>.bin` →
training reads it with `np.memmap`. Memory use is independent of corpus size. The cache
fingerprint = source absolute path + per-file (relative path, size, mtime, format, Q&A fields)
+ `max_vocab` + `val_frac`; **changing any of these re-tokenises**.

**Validation** is sliced from the **tail** of the corpus. Shuffle categorised data first,
otherwise validation loss is meaningless.

**Context shrinks automatically**: `_safe_T = min(train tokens, val tokens) - 2`; a larger
`block_size` is reduced with a warning. On tiny corpora the configured value is not what
training actually uses.

**Saving**: only when validation loss improves; `save_every > 0` adds periodic checkpoints.

**Dialogue parameter priority** (implemented twice — `chat.py` and `http_server.py` — and the
two must stay identical):
`explicit CLI argument` > `properties\<model>.properties` > `value stored in the checkpoint` > `default`.

**Generation stop**: prompts use `用户：` / `助手：`; generation stops at a complete marker
(with colon), trailing bare markers are cleaned up, and the text is cut at the next `用户：`.

**The bridge is opinion-free** (`script\http_server.py`):
- drops `system` / `developer` roles (the model was never trained with them);
- ignores `tools` / `tool_choice` / `functions` **without erroring** and **never returns
  `tool_calls`** (`--reject-tools` turns it into a 400 if you want strictness);
- on context overflow drops the **oldest** turns; a single over-long turn keeps its tail;
- no content filtering, scoring or keyword matching; `max_tokens` counts **characters**, with
  a hard cap of 600.

## 5. Workflow (binding on the AI)

1. **Read before editing**: read the whole target file first; never edit from memory.
2. **Static checks only after changes**: e.g. `python -c "import ast; ast.parse(open(...).read())"`.
   **Do not** run training or downloads just to verify.
3. **Explain the impact of default changes**: `max_vocab` / `val_frac` invalidate caches,
   `block_size` changes VRAM and context, `save_dtype` changes checkpoint size — say so.
4. **Keep the four configs in sync**: `chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem`
   must always share the same key set.
5. **Keep docs bilingual**: `README.md` ↔ `README.en.md`, `AGENTS.md` ↔ `AGENTS.en.md`;
   sections, commands and numbers must match — never update only one side.
6. **Do not touch weights**: never modify, copy or delete `model\*.pt`; retraining is the
   user's job.

### Commit messages

- **Where they live**: write each one to `.ai_res\git_log\<yyyy-MM-dd_HH-mm-ss>.txt` — UTF-8
  without BOM, CRLF, trailing newline. The file holds the message and nothing else (no "here
  is the commit message" preamble), ready to copy verbatim when pushing.
- **Format** (Conventional Commits 1.0.0):
  - first line `<type>: <description>`, type from `feat` / `fix` / `docs` / `refactor` /
    `test` / `chore` / `build` / `perf` / `style`; imperative mood, no trailing period
  - first line at most 50 columns (a CJK character counts as two)
  - blank line, then the body: **why** first (context, motivation, impact), then a `- `
    list of **what changed**
  - body lines at most 72 columns; never paste code
- **Only what is actually committed**: files excluded by `.gitignore` must not appear, **not
  even as paths** — here that is `model\` except `chat_demo*`, `src\` except `chat_demo*`, and
  all of `.ai_res\`. Even when describing the behaviour of a tracked file ("its default output
  now goes to the exam corpus directory"), never write the ignored path; phrase it without one.
  Conversely, every tracked file touched by the commit must appear, especially weights, corpora
  and `.gitignore`.
- **Check the state first**: the AI can neither run git commands nor read the `.git` directory,
  so only the user can tell committed from still-staged work. Never write from assumption.

## 6. Code style

- **File header**: `# -*- coding: utf-8 -*-` followed by an `r"""..."""` block stating purpose,
  usage and caveats.
- **Encoding guard**: every `.py` entry point starts with
  `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")`. Otherwise the Windows
  console (GBK) raises `UnicodeEncodeError` on `→` or `⚠️` — training finishes, then the last
  screen dies.
- **Comments in Chinese**: explain *why*, never restate the code. Record the measured basis for
  a value (VRAM, wall time, corpus size) so the next person knows whether it can be changed.
- **Fail fast**: invalid arguments, environment or data exit with `sys.exit(1)` plus an
  **actionable fix** (which line to edit, which `.bat` to run). **Never guess or fall back** —
  no automatic interpreter search, no automatic directory swapping.
- **Never swallow exceptions**: `except` must name a concrete type, then either print or
  re-raise; `except: pass` is forbidden.
- **Print format**: `[tag] message`, with tags `[数据]` / `[训练]` / `[设备]` / `[错误]` /
  `[警告]`; use `flush=True` inside the training loop, otherwise the log sits in the buffer and
  looks like a hang.
- **Naming**: `snake_case` for functions and variables, `UPPER_SNAKE_CASE` for constants,
  `CamelCase` for classes; config keys stay `snake_case` and must also exist in `CONFIG`.
- **No type hints**: consistent with the existing code — describe parameters in the docstring.
- **The model definition exists three times**: `code/train_llm.py` (training), `code/chat.py`
  (chat) and `script/http_server.py` (bridge). Any structural change (layers, dimensions,
  weight tying) must be applied to all three, or loading a checkpoint fails with key mismatches.
- **No logging framework**: `print(..., flush=True)` is enough; training and chat logs go
  straight to the console.

## 7. Known limitations

- **Character level, short context**: `chat_demo` uses `block_size` 128, `dlqq_exam` 256; a
  2000-character agent message leaves only the tail visible.
- **Small models memorise**: `chat_demo` has 30 Q&A pairs and `dlqq_exam` is all exam
  questions — original questions work, rephrasings fail. Never benchmark on the same set.
- **Corpus licensing**: `dlqq_exam` includes C-Eval (CC BY-NC-SA 4.0, non-commercial); do not
  redistribute that corpus.
- **The bridge is not an agent**: no tool calls, no planning, no multi-step execution — it is
  for validating the pipeline only.
- **CPU training** is slow and bf16 is disabled there (it is slower on CPU); use the `tiny`
  tier to validate the flow.

## 8. References

- Weights: [ModelScope · donglouqiaoque/dlqq-llm](https://www.modelscope.cn/models/donglouqiaoque/dlqq-llm)
- Source mirrors: [GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch docs: <https://pytorch.org/docs/stable/index.html>
- nanoGPT (same order of magnitude): <https://github.com/karpathy/nanoGPT>
