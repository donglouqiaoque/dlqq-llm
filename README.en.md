# dlqq-llm-train · Character-level GPT: training and chat

[中文](README.md) ｜ **English**

Pretrained weights ｜ [ModelScope model hub](https://www.modelscope.cn/models/donglouqiaoque/dlqq-llm)

A minimal, from-scratch language-model toolchain: **no tokenizer, no HuggingFace**.
Corpus in, model out — and you can chat with it when training finishes.

```
corpus  →  character-level vocab  →  GPT (Pre-Norm)  →  AdamW training  →  sampling
```

The model is a hand-written multi-head attention block with Pre-Norm, weight tying and
Flash Attention. The default tier has ~0.8 M parameters; an 8 GB laptop GPU trains it
in minutes.

---

## Contents

- [Quick start](#quick-start)
- [Requirements](#requirements)
- [Training a model](#training-a-model)
- [Chatting with a model](#chatting-with-a-model)
- [Using your own corpus](#using-your-own-corpus)
- [Tiers and parameters](#tiers-and-parameters)
- [Runtime modes and VRAM](#runtime-modes-and-vram)
- [Directory layout](#directory-layout)
- [FAQ](#faq)
- [Serving the model to DSH](#serving-the-model-to-dsh)

---

## Quick start
Double-click `script\start.bat`; the browser opens the local console at an address like
`http://127.0.0.1:49134/?t=...`. Five tabs:

| Tab | What it does |
|---|---|
| Train | pick a config and corpus, press Start; logs stream live and land in `.ai_res\cache\train-logs\` |
| Chat | pick a model and type; the reply streams character by character and can be stopped |
| Server | run a local OpenAI-compatible bridge; every request shows up on the page in real time |
| Config | edit `properties\*.properties` in a form; comments and key order are preserved |
| Signature | compute or verify SHA-256 for every `.pt` under `model\` |

> **Closing the cmd window of `start.bat` means exit**: training, the bridge and any signature job
> stop with it. Closing the browser tab is not an exit — training keeps running; reopen the URL.

Phones and other machines on the same LAN can open `http://<host-LAN-IP>:49135/` — a **read-only
chat page**: no token, chat only, and the train/config/signature tabs are unreachable. The first
launch may prompt Windows Firewall; allow Python on **private networks**. Pass `start.bat --no-lan`
to skip the LAN page.

The command line is no longer a user entry point; `--config` / `--parent-pid` are an internal
protocol between the WebUI and its child processes.

## Requirements

| Item | Requirement |
|---|---|
| Python | 3.10 or newer (verified on 3.14) |
| torch | 2.x; install the CUDA build to use the GPU |
| GPU | 8 GB VRAM runs the largest tier; CPU works but is much slower |

Install (**torch is the only dependency**):

```bat
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Two notes:

1. Install with `python -m pip install` into the interpreter you actually use (not `conda install`);
2. Then run `script\start.bat (Train tab) chat_demo`; if training starts, the environment is fine (CPU fallback included).

### Which Python is used

There are exactly two sources, and **no automatic search**:

```
1. python= in the config file     ← if set, it must work, or training aborts
2. python / python3 / py on PATH  ← used when the field is empty
```

- If the config names an interpreter, **that one only**. Missing file, a directory,
  a broken interpreter, missing `torch` / `numpy` — all abort immediately,
  **never falling back to another interpreter**.
- If the field is empty, PATH is used (`python`, then `python3`, then `py`); missing
  dependencies abort just the same.

The chosen interpreter is printed at startup:

```
[环境] D:\Soft\Python\Python3_13_9\python.EXE
```

To pin one, add a line to `properties\*.properties` (point at `python.exe`, not a directory):

```
python=D:\Soft\Python\Python3_13_9\python.exe
```

A wrong setting looks like this — training never starts:

```
[ERROR] python from config not found:
        D:\NoSuchDir\python.exe
[错误] 配置里指定的 python 缺少依赖：torch、numpy
       D:\...\python.exe
       装一下："D:\...\python.exe" -m pip install torch numpy
[终止] 解释器不可用，未开始训练。
```

### When the environment is wrong

The training and chat scripts **do not self-check**; a missing dependency stops them
immediately with the install command.

```
[错误] 运行环境缺少依赖：torch
       请先安装：python -m pip install torch numpy
```

Without a GPU it falls back to CPU automatically (just slower); a missing CUDA never aborts the run.

It also picks a torch-capable interpreter from PATH by itself, trying `python` /
`python3` / `py`; if none works it says torch is missing instead of guessing install paths.

---

## Training a model

### Train from the Train tab (the only entry point)

1. Double-click `script\start.bat`; the page opens on the **Train** tab;
2. pick a `properties\*.properties` in **Config** (the dropdown shows each file's header comment);
3. **Corpus** lists every directory under `src\` with file count and size — informational only,
   the directory actually used is `data_path` in the config;
4. press **Start**. The log streams live and is also written to
   `.ai_res\cache\train-logs\<config>-<timestamp>.log`;
5. press **Stop** to stop. To continue from the last checkpoint, press **Resume training**
   (the checkpoint must contain optimizer state).

### Where the configs are and how to edit them

All under **`properties\`**, one `key=value` per line, `#` starts a comment:

| File | Corpus | Tier | Notes |
|---|---|---|---|
| `chat_demo.properties` | `src\chat_demo\` | small (~0.9 M params) | 30 hand-written Q&A pairs, seconds |
| `dlqq_exam.properties` | `src\dlqq_exam\` | medium (13.7 M params) | 4,562 exam questions, a few minutes |
| `dlqq_llm.properties` | `src\dlqq_llm\` | large (31.5 M params) | GB-scale corpus |
| `dlqq_poem.properties` | `src\dlqq_poem\` | small (14.1 M params) | poetry (classical + Song ci + modern) |

**Every parameter lives in the file.** You can edit it on the **Config** tab: saving only rewrites
the values, keeping comments, blank lines and key order, and backs the old file up under
`.ai_res\cache\cfg-backup\`. Changes apply to the next training run.

| Key | Meaning |
|---|---|
| `data_path` | Corpus **directory** (relative to the repo root; any number of `txt` / `jsonl` files) |
| `model` | Model name (no `.pt`; saved to `model\<name>\<name>.pt`) |
| `cache_dir` | Where tokenised caches go (a folder under the scratch area by default) |
| `n_layer` / `n_head` / `n_embd` | Model shape |
| `block_size` / `batch_size` | Context length / samples per batch |
| `max_iters` / `eval_interval` | Training steps / evaluate every N steps |
| `lr` / `dropout` / `weight_decay` / `grad_clip` / `seed` | Optimisation |
| `max_vocab` / `val_frac` | Vocabulary cap / validation split |
| `python` | **Pin an interpreter**; empty = use PATH |
| `confirm` | `no` = start immediately, `yes` = ask once more |

Copy an existing file to add a config — the dropdown is scanned from `properties\` on each request,
so the new file shows up immediately.

### What happens without a config

If `data_path` is empty, the trainer falls back to downloading TinyShakespeare (~1.1 MB); if that
download fails it stops with a clear message instead of training on nothing.

Every evaluation point prints the validation loss, and **only the best validation checkpoint is
saved**. After training a sample is generated so you can see where the model stands.

---

## Chatting with a model

Use the **Chat** tab:

1. **Model** lists every `model\**\*.pt` with size, mtime and signature status;
2. adjust temperature / top_k / length (defaults come from `properties\<model>.properties`);
3. type and press Enter — the reply streams character by character, and **Stop** interrupts it.

- Switching models does not need a page reload: the old model is released first (VRAM included);
- while training is running you may only pick a **different** model; the model named by the running
  config is rejected, so training is not slowed down and you never read a half-written file;
- the conversation is truncated to the model's `block_size`.

### Two modes

- **Dialogue mode**: your text becomes `用户：xxx` + `助手：`, and the model continues as the assistant;
- **Continuation mode**: your text is treated as ordinary prose and the model writes on.

If the model's vocabulary has no `用户：` / `助手：` characters it was not trained on dialogue data,
and the session **switches to continuation mode automatically**.

---

## Using your own corpus

**All corpora live in one directory** (`data_path` accepts directories only, not single files).

Create a folder under `src\`, e.g. `src\my_corpus\`, and drop files in:

```
src\
  my_corpus\
    part1.jsonl
    part2.jsonl        ← any number of files; merged in filename order
    notes.txt
```

Supported files:

| File | Handling |
|---|---|
| `.jsonl` with Q&A fields (`input`/`target`, …) | Joined as `用户：… / 助手：…` |
| `.jsonl` plain text (`{"text": "..."}`) | Fed as plain text |
| `.txt` | Fed as plain text (UTF-8 / GBK / BIG5 auto-detected) |

For a chat model, write the jsonl like this:

```json
{"input": "什么是过拟合", "target": "模型把训练数据背下来了，换一批新数据就不行了。"}
{"input": "那怎么缓解", "target": "加数据、加正则、减小模型容量，三条路一起走。"}
```

Then train:

```bat
# create a new file under properties\ with these three lines:
#   data_path=src/my_corpus
#   model=mymodel
#   max_iters=5000
```

Rough expectations:

| Corpus size | What you get |
|---|---|
| < 100 KB | Mostly **memorised**; exact questions work, rephrasings break |
| ~1 MB | Sentence patterns start to generalise |
| 10 MB+ | Short sentences are fluent, long ones drift |

Encodings are detected per file (UTF-8 / utf-8-sig / gb18030 / big5), so Notepad's GBK
files work as-is.

### Clean downloaded corpora first

Wikipedia dumps and web crawls cannot be fed directly (`{{templates}}`, `[[links]]`,
mixed languages and ads). Cleaning strips wiki markup, normalises whitespace, filters by
length and Chinese ratio, and de-duplicates lines. The helper scripts that do this are local
tools, not shipped with the repo — usage in AIRES.md.

It strips wiki markup, normalises whitespace, filters by length (`--min-chars`) and
Chinese ratio (`--min-zh`), de-duplicates lines, and writes a `.txt` (blank line between
paragraphs) or `.jsonl` the trainer can read.

---

## Tiers and parameters

| Tier | n_layer | n_head | n_embd | block | batch | Params | VRAM | Notes |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 2 | 2 | 64 | 64 | 16 | ~0.2 M | < 0.5 G | Seconds per run; sanity check |
| `small` (default) | 4 | 4 | 128 | 128 | 32 | ~0.8 M | < 1 G | Visible results |
| `medium` | 6 | 6 | 384 | 256 | 64 | ~10 M | ~2–3 G | Clearly better with enough data |
| `large` | 8 | 8 | 512 | 256 | 32 | ~25 M | ~6–7 G | At the edge of 8 GB; do not go bigger |

Every config key can be overridden on the command line:

```bat
# change in the config: n_layer=6, n_embd=384, lr=0.001, max_iters=5000
```

Common keys:

| Key | Default | Meaning |
|---|---|---|
| `data_path` | none (auto-download) | Corpus directory; empty downloads TinyShakespeare |
| `out_path` | `model/mymodel.pt` | Where the checkpoint is written |
| `max_iters` | 2000 | Training steps — the simplest "how long" knob |
| `batch_size` | 32 | Samples per step; lower it when VRAM is tight |
| `block_size` | 128 | Context length; shrinks automatically on tiny corpora |
| `lr` | 3e-4 | Learning rate |
| `dropout` | 0.1 | Raise it for small corpora |
| `eval_interval` | 200 | Evaluate (and maybe save) every N steps |
| `max_vocab` | 8000 | Vocabulary cap: keep the N most frequent characters |
| `val_frac` | 0.05 | Tail fraction used for validation |
| `cache_dir` | cache under the scratch area | Tokenised binary cache lives here |
| `prompt` | empty | Sampling prompt after training; auto-picked when empty |
| `data_format` | `auto` | `auto` / `txt` / `jsonl`; auto sniffs extension and header |
| `q_field` | empty | jsonl question field; inferred when empty |
| `a_field` | empty | jsonl answer field; inferred when empty |
| `eval_iters` | 50 | Batches per evaluation; 20 saves time on long runs |
| `grad_accum` | 1 | Gradient accumulation to fake a larger batch |
| `warmup_iters` | 0 | LR warmup steps; 1000–2000 for long runs |
| `lr_decay` | `cosine` | `cosine` decay / `none` constant |
| `min_lr_frac` | 0.1 | Decay floor = `lr × this` |
| `grad_checkpoint` | `False` | Checkpointing: less VRAM, ~20–30% slower |
| `save_every` | 0 | Force a save every N steps (0 = only on improvement) |
| `chat_mode` | `True` | `False` = pure continuation model (poetry), no `用户：/助手：` |
| `chat_temperature` | 0.8 | Sampling temperature; higher is wilder |
| `chat_top_k` | 40 | Sample from the top K; 0 = no truncation |
| `chat_max_new` | 180 | Max characters generated per turn |

`chat_*` keys are **dialogue parameters** stored in the checkpoint. `chat.py` and
`http_server.py` resolve them by this priority (editing properties takes effect without
retraining):

```
explicit CLI argument  >  properties\<model>.properties  >  value in the checkpoint  >  default
```

---

## Training `dlqq_llm` longer (optional)

`dlqq_llm` trains directly from `properties\dlqq_llm.properties`; to train longer, edit
that file — no extra config needed:

| Parameter | Effect |
|---|---|
| `max_iters` | Steps; multiply by `batch_size × block_size` to estimate tokens seen |
| `grad_accum` | Accumulate gradients to fake a bigger batch |
| `warmup_iters` / `lr_decay` / `min_lr_frac` | Warmup + cosine decay; a must for long runs |
| `grad_checkpoint` | Checkpointing: less VRAM, 20–30% slower |
| `save_every` | Force a checkpoint every N steps, so a flat validation loss cannot block saving |

Example: set `max_iters=60000`, run `script\start.bat`, pick `dlqq_llm` on the Train tab and press
**Resume training** (the checkpoint must contain optimizer state).

---

## A tiny exam demo (`dlqq_exam`)

Repo size is limited (Gitee caps a single file at 50 MB). `chat_demo` is 30 hand-written
Q&A pairs (13 KB) — enough to smoke-test the pipeline, but thin to play with.
`dlqq_exam` uses an **exam question bank** instead: far more information per byte.

Two steps:

```bat
REM 1) Prepare the exam corpus: merge C-Eval / gaokao / biology into src\dlqq_exam\dlqq_exam.jsonl
REM    (the corpus is not in the repo — download and merge it yourself)

REM 2) Train and chat
script\start.bat (Train tab) dlqq_exam
script\start.bat (Chat tab) dlqq_exam
```

Staying under 50 MB comes down to two lines in `properties\dlqq_exam.properties`:

```properties
save_dtype=fp16        # half-precision weights, half the size
save_optimizer=false   # skip optimizer state, another half (demos don't need resume)
```

Measured: 4,562 records tokenise to ~2.7 M tokens; 13.7 M parameters at fp16 without
optimizer ≈ **27 MB**. `chat_demo` is smaller still: 0.9 M parameters ≈ **1.9 MB**.

> **⚠ Attribute the source**: `dlqq_exam` trains on the questions themselves, so the model
> **memorises them**. Credit the datasets (ModelScope: C-Eval / gaokao-benchmark / …) and
> **do not** report benchmark scores on the same questions — that is self-grading, not ability.
> C-Eval is licensed CC BY-NC-SA 4.0 (non-commercial); do not redistribute this corpus.

### Where corpora come from (ModelScope only, direct in mainland China)

| Dataset (ModelScope repo) | Size | Use |
|---|---|---|
| `philexohf/LLM-Pretrain-Data` | ~24.6 GB / 4.96 B tokens | **Preferred pretraining**: web 55% + news 27% + wiki 12% + encyclopedia 6%, cleaned and deduped, one document per line |
| `AI-ModelScope/chinese-fineweb-edu` | ~290 GB | Subset when you only want web text |
| `OmniData/C-Eval` | 1.5 MB | 52-subject multiple choice |
| `opencompass/gaokao-benchmark` | 5.7 MB | Gaokao multiple choice |
| `pandayi/Biology_subject_of_college_entrance_examination` | 0.2 MB | Gaokao biology (with explanations) |
| `SHPDataGR/medical-exam-question-bank-json` | 35 MB | Medical exam bank |
| `tongyi_dianjin/CFLUE` | 66 MB | Finance qualification questions |

Downloads:

```bat
REM Exam data: download C-Eval / gaokao-benchmark / gaokao biology from ModelScope and merge
REM            (ModelScope only — no GitHub, no HuggingFace, no login)

REM Pretraining text: download, clean into jsonl, drop it into the corpus folder the config points at
```

Downloading and cleaning are done by local helpers (not shipped with the repo) — usage in
AIRES.md. The resulting jsonl goes straight into the corpus folder the config points at, so
download and train.

### How big is the pretraining corpus, and why 2 GB by default?

The helper can list the corpora and their sizes — usage in AIRES.md:

```
mix      philexohf/LLM-Pretrain-Data
         1 file available, 12.72 GB total
fineweb  AI-ModelScope/chinese-fineweb-edu
         302 files available, 311.44 GB total
```

| Corpus | Total | Notes |
|---|---|---|
| `mix` (train.txt) | **12.72 GB** ≈ 4.5 B tokens | web + news + wiki + encyclopedia, one doc per line |
| `fineweb` | **311 GB** (302 parquet shards) | Education-oriented web text, shard by shard |

2 GB is chosen from the model size:

- `dlqq_llm` is 31.5 M parameters; at "20 tokens per parameter" it needs ≈ **630 M tokens**;
- 2 GB of Chinese text ≈ 700 M tokens, plus the existing `firefly-train-1.1M.jsonl`
  (1.1 GB ≈ 400 M tokens) gives ~1.1 B tokens — already more than enough;
- Downloading all 12.7 GB buys a 29.4 M model nothing but 1–2 hours and 13 GB of disk.

For the full set pass `--max-mb 0` (download until done), or `--max-mb 8000`.

---

## Training on GB-scale corpora

1 GB of Chinese is roughly 350 M characters. Reading a whole file into memory and
encoding it into a Python list would take tens of GB — so the pipeline is **streaming**:

```
read in chunks → numpy counts characters → keep top max_vocab → encode to uint16 → write a cache file
training uses np.memmap and reads a few KB at random
```

**Memory use is independent of corpus size**; only `block_size` / `batch_size` matter.

### Feed JSONL directly, no conversion to txt

Point `data_path` at a directory containing `.jsonl` and the script finds the Q&A fields
and joins them into `用户：… / 助手：…` (**feeding raw jsonl teaches the model braces and
quotes, so this step matters**).

Field pairs tried in order:

```
input/target · q/a · question/answer · src/tgt · instruction/output
prompt/response · query/response · ask/answer · title/content
```

If none match, specify them: `q_field=... a_field=...`.

### Measured (1.12 GB in `src\dlqq_llm\`, 1,649,399 Q&A records)

| Metric | Value |
|---|---|
| Tokenisation time | **30 s** (including startup; reused afterwards) |
| Tokens produced | 425,220,455 |
| Vocabulary | 8000 (capped); only 210 k characters (0.05%) dropped |
| `data.bin` | 850 MB |
| Peak memory | ~1.5 GB (independent of corpus size) |
| Second run | 4.9 s (cache reused) |

Key settings to put into `properties\dlqq_llm.properties`:

```properties
data_path=src/dlqq_llm
model=dlqq_llm
max_vocab=8000
n_layer=8
n_head=8
n_embd=512
block_size=256
batch_size=32
max_iters=60000
lr=0.0003
eval_interval=2000
```

Measured speed on this corpus (29.4 M parameters, fits in 8 GB):

| Tier | Config | Speed | 60,000 steps |
|---|---|---|---|
| small | 4 layers / 128 dim / block 128 / batch 32 | 143 steps/s | ~7 min |
| large | 8 layers / 512 dim / block 256 / batch 32 | ~5 steps/s | ~3.3 h |

> The small tier (0.8 M parameters) badly underfits 1 GB — **use large for GB-scale data**.
> In the smoke test, large pushed validation loss from 9.06 to 5.72 within 30 steps,
> so the data path is sound.

Two caveats:

- Each corpus (together with `max_vocab` / `val_frac` / format / field settings) gets its
  own cache file; changing any of them writes a new one. So the demo `chat_demo\` and the
  GB-scale `dlqq_llm\` can **coexist** — switching back and forth never re-tokenises.
  Delete the cache folder to start over;
- The validation split is taken from the **tail** of the corpus. If the data is ordered by
  category (news first, dialogue later), shuffle the lines first — otherwise validation
  loss will not reflect real quality.

### Adding data: drop files into the directory

`data_path` accepts a directory; `dlqq_llm.properties` points at one:

```properties
data_path=src/dlqq_llm
```

The directory can hold any number of files; they are read in filename order and
concatenated into one stream:

| What you put in | Handling |
|---|---|
| `*.jsonl` with Q&A fields | Joined as `用户：… / 助手：…` |
| `*.jsonl` with only `{"text": "..."}` | Fed as plain text (pretraining data looks like this) |
| `*.txt` | Fed as plain text |
| Subdirectories | Scanned too (hidden ones skipped) |

Adding, removing or editing files changes the cache fingerprint, so **the next run
re-tokenises automatically** — no manual cache cleanup.

To grow the corpus, download from ModelScope (no GitHub / HuggingFace / login); downloading
and cleaning are done by local helpers — usage in AIRES.md.

It filters by length and Chinese ratio, de-duplicates lines, and writes
`pretrain-mix-000.jsonl`, `pretrain-mix-001.jsonl`, … at ~200 MB each. **Existing files are
never overwritten** (numbering continues). A progress bar runs during download and it prints
"corpus directory now has N files / X GB" at the end. Then `script\start.bat (Train tab) dlqq_llm`.

---

## Training a poetry model (`dlqq_poem`)

A small model dedicated to classical Chinese poetry: 6 layers × 384 dim (14.1 M parameters),
trained on **Tang/Song/Yuan/Ming/Qing poems + Song ci + modern poetry**; about 30 minutes on
8 GB VRAM.

The corpus holds **poem text only — no `诗：` / `词：` labels**, so the model writes poems
instead of repeating labels.

### 1. Download the corpus

The three sources (poems 43 MB, ci 19 MB, modern 57 MB) are downloaded and cleaned by local
helpers — usage in AIRES.md.

| Source | Repo | Size | Content |
|---|---|---|---|
| `shici` | `modelscope/chinese-poetry-collection` | 43.4 MB | Tang to Qing poems |
| `songci` | `peaceminuczy/songci` | 18.7 MB | Song ci |
| `modern` | `Iess/chinese_modern_poetry` (hf-mirror.com) | 56.6 MB | Modern poetry (Chinese poets + translations) |

Everything lands in `src\dlqq_poem\` as `shici-000.jsonl`, `songci-000.jsonl`,
`modern-000.jsonl`. Existing files are kept, a progress bar runs, and lines break at
punctuation.

> Older corpora carried `诗：` / `词：` prefixes. The script detects such files and **stops,
> telling you to delete them** (`del src\dlqq_poem\*.jsonl`); otherwise old and new files
> mix the labels into the text.

### 2. Train

```bat
script\start.bat (Train tab) dlqq_poem
```

`properties\dlqq_poem.properties` is tuned to the actual corpus:

| Item | Value | Rationale |
|---|---|---|
| Shape | 6 layers, 384 dim (14.1 M) | ~40 M characters / 14 M params ≈ 2.8 tokens per parameter |
| `block_size` | 256 | Modern poems average 210 chars, 90% under 136 — 256 fits a whole poem |
| `batch_size` | 32 | 32 × 256 = 8,192 tokens/step (same steps as 64×128, double the context) |
| `max_iters` | 15000 | ≈ 123 M tokens ≈ 3.1 passes over the 39.2 M-token corpus |
| `max_vocab` | 7000 | Covers >99.9% of characters |
| `chat_mode` | `false` | Tells `chat.py` this is a continuation model — no `用户：/助手：` |

Measured ~70 k tokens/s, **about 29 minutes** end to end; a checkpoint every 2,500 steps (fp16).

### 3. Let it write

The corpus is bare text, so "chatting" means **give it a start and it continues**:

```bat
script\start.bat (Chat tab) dlqq_poem
```

```
你  > 春
它  > 风暖鸟声碎，日高花影重。
      ...
```

- One or two characters → it starts a line from there;
- A line of poetry → it continues;
- A modern-poetry sentence → it continues in that style.

Because `chat_mode=false` is stored in the checkpoint, `chat.py` uses continuation mode
and never prepends `用户：/助手：`.

Sampling parameters also live in `properties\dlqq_poem.properties` and take effect
**without retraining**:

```properties
chat_mode=false
chat_temperature=1.0     # 0.8 makes a small model loop ("他的女人，他的女人")
chat_top_k=0             # 0 = sample from the full vocabulary
chat_max_new=200
```

`script\start.bat (Chat tab)` and `script\start.bat (Server tab)` both read these; override temporarily with
`temperature=0.8 top_k=40` on the command line.

---

## Runtime modes and VRAM

Before every training or chat run, the environment is reported once:

```
[模式] GPU —— 混合精度 bf16（推荐：显存占用更低、张量核更快）
[设备] NVIDIA GeForce RTX 4060 Laptop GPU | sm_89
[显存] 总 8.0 GB，已用 1.1 GB（可用 6.9 GB）
```

The three modes:

| Mode | Trigger | Notes |
|---|---|---|
| **GPU · bf16 mixed precision** | CUDA device present | Forward/backward in bf16; less VRAM, faster tensor cores |
| **GPU · fp32** | CUDA present but AMP off | More precise, more VRAM |
| **CPU · fp32** | No usable CUDA device | Also reports **RAM** usage; bf16 on CPU is slower, so it is off |

When VRAM is short: lower `batch_size` → lower `block_size` → drop to a smaller tier.

---

## Directory layout
```
script\                 double-click entry (the only one)
  start.bat             find python → run code\webui.py in the foreground → open browser
  logo.ico              shortcut icon
code\                   all Python
  webui.py              local web service: static files + JSON API + SSE + child processes
  train.py              training (data pipeline + model + loop + sampling)
  chat.py               chat core: ChatSession (load / streaming reply / set params)
  http_server.py        OpenAI-compatible bridge (lazy multi-model loading)
  common.py             find_root / order-preserving properties I/O / watchdog / hide dir
  model_signature.py    weight SHA-256 signatures (stdlib only)
web\                    frontend served by webui.py (index.html: five tabs)
  index.html            five tabs: train / chat / server / config / signature
  chat.html             LAN read-only chat page
properties\             four training configs
src\                    corpora (not in git)
model\                  training output (not in git; model\<name>\<name>.pt plus .pt.sha256)
.ai_res\                scratch area (caches, training logs, config backups; not in git)
```

- The project root is found by walking up to the directory containing `code\train.py`.
- Model id = directory name: `model\chat_demo\chat_demo.pt` → `chat_demo`.

## FAQ

**Q: The window flashes and disappears.**

Every `.bat` ends with `pause`, so a normal run stays open. A real flash usually means
`python.exe` was not found — run it from a terminal to see the full error:

```bat
script\start.bat (Train tab)
```

**Q: "Missing dependency" at startup.**

Look at the missing module in the error and install it into the current interpreter: `python -m pip install torch numpy`.
The usual cause: the Python on PATH is not the one torch was installed into.

**Q: Running from an IDE's Run button fails with "cannot read keyboard input".**

That mode has no interactive terminal, so `input()` gets EOF and the number prompt cannot
work. Two options:

1. **Double-click** `script\start.bat (Train tab)` (a real console);
2. **Name the config** in the IDE to skip the prompt:

```bat
script\start.bat (Train tab) chat_demo
```

**Q: Will retraining with a different corpus collide with the model name?**

No. If the name exists you are asked whether to overwrite; answer `n` and pick another name.

**Q: I want to train but not with `chat_demo`.**

Create a corpus folder under `src\`, then copy a config and change two lines (the new
config is listed automatically):

```
# properties\my_corpus.properties
data_path=src\my_corpus      ← a directory holding jsonl / txt
model=mymodel
...（copy the rest of chat_demo.properties; adjust n_layer / max_iters as needed）
```

**`data_path` accepts directories only**; `txt` and `jsonl` are both supported and the
format is inferred from extension and header. For jsonl the Q&A fields are auto-detected
(single-field `{"text": ...}` is treated as plain text).

**Q: I want to change one parameter without editing the config.**

Call the training script directly; command-line values override the config:

```bat
# change in the config: data_path=src/chat_demo, model=chat_demo, max_iters=500
```

**Q: The model still parrots after a long training run.**

The corpus is too small. A character-level model under 100 KB is mostly reciting;
aim for 1 MB+ to see generalisation.

**Q: Can I train on CPU?**

Yes, but it is slow, and bf16 is disabled on CPU (it is slower there). Start with the
`tiny` tier to validate the pipeline.

**Q: How do I know a weight file was not altered or corrupted?**

Double-click `Verify on the Signature tab`: it walks every `.pt` under `model\`, recomputes
SHA-256 and compares it with the adjacent `<name>.pt.sha256`; every file must print PASS.
Run it after training, or before uploading weights to the model hub. Re-generate signatures
with `Compute on the Signature tab` (mandatory once the weights change).

---

## Serving the model to DSH
On the **Server** tab: set the port (default 49128) and press Start; the page shows
`http://127.0.0.1:49128/v1`, which you put into the DSH provider settings. Every request then
appears in the session list below (fed by the `@@EV@@` lines the service prints to stdout), so you
debug from the page instead of the console. Press Stop to shut it down — closing the `start.bat`
window also stops it.

The service is OpenAI-compatible: `/v1/models` and `/v1/chat/completions` (with `stream`).
Models are scanned from `model\**\*.pt` and loaded only on first use.

