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

With Python and torch installed, double-click two files:

| Step | Double-click | What you do |
|---|---|---|
| 1 | `script\train.bat` | Pick a config by number, press Enter, training starts |
| 2 | `script\chat.bat` | Press Enter, then start typing |

**Every training parameter lives in `properties\`** — corpus, model size, steps, learning
rate, even which Python to use. Edit the config; there is nothing to memorise on the
command line.

To verify the environment first, double-click `script\torch_check.bat`.

> Four configs ship with the repo: `chat_demo` (30 hand-written Q&A pairs, 0.9 M params,
> a few seconds), `dlqq_exam` (4,562 exam questions, 13.7 M params, a few minutes),
> `dlqq_llm` (GB-scale corpus, 29.4 M params) and `dlqq_poem` (poetry, 14.1 M params).
> Chat defaults to `model\chat_demo\chat_demo.pt`.
> **A missing corpus or model aborts immediately** and tells you how to fix it.
>
> **Rather not train?** Released weights live on the [ModelScope model hub](https://www.modelscope.cn/models/donglouqiaoque/dlqq-llm);
> drop them in as `model\<name>\<name>.pt` and chat directly — the model card describes each one.
> Every weight ships with a matching `.sha256`; run `script\verify_signature.bat` to check it before loading.

---

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
2. Run `script\torch_check.bat` afterwards — `CUDA available: True` plus your GPU name means you are set.

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
immediately and points at the checker.

```
[错误] 运行环境缺少依赖：torch
       请先运行 script\torch_check.bat 检查运行环境。
```

`script\torch_check.bat` runs 9 checks (environment info, CUDA availability, tensor
transfers, matmul correctness, autograd, conv + half precision, torchvision, torchaudio,
CPU/GPU speed comparison). **A failure does not stop the run** — each item reports
`[ OK ] / [SKIP] / [FAIL]`.

It also picks a torch-capable interpreter from PATH by itself, trying `python` /
`python3` / `py`; if none works it says torch is missing instead of guessing install paths.

---

## Training a model

### Double-click → pick a config (recommended)

`script\train.bat` asks exactly one question:

```
Available configs:
  [1] chat_demo.properties
  [2] dlqq_exam.properties
  [3] dlqq_llm.properties
  [4] dlqq_poem.properties

Select a number (Enter = 1): 1
```

Type a number and press Enter; everything else comes from the config file.

```
[config]  properties\chat_demo.properties
[python]  D:\Soft\Python\Python3_13_9\python.exe

  训练（配置文件：chat_demo.properties）
[语料] src\chat_demo（目录，1 个文件）
[输出] model\chat_demo\chat_demo.pt
[参数] n_layer=4 n_head=4 n_embd=128 block_size=128 batch_size=32 max_iters=3000 ...
[环境] D:\Soft\Python\Python3_13_9\python.EXE
```

The window stays open afterwards (`pause` at the end) so you can read the log and the
sample generation.

### Where the configs are and how to edit them

All under **`properties\`**, one `key=value` per line, `#` starts a comment:

| File | Corpus | Tier | Notes |
|---|---|---|---|
| `chat_demo.properties` | `src\chat_demo\` | small (~0.9 M params) | 30 hand-written Q&A pairs, seconds |
| `dlqq_exam.properties` | `src\dlqq_exam\` | medium (13.7 M params) | 4,562 exam questions, a few minutes |
| `dlqq_llm.properties` | `src\dlqq_llm\` | large (29.4 M params) | GB-scale corpus, ~3 hours (60,000 steps) |
| `dlqq_poem.properties` | `src\dlqq_poem\` | small (14.1 M params) | poetry (classical + Song ci + modern), ~30 min |

**Every parameter is in the file**; save and it takes effect. No code changes needed:

| Key | Meaning |
|---|---|
| `data_path` | Corpus **directory** (relative to the repo root; put `txt` / `jsonl` files inside, any number) |
| `model` | Model name (no `.pt`; saved to `model\<name>\<name>.pt`) |
| `cache_dir` | Where tokenised caches go (default `.ai_res/cache`) |
| `n_layer` / `n_head` / `n_embd` | Model shape |
| `block_size` / `batch_size` | Context length / samples per batch |
| `max_iters` / `eval_interval` | Training steps / evaluate every N steps |
| `lr` / `dropout` / `weight_decay` / `grad_clip` / `seed` | Optimisation |
| `max_vocab` / `val_frac` | Vocabulary cap / validation split |
| `python` | **Pin an interpreter**; empty = use PATH |
| `confirm` | `no` = start immediately, `yes` = ask once more |

To add a config, copy one and edit it — `train.bat` lists it automatically.

### Usage examples

```bat
script\train.bat                        REM same as double-clicking: pick from a list
script\train.bat dlqq_llm               REM use properties\dlqq_llm.properties
script\train.bat my.properties          REM any config file name
```

### Calling the training script directly (advanced)

For one-off overrides the config does not cover, pass values on the command line:

```bat
python code\train_llm.py data_path=src\chat_demo out_path=model\chat_demo\chat_demo.pt max_iters=3000
```

> Arguments are `key=value`, **not** `--key value`.
> Parent directories of `out_path` are created automatically.

### What happens without arguments

```bat
python code\train_llm.py
```

With no `data_path`, the script downloads TinyShakespeare (~1.1 MB) as the corpus.
If the download fails it stops with two options instead of training on nothing.

Every evaluation point prints the validation loss, and **only the best validation
checkpoint is saved**. After training the model writes a sample so you can see where
it stands.

---

## Chatting with a model

### Option 1: double-click (recommended)

Double-click `script\chat.bat`:

- **Press Enter** → use the default model `model\chat_demo\chat_demo.pt`;
- **Type a number** → pick another model from the list;
- **Type a name** → resolve it to a `.pt`.

### Option 2: command line

```bat
script\chat.bat                  REM Enter = default model
script\chat.bat mymodel          REM skip the question

python code\chat.py model\chat_demo\chat_demo.pt                     REM interactive
python code\chat.py model\chat_demo\chat_demo.pt --once "用户：hello"  REM one shot
```

### Commands inside the chat

| Command | Effect |
|---|---|
| `:q` | Quit (`:quit` / `:exit` also work) |
| `:t 0.7` | Change temperature — lower is safer, higher is wilder |
| `:n 200` | Change the generation length |
| `:raw` | Toggle dialogue / continuation mode |
| `:p` | Show the current settings |

### Two modes

- **Dialogue mode**: your text becomes `用户：xxx` + `助手：`, and the model continues as the assistant;
- **Continuation mode**: your text is treated as ordinary prose and the model writes on.

If the model's vocabulary has no `用户：` / `助手：` characters, it was not trained on
dialogue data; `chat.py` **switches to continuation mode automatically**.

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
python code\train_llm.py data_path=src\my_corpus out_path=model\mymodel\mymodel.pt max_iters=5000
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
mixed languages and ads). `.ai_res\script\prepare_corpus.py` converts and cleans:

```bat
REM Wikipedia dump (wikiextractor output, one JSON per line)
python .ai_res\script\prepare_corpus.py --in .ai_res\doc\zhwiki --out .ai_res\doc\zhwiki.txt --wikitext

REM parquet (fineweb and friends; finds the text column)
python .ai_res\script\prepare_corpus.py --in .ai_res\doc\fineweb --out .ai_res\doc\fineweb.txt --min-chars 100
```

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
python code\train_llm.py n_layer=6 n_embd=384 lr=1e-3 max_iters=5000
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
| `cache_dir` | `.ai_res/cache` | Tokenised cache; temp files stay in `.ai_res` |
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
| `resume` | `False` | Continue from `out_path` (needs optimizer/iter inside) |
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
| `resume` | `true` continues from `out_path` (optimizer state and LR schedule included) |
| `save_every` | Force a checkpoint every N steps, so a flat validation loss cannot block saving |

Example: set `max_iters=60000` and `resume=true`, then run `script\train.bat` and pick
`dlqq_llm`.

---

## A tiny exam demo (`dlqq_exam`)

Repo size is limited (Gitee caps a single file at 50 MB). `chat_demo` is 30 hand-written
Q&A pairs (13 KB) — enough to smoke-test the pipeline, but thin to play with.
`dlqq_exam` uses an **exam question bank** instead: far more information per byte.

Two steps:

```bat
REM 1) Download exam data (ModelScope: C-Eval + gaokao + biology) into src\dlqq_exam\dlqq_exam.jsonl
script\fetch_chat_demo.bat

REM 2) Train and chat
script\train.bat dlqq_exam
script\chat.bat dlqq_exam
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
REM Exam data: bundled script, ModelScope only — no GitHub, no HuggingFace, no login
script\fetch_chat_demo.bat --datasets c-eval gaokao bio

REM Large pretraining corpora: scripts under .ai_res\script, fetch N MB via HTTP Range
.ai_res\script\fetch_pretrain_corpus.bat --source mix --max-mb 2000
```

`fetch_pretrain_corpus.bat` writes cleaned jsonl straight into `src\dlqq_llm\`, which is
what the config points at — download and train. It shows a progress bar (percent, done/total, speed).

### How big is the pretraining corpus, and why 2 GB by default?

```bat
.ai_res\script\fetch_pretrain_corpus.bat --list
```

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

- `dlqq_llm` is 29.4 M parameters; at "20 tokens per parameter" it needs ≈ **590 M tokens**;
- 2 GB of Chinese text ≈ 700 M tokens, plus the existing `firefly-train-1.1M.jsonl`
  (1.1 GB ≈ 400 M tokens) gives ~1.1 B tokens — already more than enough;
- Downloading all 12.7 GB buys a 29.4 M model nothing but 1–2 hours and 13 GB of disk.

For the full set pass `--max-mb 0` (download until done), or `--max-mb 8000`.

---

## Training on GB-scale corpora

1 GB of Chinese is roughly 350 M characters. Reading a whole file into memory and
encoding it into a Python list would take tens of GB — so the pipeline is **streaming**:

```
read in chunks → numpy counts characters → keep top max_vocab → encode to uint16 → write .ai_res\cache\data.bin
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

Command:

```bat
python code\train_llm.py data_path=src\dlqq_llm out_path=model\dlqq_llm\dlqq_llm.pt ^
    max_vocab=8000 n_layer=8 n_head=8 n_embd=512 block_size=256 batch_size=32 ^
    max_iters=60000 lr=3e-4 eval_interval=2000
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
  own cache file `.ai_res\cache\data-<fingerprint>.bin`; changing any of them writes a new
  one. So the demo `chat_demo\` and the GB-scale `dlqq_llm\` can **coexist** — switching
  back and forth never re-tokenises. Delete `.ai_res\cache\` to start over;
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

To grow the corpus, use `.ai_res\script\fetch_pretrain_corpus.bat` (ModelScope, no
GitHub / HuggingFace / login):

```bat
REM Default: mixed corpus (web + news + wiki + encyclopedia) 2 GB into src\dlqq_llm\
.ai_res\script\fetch_pretrain_corpus.bat

REM Only 500 MB to try; the education web corpus; or everything
.ai_res\script\fetch_pretrain_corpus.bat --max-mb 500
.ai_res\script\fetch_pretrain_corpus.bat --source fineweb --max-mb 4000
.ai_res\script\fetch_pretrain_corpus.bat --max-mb 0

REM List available corpora and their sizes
.ai_res\script\fetch_pretrain_corpus.bat --list
```

It filters by length and Chinese ratio, de-duplicates lines, and writes
`pretrain-mix-000.jsonl`, `pretrain-mix-001.jsonl`, … at ~200 MB each. **Existing files are
never overwritten** (numbering continues). A progress bar runs during download and it prints
"corpus directory now has N files / X GB" at the end. Then `script\train.bat dlqq_llm`.

---

## Training a poetry model (`dlqq_poem`)

A small model dedicated to classical Chinese poetry: 6 layers × 384 dim (14.1 M parameters),
trained on **Tang/Song/Yuan/Ming/Qing poems + Song ci + modern poetry**; about 30 minutes on
8 GB VRAM.

The corpus holds **poem text only — no `诗：` / `词：` labels**, so the model writes poems
instead of repeating labels.

### 1. Download the corpus

```bat
REM All three sources (poems 43 MB + ci 19 MB + modern 57 MB)
.ai_res\script\fetch_poem_corpus.bat

REM One source / only 5 MB to try / list sources
.ai_res\script\fetch_poem_corpus.bat --source modern
.ai_res\script\fetch_poem_corpus.bat --source shici songci
.ai_res\script\fetch_poem_corpus.bat --max-mb 5
.ai_res\script\fetch_poem_corpus.bat --list
```

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
script\train.bat dlqq_poem
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
script\chat.bat dlqq_poem
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

`script\chat.bat` and `script\http_server.bat` both read these; override temporarily with
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
code/                     Training and chat scripts
  train_llm.py            Training
  chat.py                 Chat / inference

script/                   Entry points (double-click)
  train.bat               Train (lists properties\ and asks for a number)
  chat.bat                Chat
  fetch_chat_demo.bat     Download exam data into dlqq_exam (ModelScope, direct)
  torch_check.bat         Environment self-check
  http_server.bat         Start the bridge server (for DSH)
  train_chat_helper.py    Interactive logic behind the .bat files
  fetch_chat_demo.py      Exam download and format conversion
  torch_check.py          The individual checks
  http_server.py          The bridge server itself (OpenAI-compatible, self-contained)
  model_signature.py      Weight signatures: compute / verify SHA-256 (stdlib only, no torch)
  compute_signature.bat   Double-click: sign every .pt under model\
  verify_signature.bat    Double-click: verify every signature

properties/               Training configs (one key=value per line, live on save)
  chat_demo.properties    30 hand-written Q&A pairs (0.9 M params, small)
  dlqq_exam.properties    4,562 exam questions (13.7 M params, corpus from script\fetch_chat_demo.bat)
  dlqq_llm.properties     GB-scale corpus (src\dlqq_llm\) + large tier
  dlqq_poem.properties    Poetry (src\dlqq_poem\, 14.1 M params)

.ai_res/                  Scratch area for helper scripts and intermediates, **git-ignored**
  cache/                  Tokenised caches (one data-<fingerprint>.bin per corpus)
  doc/                    Raw corpora / downloaded datasets
  script/                 Helpers: corpus export, PDF conversion, plotting, benchmarks, corpus downloads

src/                      Corpora (one folder per corpus, e.g. src\chat_demo\, src\dlqq_exam\)
model/                    Models (one folder per model: model\chat_demo\chat_demo.pt)
release/                  Packaging output
```

> Everything under `src/` and `model/` except `.gitkeep` and `chat_demo.*` is git-ignored
> and stays local.

---

## FAQ

**Q: The window flashes and disappears.**

Every `.bat` ends with `pause`, so a normal run stays open. A real flash usually means
`python.exe` was not found — run it from a terminal to see the full error:

```bat
script\train.bat
```

**Q: "Missing dependency" at startup.**

Run `script\torch_check.bat`. It reports what is missing and which Python is in use.
The usual cause: the Python on PATH is not the one torch was installed into.

**Q: Running from an IDE's Run button fails with "cannot read keyboard input".**

That mode has no interactive terminal, so `input()` gets EOF and the number prompt cannot
work. Two options:

1. **Double-click** `script\train.bat` (a real console);
2. **Name the config** in the IDE to skip the prompt:

```bat
script\train.bat chat_demo
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
python code\train_llm.py data_path=src\chat_demo out_path=model\chat_demo\chat_demo.pt max_iters=500
```

**Q: The model still parrots after a long training run.**

The corpus is too small. A character-level model under 100 KB is mostly reciting;
aim for 1 MB+ to see generalisation.

**Q: Can I train on CPU?**

Yes, but it is slow, and bf16 is disabled on CPU (it is slower there). Start with the
`tiny` tier to validate the pipeline.

**Q: How do I know a weight file was not altered or corrupted?**

Double-click `script\verify_signature.bat`: it walks every `.pt` under `model\`, recomputes
SHA-256 and compares it with the adjacent `<name>.pt.sha256`; every file must print PASS.
Run it after training, or before uploading weights to the model hub. Re-generate signatures
with `script\compute_signature.bat` (mandatory once the weights change).

---

## Serving the model to DSH

A trained model can run as an **OpenAI-compatible local service** and be used in
DeepSeek Harness like any other provider — **no code or config changes on the DSH side,
everything happens in the UI**.

### Step 1: start the bridge server

Double-click `script\http_server.bat` — **no model selection**. It scans every `.pt` under
`model\` and `release\`, lists them, and **loads none**:

```
[清单] 共 4 个模型（现在不加载，谁被调用才加载谁）
       chat_demo                     1.9 MB  10-27 09:42  ← 默认
       dlqq_exam                    27.6 MB  10-27 09:53
       dlqq_llm                    337.1 MB  10-20 10:04
       dlqq_poem                   133.3 MB  10-20 14:29
```

Only a model that DSH actually calls is loaded into VRAM; switching models unloads the
previous one (1 resident by default, adjustable with `--max-loaded`).

You can also pass arguments directly:

```bat
script\http_server.bat                      REM port 49128, scan model dirs
script\http_server.bat 9000                 REM different port
script\http_server.bat 9000 cpu             REM force CPU inference
script\http_server.bat 9000 cuda 2          REM keep 2 models resident
python script\http_server.py --help         REM all options
```

On startup it prints the endpoints and model list:

```
  接口     http://127.0.0.1:49128/v1
  模型清单 http://127.0.0.1:49128/v1/models
  健康检查 http://127.0.0.1:49128/health
  停止服务：在这个窗口按 Ctrl+C
```

**Keep this window open** so DSH can reach the models.

### Step 2: add a custom provider in DSH

Open DSH → **Settings → Models → Add provider → Custom provider** and fill in:

| Field | Value |
|---|---|
| Provider ID | `local-gpt` (lowercase and hyphens; **cannot be changed after saving**) |
| Display name | anything, e.g. `Local small model` |
| Base URL | `http://127.0.0.1:49128/v1` |
| API protocol | `openai-completions` |
| Credential / API key | `local` (not validated, but **must not be empty**) |
| Model | Click "fetch available models" to list everything under `model\`; pick any, switch later from the model selector |

The model id is the directory name (`model\chat_demo\chat_demo.pt` → `chat_demo`); it
becomes `dir/file` only when the directory and file names differ, plus `@parent` on
collisions. Requests may use `chat_demo`, `chat_demo.pt` or
`model/chat_demo/chat_demo.pt` (case-insensitive).

### What the service implements

| Endpoint | Behaviour |
|---|---|
| `GET /v1/models` | Full model list (filenames only, no weights loaded) |
| `GET /v1/models/{id}` | Single model info |
| `POST /v1/chat/completions` | Chat, **streaming and non-streaming**; picks the model from the request, loads it on first use |
| `GET /health` | Health check with **loaded** models, device and default model |

**Lazy loading**: `/v1/models` and `/health` never read weights; only a chat request
`torch.load`s the `.pt`. Switching unloads via LRU and frees VRAM.

Behaviour contract — **the bridge does two things only; judging content is the model's job**:

- **`system` / `developer` roles are dropped** — this model was never trained with system prompts;
- **`tools` / `tool_choice` / `functions` are ignored**, and the reply is plain text —
  **`tool_calls` are never returned** (the DSH UI has no "disable tools" switch, so erroring out is not an option);
- Multi-turn history becomes `用户：… / 助手：…`; **when it overflows the context the oldest
  turns are dropped**, and a single over-long turn keeps its tail;
- No filtering, scoring or keyword matching beyond that — quality is the model's business;
- Generation stops at `用户：` / `助手：` and trailing markers are cleaned up;
- A hard cap on generation length (600 characters by default);
- Add `--reject-tools` to error on tool requests, `--debug` to dump raw client messages.

### Known limitations

| Limitation | Why |
|---|---|
| No tool calls | Tool definitions are ignored; it only returns text — no file reads, no commands |
| Very short context | `chat_demo` has `block_size` 128 (`dlqq_exam` is 256). **A single DSH agent message is usually 2000+ characters**, so the model only sees the tail |
| Weak answers | `chat_demo` is 0.9 M parameters over 30 Q&A pairs — mostly recitation; `dlqq_exam` is 13.7 M parameters trained on the question bank, so **it answers those questions and fails on rephrasings**; neither can reason |
| Not an agent model | It continues text; it does not plan or execute multi-step tasks |

> Conclusion: **hooking it into DSH validates the pipeline** — request out, model receives,
> reply back — and that works. For fluent answers, grow the corpus past 1 MB and retrain,
> or move up a tier.
