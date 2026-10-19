# llm-train · 字符级 GPT 训练与对话

一套从零实现的极小语言模型工具链：**不用 tokenizer、不用 HuggingFace**，
语料进、模型出，训练完就能跟它聊天。

```
语料 txt  →  字符级词表  →  GPT(Pre-Norm)  →  AdamW 训练  →  采样生成
```

模型是自己实现的多头注意力 + Pre-Norm + 权重共享 + Flash Attention，
默认档位约 80 万参数，8 GB 显存的笔记本显卡几分钟就能跑出结果。

---

## 目录

- [快速开始](#快速开始)
- [环境准备](#环境准备)
- [训练模型](#训练模型)
- [跟模型对话](#跟模型对话)
- [用自己的语料](#用自己的语料)
- [参数与档位](#参数与档位)
- [运行模式与显存](#运行模式与显存)
- [目录结构](#目录结构)
- [常见问题](#常见问题)
- [把模型接进 DSH](#把模型接进-dsh)

---

## 快速开始

准备好 Python 与 torch 之后，双击两个文件即可：

| 步骤 | 双击 | 你要做的 |
|---|---|---|
| 1 | `script\train.bat` | 按编号选一个配置文件，回车即开始训练 |
| 2 | `script\chat.bat` | 直接回车，然后开始打字聊天 |

**所有训练参数都写在 `properties\` 的配置文件里**（语料、模型大小、步数、
学习率、甚至用哪个 Python），改配置就行，不需要记任何命令行参数。

想先确认环境是否装好，双击 `script\torch_check.bat`，它会逐项检查并在最后给出结论。

> 自带两个配置：`chat_demo.properties`（考题库 demo，13.7 M 参数，几分钟）和
> `dlqq_llm.properties`（`src\dlqq_llm\` 目录，GB 级语料）。对话默认用 `model\chat_demo\chat_demo.pt`。
> **语料或模型缺失时会直接终止**，并提示你怎么补上。

---

## 环境准备

| 项 | 要求 |
|---|---|
| Python | 3.10 或更高（已在 3.14 上验证） |
| torch | 2.x；要用显卡就装 CUDA 版 |
| 显卡 | 8 GB 显存即可跑满默认档位，没有显卡也能用 CPU（会慢很多） |

装依赖（**只需要 torch**）：

```bat
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

注意两点：

1. 用 `python -m pip install` 装到你要用的那个解释器里（不是 `conda install`）；
2. 装完跑一次 `script\torch_check.bat`，看到 `CUDA available: True` 和你的显卡名就说明可用。

### 用哪个 Python

只有两种来源，**不做任何自动查找**：

```
1. 配置文件里的 python=      ← 填了就必须能用，不能用就直接报错退出
2. PATH 上的 python / python3 / py   ← 没填时用这个
```

- 配置里填了路径：**只认这个**。文件不存在、是目录、跑不起来、缺 `torch` / `numpy`
  —— 一律当场报错退出，**不会退而求其次去找别的解释器**；
- 配置里留空：用 PATH 上的 `python`（找不到就试 `python3`、`py`），
  它缺依赖同样直接报错，不换别的。

启动时会打印实际用的是哪个：

```
[环境] D:\Soft\Python\Python3_13_9\python.EXE
```

想固定用某个解释器，就在 `properties\*.properties` 里写一行（要指向 `python.exe`，不是目录）：

```
python=D:\Soft\Python\Python3_13_9\python.exe
```

配错了会是这样，训练不会开始：

```
[ERROR] python from config not found:
        D:\NoSuchDir\python.exe
[错误] 配置里指定的 python 缺少依赖：torch、numpy
       D:\...\python.exe
       装一下："D:\...\python.exe" -m pip install torch numpy
[终止] 解释器不可用，未开始训练。
```

### 环境不对时会怎样

训练与对话脚本**不做环境自检**：启动时依赖缺失就立刻停下，并提示你运行自检脚本。

```
[错误] 运行环境缺少依赖：torch
       请先运行 script\torch_check.bat 检查运行环境。
```

`script\torch_check.bat` 会检查 9 项（环境信息、CUDA 可用性、张量搬运、矩阵乘正确性、
自动求导、卷积与半精度、torchvision、torchaudio、CPU/GPU 速度对比），**失败不中断**，
逐条给出 `[ OK ] / [SKIP] / [FAIL]`。

它还会自动在 `PATH` 里挑一个装了 torch 的解释器来跑：逐个试 `python` / `python3` / `py`，
都不行就明确报「缺 torch」，不会去猜别的安装目录。

---

## 训练模型

### 双击 → 选一个配置文件（推荐）

双击 `script\train.bat`，只会问你一件事：

```
Available configs:
  [1] chat_demo.properties
  [2] dlqq_llm.properties

Select a number (Enter = 1): 1
```

输编号回车，剩下的全部由配置文件决定，直接开始训练。

```
[config]  properties\chat_demo.properties
[python]  D:\Soft\Python\Python3_13_9\python.exe

  训练（配置文件：chat_demo.properties）
[语料] src\chat_demo（目录，1 个文件）
[输出] model\chat_demo\chat_demo.pt
[参数] n_layer=6 n_head=6 n_embd=384 block_size=256 batch_size=32 max_iters=3000 ...
[环境] D:\Soft\Python\Python3_13_9\python.EXE
```

训练结束窗口会停住（脚本末尾有 `pause`），方便你看日志和模型生成样例。

### 配置文件在哪、怎么改

都在 **`properties\`** 目录下，一行一个 `key=value`，`#` 开头是注释：

| 文件 | 语料 | 档位 | 说明 |
|---|---|---|---|
| `chat_demo.properties` | `src\chat_demo\` | medium（13.7 M 参数） | 考题库 demo，几分钟 |
| `dlqq_llm.properties` | `src\dlqq_llm\` | large（33 M 参数） | GB 级语料，约 1 小时 |

**所有参数都在文件里**，改完保存就生效，不需要碰代码：

| 键 | 含义 |
|---|---|
| `data_path` | 语料**目录**（相对工程根目录；目录里放 `txt` / `jsonl`，可以多个） |
| `model` | 模型名（不用写 `.pt`，自动存到 `model\<名字>\<名字>.pt`） |
| `cache_dir` | 切词缓存放哪（默认 `ai_res/cache`） |
| `n_layer` / `n_head` / `n_embd` | 模型结构 |
| `block_size` / `batch_size` | 上下文长度 / 每批条数 |
| `max_iters` / `eval_interval` | 训练步数 / 多少步评估一次 |
| `lr` / `dropout` / `weight_decay` / `grad_clip` / `seed` | 优化相关 |
| `max_vocab` / `val_frac` | 词表上限 / 验证集比例 |
| `python` | **指定解释器**；留空 = 自动找一个装了 torch 的 |
| `confirm` | `no` = 选完直接开训，`yes` = 再问一次 |

想加一套新配置，复制一份改改即可，`train.bat` 会自动列出来。

### 用法示例

```bat
script\train.bat                        REM 双击等价，列出配置让你选
script\train.bat dlqq_llm               REM 直接指定 properties\dlqq_llm.properties
script\train.bat my.properties          REM 指定任意配置文件名
```

### 直接调用训练脚本（进阶）

配置文件覆盖不到的临时调整，可以直接传参：

```bat
python code\train_llm.py data_path=src\chat_demo out_path=model\chat_demo\chat_demo.pt max_iters=3000
```

> 参数是 `key=value` 形式，**不是** `--key value`。
> `out_path` 的上级目录不存在时会自动创建。

### 不带参数会怎样

```bat
python code\train_llm.py
```

不指定 `data_path` 时，脚本会自动下载 TinyShakespeare（约 1.1 MB）作为训练语料。
下载失败会直接停下并给出两个选择，不会拿空数据硬跑。

训练过程中每个评估点都会打印验证损失，**只保存验证损失最好的那一版**。
训练结束时会用模型自己生成一段文字，让你直观看到当前效果。

---

## 跟模型对话

### 方式一：双击（推荐）

双击 `script\chat.bat`：

- **直接回车** → 用默认模型 `model\chat_demo\chat_demo.pt`；
- 输入**编号** → 选清单里的其它模型；
- 输入**名称** → 用它找到对应的 `.pt`。

### 方式二：命令行

```bat
script\chat.bat                  REM 回车用默认模型
script\chat.bat 我的模型          REM 直接指定，跳过询问

python code\chat.py model\chat_demo\chat_demo.pt               REM 交互式
python code\chat.py model\chat_demo\chat_demo.pt --once "用户：你好"  REM 只生成一次
```

### 对话中的命令

| 命令 | 作用 |
|---|---|
| `:q` | 退出（`:quit` / `:exit` 也行） |
| `:t 0.7` | 改温度。越小越保守、越大越野 |
| `:n 200` | 改单次生成长度 |
| `:raw` | 在「对话模式」和「续写模式」之间切换 |
| `:p` | 查看当前设置 |

### 两种模式的区别

- **对话模式**：把你的话拼成 `用户：xxx` + `助手：`，让模型接着回答；
- **续写模式**：把你的话当成一段普通文本，让模型往下写。

如果模型的词表里没有「用户：」「助手：」这些字，说明它不是用对话语料训出来的，
`chat.py` 会**自动切到续写模式**，不需要你手动设置。

---

## 用自己的语料

**语料统一放在一个目录里**（`data_path` 只接受目录，不再接受单个文件）。

先在 `src\` 下建一个目录，例如 `src\my_corpus\`，把语料文件丢进去：

```
src\
  my_corpus\
    part1.jsonl
    part2.jsonl        ← 可以多个文件，训练时按文件名排序合并
    notes.txt
```

支持的文件：

| 文件 | 怎么处理 |
|---|---|
| `.jsonl` 问答（`input`/`target` 等） | 拼成「用户：…／助手：…」 |
| `.jsonl` 纯文本（`{"text": "..."}`） | 原样当纯文本 |
| `.txt` | 原样当纯文本（UTF-8 / GBK / BIG5 都能自动识别） |

想让它会对话，jsonl 这样写：

```json
{"input": "什么是过拟合", "target": "模型把训练数据背下来了，换一批新数据就不行了。"}
{"input": "那怎么缓解", "target": "加数据、加正则、减小模型容量，三条路一起走。"}
```

然后训练：

```bat
python code\train_llm.py data_path=src\my_corpus out_path=model\我的模型\我的模型.pt max_iters=5000
```

几个经验值：

| 语料规模 | 大致效果 |
|---|---|
| < 100 KB | 基本是**背下来**的，问答能对上，换个问法就崩 |
| 1 MB 左右 | 开始能看出「学到了句式」，泛化明显变好 |
| 10 MB 以上 | 短句通顺，长句仍会飘 |

目录里的文件按 UTF-8 / utf-8-sig / gb18030 / big5 自动识别，记事本存的 GBK 文件也能直接读。

### 下载来的语料要先清洗

从维基、fineweb 这类源下回来的东西不能直接喂（维基是 `{{模板}}` `[[链接]]`，网页语料混着英文和广告）。
`ai_res\script\prepare_corpus.py` 负责转换和清洗：

```bat
REM 维基 dump（wikiextractor 的输出，每行一个 JSON）
python ai_res\script\prepare_corpus.py --in ai_res\doc\zhwiki --out ai_res\doc\zhwiki.txt --wikitext

REM parquet（fineweb 之类，自动找 text 列）
python ai_res\script\prepare_corpus.py --in ai_res\doc\fineweb --out ai_res\doc\fineweb.txt --min-chars 100
```

它会做：剥维基标记、统一空白、长度过滤（`--min-chars`）、中文占比过滤（`--min-zh`）、按行去重，
最后输出训练脚本能直接吃的 `.txt`（段落间空行）或 `.jsonl`。

---

## 参数与档位

| 档位 | n_layer | n_head | n_embd | block | batch | 参数量 | 显存 | 说明 |
|---|---|---|---|---|---|---|---|---|
| `tiny` | 2 | 2 | 64 | 64 | 16 | ~0.2 M | < 0.5 G | 几秒一轮，用来验证流程 |
| `small`（默认） | 4 | 4 | 128 | 128 | 32 | ~0.8 M | < 1 G | 能看出效果 |
| `medium` | 6 | 6 | 384 | 256 | 64 | ~10 M | ~2–3 G | 语料够大时明显更好 |
| `large` | 8 | 8 | 512 | 256 | 32 | ~25 M | ~6–7 G | 8 G 显存的上限附近，别再往上 |

所有配置项都能在命令行覆盖：

```bat
python code\train_llm.py n_layer=6 n_embd=384 lr=1e-3 max_iters=5000
```

常用的几个键：

| 键 | 默认值 | 含义 |
|---|---|---|
| `data_path` | 无（自动下载） | 语料文件；不填就下载 TinyShakespeare |
| `out_path` | `model/mymodel.pt` | 模型保存位置 |
| `max_iters` | 2000 | 训练步数，最直观的「训练多久」 |
| `batch_size` | 32 | 每步几条样本，显存不够就调小 |
| `block_size` | 128 | 上下文长度；语料太短会被自动调小 |
| `lr` | 3e-4 | 学习率 |
| `dropout` | 0.1 | 语料小的时候可以调大一点 |
| `eval_interval` | 200 | 多少步评估一次、顺便决定要不要保存 |
| `max_vocab` | 8000 | 词表上限，只保留最常出现的前 N 个字 |
| `val_frac` | 0.05 | 从语料尾部切这个比例做验证集 |
| `cache_dir` | `ai_res/cache` | 切好词的二进制缓存放这里（临时文件统一进 `ai_res`） |
| `prompt` | 空 | 训练结束采样用的开头；留空自动挑 |
| `data_format` | `auto` | `auto` / `txt` / `jsonl`；auto 看扩展名和文件开头 |
| `q_field` | 空 | jsonl 里「问」的字段名；留空自动推断 |
| `a_field` | 空 | jsonl 里「答」的字段名；留空自动推断 |
| `eval_iters` | 50 | 每次评估跑几批；长训练建议调到 20 省时间 |
| `grad_accum` | 1 | 梯度累积：显存不够时用它凑等效 batch |
| `warmup_iters` | 0 | 学习率预热步数，长训练建议 1000–2000 |
| `lr_decay` | `cosine` | `cosine` 余弦退火 / `none` 恒定 |
| `min_lr_frac` | 0.1 | 退火终点 = `lr × 这个比例` |
| `grad_checkpoint` | `False` | 梯度检查点：省显存，慢约 20–30% |
| `resume` | `False` | 从 `out_path` 接着训（要里面存过 optimizer/iter） |
| `save_every` | 0 | 每 N 步强制存档（0 = 只在验证变好时存） |

---

## 让 dlqq_llm 训得更久（可选）

`dlqq_llm` 现在直接在 `properties\dlqq_llm.properties` 上训练，想训得更久就改这个文件里的参数，
不需要另建配置：

| 参数 | 作用 |
|---|---|
| `max_iters` | 训练步数；配合 `batch_size × block_size` 估算见过的 token 数 |
| `grad_accum` | 显存不够时用梯度累积凑等效 batch |
| `warmup_iters` / `lr_decay` / `min_lr_frac` | 学习率预热 + 余弦退火，长训练必开 |
| `grad_checkpoint` | 梯度检查点，省显存（慢 20–30%） |
| `resume` | 改成 `true` 后从 `out_path` 接着训（优化器状态、学习率进度都接得上） |
| `save_every` | 每 N 步强制存档，防止「验证损失一直没变好 → 一直不存盘」 |

例：把 `max_iters` 调到 60000、`resume=true`，双击 `script\train.bat` 选 `dlqq_llm` 即可继续。

---

## 做一个小体积的答题 demo

仓库体积有限（Gitee 单文件 50 MB），原来那份 5 K 字的演示语料互动性很弱。
`chat_demo` 现在用**考题库**当语料：同样体积下信息密度高得多，demo 立刻变得能玩。

两步：

```bat
REM 1) 下载考题（魔搭：C-Eval + 高考 + 生物），
REM    直接写成 demo 语料 src\chat_demo\chat_demo.jsonl
script\fetch_chat_demo.bat

REM 2) 训练 + 聊天（配置就是原来的 chat_demo，没有新增配置）
script\train.bat chat_demo
script\chat.bat
```

体积控制在 50 MB 以内的关键是 `properties\chat_demo.properties` 里这两行：

```properties
save_dtype=fp16        # 权重存半精度，体积减半
save_optimizer=false   # 不存优化器状态，再省一半（demo 不需要续训）
```

实测：3.24 M 参数，fp32 + 优化器 31 MB → fp16 无优化器 **6.2 MB**；
10.76 M 参数（6 层 384 维，小词表）→ **20.6 MB**。按这个比例，13.7 M 参数的
demo 模型约 **27 MB**，塞进 50 MB 没问题。

> **⚠ 数据来源要写明**：demo 把题库全部用于训练，模型是**记住了这些题**，
> 所以要标注语料来源（魔搭上的 C-Eval / gaokao-benchmark 等），
> 并且**不要**把同一批题当 benchmark 报分数 —— 那是自问自答，不是能力。

### 语料从哪来（全部走魔搭，国内直连）

| 数据集（魔搭仓库） | 规模 | 用途 |
|---|---|---|
| `philexohf/LLM-Pretrain-Data` | 约 24.6 GB / 49.6 亿 token | **预训练首选**：网页 55% + 新闻 27% + 维基 12% + 百科 6%，已清洗去重，txt 每行一篇 |
| `AI-ModelScope/chinese-fineweb-edu` | 约 290 GB | 只要网页语料时取子集 |
| `OmniData/C-Eval` | 1.5 MB | 52 学科选择题 |
| `opencompass/gaokao-benchmark` | 5.7 MB | 高考选择题 |
| `pandayi/Biology_subject_of_college_entrance_examination` | 0.2 MB | 高考生物（含解析） |
| `SHPDataGR/medical-exam-question-bank-json` | 35 MB | 医学考试题库 |
| `tongyi_dianjin/CFLUE` | 66 MB | 金融资格考试选择题 |

下载方式：

```bat
REM 考题：用自带脚本，直连魔搭，不需要 GitHub / HuggingFace / 登录
script\fetch_chat_demo.bat --datasets c-eval gaokao bio

REM 预训练大语料：用 ai_res\script 下的脚本，按需取多少 MB（HTTP Range，不整包下）
ai_res\script\fetch_pretrain_corpus.bat --source mix --max-mb 2000
```

`fetch_pretrain_corpus.bat` 会把清洗后的文本写成 jsonl 直接放进 `src\dlqq_llm\`，
训练配置指向的就是这个目录，所以下完直接训即可。下载时会显示进度条（百分比、已下/总量、速度）。

### 预训练语料有多大？为什么默认只下 2 GB？

```bat
ai_res\script\fetch_pretrain_corpus.bat --list
```

```
mix      philexohf/LLM-Pretrain-Data
         可用 1 个文件，合计 12.72 GB
fineweb  AI-ModelScope/chinese-fineweb-edu
         可用 302 个文件，合计 311.44 GB
```

| 语料 | 总量 | 说明 |
|---|---|---|
| `mix`（train.txt） | **12.72 GB** ≈ 45 亿 token | 网页+新闻+维基+百科，一行一篇 |
| `fineweb` | **311 GB**（302 个 parquet 分片） | 教育向网页，按片下载 |

默认 2 GB 是按模型规模算的：

- `dlqq_llm` 是 33 M 参数，按「每参数 20 token」≈ **6.6 亿 token** 就够；
- 2 GB 中文文本 ≈ 7 亿 token，加上原来的 `firefly-train-1.1M.jsonl`（1.1 GB ≈ 4 亿 token），
  总量约 11 亿 token，已经超过所需；
- 全下 12.7 GB 对 33 M 模型没有收益，只是多花 1–2 小时下载和 13 GB 硬盘。

真要全量就传 `--max-mb 0`（一直下到结束），或者用 `--max-mb 8000` 取 8 GB。

---

## 训练 GB 级大语料

1 GB 中文约 3.5 亿字。如果先把整个文件读进内存、再转成 Python 列表，光这一步就要
吃掉十几 GB —— 所以数据管线是**流式**的：

```
分块读 → numpy 统计字频 → 取前 max_vocab 个字 → 分块编码成 uint16 → 写 ai_res\cache\data.bin
训练时 np.memmap 内存映射，随机读几 KB
```

**内存占用与语料大小无关**，只跟 `block_size` / `batch_size` 有关。

### 直接喂 JSONL，不用先转 txt

`data_path` 指到 `.jsonl` 就行，脚本会自动找问答字段并拼成 `用户：…／助手：…`
（**直接把 jsonl 喂进去会把括号引号也学进文法，所以这一步必须做**）。

自动识别的字段名，按顺序尝试：

```
input/target · q/a · question/answer · src/tgt · instruction/output
prompt/response · query/response · ask/answer · title/content
```

都对不上就显式指定：`q_field=问  a_field=答`。

### 实测（1.12 GB 的 `src\dlqq_llm\`，1,649,399 条问答）

| 指标 | 数值 |
|---|---|
| 切词耗时 | **30 秒**（含启动，建一次后续复用） |
| 产出 token | 425,220,455 |
| 词表 | 8000（上限截断），词表外只丢弃 21 万字（0.05%） |
| `data.bin` | 850 MB |
| 峰值内存 | 约 1.5 GB（与语料大小无关） |
| 第二次运行 | 4.9 秒（直接复用缓存） |

命令：

```bat
python code\train_llm.py data_path=src\dlqq_llm out_path=model\dlqq_llm\dlqq_llm.pt ^
    max_vocab=8000 n_layer=8 n_head=8 n_embd=512 block_size=256 batch_size=32 ^
    max_iters=60000 lr=3e-4 eval_interval=2000
```

这份语料的实测训练速度（29.43 M 参数，8 GB 显存跑得下）：

| 档位 | 配置 | 速度 | 60000 步 |
|---|---|---|---|
| small | 4 层 / 128 维 / block 128 / batch 32 | 143 步/秒 | 约 7 分钟 |
| large | 8 层 / 512 维 / block 256 / batch 32 | 约 5 步/秒 | 约 3.3 小时 |

> small 档只有 0.8 M 参数，喂 1 GB 会严重欠拟合；**GB 级语料请用 large**。
> 冒烟测试里 large 档 30 步就把验证损失从 9.06 压到 5.72，说明数据是通的。

两点注意：

- 每份语料（连同 `max_vocab` / `val_frac` / 格式 / 字段这些设置）对应一个独立缓存文件
  `ai_res\cache\data-<指纹>.bin`，改任何一项都会自动写一份新的。
  所以 demo 的 `chat_demo\` 和 GB 级的 `dlqq_llm\` 可以**同时留着**，
  来回切换不用重新切词；想全部重来就删 `ai_res\cache\`；
- 验证集是从**语料尾部**切的。语料若按类别顺序排列（比如前半新闻、后半对话），
  建议先打乱行序，否则验证损失反映不了真实水平。

### 加语料：往目录里丢文件就行

`data_path` 既可以是**文件**，也可以是**目录**。`dlqq_llm.properties` 现在指向目录：

```properties
data_path=src/dlqq_llm
```

这个目录里可以放任意多个文件，训练时按文件名排序依次读、拼成一条流：

| 放进去的东西 | 怎么处理 |
|---|---|
| `*.jsonl`（`input`/`target` 等问答字段） | 拼成 `用户：…／助手：…` |
| `*.jsonl`（只有 `{"text": "..."}`） | 当纯文本直接喂（预训练语料就长这样） |
| `*.txt` | 原样当纯文本 |
| 子目录 | 也会扫（跳过隐藏目录） |

新增/删除/改动文件后，缓存指纹会变，**下次训练自动重新切词**，不用手动删缓存。

扩充语料用 `ai_res\script\fetch_pretrain_corpus.bat`（魔搭直连，不需要 GitHub / HuggingFace / 登录）：

```bat
REM 默认：混合语料（网页+新闻+维基+百科，已清洗）取 2 GB，写进 src\dlqq_llm\
ai_res\script\fetch_pretrain_corpus.bat

REM 只取 500 MB 先试；换教育向网页语料；或者全下
ai_res\script\fetch_pretrain_corpus.bat --max-mb 500
ai_res\script\fetch_pretrain_corpus.bat --source fineweb --max-mb 4000
ai_res\script\fetch_pretrain_corpus.bat --max-mb 0

REM 看有哪些语料、各有多大
ai_res\script\fetch_pretrain_corpus.bat --list
```

它会做长度过滤、中文占比过滤、按行去重，输出 `pretrain-mix-000.jsonl`、`pretrain-mix-001.jsonl`……
每个默认 200 MB，**已有文件绝不覆盖**（序号往后顺延）。下载时有进度条，
结束后会打印「语料目录现在共 N 个文件 / X GB」。下完直接 `script\train.bat dlqq_llm`。

---

## 运行模式与显存

每次开始训练或对话前，都会先报一次运行环境：

```
[模式] GPU —— 混合精度 bf16（推荐：显存占用更低、张量核更快）
[设备] NVIDIA GeForce RTX 4060 Laptop GPU | sm_89
[显存] 总 8.0 GB，已用 1.1 GB（可用 6.9 GB）
```

三种模式的含义：

| 模式 | 触发条件 | 说明 |
|---|---|---|
| **GPU · 混合精度 bf16** | 有 CUDA 设备 | 前向与反向走 bf16，显存占用更低、张量核更快 |
| **GPU · 单精度 fp32** | 有 CUDA 设备但关掉 AMP | 精度更高，显存占用更大 |
| **CPU · 单精度 fp32** | 没有可用的 CUDA 设备 | 会额外报**内存**总量与占用；CPU 上开 bf16 反而更慢，所以不启用 |

显存不够时的处理顺序：调小 `batch_size` → 调小 `block_size` → 换更低的档位。

---

## 目录结构

```
code/                     训练与对话脚本
  train_llm.py            训练
  chat.py                 对话 / 推理

script/                   入口脚本（双击即用）
  train.bat               训练（列出 properties\ 里的配置让你选）
  chat.bat                进入对话
  fetch_chat_demo.bat     下载考题库、生成 demo 语料（魔搭，国内直连）
  torch_check.bat         环境自检
  http_server.bat         启动桥接服务（供 DSH 接入）
  train_chat_helper.py    上面几个 bat 的交互逻辑
  fetch_chat_demo.py      考题下载与格式转换
  torch_check.py          自检的具体检查项
  http_server.py          桥接服务本体（OpenAI 兼容，独立自足）

properties/               训练配置（一行一个 key=value，改完即生效）
  chat_demo.properties    demo 配置（13.7 M 参数，语料来自 script\fetch_chat_demo.bat）
  dlqq_llm.properties     GB 级语料（src\dlqq_llm\）+ large 档

ai_res/                   临时目录，放辅助脚本与中间产物，**已被 .gitignore 忽略**
  cache/                  切词缓存（每份语料一个 data-<指纹>.bin）
  doc/                    原始语料 / 下载的题库
  script/                 辅助工具：语料导出、PDF 转换、出图、跑分、预训练语料下载等

src/                      语料目录（每个语料一个文件夹，例如 src\chat_demo\、src\dlqq_llm\）
model/                    模型目录（每个模型一个文件夹：model\chat_demo\chat_demo.pt）
release/                  打包输出目录
```

> `src/` 与 `model/` 里除 `.gitkeep` 与 `chat_demo.*` 之外的内容都在 `.gitignore` 中，
> 属于本地数据，不进版本库。

---

## 常见问题

**Q：窗口一闪就没了，看不到结果？**

所有 bat 结尾都有 `pause`，正常情况不会闪退。如果确实闪退，多半是没找到 `python.exe`，
在命令行里手动跑一次就能看到完整报错：

```bat
script\train.bat
```

**Q：提示「运行环境缺少依赖」怎么办？**

跑 `script\torch_check.bat`。它会告诉你缺什么、以及当前用的是哪个 Python。
最常见的原因是 PATH 上的 Python 和你装 torch 的那个不是同一个。

**Q：在 IDE 的「运行」按钮里跑，报「读不到键盘输入」？**

那种方式没有可交互的终端，`input()` 直接拿到 EOF，所以没法选编号。两个办法：

1. **双击** `script\train.bat`（真实控制台窗口），正常答题；
2. 在 IDE 里**直接指定配置**，跳过提问：

```bat
script\train.bat chat_demo
```

**Q：训练完想换语料重训，模型名会冲突吗？**

不会。输入已存在的名称时会问你是否覆盖，回答 `n` 就可以换个名字。

**Q：想训练但不想用 `chat_demo`？**

先在 `src\` 下建一个语料目录，再复制一份配置改两行（`train.bat` 会自动列出新配置）：

```
# properties\my_corpus.properties
data_path=src\my_corpus      ← 目录，里面放 jsonl / txt
model=我的模型
...（其余照抄 chat_demo.properties，按需改 n_layer / max_iters 等）
```

**`data_path` 只接受目录**；目录里 `txt` 和 `jsonl` 都支持，格式按扩展名和文件开头自动判断，
jsonl 会自动找问答字段（`{"text": ...}` 这种单字段的当纯文本用）。

**Q：想临时改一个参数，不想动配置文件？**

直接调训练脚本，命令行覆盖会盖掉配置文件里的同名项：

```bat
python code\train_llm.py data_path=src\chat_demo out_path=model\chat_demo\chat_demo.pt max_iters=500
```

**Q：模型训练很久还是只会复读？**

语料太小。字符级模型在 100 KB 以下的语料上主要是在背诵，
想看到泛化，把语料做到 1 MB 以上再试。

**Q：能用 CPU 训练吗？**

能，但会很慢，而且 CPU 上不会启用 bf16（反而更慢）。建议至少 `tiny` 档位先跑通流程。

---

## 把模型接进 DSH

训练出来的模型可以起成一个 **OpenAI 兼容的本地服务**，然后在 DeepSeek Harness 里像
普通提供方一样使用 —— **DSH 侧不改任何代码或配置文件，全部在界面里完成**。

### 第一步：启动桥接服务

双击 `script\http_server.bat` 就行 —— **不用选模型**。它会自动扫 `model\` 和 `release\`
下的所有 `.pt`，启动时只列清单，**一个都不加载**：

```
[清单] 共 3 个模型（现在不加载，谁被调用才加载谁）
       chat_demo                     3.6 MB  10-17 16:25  ← 默认
       dlqq_llm                    117.9 MB  10-17 17:27
       chat_demo_exam               27.0 MB  10-19 11:40
```

谁被 DSH 真正调用，才把谁读进显存；切到另一个模型时，上一个自动卸载
（默认只留 1 个在显存里，可用 `--max-loaded` 调）。

也可以直接用参数跳过提问：

```bat
script\http_server.bat                      REM 监听 49128，自动扫模型
script\http_server.bat 9000                 REM 换端口
script\http_server.bat 9000 cpu             REM 强制用 CPU 推理（不占显卡）
script\http_server.bat 9000 cuda 2          REM 同时在显存里留 2 个模型
python script\http_server.py --help         REM 看全部参数
```

启动后窗口会打印接口地址与模型清单：

```
  接口     http://127.0.0.1:49128/v1
  模型清单 http://127.0.0.1:49128/v1/models
  健康检查 http://127.0.0.1:49128/health
  停止服务：在这个窗口按 Ctrl+C
```

**这个窗口要一直开着**，DSH 才能访问到模型。

### 第二步：在 DSH 里添加自定义提供方

打开 DSH → **设置 → 模型 → 添加提供方 → 自定义提供方**，按下表填写：

| 字段 | 填什么 |
|---|---|
| Provider ID | `local-gpt`（小写连字符；**保存后不可改**） |
| 显示名称 | 随便写，例如 `本地小模型` |
| 基础 URL | `http://127.0.0.1:49128/v1` |
| API 协议 | `openai-completions` |
| 凭据 / API Key | `local`（本地服务不校验，但**这一栏不能留空**） |
| 模型 | 点「获取可用模型」会列出 `model\` 下所有模型；**随便选，之后在模型选择器里随时切换** |

模型 id 就是模型目录名（`model\chat_demo\chat_demo.pt` → id `chat_demo`）；
目录名和文件名不一致时才写成 `目录/文件名`，重名再加 `@父目录` 区分。
请求里的 `model` 写 `chat_demo`、`chat_demo.pt`、`model/chat_demo/chat_demo.pt` 都能认（大小写不敏感）。

保存后回到会话页，在模型选择器里换模型，服务端会自动加载对应权重。

### 服务实现了什么

| 接口 | 说明 |
|---|---|
| `GET /v1/models` | 全部模型清单（只读文件名，不加载权重） |
| `GET /v1/models/{id}` | 单个模型信息 |
| `POST /v1/chat/completions` | 对话，**流式与非流式都支持**；按请求里的 `model` 选模型，首次用到才加载 |
| `GET /health` | 健康检查，返回当前**已加载**的模型、设备、默认模型 |

**懒加载**：`/v1/models`、`/health` 都不会把权重读进内存；只有对话请求到了，
才 `torch.load` 对应的那个 `.pt`。切换时按 LRU 卸载旧模型并释放显存缓存。

行为约定 —— **桥接层只做两件事，内容判断一律交给模型**：

- **system / developer 角色直接忽略** —— 这个模型从没用系统提示训过；
- **`tools` / `tool_choice` / `functions` 一律忽略**，照常返回文本，
  **永远不返回 `tool_calls`**（DSH 界面里没有「关闭工具」的开关，所以不能报错）；
- 多轮对话拼成 `用户：…／助手：…`；**超出上下文时从最早的一轮开始丢**，
  只剩一轮还太长就按字符保留尾部；
- 除此之外没有任何过滤、打分或关键词匹配 —— 答成什么样是模型的事；
- 生成到 `用户：` / `助手：` 就停，并清理结尾残留的标记；
- 单次生成长度有硬上限（默认 600 字符）；
- 想恢复「带 tools 就报错」的严格行为，启动时加 `--reject-tools`；
  想看客户端发来的原始消息，启动时加 `--debug`。

### 已知限制

| 限制 | 说明 |
|---|---|
| 不会调用工具 | 请求里带的工具定义会被忽略，模型只回文本；它不会读文件、不会跑命令 |
| 上下文很短 | 默认档位只有 128 字元（`block_size`）。**DSH 一条 agent 消息通常有 2000 字以上**，所以模型实际只看到最后那一小段 |
| 回答质量受限 | demo 模型约 13.7 M 参数，训练自考题库（魔搭：C-Eval / 高考 / 生物），**能答题库里的题，但换个问法就露馅**；它没有推理能力，只是把语料背下来了 |
| 不是 agent 模型 | 它只会接着你的话往下写，不会规划、不会分步执行任务 |

> 结论：**接进 DSH 用来验证链路 —— 请求发得出、模型收得到、回答回得来 —— 是完全可行的**。
> 想要通顺的回答，得先把语料做大（1 MB 以上）再重训，或者换更大的档位。
