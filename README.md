# dlqq-llm-train · 字符级 GPT 训练与对话

**中文** ｜ [English](README.en.md)

预训练权重 ｜ [ModelScope 模型仓库](https://www.modelscope.cn/models/donglouqiaoque/dlqq-llm)

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
双击 `script\start.bat`，浏览器会自动打开本地控制台（地址形如
`http://127.0.0.1:7788/?t=...`）。页面有五个标签：

| 标签 | 用途 |
|---|---|
| 训练 | 选配置和语料，点【开始训练】；日志实时滚动，同时落盘到 `.ai_res\cache\train-logs\` |
| 对话 | 选模型，直接打字；回复逐字出现，可随时【停止】 |
| 服务 | 起一个 OpenAI 兼容的本地桥接服务，页面上能实时看到每次问答 |
| 配置 | 在表单里改 `properties\*.properties`，保存时保留注释与键顺序 |
| 签名 | 给 `model\` 下的 `.pt` 计算或校验 SHA-256 |

> **关掉 `start.bat` 的 cmd 窗口 = 退出**：训练、桥接服务、签名任务都会一起结束。
> 关掉浏览器标签**不算**退出，训练照跑，重新访问控制台地址即可。
> 每个模型都带同名 `.pt.sha256`，可在「签名」页核对后再加载。

命令行不再是使用入口：`--config` / `--parent-pid` 这些参数只由 WebUI 内部调用。

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
2. 装完直接双击 `script\start.bat`，在「训练」页选 `chat_demo`，能开训就说明环境可用（没显卡会自动用 CPU）。

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

训练与对话脚本**不做环境自检**：启动时依赖缺失就立刻停下，并给出安装命令。

```
[错误] 运行环境缺少依赖：torch
       请先安装：python -m pip install torch numpy
```

没有显卡时自动回退 CPU（只是慢），不会因为找不到 CUDA 而报错。

它还会自动在 `PATH` 里挑一个装了 torch 的解释器来跑：逐个试 `python` / `python3` / `py`，
都不行就明确报「缺 torch」，不会去猜别的安装目录。

---

## 训练模型
### 在「训练」页训练（唯一入口）

1. 双击 `script\start.bat`，页面默认就在「训练」标签；
2. **配置**下拉里选一个 `properties\*.properties`（下拉项后面会显示该文件头部的说明）；
3. **语料**下拉显示 `src\` 下每个语料目录的文件数与体积，仅供核对——实际读哪个目录由配置里的
   `data_path` 决定；
4. 点【开始训练】。日志会实时滚动，同时写入 `.ai_res\cache\train-logs\<配置名>-<时间>.log`；
5. 想停就点【停止】；想续训，把配置里的 `resume` 改成 `true` 再点开始（需要有带优化器的存档）。

页面上看不到参数？参数都在配置文件里：`properties\` 下四个文件（`chat_demo` / `dlqq_exam` /
`dlqq_llm` / `dlqq_poem`），键和含义见下文的参数表，也可以在「配置」页直接改。

### 配置文件在哪、怎么改

- 文件：`properties\<名字>.properties`，一行一个 `key=value`，`#` 开头是整行注释；
- **不要写行尾注释**：解析器按 `split("=", 1)` 取值，`block_size=128  # 说明` 会把注释当成值；
- 四个文件的键集合、顺序、分节保持一致，新增键要同时出现在四个文件里；
- 在「配置」页保存时，只改 value，注释、空行、键顺序都保留，改完下次训练生效。

### 想加一套新配置

复制一份现有的改改名字，它会自动出现在「训练」页的下拉里（下拉内容是实时扫描 `properties\`）。

## 跟模型对话
在「对话」标签里：

1. **模型**下拉会自动列出 `model\**\*.pt`（显示体积、修改时间、是否已签名）；
2. 调好温度 / top_k / 生成长度（默认取 `properties\<模型名>.properties` 里的值）；
3. 输入内容回车发送，回复会**逐字**出现，生成中可点【停止】。

- 切换模型不用重启页面：旧模型会被释放（含显存）再加载新的；
- 训练正在跑时，**只能选不同的模型**——与训练配置同名的模型会被拒绝，避免拖慢训练或读到半成品；
- 对话上下文按模型的 `block_size` 截断，拼提示词的规则是 `用户：…` / `助手：…`；
  词表里没有这两个标记的模型会自动切「续写模式」。

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
# 在 properties\ 里新建一份配置，改这三行：
#   data_path=src/my_corpus
#   model=我的模型
#   max_iters=5000
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
清洗要做的是：剥维基标记、统一空白、长度过滤、中文占比过滤、按行去重，
最后输出训练脚本能直接吃的 `.txt`（段落间空行）或 `.jsonl`。
本机有现成的清洗脚本（不随仓库发布），用法见 AIRES.md。

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
# 在配置里改：n_layer=6、n_embd=384、lr=0.001、max_iters=5000
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
| `cache_dir` | 临时目录下的 cache | 切好词的二进制缓存放这里 |
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
| `chat_mode` | `True` | `False` = 纯续写模型（诗词），提示词不加「用户：／助手：」 |
| `chat_temperature` | 0.8 | 对话采样温度，越大越发散 |
| `chat_top_k` | 40 | 只从概率最高的 K 个里采样；0 = 不截断 |
| `chat_max_new` | 180 | 一次最多生成多少字 |

`chat_*` 这几个是**对话参数**，训练时写进模型文件；`chat.py` 和 `http_server.py`
加载模型时会按下面的优先级取用（改 properties 不用重训就生效）：

```
命令行显式参数  >  properties\<模型名>.properties  >  模型文件里存的  >  默认值
```

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

例：把 `max_iters` 调到 60000、`resume=true`，双击 `script\start.bat`，在「训练」页选 `dlqq_llm` 即可继续。

---

## 做一个小体积的考题 demo（dlqq_exam）

仓库体积有限（Gitee 单文件 50 MB）。`chat_demo` 是 30 条手写问答（13 KB），
跑通流程够用，但互动性弱；`dlqq_exam` 改用**考题库**当语料，同样体积下信息密度高得多。

两步：

```bat
REM 1) 准备考题语料：把 C-Eval / 高考 / 生物 合并成 src\dlqq_exam\dlqq_exam.jsonl
REM    （语料不进仓库，需自行下载并合并）

REM 2) 训练 + 聊天
script\start.bat    （训练页选 dlqq_exam）
script\start.bat    （对话页选 dlqq_exam）
```

体积控制在 50 MB 以内的关键是 `properties\dlqq_exam.properties` 里这两行：

```properties
save_dtype=fp16        # 权重存半精度，体积减半
save_optimizer=false   # 不存优化器状态，再省一半（demo 不需要续训）
```

实测：4,562 条语料切出约 270 万 token，13.7 M 参数存成 fp16 + 无优化器约 **27 MB**；
`chat_demo` 更小，0.9 M 参数约 **1.9 MB**。

> **⚠ 数据来源要写明**：`dlqq_exam` 把题库全部用于训练，模型是**记住了这些题**，
> 所以要标注语料来源（魔搭上的 C-Eval / gaokao-benchmark 等），
> 并且**不要**把同一批题当 benchmark 报分数 —— 那是自问自答，不是能力。
> 另外 C-Eval 的许可是 CC BY-NC-SA 4.0（非商业），这份语料不要随仓库公开分发。

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
REM 考题：从魔搭自行下载 C-Eval / gaokao-benchmark / 高考生物，合并成一个 jsonl
REM       （直连魔搭，不需要 GitHub / HuggingFace / 登录）

REM 预训练文本：下载后清洗成 jsonl，放进训练配置指向的语料目录
```

下载和清洗都用本机的辅助脚本完成（不随仓库发布），用法见 AIRES.md。
产出的 jsonl 直接进语料目录，训练配置指向的就是那里，下完直接训即可。

### 预训练语料有多大？为什么默认只下 2 GB？

脚本支持 `--list` 查看各语料的总量（用法见 AIRES.md）：

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

- `dlqq_llm` 是 29.4 M 参数，按「每参数 20 token」≈ **5.9 亿 token** 就够；
- 2 GB 中文文本 ≈ 7 亿 token，加上原来的 `firefly-train-1.1M.jsonl`（1.1 GB ≈ 4 亿 token），
  总量约 11 亿 token，已经超过所需；
- 全下 12.7 GB 对 29.4 M 模型没有收益，只是多花 1–2 小时下载和 13 GB 硬盘。

真要全量就传 `--max-mb 0`（一直下到结束），或者用 `--max-mb 8000` 取 8 GB。

---

## 训练 GB 级大语料

1 GB 中文约 3.5 亿字。如果先把整个文件读进内存、再转成 Python 列表，光这一步就要
吃掉十几 GB —— 所以数据管线是**流式**的：

```
分块读 → numpy 统计字频 → 取前 max_vocab 个字 → 分块编码成 uint16 → 写缓存文件
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

配置要点（写进 `properties\dlqq_llm.properties`）：

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

这份语料的实测训练速度（29.43 M 参数，8 GB 显存跑得下）：

| 档位 | 配置 | 速度 | 60000 步 |
|---|---|---|---|
| small | 4 层 / 128 维 / block 128 / batch 32 | 143 步/秒 | 约 7 分钟 |
| large | 8 层 / 512 维 / block 256 / batch 32 | 约 5 步/秒 | 约 3.3 小时 |

> small 档只有 0.8 M 参数，喂 1 GB 会严重欠拟合；**GB 级语料请用 large**。
> 冒烟测试里 large 档 30 步就把验证损失从 9.06 压到 5.72，说明数据是通的。

两点注意：

- 每份语料（连同 `max_vocab` / `val_frac` / 格式 / 字段这些设置）对应一个独立的缓存文件，
  改任何一项都会自动写一份新的。所以 demo 的 `chat_demo\` 和 GB 级的 `dlqq_llm\`
  可以**同时留着**，来回切换不用重新切词；想全部重来就删掉缓存目录；
- 验证集是从**语料尾部**切的。语料若按类别顺序排列（比如前半新闻、后半对话），
  建议先打乱行序，否则验证损失反映不了真实水平。

### 加语料：往目录里丢文件就行

`data_path` 只接受**目录**（传文件会直接报错并打印迁移命令）。`dlqq_llm.properties` 指向：

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

扩充语料从魔搭下载（不需要 GitHub / HuggingFace / 登录），下载与清洗由本机的辅助脚本
完成，用法见 AIRES.md。常用的几种取法：

它会做长度过滤、中文占比过滤、按行去重，输出 `pretrain-mix-000.jsonl`、`pretrain-mix-001.jsonl`……
每个默认 200 MB，**已有文件绝不覆盖**（序号往后顺延）。下载时有进度条，
结束后会打印「语料目录现在共 N 个文件 / X GB」。下完直接双击 `script\start.bat`，在「训练」页选 `dlqq_llm`。

---

## 训练一个写诗模型 dlqq_poem

专门写诗的小模型：6 层 384 维（14.1 M 参数），语料是**古诗 + 宋词 + 近现代诗**，
在 8 GB 显存上半小时左右训完。

语料里**只有诗词正文，没有「诗：」「词：」这类标注**，所以它写出来的就是诗本身，
不会把标注当内容复读。

### 1. 下语料

三个源（诗 43 MB、宋词 19 MB、近现代诗 57 MB）分别下载、清洗，由本机的辅助脚本完成，
用法见 AIRES.md。

| 源 | 仓库 | 大小 | 内容 |
|---|---|---|---|
| `shici` | `modelscope/chinese-poetry-collection` | 43.4 MB | 唐宋元明清诗 |
| `songci` | `peaceminuczy/songci` | 18.7 MB | 宋词 |
| `modern` | `Iess/chinese_modern_poetry`（国内镜像 hf-mirror.com） | 56.6 MB | 近现代诗（中国诗人 + 外国诗中译） |

都会下载到 `src\dlqq_poem\`，输出 `shici-000.jsonl`、`songci-000.jsonl`、`modern-000.jsonl`，
已有文件不覆盖、进度条、按句号断行排版。

> 旧版本生成的语料每首带「诗：」「词：」前缀。脚本会检测这种文件并**停下提醒你删掉**
> （`del src\dlqq_poem\*.jsonl`），否则新旧混在一起会把标注学进去。

### 2. 训练

```bat
script\start.bat    （训练页选 dlqq_poem）
```

配置在 `properties\dlqq_poem.properties`，是按实际语料调过的：

| 项 | 值 | 依据 |
|---|---|---|
| 结构 | 6 层 384 维（14.1 M） | 约 4000 万字 / 14 M 参数 ≈ 2.8 token/参数 |
| `block_size` | 256 | 现代诗平均 210 字、90% 在 136 字内，256 能完整装下一首 |
| `batch_size` | 32 | 32 × 256 = 每步 8192 token（与 64×128 步数相同，上下文翻倍） |
| `max_iters` | 15000 | ≈ 1.23 亿 token ≈ 语料（3920 万 token）3.1 轮 |
| `max_vocab` | 7000 | 覆盖 99.9% 以上的字 |
| `chat_mode` | `false` | 告诉 `chat.py` 这是续写模型，别套「用户：／助手：」 |

实测约 7.0 万 token/秒，**全程约 29 分钟**；每 2500 步存一次档（fp16）。

### 3. 让它写

语料是纯正文，所以对话就是**给它一个开头，它往下写**：

```bat
script\start.bat    （对话页选 dlqq_poem）
```

```
你  > 春
它  > 风暖鸟声碎，日高花影重。
      ...
```

- 输入一两个字 → 以它起句；
- 输入一句诗 → 接着往下续；
- 直接输入一段现代诗的句子 → 按现代诗的断行风格续写。

因为 `chat_mode=false` 存在模型文件里，`chat.py` 会自动用续写模式，不会再加「用户：／助手：」。

采样参数也放在 `properties\dlqq_poem.properties` 里，改完**不用重训**就生效：

```properties
chat_mode=false
chat_temperature=1.0     # 小模型用 0.8 容易陷入「他的女人，他的女人」循环
chat_top_k=0             # 0 = 不截断，从全词表采样
chat_max_new=200
```

`script\start.bat` 的「对话」页和「服务」页加载这个模型时都会读这些值；
临时想改，命令行加 `temperature=0.8 top_k=40` 就能覆盖。

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
script\                 双击入口（只有这一处）
  start.bat             找 python → 前台起 code\webui.py → 打开浏览器
  logo.ico              快捷方式图标
code\                   所有 Python
  webui.py              本地 Web 服务：静态页 + JSON API + SSE + 子进程管理
  train.py              训练（数据管线 + 模型 + 训练循环 + 采样）
  chat.py               对话核心：ChatSession（加载 / 流式回复 / 改参数）
  http_server.py        OpenAI 兼容桥接服务（多模型懒加载）
  common.py             find_root / properties 保序读写 / 父进程看门狗 / 隐藏目录
  model_signature.py    权重 SHA-256 签名（只依赖标准库）
web\                    前端页面（由 webui.py 托管）
  index.html            五个标签：训练 / 对话 / 服务 / 配置 / 签名
properties\             四个训练配置
src\                    语料目录（不进版本库）
model\                  训练产物（不进版本库；model\<名字>\<名字>.pt 与同名 .pt.sha256）
.ai_res\                临时目录（缓存、训练日志、配置备份，不进版本库）
```

- 工程根目录靠「向上找到含 `code\train.py` 的那一级」定位，整个目录搬到哪都能跑；
- 模型 id = 目录名：`model\chat_demo\chat_demo.pt` 的 id 就是 `chat_demo`。

## 常见问题

**Q：窗口一闪就没了，看不到结果？**

所有 bat 结尾都有 `pause`，正常情况不会闪退。如果确实闪退，多半是没找到 `python.exe`，
在命令行里手动跑一次就能看到完整报错：

```bat
script\start.bat    （训练页）
```

**Q：提示「运行环境缺少依赖」怎么办？**

看报错里的依赖名，用当前解释器装：`python -m pip install torch numpy`。
最常见的原因是 PATH 上的 Python 和你装 torch 的那个不是同一个。

**Q：在 IDE 的「运行」按钮里跑，报「读不到键盘输入」？**

那种方式没有可交互的终端，`input()` 直接拿到 EOF，所以没法选编号。两个办法：

1. **双击** `script\start.bat`，在「训练」页选 `dlqq_exam`，正常答题；
2. 在 IDE 里**直接指定配置**，跳过提问：

```bat
script\start.bat    （训练页选 chat_demo）
```

**Q：训练完想换语料重训，模型名会冲突吗？**

不会。输入已存在的名称时会问你是否覆盖，回答 `n` 就可以换个名字。

**Q：想训练但不想用 `chat_demo`？**

先在 `src\` 下建一个语料目录，再复制一份配置改两行（「训练」页的下拉会实时列出新配置）：

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
# 在配置里改：data_path=src/chat_demo、model=chat_demo、max_iters=500
```

**Q：模型训练很久还是只会复读？**

语料太小。字符级模型在 100 KB 以下的语料上主要是在背诵，
想看到泛化，把语料做到 1 MB 以上再试。

**Q：能用 CPU 训练吗？**

能，但会很慢，而且 CPU 上不会启用 bf16（反而更慢）。建议至少 `tiny` 档位先跑通流程。

**Q：怎么确认权重文件没被改过、没传坏？**

双击 `WebUI「签名」页的【校验签名】`：它扫描 `model\` 下所有 `.pt`，逐个重算 SHA-256，
和旁边的 `<名字>.pt.sha256` 比对，四个都 PASS 才算没问题。训练完、或把权重传到模型仓库
之前跑一次。签名用 `WebUI「签名」页的【计算签名】` 重新生成（改过权重后必须重新生成）。

---

## 把模型接进 DSH
在「服务」标签里：

1. 填端口（默认 49128），点【启动服务】；
2. 页面显示 `http://127.0.0.1:49128/v1`，把这个地址填进 DSH 的模型提供方设置；
3. 之后每次请求的问答都会实时出现在页面下方的「会话记录」里（来自服务进程输出的
   `@@EV@@` 事件），排错时不用再翻控制台；
4. 点【停止】关闭服务；关掉 `start.bat` 的窗口时它也会一起退出。

服务本身是 OpenAI 兼容的：`/v1/models`、`/v1/chat/completions`（支持 `stream`），
模型列表自动扫描 `model\**\*.pt`，谁被调用才加载谁。

