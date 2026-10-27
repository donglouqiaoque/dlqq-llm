# AGENTS.md — dlqq-llm-train 工程约定

> 本文给 AI 助手（DeepSeek Harness / Qwen Code / Claude Code / Cursor 等）阅读：
> 硬性规则、目录结构、配置语法与运行时行为。
> 使用说明见 [README.md](README.md)（英文版 [README.en.md](README.en.md)）；
> 本文英文版见 [AGENTS.en.md](AGENTS.en.md)。
> README 与 AGENTS 是**中英文双份**的唯一说明文档，改一份必须同步改另一份。

## 一、硬性规则（必须遵守）

1. **技术栈固定**：Python 3.10+ / PyTorch 2.x。训练与对话链路只允许 `torch`、`numpy`
   两个第三方库，其余一律用标准库。
2. **不引入 tokenizer / HuggingFace / transformers**：这是字符级模型，词表由
   `code/train_llm.py` 自己统计、自己切。
3. **禁止执行任何 git 命令**：包括只读的 `git status` / `diff` / `log` / `show`，
   以及 `git add` / `commit` / `push` / `restore`。版本管理由用户本人操作，AI 一律不碰。
4. **禁止用脚本批量改写源码**：不得用 PowerShell / Python / sed / awk 对源码做批量
   查找替换或自动改写。所有代码修改必须逐文件、逐处确认后用编辑工具完成；确需大范围
   调整时，先列清单交用户确认，再逐项修改。
5. **禁止 AI 执行有副作用的脚本**：训练（`script\train.bat`、`code\train_llm.py`）、
   下载语料（`script\fetch_chat_demo.bat`、`.ai_res\script\fetch_*.bat`）、删除、
   批量清理，一律交给用户手动执行。AI 只做只读检查：读文件、列目录、语法与静态检查。
6. **`.bat` 只能出现 ASCII**：注释、`echo`、提示语全部用英文；中文提示写在 Python 里。
   原因：cmd 默认代码页是 GBK，UTF-8 中文会乱码，甚至让批处理解析失败。
7. **编码与换行**：`.py` / `.properties` / `.md` 用 UTF-8（无 BOM）；`.bat` 用纯 ASCII；
   Windows 下文本文件统一 CRLF。
8. **临时文件只进 `.ai_res\`**：缓存、下载、临时产物一律放 `.ai_res\`（已被 gitignore），
   不要写系统临时目录；辅助脚本放 `.ai_res\script\`，不要堆在 `.ai_res\` 根目录。
9. **禁止硬编码本机路径**：`properties` 里的 `python=` 默认留空（走 PATH）；注释里的
   示例只能是通用形式（如 `C:\Python312\python.exe`），不得出现某台机器的
   `D:\Soft\...`、用户名或 conda 环境路径。
10. **权重与语料不进版本库**：`model/*`、`src/*` 都被 `.gitignore` 忽略（`chat_demo*`
    例外）；语料统一放**目录**，不接受单个文件。发布出去的权重在模型仓库
    `dlqq-llm-model`，应与本地 `model\` 保持同一份字节，并各自带 `.pt.sha256` 签名。
11. **禁止读写 `.git` 目录**：包括 `.git\index`、`.git\COMMIT_EDITMSG`、`.git\logs\*`
    等一切文件，**只读查看也不行**（看修改时间同样算）。判断提交/暂存状态只能问用户。

## 二、配置文件（properties）约定

- **语法**：一行一个 `key=value`，`#` 开头是**整行**注释。
- **禁止行尾注释**：解析器按 `split("=", 1)` 取值，`block_size=128  # 说明` 会把
  `# 说明` 一起当成值，直接报错或静默出错。注释只能单独占一行。
- **四个配置保持同构**：`chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem`
  键集合、顺序、分节一致；改一个文件的结构时，另外三个同步改。
- **只能写 `CONFIG` 里已有的键**：`code/train_llm.py` 的 `CONFIG` 是唯一来源，
  写了未知键会打印「未知配置项」并忽略。
- **`python=` 语义**：留空 = 用 PATH 上的 `python` / `python3` / `py`；填了 = **只认它**，
  不存在、是目录、缺 `torch`/`numpy` 都直接终止，**不做任何兜底查找**。
- **`resume=true`** 需要有带 `iter` / `optimizer` 的存档才会续训；没有存档时自动从头开始。

| 键 | 含义 |
|---|---|
| `data_path` | 语料**目录**（相对工程根）；传文件会报错并打印迁移命令 |
| `model` | 模型名，自动落到 `model\<名字>\<名字>.pt` |
| `cache_dir` | 切词缓存目录，保持 `.ai_res/cache` |
| `chat_mode` / `chat_temperature` / `chat_top_k` / `chat_max_new` | 对话参数，加载时读取 |
| `n_layer` / `n_head` / `n_embd` / `block_size` / `batch_size` | 模型结构与每步 token 数 |
| `max_iters` / `eval_interval` / `eval_iters` | 训练步数与评估节奏 |
| `lr` / `warmup_iters` / `lr_decay` / `min_lr_frac` / `weight_decay` / `grad_clip` / `dropout` / `seed` | 优化 |
| `grad_accum` / `grad_checkpoint` | 显存不够时的等效 batch / 省显存 |
| `resume` / `save_every` / `save_dtype` / `save_optimizer` | 续训与存档体积 |
| `max_vocab` / `val_frac` | 词表上限 / 从语料尾部切验证集的比例 |
| `python` / `confirm` | 解释器 / 是否二次确认 |

## 三、目录结构速查

```
code/                 训练与推理（不依赖 properties，可独立传参运行）
  train_llm.py        数据管线 + 模型 + 训练循环 + 采样，单文件 840 行
  chat.py             加载 .pt 对话；模型定义与 train_llm.py 保持一致（复制一份，不 import）
script/               入口脚本（双击即用）
  train.bat           pick 配置 → helper → train_llm.py
  chat.bat            选模型 → helper → chat.py
  http_server.bat     起 OpenAI 兼容桥接服务
  torch_check.bat     9 项环境自检
  fetch_chat_demo.bat 下载考题库 → src\dlqq_exam\dlqq_exam.jsonl
  train_chat_helper.py  上述 bat 的交互逻辑（选配置、验解释器、探设备）
  http_server.py      桥接服务本体（独立自足，只用 torch + 标准库）
  torch_check.py      自检项实现
  model_signature.py  权重 SHA-256 签名（计算 / 校验，纯标准库，不需要 torch）
  compute_signature.bat / verify_signature.bat  双击入口
  fetch_chat_demo.py  考题下载与格式转换
properties/           四个训练配置（要开源，注释写给外人看）
src/                  语料目录（不进版本库）
model/                模型目录（不进版本库；model\<名字>\<名字>.pt）
.ai_res/              临时目录（不进版本库）：cache / doc / script / git_log
```

- 工程根目录靠「向上找到含 `code\train_llm.py` 的那一级」定位，脚本换位置不会失效。
- 模型 id = 目录名（`model\chat_demo\chat_demo.pt` → `chat_demo`）；目录名与文件名不一致时
  才写成 `目录/文件名`。

## 四、运行时行为约定（改代码前先理解）

**数据管线**：分块读（4M 字）→ `np.bincount` 统计字频 → 取前 `max_vocab` 个字符 →
`searchsorted` 编码成 uint16 写 `.ai_res\cache\data-<指纹>.bin` → 训练用 `np.memmap` 随机读。
内存占用与语料大小无关。缓存指纹 = 源绝对路径 + 每个文件的（相对路径、大小、mtime、格式、
问答字段）+ `max_vocab` + `val_frac`；**改这些会导致重新切词**。

**验证集**从语料**尾部**切。语料按类别排序时要先打乱，否则验证损失失真。

**上下文自动缩水**：`_safe_T = min(训练 token, 验证 token) - 2`，`block_size` 超过它会被
自动调小并打警告。小语料配大 `block_size` 时，实际训练用的不是配置里的值。

**保存策略**：只在验证损失变好时存盘，`save_every > 0` 时额外定时存。

**对话参数优先级**（`chat.py` 与 `http_server.py` 各实现一份，必须保持一致）：
`命令行显式参数` > `properties\<模型名>.properties` > `模型文件里存的` > `默认值`。

**生成停止**：拼提示词用 `用户：` / `助手：`；生成到完整标记（含冒号）就停，收尾再清掉
结尾残留的裸标记；碰到下一个 `用户：` 直接截断。

**桥接服务「无观点」**（`script\http_server.py`）：
- 丢弃 `system` / `developer` 角色（模型没这么训过）；
- 忽略 `tools` / `tool_choice` / `functions` 但**不报错**，**永不返回 `tool_calls`**
  （可加 `--reject-tools` 改成 400）；
- 超上下文从**最早一轮**开始丢，只剩一轮就按字符留尾部；
- 不做任何内容过滤、打分或关键词匹配；`max_tokens` 按**字符**算，硬上限 600。

## 五、改代码的流程（AI 必须遵守）

1. **先读后改**：修改前完整读目标文件，不做「凭印象」的编辑。
2. **改完做静态检查**：`python -c "import ast; ast.parse(open(...).read())"` 之类，确认语法；
   **不要**为了验证去跑训练或下载。
3. **改默认值要说明影响**：比如改 `max_vocab` / `val_frac` 会让缓存失效、改 `block_size`
   会影响显存与上下文、改 `save_dtype` 会改变存档体积，必须写清后果。
4. **改配置结构要四份同步**：`chat_demo` / `dlqq_exam` / `dlqq_llm` / `dlqq_poem` 的键
   必须始终一致。
5. **改文档要中英同步**：`README.md` ↔ `README.en.md`，`AGENTS.md` ↔ `AGENTS.en.md`；
   章节、命令、数字必须一致，不允许只改一份。
6. **不动权重**：不修改、不复制、不删除 `model\*.pt`；需要重训时由用户执行。

### 提交信息（commit message）

- **放在哪**：写成 `.ai_res\git_log\<yyyy-MM-dd_HH-mm-ss>.txt`，UTF-8 无 BOM、CRLF、
  结尾带换行；文件里只有提交信息原文，不要加「以下是提交信息」这类说明，推送时整段复制。
- **格式**（Conventional Commits 1.0.0）：
  - 首行 `<type>: <描述>`，type 取 `feat` / `fix` / `docs` / `refactor` / `test` /
    `chore` / `build` / `perf` / `style`；祈使句，结尾不加句号
  - 首行不超过 50 列（中文按 2 列算）
  - 空一行写正文：先讲**为什么**（背景、动机、影响），再用 `- ` 列表讲**改了什么**
  - 正文每行不超过 72 列，不贴代码
- **只写会进这次提交的内容**：被 `.gitignore` 忽略的东西，**连路径都不许出现**——
  本仓库是 `model\` 里除 `chat_demo*` 之外的权重、`src\` 里除 `chat_demo*` 之外的语料、
  `.ai_res\` 全部。哪怕只是在描述某个已跟踪文件的行为（比如「默认输出改到哪个目录」），
  也不能把被忽略的路径写进去；写成「改到考题语料目录」这种不含路径的说法。
  反过来，本次改过的受版本控制文件一个都不能漏，尤其是权重、语料、`.gitignore`。
- **先确认提交状态**：AI 不能执行 git 命令，也不能读写 `.git` 目录，判断哪些已提交、
  哪些还在暂存区只能问用户，不要按猜测写。

## 六、代码风格

- **文件头**：`# -*- coding: utf-8 -*-`，紧跟一段 `r"""..."""`，写清用途、用法和注意事项。
- **编码兜底**：所有 `.py` 入口先 `sys.stdout/stderr.reconfigure(encoding="utf-8", errors="replace")`。
  否则 Windows 控制台（GBK）打印 `→`、`⚠️` 会直接抛 `UnicodeEncodeError`，训练跑完却在最后一屏崩掉。
- **注释用中文**：解释「为什么这么做」，不要复述代码。参数取值的依据（实测显存、耗时、语料规模）
  写进注释，后来者才知道能不能改。
- **失败即退出**：参数、环境、数据不合法时直接 `sys.exit(1)`，并打印**可执行的修复提示**
  （该改哪一行、该跑哪个 bat）。**不做兜底猜测**：不自动找解释器、不自动换目录。
- **不吞异常**：`except` 必须写具体类型；要么打印，要么重抛，禁止 `except: pass`。
- **打印格式**：统一 `[标签] 内容`，标签用 `[数据]` / `[训练]` / `[设备]` / `[错误]` / `[警告]`；
  训练循环里带 `flush=True`，否则日志会憋在缓冲区里，看起来像卡死。
- **命名**：函数与变量 `snake_case`，常量 `UPPER_SNAKE_CASE`，类 `CamelCase`；配置键沿用
  `CONFIG` 里的 `snake_case`，新增键必须同时出现在 `CONFIG` 中。
- **不用类型注解**：与现有代码保持一致，参数含义写在 docstring 里。
- **模型定义有三份**：`code/train_llm.py`（训练）、`code/chat.py`（对话）、
  `script/http_server.py`（桥接）。改网络结构（层数、维度、权重共享）必须三处同步，
  否则加载权重会报 key 不匹配。
- **不引入日志框架**：用 `print(..., flush=True)` 即可，训练与对话日志直接看控制台。

## 七、已知限制

- **字符级 + 短上下文**：`chat_demo` 的 `block_size` 是 128，`dlqq_exam` 是 256；
  一条 2000 字的 agent 消息只能看到最后一段。
- **小模型会背语料**：`chat_demo` 30 条问答、`dlqq_exam` 全是考题，原题能答、换问法就露馅；
  不要拿同一批题报 benchmark 分数。
- **语料许可**：`dlqq_exam` 含 C-Eval（CC BY-NC-SA 4.0，非商业），语料不要随仓库公开分发。
- **桥接服务不是 agent**：不会调用工具、不会规划、不会分步执行，只适合验证链路。
- **CPU 训练**很慢且不启用 bf16（CPU 上更慢），只建议用 `tiny` 跑通流程。

## 八、参考资料

- 模型权重：[ModelScope · donglouqiaoque/dlqq-llm](https://www.modelscope.cn/models/donglouqiaoque/dlqq-llm)
- 源码镜像：[GitHub](https://github.com/donglouqiaoque/dlqq-llm) ｜ [Gitee](https://gitee.com/donglouqiaoque/dlqq-llm)
- PyTorch 文档：<https://pytorch.org/docs/stable/index.html>
- nanoGPT（同量级的参考实现）：<https://github.com/karpathy/nanoGPT>
