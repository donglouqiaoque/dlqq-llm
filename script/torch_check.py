# -*- coding: utf-8 -*-
"""PyTorch 安装自检 —— 逐项测试，失败不中断
==============================================
用法：双击 script\\torch_check.bat，或者
      python script\\torch_check.py

如果当前解释器没有 torch，本脚本会再试 PATH 上的 python / python3 / py，
谁有 torch 就用它重新跑一遍自己 —— 所以从哪个 python 启动都能出正确结果。
"""
import os
import re
import sys
import time
import shutil
import platform
import subprocess

# Windows 控制台默认是 GBK，打印 ✅ / ⚠ 这类字符会直接抛 UnicodeEncodeError。
# 这里把标准输出切成 UTF-8，并显式指定 errors="replace"：个别终端仍不支持时
# 顶多显示成占位符，不会让脚本半路崩掉。
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


# ==================================================================
# 自我修复：当前解释器没有 torch 时，换一个有的重新执行自己
# ==================================================================
def _torch_ok(exe):
    """问 pip 的清单最快；pip 不可用再退回真的 import 一次"""
    if not exe or not os.path.isfile(exe):
        return False
    try:
        r = subprocess.run([exe, "-m", "pip", "list"],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0 and r.stdout:
            return re.search(r"(?m)^torch\s", r.stdout) is not None
    except Exception:
        pass
    try:
        r = subprocess.run([exe, "-c", "import torch"],
                           capture_output=True, timeout=600)
        return r.returncode == 0
    except Exception:
        return False


def _candidates():
    """候选解释器，只认 PATH（和显式指定），不去猜安装目录。
    顺序：显式指定 > PATH 上的 python > 当前解释器。"""
    out, seen = [], set()

    def add(p):
        if not p:
            return
        if "\\windowsapps\\" in os.path.normcase(os.path.abspath(p)):
            return
        key = os.path.normcase(os.path.abspath(p))
        if key not in seen:
            seen.add(key)
            out.append(p)

    add(os.environ.get("PYTHON") or os.environ.get("TLLM_PYTHON"))
    for name in ("python", "python3", "py"):
        exe = shutil.which(name)
        if exe:
            add(exe)
    add(sys.executable)
    return out


def relaunch_if_needed():
    """当前解释器没 torch 就换一个；换成功则本进程不再返回"""
    try:
        import torch  # noqa: F401
        return
    except Exception:
        pass

    me = os.path.abspath(__file__)
    print(f"  当前解释器没有 torch：{sys.executable}")
    print("  正在查找其它装了 torch 的 python …")
    for exe in _candidates():
        if os.path.abspath(exe).lower() == os.path.abspath(sys.executable).lower():
            continue
        if _torch_ok(exe):
            print(f"  找到：{exe}")
            print("  用它重新执行自检 …\n")
            sys.stdout.flush()
            try:
                sys.exit(subprocess.call([exe, me], cwd=os.getcwd()))
            except KeyboardInterrupt:
                sys.exit(130)
    print("  [FAIL] 没找到任何装了 torch 的 python。")
    print("         装一个：python -m pip install torch")
    sys.exit(1)


relaunch_if_needed()

FAILS, SKIPS = [], []

class Skip(Exception):
    """本项在当前环境下不适用（例如没有 GPU）"""

def head(t):
    print("\n" + "=" * 60)
    print("  " + t)
    print("=" * 60)

def run(name, fn):
    try:
        r = fn()
        print("  [ OK ]  %-36s %s" % (name, "" if r is None else r))
    except Skip as e:
        print("  [SKIP]  %-36s %s" % (name, e))
        SKIPS.append(name)
    except Exception as e:
        print("  [FAIL]  %-36s %s: %s" % (name, type(e).__name__, e))
        FAILS.append(name)

# ---------------------------------------------------------------- 1
head("1 / 9   环境信息")
print("  Python   :", sys.version.split()[0], "|", platform.architecture()[0])
print("  系统     :", platform.platform())

try:
    import torch
except Exception as e:
    print("  [FAIL] import torch ->", type(e).__name__, e)
    sys.exit(1)

print("  torch    :", torch.__version__)
print("  cuda 编译:", torch.version.cuda)
print("  cudnn    :", torch.backends.cudnn.version())
print("  安装位置 :", torch.__file__)
if "+cpu" in torch.__version__:
    print("  ⚠⚠ 这是 CPU-only 版本！想要 GPU 加速请改装 cu126 版：")
    print("      python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126")

# ---------------------------------------------------------------- 2
head("2 / 9   GPU 是否可用")
HAS_CUDA = torch.cuda.is_available()
print("  torch.cuda.is_available() =", HAS_CUDA)
if HAS_CUDA:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        mem = getattr(p, "total_memory", None) or getattr(p, "total_mem", 0)
        print("  [%d] %s" % (i, p.name))
        print("      算力 sm_%d%d   显存 %.1f GB" % (p.major, p.minor, mem / 1024**3))
    print("  当前设备 :", torch.cuda.get_device_name(0))
else:
    print("  ⚠ 没有可用 CUDA 设备。可能原因：")
    print("     1) 装成了 +cpu 版         → 看上面 torch 版本号有没有 +cpu")
    print("     2) 显卡驱动太旧           → cmd 里跑 nvidia-smi 看右上角 CUDA Version")
    print("     3) 这台机器本来就没 N 卡  → 那就只能用 CPU")
    FAILS.append("torch.cuda.is_available() 为 False")

DEV = "cuda" if HAS_CUDA else "cpu"

def need_gpu():
    if not HAS_CUDA:
        raise Skip("无 GPU，本机不适用")

# ---------------------------------------------------------------- 3
head("3 / 9   基础张量运算")
run("创建张量与四则运算", lambda: str((torch.randn(3, 4) * 2 + 1).shape))
run("把张量放到 %s 上" % DEV, lambda: str(torch.zeros(2, 2, device=DEV).device))
def move_check():
    need_gpu()
    return str(torch.randn(2, 2).cuda().cpu().shape)
run("CPU <-> GPU 来回搬运", move_check)

# ---------------------------------------------------------------- 4
head("4 / 9   矩阵乘法正确性（GPU 与 CPU 对比）")
def matmul_check():
    if not HAS_CUDA:
        torch.manual_seed(0)
        return "无 GPU，只算了 CPU: %s" % str(tuple((torch.randn(256,256) @ torch.randn(256,256)).shape))
    torch.manual_seed(0)
    a = torch.randn(256, 256); b = torch.randn(256, 256)
    g = (a.cuda() @ b.cuda()).cpu()
    d = (g - a @ b).abs().max().item()
    assert d < 1e-2, "误差过大 %.4f（结果不正确）" % d
    return "与 CPU 的最大误差 %.2e  (正常)" % d
run("256x256 矩阵乘", matmul_check)

# ---------------------------------------------------------------- 5
head("5 / 9   自动求导（反向传播）")
def autograd_check():
    x = torch.randn(128, 128, device=DEV, requires_grad=True)
    ((x ** 2).sum()).backward()
    assert x.grad is not None and x.grad.shape == x.shape
    return "grad 形状 %s" % str(tuple(x.grad.shape))
run("autograd 反向传播", autograd_check)

# ---------------------------------------------------------------- 6
head("6 / 9   神经网络算子")
def conv_check():
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(3, 8, 3, padding=1).to(DEV)
    return "Conv2d 输出 %s" % str(tuple(conv(torch.randn(1, 3, 32, 32, device=DEV)).shape))
run("Conv2d 卷积", conv_check)

def pool_check():
    x = torch.randn(1, 8, 32, 32, device=DEV)
    return "MaxPool 输出 %s" % str(tuple(torch.nn.MaxPool2d(2)(x).shape))
run("MaxPool2d 池化", pool_check)

def half_check():
    need_gpu()
    x = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    return "fp16 矩阵乘 %s" % str(tuple((x @ x).shape))
run("半精度 fp16", half_check)

def bf16_check():
    need_gpu()
    x = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
    return "bf16 矩阵乘 %s" % str(tuple((x @ x).shape))
run("bfloat16", bf16_check)

# ---------------------------------------------------------------- 7
head("7 / 9   torchvision（真跑一个网络）")
def tv_check():
    import torchvision
    return "版本 " + torchvision.__version__
run("import torchvision", tv_check)

def resnet_check():
    from torchvision.models import resnet18
    m = resnet18(weights=None).to(DEV).eval()      # weights=None -> 不联网下载
    with torch.no_grad():
        y = m(torch.randn(1, 3, 224, 224, device=DEV))
    return "ResNet18 前向输出 %s" % str(tuple(y.shape))
run("ResNet18 前向传播", resnet_check)

# ---------------------------------------------------------------- 8
head("8 / 9   torchaudio（版本不匹配的重灾区）")
def ta_check():
    import torchaudio
    return "版本 " + torchaudio.__version__
run("import torchaudio", ta_check)

def mel_check():
    import torchaudio
    sr = 16000
    wav = torch.randn(1, sr, device=DEV)
    mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=64).to(DEV)
    return "MelSpectrogram 输出 %s" % str(tuple(mel(wav).shape))
run("真跑一次 MelSpectrogram（暴露 ABI 问题）", mel_check)

# ---------------------------------------------------------------- 9
head("9 / 9   速度对比：CPU vs GPU")
def bench(dev, n=1024, iters=30):
    a = torch.randn(n, n, device=dev); b = torch.randn(n, n, device=dev)
    for _ in range(3):
        a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        a @ b
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1000.0

def speed_check():
    tc = bench("cpu")
    print("  CPU : %9.2f ms / 次   (1024x1024 矩阵乘)" % tc)
    if HAS_CUDA:
        tg = bench("cuda")
        print("  GPU : %9.2f ms / 次" % tg)
        print("  → GPU 快了 %.1f 倍" % (tc / tg))
        return None
    raise Skip("无 GPU 可比")
run("速度对比", speed_check)

# ---------------------------------------------------------------- 总结
head("总结")
if FAILS:
    print("  ❌ 有问题的项（需要处理）：")
    for f in FAILS:
        print("     -", f)
else:
    print("  ✅ 全部通过 —— 这套 PyTorch 可以正常用了")
if SKIPS:
    print("  ⏭  跳过（本机不适用，不是错误）：" + "、".join(SKIPS))
print("\n  把上面整段输出贴回来即可。")
