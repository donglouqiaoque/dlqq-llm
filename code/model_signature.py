# -*- coding: utf-8 -*-
r"""模型权重数字签名：计算与校验
================================================================
dlqq-llm-train 训练出来的权重放在 model\<名字>\<名字>.pt。本脚本给每个 .pt
生成一份 SHA-256 摘要文件：训练完核对一遍，之后传到模型仓库（dlqq-llm-model）
的就是同一份字节，两边对得上。

约定 —— 一个 .pt 对应一个签名文件，文件名 = pt 文件名 + ".sha256"：
    model\chat_demo\chat_demo.pt
    model\chat_demo\chat_demo.pt.sha256

签名文件是纯文本 key=value（algorithm / file / size / sha256），
内容只跟权重字节有关：重复计算不会产生差异。

计算和校验都扫描 model\ 下所有 .pt —— 有几个处理几个，
以后往目录里加新模型，不用改脚本。

注意：model\ 整个目录在 .gitignore 里（chat_demo* 例外），签名文件默认
不进版本库，属于本地校验用。

用法（在仓库根目录执行）：
    python script\sign\create\model_signature.py --mode compute        # 生成/更新全部签名
    python script\sign\create\model_signature.py --mode verify         # 校验全部
    python script\sign\create\model_signature.py --mode verify --file model\dlqq_poem\dlqq_poem.pt
    python script\sign\create\model_signature.py --mode compute --quiet

也可以直接双击：
    script\compute_signature.bat      生成签名
    script\verify_signature.bat       校验签名

只依赖 Python 标准库（hashlib），不需要 torch —— 只想核对权重的人
不必安装训练环境。
"""
import argparse
import hashlib
import os
import sys
import time

# Windows 控制台默认 GBK，输出里的中文/符号需要 UTF-8 兜底
for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))

ALGORITHM = "sha256"
SIG_SUFFIX = ".sha256"
CHUNK = 4 * 1024 * 1024          # 4 MB 一块：内存占用固定，速度也够

SIG_HEADER = """\
# dlqq-llm 模型权重数字签名
# 由 script/model_signature.py 生成，请勿手工修改（改一个字符校验就会失败）。
# algorithm=sha256；校验：script\\verify_signature.bat
"""


# ==================================================================
# 小工具
# ==================================================================
def find_root(start):
    """往上找到同时含 model\\ 与 script\\ 的那一级，作为仓库根目录。
    这样脚本被移动位置、或从别的目录调用，都不会算错路径。"""
    cur = os.path.abspath(start)
    for _ in range(8):
        if (os.path.isdir(os.path.join(cur, "model"))
                and os.path.isdir(os.path.join(cur, "script"))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.dirname(os.path.abspath(start))


def human_size(n):
    """1536 → '1.5 KB'；给控制台看的，签名文件里始终写精确字节数"""
    if n < 1024:
        return f"{n} B"
    for unit in ("KB", "MB", "GB", "TB"):
        n /= 1024.0
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
    return f"{n:.1f} TB"


def sha256_file(path, show_progress=False):
    """流式计算 SHA-256，返回 (十六进制摘要, 耗时秒)。
    分块读：337 MB 的权重也只占几 MB 内存。"""
    h = hashlib.sha256()
    total = os.path.getsize(path)
    done = 0
    t0 = time.time()
    tick = 0.0
    with open(path, "rb") as f:
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            h.update(buf)
            done += len(buf)
            if show_progress:
                now = time.time()
                if now - tick >= 0.12 or done == total:
                    tick = now
                    pct = (done * 100.0 / total) if total else 100.0
                    speed = (done / 1048576.0) / max(now - t0, 1e-6)
                    sys.stdout.write(
                        f"\r        {human_size(done)} / {human_size(total)}"
                        f"  {pct:5.1f}%  {speed:6.1f} MB/s   ")
                    sys.stdout.flush()
    if show_progress and total:
        sys.stdout.write("\r" + " " * 78 + "\r")
        sys.stdout.flush()
    return h.hexdigest(), (time.time() - t0)


# ==================================================================
# 签名文件：写 / 读
# ==================================================================
def write_signature(pt_path, digest):
    """在 .pt 旁边写 <pt 文件名>.sha256，返回 (签名路径, 之前是否已存在)"""
    sig_path = pt_path + SIG_SUFFIX
    existed = os.path.isfile(sig_path)
    lines = [
        f"algorithm={ALGORITHM}",
        f"file={os.path.basename(pt_path)}",
        f"size={os.path.getsize(pt_path)}",
        f"sha256={digest}",
    ]
    # newline="\n"：固定 LF 换行，Windows / Linux 上生成的签名文件逐字节一致
    with open(sig_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(SIG_HEADER)
        f.write("\n".join(lines) + "\n")
    return sig_path, existed


def read_signature(sig_path):
    """读 key=value；忽略空行和 # 注释；键名统一小写"""
    data = {}
    with open(sig_path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            data[k.strip().lower()] = v.strip()
    return data


# ==================================================================
# 找目标：model\ 下所有 .pt
# ==================================================================
def collect_targets(root, files):
    """返回要处理的 .pt 绝对路径列表；参数不合法时返回 None"""
    if files:
        out = []
        for name in files:
            p = name if os.path.isabs(name) else os.path.join(root, name)
            p = os.path.abspath(p)
            if not os.path.isfile(p):
                print(f"[错误] 找不到文件：{name}")
                return None
            if not p.lower().endswith(".pt"):
                print(f"[错误] 不是 .pt 权重文件：{name}")
                return None
            out.append(p)
        return out

    model_dir = os.path.join(root, "model")
    found = []
    for cur, dirs, names in os.walk(model_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for n in sorted(names):
            if n.lower().endswith(".pt") and not n.startswith("."):
                found.append(os.path.join(cur, n))
    return sorted(found)


def print_banner(title):
    print("=" * 74)
    print(f"  dlqq-llm 模型权重数字签名 / {title}")
    print("=" * 74)


# ==================================================================
# 计算
# ==================================================================
def do_compute(root, files, quiet):
    targets = collect_targets(root, files)
    if targets is None:
        return 2
    if not targets:
        print("[错误] model\\ 下一个 .pt 都没找到，确认权重已经放好。")
        return 1

    print_banner("计算")
    print(f"[算法] SHA-256（流式读取，内存占用固定 {CHUNK // 1048576} MB）")
    print(f"[仓库] {root}")
    print(f"[扫描] 共 {len(targets)} 个 .pt 文件，逐个生成签名")
    print("-" * 74)

    t_all = time.time()
    for i, p in enumerate(targets, 1):
        rel = os.path.relpath(p, root)
        size = os.path.getsize(p)
        print(f"  [{i}/{len(targets)}] {rel}  ({human_size(size)}, {size:,} 字节)")
        progress = (not quiet) and sys.stdout.isatty()
        digest, dt = sha256_file(p, show_progress=progress)
        sig_path, existed = write_signature(p, digest)
        print(f"        sha256 {digest}")
        print(f"        {'覆盖' if existed else '新增'} {os.path.basename(sig_path)}"
              f"  (耗时 {dt:.1f}s)")

    print("-" * 74)
    print(f"[完成] {len(targets)} 个签名文件已写入，总用时 {time.time() - t_all:.1f}s")
    print("       复核：script\\verify_signature.bat")
    return 0


# ==================================================================
# 校验
# ==================================================================
def do_verify(root, files, quiet):
    targets = collect_targets(root, files)
    if targets is None:
        return 2
    if not targets:
        print("[错误] model\\ 下一个 .pt 都没找到，确认权重已经放好。")
        return 1

    print_banner("校验")
    print("[算法] SHA-256")
    print(f"[仓库] {root}")
    print(f"[扫描] 共 {len(targets)} 个 .pt 文件，逐个比对签名")
    print("-" * 74)

    t_all = time.time()
    passed = failed = 0

    for i, p in enumerate(targets, 1):
        rel = os.path.relpath(p, root)
        size = os.path.getsize(p)
        sig_path = p + SIG_SUFFIX
        print(f"  [{i}/{len(targets)}] {rel}  ({human_size(size)}, {size:,} 字节)")

        # ---- 签名文件本身在不在 ----
        if not os.path.isfile(sig_path):
            print(f"        FAIL  缺少签名文件 {os.path.basename(sig_path)}")
            print( "              先跑 script\\compute_signature.bat 生成")
            failed += 1
            continue

        rec = read_signature(sig_path)
        problems = []

        # ---- 先把签名文件的字段核对一遍（便宜，不用读权重）----
        algo = (rec.get("algorithm") or "").lower()
        if algo != ALGORITHM:
            problems.append(f"签名文件里的算法是「{algo or '空'}」，本脚本只认 {ALGORITHM}")

        want_name = rec.get("file") or ""
        if want_name != os.path.basename(p):
            problems.append(f"签名文件记录的文件名是「{want_name or '空'}」，"
                            f"与当前的 {os.path.basename(p)} 不符（可能是复制错了签名）")

        want_size = -1
        if "size" not in rec:
            problems.append("签名文件缺少 size 字段")
        else:
            try:
                want_size = int(rec["size"])
            except (TypeError, ValueError):
                problems.append(f"签名文件里的 size 不是整数：{rec['size']}")
            else:
                if want_size != size:
                    problems.append(f"大小不符：签名记录 {want_size:,} 字节，"
                                    f"实际 {size:,} 字节")

        want_digest = (rec.get("sha256") or "").lower()
        if len(want_digest) != 64 or any(c not in "0123456789abcdef" for c in want_digest):
            problems.append("签名文件里的 sha256 不是 64 位十六进制")

        # ---- 再算实际摘要 ----
        progress = (not quiet) and sys.stdout.isatty()
        digest, dt = sha256_file(p, show_progress=progress)
        if want_digest and digest != want_digest:
            problems.append("摘要不一致：文件内容和签名对不上")

        if problems:
            print(f"        FAIL  {problems[0]}")
            for extra in problems[1:]:
                print(f"              {extra}")
            if want_digest:
                print(f"              签名记录：{want_digest}")
            print(f"              实际计算：{digest}")
            failed += 1
        else:
            print(f"        PASS  大小 {size:,} 字节 ✓ | "
                  f"摘要 {digest[:16]}…{digest[-8:]} 与签名一致  ({dt:.1f}s)")
            passed += 1

    print("-" * 74)
    cost = time.time() - t_all
    if failed == 0:
        print(f"[结果] 全部通过（{passed} 个文件），用时 {cost:.1f}s")
        return 0
    print(f"[结果] 通过 {passed} 个，失败 {failed} 个，用时 {cost:.1f}s")
    print("       权重可能没下载完整、被改过，或者签名文件不是这一版。")
    print("       建议重新下载权重后再校验一次。")
    return 1


# ==================================================================
# 入口
# ==================================================================
def main():
    ap = argparse.ArgumentParser(
        description="计算 / 校验 dlqq-llm 模型权重的 SHA-256 数字签名",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例：\n"
               "  python script\\model_signature.py --mode compute\n"
               "  python script\\model_signature.py --mode verify\n"
               "  python script\\model_signature.py --mode verify --file model\\dlqq_llm\\dlqq_llm.pt\n")
    ap.add_argument("--mode", required=True, choices=["compute", "verify"],
                    help="compute = 生成签名；verify = 校验签名")
    ap.add_argument("--root", default=None,
                    help="仓库根目录；默认自动向上查找（同时含 model\\ 与 script\\ 的那一级）")
    ap.add_argument("--file", action="append", default=[], metavar="PT",
                    help="只处理指定的 .pt（可写多次；相对路径按仓库根解析）")
    ap.add_argument("--quiet", action="store_true", help="不显示进度条")
    ap.add_argument("--parent-pid", type=int, default=0,
                    help="父进程 pid（WebUI 传入时用来判断是否已被关闭）")
    a = ap.parse_args()

    root = os.path.abspath(a.root) if a.root else find_root(HERE)
    if not os.path.isdir(os.path.join(root, "model")):
        print(f"[错误] 这个目录下没有 model\\：{root}")
        return 2

    if a.mode == "compute":
        return do_compute(root, a.file, a.quiet)
    return do_verify(root, a.file, a.quiet)


if __name__ == "__main__":
    sys.exit(main() or 0)
