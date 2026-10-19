# -*- coding: utf-8 -*-
"""从 ModelScope（魔搭）下载中文考题数据集，转成训练 / 评测两份 jsonl
=====================================================================
只走国内源，不需要 GitHub、不需要 HuggingFace、不需要登录。

可用数据集（都是魔搭上的公开数据集）：
    c-eval        C-Eval 52 学科选择题（约 1.5 MB）
    gaokao        高考选择题（OpenCompass 镜像，约 5.7 MB）
    gaokao-damo   高考选择题（另一个镜像，约 2 MB）
    bio           高考生物（含解析，约 0.2 MB）
    medical       医学考试题库（约 35 MB）
    tcm           中医考试题（约 10 MB）
    cflue         金融资格考试选择题（约 66 MB）

用法（也可以直接双击 script\\fetch_exam_data.bat）：
    python script\\fetch_exam_data.py --list          （看有哪些）
    python script\\fetch_exam_data.py --out ai_res\\doc\\exam
    python script\\fetch_exam_data.py --out ai_res\\doc\\exam --datasets c-eval gaokao
    python script\\fetch_exam_data.py --from-dir ai_res\\doc\\exam --out ai_res\\doc\\exam

产出（--out 目录下）：
    exam_train.jsonl   训练用（用户：…／助手：…）
    exam_eval.jsonl    评测用（只留着打分，不要拿去训练）
    raw\\               原始下载文件

做成 demo 语料时，bat 会加 --all-to-train --single src\\chat_demo.jsonl，
把题目全部写进 src\\chat_demo.jsonl（chat_demo 配置直接吃这个文件）。

⚠ 评测集拿去训练，你报出来的分数就是自己背的答案，没有任何意义。
"""
import argparse
import fnmatch
import glob
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
import zipfile

for _s in ("stdout", "stderr"):
    try:
        getattr(sys, _s).reconfigure(encoding="utf-8", errors="replace",
                                     line_buffering=True, write_through=True)
    except Exception:
        pass

API = ("https://modelscope.cn/api/v1/datasets/{repo}/repo/tree"
       "?Revision=master&Recursive=true&PageSize=200&PageNumber={page}")
DL = "https://modelscope.cn/api/v1/datasets/{repo}/repo?Revision=master&FilePath={path}"
UA = {"User-Agent": "Mozilla/5.0"}

# 名字 -> (魔搭仓库名, 要下的文件通配符)
MODELSCOPE_DATASETS = {
    "c-eval":      ("OmniData/C-Eval", ["*.zip", "*.json", "*.jsonl"]),
    "gaokao":      ("opencompass/gaokao-benchmark", ["*.json", "*.jsonl", "*.zip"]),
    "gaokao-damo": ("damotest/gaokao-benchmark", ["*.json", "*.jsonl", "*.zip"]),
    "bio":         ("pandayi/Biology_subject_of_college_entrance_examination", ["*.json", "*.jsonl", "*.xlsx", "*.csv"]),
    "medical":     ("SHPDataGR/medical-exam-question-bank-json", ["*.json", "*.jsonl"]),
    "tcm":         ("huangxp/hwtcm", ["*.json", "*.jsonl", "*.xlsx", "*.csv", "*.txt"]),
    "cflue":       ("tongyi_dianjin/CFLUE", ["*.json", "*.jsonl", "*.zip"]),
}

# 默认只下小的三个，够 demo 用；要更多自己加 --datasets
DEFAULT_SET = ("c-eval", "gaokao", "bio")

Q_KEYS = ("question", "problem", "stem", "query", "prompt", "instruction", "问", "题目", "题干")
A_KEYS = ("answer", "target", "label", "output", "response", "答案", "correct", "标准答案")
EXP_KEYS = ("explanation", "analysis", "reasoning", "解析", "rationale", "详解")
OPT_KEYS = ("options", "choices", "选项")


def http_get(url, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=timeout).read()


def http_json(url, timeout=60):
    return json.loads(http_get(url, timeout).decode("utf-8", "replace"))


def list_files(repo):
    files, page = [], 1
    while True:
        data = http_json(API.format(repo=repo, page=page))
        block = data.get("Data") or {}
        batch = block.get("Files") or []
        files.extend(batch)
        if len(files) >= (block.get("TotalCount") or 0) or not batch:
            break
        page += 1
    return files


def download(datasets, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    got = []
    for name in datasets:
        if name not in MODELSCOPE_DATASETS:
            print(f"  [跳过] 不认识的数据集：{name}（可选：{', '.join(MODELSCOPE_DATASETS)}）")
            continue
        repo, patterns = MODELSCOPE_DATASETS[name]
        print(f"\n[下载] {name}  ({repo})")
        try:
            files = list_files(repo)
        except Exception as e:
            print(f"  [失败] 列文件出错：{e}")
            continue
        picked = [f for f in files if any(fnmatch.fnmatch(f.get("Path", ""), p) for p in patterns)]
        if not picked:
            print(f"  [跳过] 仓库里没有匹配 {patterns} 的文件")
            continue
        target = os.path.join(raw_dir, name)
        os.makedirs(target, exist_ok=True)
        ok = 0
        for f in picked:
            rel = f["Path"].replace("/", os.sep)
            dst = os.path.join(target, rel)
            if os.path.isfile(dst) and os.path.getsize(dst) == f.get("Size", -1):
                ok += 1
                continue
            os.makedirs(os.path.dirname(dst) or target, exist_ok=True)
            url = DL.format(repo=repo, path=urllib.parse.quote(f["Path"]))
            try:
                data = http_get(url, timeout=180)
                with open(dst, "wb") as out:
                    out.write(data)
                print(f"  {len(data)/1e6:6.1f} MB  {f['Path']}")
                ok += 1
                if dst.lower().endswith(".zip"):
                    with zipfile.ZipFile(dst) as z:
                        z.extractall(target)
                    print("           已解压")
            except Exception as e:
                print(f"  [失败] {f['Path']}: {e}")
        if ok:
            got.append(name)
    return got


def pick(obj, keys):
    for k in keys:
        if k in obj and obj[k] not in (None, "", []):
            return obj[k]
    return None


def letter_of_answer(ans, opt_map):
    if ans is None:
        return None
    a = str(ans).strip()
    if len(a) == 1 and a.upper() in "ABCDEFGH":
        return a.upper()
    for letter, text in opt_map.items():
        if text and a and (a == text.strip() or a in text):
            return letter
    for ch in a[:6]:
        if ch.upper() in "ABCDEFGH":
            return ch.upper()
    return None


def to_record(obj):
    if not isinstance(obj, dict):
        return None
    q = pick(obj, Q_KEYS)
    if not isinstance(q, str) or len(q.strip()) < 4:
        return None
    # 有些数据集把补充材料放在 input 里（instruction + input 拼起来才是完整题面）
    extra = obj.get("input")
    if isinstance(extra, str) and extra.strip() and extra.strip() != q.strip():
        q = q.strip() + "\n" + extra.strip()

    opt_map = {}
    options = pick(obj, OPT_KEYS)
    if isinstance(options, list):
        for i, t in enumerate(options[:8]):
            if isinstance(t, str) and t.strip():
                opt_map["ABCDEFGH"[i]] = t.strip()
    for letter in "ABCDEFGH":
        v = obj.get(letter) or obj.get(letter.lower())
        if isinstance(v, str) and v.strip():
            opt_map[letter] = v.strip()

    ans_raw = pick(obj, A_KEYS)
    # 有些数据集（gaokao 选择题）答案是列表：["D"] 或 ["A","B"]
    if isinstance(ans_raw, (list, tuple)):
        parts = [str(x).strip() for x in ans_raw if str(x).strip()]
        if not parts:
            return None
        if all(len(p) == 1 and p.upper() in "ABCDEFGH" for p in parts):
            ans_raw = "".join(parts)          # 单选/多选：连成 "D" / "AB"
        else:
            ans_raw = "；".join(parts)         # 主观题：多行答案拼起来
    letter = letter_of_answer(ans_raw, opt_map) if opt_map else None
    if letter:
        ans_text = f"{letter}. {opt_map[letter]}"
    elif isinstance(ans_raw, str) and ans_raw.strip():
        ans_text = ans_raw.strip()
    else:
        return None

    exp = pick(obj, EXP_KEYS)
    if isinstance(exp, str) and exp.strip() and exp.strip() != ans_text:
        ans_text += "\n解析：" + exp.strip()[:400]

    if opt_map:
        body = "\n".join(f"{k}. {v}" for k, v in sorted(opt_map.items()))
        user = f"{q.strip()}\n{body}"
    else:
        user = q.strip()
    return user, ans_text


def iter_files(root):
    """要转换的原始文件：json / jsonl / csv。
    zip 会在 convert 开头先解压，所以这里不用重复读。"""
    for path in glob.glob(os.path.join(root, "**", "*"), recursive=True):
        if os.path.isfile(path) and path.lower().endswith((".json", ".jsonl", ".csv")):
            yield path


def _parse_stream(f, kind):
    """从文本流里逐条产出记录；kind = csv / jsonl / json"""
    if kind == "csv":
        import csv
        for row in csv.DictReader(f):
            yield row
        return
    if kind == "jsonl":
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
        return
    # json：可能是数组，也可能是一行一个对象
    head = f.read(1)
    f.seek(0)
    if head == "[":
        data = json.load(f)
        if isinstance(data, list):
            for item in data:
                yield item
        return
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


def _kind_of(name):
    low = name.lower()
    if low.endswith(".csv"):
        return "csv"
    if low.endswith(".jsonl"):
        return "jsonl"
    return "json"


def extract_zips(root):
    """把没解压过的 zip 解到它旁边（已解压的跳过）"""
    for path in glob.glob(os.path.join(root, "**", "*.zip"), recursive=True):
        try:
            with zipfile.ZipFile(path) as z:
                names = z.namelist()
                if not names:
                    continue
                marker = os.path.join(os.path.dirname(path), names[0].split("/")[0])
                if os.path.exists(marker):
                    continue
                z.extractall(os.path.dirname(path))
                print(f"  [解压] {os.path.relpath(path, root)} → {len(names)} 个条目")
        except (OSError, zipfile.BadZipFile) as e:
            print(f"  [跳过] 解压失败 {os.path.basename(path)}: {e}")


def iter_records(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            yield from _parse_stream(f, _kind_of(path))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"  [跳过] {os.path.basename(path)}: {e}")


def is_eval_path(path):
    low = path.lower().replace("\\", "/")
    marks = ("/test/", "/val/", "/valid/", "/dev/", "test.", "val.", "_test", "_val")
    return any(m in low for m in marks)


def convert(raw_dir, out_dir, max_per_set=0, all_to_train=False, single=None):
    os.makedirs(out_dir, exist_ok=True)
    if single:
        single = os.path.abspath(single)
        os.makedirs(os.path.dirname(single) or ".", exist_ok=True)
        train_path = single
        eval_path = None
        all_to_train = True
    else:
        train_path = os.path.join(out_dir, "exam_train.jsonl")
        eval_path = os.path.join(out_dir, "exam_eval.jsonl")
    seen = set()
    n_train = n_eval = 0
    skipped = no_answer = 0

    extract_zips(raw_dir)

    with open(train_path, "w", encoding="utf-8") as ft, \
         (open(eval_path, "w", encoding="utf-8") if eval_path else open(os.devnull, "w")) as fe:
        for path in sorted(iter_files(raw_dir)):
            if "/.git/" in path.replace("\\", "/"):
                continue
            n_here = 0
            for obj in iter_records(path):
                rec = to_record(obj)
                if not rec:
                    # 分开统计：C-Eval 的 test 分片故意不给答案，不算「格式错」
                    ans = pick(obj, A_KEYS) if isinstance(obj, dict) else None
                    if ans in (None, "") or (isinstance(ans, (list, tuple)) and not ans):
                        no_answer += 1
                    else:
                        skipped += 1
                    continue
                user, ans = rec
                h = hashlib.md5(user[:200].encode("utf-8", "ignore")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                line = json.dumps({"input": user, "target": ans}, ensure_ascii=False) + "\n"
                if eval_path and is_eval_path(path) and not all_to_train:
                    fe.write(line)
                    n_eval += 1
                else:
                    ft.write(line)
                    n_train += 1
                n_here += 1
                if max_per_set and n_here >= max_per_set:
                    break
            if n_here:
                print(f"  {n_here:>6} 条  {os.path.relpath(path, raw_dir)}")

    size_train = os.path.getsize(train_path) / 1e6
    print(f"\n[完成] 训练集 {n_train} 条 / {size_train:.1f} MB → {train_path}")
    if eval_path:
        print(f"       评测集 {n_eval} 条 → {eval_path}")
    print(f"       跳过（格式不认识）{skipped} 条；没有答案 {no_answer} 条")
    if all_to_train:
        print("\n[注意] 全部题目都进了训练集，这份数据是「演示语料」。")
        print("       别拿同一批题出去报 benchmark 分数。")
    elif n_eval == 0:
        print("\n[提醒] 没有识别到评测集（test/val 目录）。")
        print("       建议手动留出一部分不训练，否则你报的分数没有意义。")
    return 0


def main():
    ap = argparse.ArgumentParser(description="从魔搭下载考题数据集并转成 jsonl")
    ap.add_argument("--out", required=True, help="输出目录，例如 ai_res\\doc\\exam")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help=f"要下哪些（默认 {' '.join(DEFAULT_SET)}）")
    ap.add_argument("--from-dir", default=None, help="跳过下载，直接转换这个目录里已有的文件")
    ap.add_argument("--max-per-file", type=int, default=0, help="每个文件最多取多少条（调试用）")
    ap.add_argument("--all-to-train", action="store_true", help="全部转成训练集（demo 语料用）")
    ap.add_argument("--single", default=None, help="只写一个文件，例如 src\\chat_demo.jsonl")
    ap.add_argument("--list", action="store_true", help="只列出可选数据集，不下载")
    args = ap.parse_args()

    if args.list:
        print("可用数据集（魔搭仓库）：")
        for k, (repo, _) in MODELSCOPE_DATASETS.items():
            mark = "  ← 默认" if k in DEFAULT_SET else ""
            print(f"  {k:<12} {repo}{mark}")
        return 0

    out_dir = os.path.abspath(args.out)
    raw_dir = args.from_dir or os.path.join(out_dir, "raw")

    if args.from_dir:
        print(f"[跳过下载] 直接转换 {args.from_dir}")
    else:
        names = args.datasets or list(DEFAULT_SET)
        got = download(names, raw_dir)
        if not got:
            print("\n[错误] 一个数据集都没下下来。")
            print("       看看有哪些：python script\\fetch_exam_data.py --list --out .")
            return 1

    return convert(raw_dir, out_dir, args.max_per_file, args.all_to_train, args.single)


if __name__ == "__main__":
    sys.exit(main())
