#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Toolkit (single-file)
功能：
  - parse: 解析目录中的 .txt 为同名 .json（每题 fields: language, number, context, question, options）
  - merge: 将 answers 目录中同名 .json 的 answer 合并回已生成的 .json（按题号匹配）
  - stats-blanks: 统计空白选项（导出 blank_options_report.csv）
  - stats-over6: 统计“选项数 > 6”（导出 over_six_options.csv）

用法示例：
  解析：
    python qa_toolkit.py parse --src "path/to/txt_dir" [--confusables-policy preserve|ascii_approx|visual_approx]
  合并答案：
    python qa_toolkit.py merge --gen "path/to/generated_jsons" --ans "path/to/answers_jsons" [--out same|merged]
  统计空白选项：
    python qa_toolkit.py stats-blanks --dir "path/to/jsons"
  统计选项超 6：
    python qa_toolkit.py stats-over6 --dir "path/to/jsons"
"""

from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional, Union

# =========================
# Unicode 清洗与规范化
# =========================

# 把各种奇怪空格 → 普通空格；移除零宽字符；把各种 dash/minus → ASCII '-'
_UNICODE_SPACE_TO_SPACE = {
    0x00A0: ' ', 0x1680: ' ',
    **{cp: ' ' for cp in range(0x2000, 0x200B)},  # U+2000–U+200A（包含 U+2004）
    0x202F: ' ', 0x205F: ' ', 0x3000: ' ',
}
_ZERO_WIDTH_REMOVE = {
    0x180E: '', 0x200B: '', 0x200C: '', 0x200D: '',
    0x2060: '', 0xFEFF: '',
}
_DASHES_TO_MINUS = {
    0x2212: '-', 0x2010: '-', 0x2011: '-', 0x2012: '-', 0x2013: '-', 0x2014: '-', 0x2015: '-',
}

# σ 的可配置策略：'preserve'（默认，保留 σ）、'ascii_approx'（σ→"sigma"）、'visual_approx'（σ→'o'）
def build_translate_table(confusables_policy: str) -> Dict[int, Union[int, str]]:
    table: Dict[int, Union[int, str]] = {
        **{k: ord(v) for k, v in _UNICODE_SPACE_TO_SPACE.items()},
        **_ZERO_WIDTH_REMOVE,
        **{k: ord(v) for k, v in _DASHES_TO_MINUS.items()},
    }
    if confusables_policy == 'ascii_approx':
        table.update({
            0x03C3: "sigma",  # σ
            0x03C2: "sigma",  # ς
            0x03A3: "SIGMA",  # Σ
        })
    elif confusables_policy == 'visual_approx':
        table.update({
            0x03C3: "o",
            0x03C2: "o",
            0x03A3: "O",
        })
    # preserve: 不做额外替换
    return table

def sanitize_text(text: str, translate_table: Dict[int, Union[int, str]]) -> str:
    """统一换行并应用 translate 表；不改变 '\n' 本身"""
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.translate(translate_table)

# =========================
# 解析 .txt → 同名 .json
# =========================

OPTION_RE = re.compile(r'^\s*([1-6])\s*(?:[.)])?\s*(.*)$')

def detect_lang_by_filename(name: str) -> Optional[str]:
    low = name.lower()
    if low.endswith("danish.txt") or low.endswith(".danish") or low.endswith("_danish.txt"):
        return "danish"
    if low.endswith("english.txt") or low.endswith(".english") or low.endswith("_english.txt"):
        return "english"
    return None

def detect_lang_by_content(text: str) -> str:
    if re.search(r'^\s*Opgave\s+\d+\s*$', text, re.M) or "Spørgsmål" in text:
        return "danish"
    if re.search(r'^\s*Exercise\s+\d+\s*$', text, re.M) or re.search(r'^\s*Question\s+\d+\s*$', text, re.M):
        return "english"
    return "danish"  # 兜底

def normalize_para_breaks(s: str) -> str:
    """仅用于 context/question：把连续两个及以上换行折叠为一个空格；单换行保留。"""
    if not s:
        return s
    s = re.sub(r'\n{2,}', ' ', s)
    return s.strip()

def parse_file_to_items(text: str, language: str) -> List[Dict[str, Any]]:
    lines = [ln.rstrip() for ln in text.splitlines()]
    if language == "danish":
        ex_re = re.compile(r'^\s*Opgave\s+(\d+)\s*$')
        q_re  = re.compile(r'^\s*Spørgsmål\s+(\d+)\s*$')
    else:
        ex_re = re.compile(r'^\s*Exercise\s+(\d+)\s*$', re.I)
        q_re  = re.compile(r'^\s*Question\s+(\d+)\s*$', re.I)

    items: List[Dict[str, Any]] = []
    cur: Optional[Dict[str, Any]] = None
    buffer: List[str] = []
    state = "idle"  # idle/after_ex/reading_context/after_q/reading_question/reading_options/reading_notes
    seq = 0         # 顺序号兜底

    def flush_to(field: str):
        nonlocal buffer, cur
        cur[field] = ("\n".join(buffer)).strip()
        buffer = []

    def finalize():
        nonlocal cur, state, seq
        if not cur:
            return
        if state == "reading_context":
            flush_to("context")
        elif state == "reading_question":
            flush_to("question")

        # 题号：优先 Question > Exercise > 顺序
        number = cur.get("q_no") or cur.get("ex_no") or (seq + 1)
        if cur.get("q_no") or cur.get("ex_no"):
            seq = int(number)
        else:
            seq += 1

        # options：保留换行，仅 strip 两端空白
        option_texts = [opt["text"].strip() for opt in sorted(cur["options"], key=lambda x: x["label"])]
        items.append({
            "language": language,
            "number": int(number),
            "context": normalize_para_breaks(cur.get("context", "")),
            "question": normalize_para_breaks(cur.get("question", "")),
            "options": option_texts,
        })
        cur, state = None, "idle"

    for line in lines:
        # 题头
        m_ex = ex_re.match(line)
        if m_ex:
            finalize()
            cur = {"context": "", "question": "", "options": [], "ex_no": int(m_ex.group(1)), "q_no": None}
            buffer, state = [], "after_ex"
            continue
        if not cur:
            continue

        # 小题头
        m_q = q_re.match(line)
        if m_q:
            if buffer and state in ("reading_context", "after_ex"):
                flush_to("context")
            cur["q_no"] = int(m_q.group(1))
            buffer, state = [], "after_q"
            continue

        # 仅在看到 question 之后才识别选项（避免把表格/编号当成选项）
        m_opt = OPTION_RE.match(line) if state in ("reading_options", "reading_question", "after_q") else None
        if m_opt:
            label = int(m_opt.group(1))
            text_after = m_opt.group(2)
            if state in ("reading_question", "after_q"):
                flush_to("question")
            state = "reading_options"
            cur["options"].append({"label": label, "text": text_after})
            continue

        # 空行
        if line.strip() == "":
            if state == "after_ex":
                state = "reading_context"; continue
            if state == "after_q":
                state = "reading_question"; continue
            if state == "reading_options":
                # 选项阶段：空行不结束，除非已收满 6 个；否则作为上一选项的续行换行
                if len(cur["options"]) >= 6:
                    state = "reading_notes"; buffer = []
                else:
                    if cur["options"] and cur["options"][-1]["text"]:
                        cur["options"][-1]["text"] += "\n"
                continue
            if state in ("reading_context", "reading_question", "reading_notes"):
                buffer.append("")
            continue

        # 非空文本分派
        if state in ("reading_context", "after_ex"):
            buffer.append(line); state = "reading_context"
        elif state in ("reading_question", "after_q"):
            buffer.append(line); state = "reading_question"
        elif state == "reading_options":
            # 非编号行，一律并入最近一个选项文本（保持换行）
            if cur["options"]:
                prev = cur["options"][-1]["text"]
                cur["options"][-1]["text"] = (prev + ("\n" if prev else "") + line)
            else:
                buffer.append(line); state = "reading_notes"
        elif state == "reading_notes":
            buffer.append(line)
        else:
            buffer.append(line); state = "reading_context"

    finalize()
    return items

def cmd_parse(args: argparse.Namespace) -> None:
    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"[ERROR] 源目录不存在：{src}")

    table = build_translate_table(args.confusables_policy)

    txt_paths = sorted(src.glob("*.txt"))
    if not txt_paths:
        print("[INFO] 未找到任何 .txt 文件。")
        return

    total_q = 0
    for path in txt_paths:
        # 读取并清洗 Unicode
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            raw = path.read_text(encoding="utf-8-sig")
        text = sanitize_text(raw, table)

        lang = detect_lang_by_filename(path.name) or detect_lang_by_content(text)
        items = parse_file_to_items(text, lang)

        out_path = path.with_suffix(".json")
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[OK] {out_path.name}: {len(items)} questions")
        total_q += len(items)

    print(f"\n[SUMMARY] Files: {len(txt_paths)} | Total questions: {total_q}")

# =========================
# 合并答案
# =========================

def _to_int_or_str(x: Any) -> Optional[Union[int, str]]:
    if x is None:
        return None
    if isinstance(x, int):
        return x
    sx = str(x).strip()
    try:
        return int(sx)
    except Exception:
        return sx

def _get_qid_from_item(item: Dict[str, Any], default_seq: int) -> Union[int, str]:
    for k in ("number", "question_id", "id"):
        if k in item:
            v = _to_int_or_str(item[k])
            if v is not None:
                return v
    return default_seq

def _build_answer_map(ans_data: Any, our_items: List[Dict[str, Any]]) -> Dict[Union[int, str], Any]:
    """
    支持结构：
      - dict: {question_id: answer}
      - list[dict]: 每个含 question_id/number/id 和 answer
      - list[value]: 与 our_items 等长，按顺序对齐
    """
    # dict 直接用
    if isinstance(ans_data, dict):
        amap: Dict[Union[int, str], Any] = {}
        for k, v in ans_data.items():
            amap[_to_int_or_str(k)] = v
        return amap

    # list
    if isinstance(ans_data, list):
        has_dict = any(isinstance(x, dict) for x in ans_data)
        has_qid_key = any(
            isinstance(x, dict) and any(k in x for k in ("question_id", "number", "id"))
            for x in ans_data
        )
        if has_dict and has_qid_key:
            amap: Dict[Union[int, str], Any] = {}
            for obj in ans_data:
                if not isinstance(obj, dict):
                    continue
                qid = None
                for key in ("question_id", "number", "id"):
                    if key in obj:
                        qid = _to_int_or_str(obj[key]); break
                if qid is not None and "answer" in obj:
                    amap[qid] = obj["answer"]
            return amap

        # 纯数组且长度对齐：按顺序对齐
        if len(ans_data) == len(our_items):
            amap = {}
            for idx, item in enumerate(our_items):
                qid = _get_qid_from_item(item, idx + 1)
                amap[qid] = ans_data[idx]
            return amap

    return {}

def merge_one_file(gen_path: Path, ans_path: Path, out_path: Path) -> Tuple[int, int]:
    items = json.loads(gen_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        print(f"[WARN] {gen_path.name}: 我们的 JSON 不是数组，跳过。")
        return 0, 0

    if not ans_path.exists():
        # 没有答案文件也要写出原始 items（覆盖/输出）
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(items), 0

    ans_raw = json.loads(ans_path.read_text(encoding="utf-8"))
    if not isinstance(ans_raw, (dict, list)):
        print(f"[WARN] {ans_path.name}: 答案文件结构非常规（{type(ans_raw).__name__}），跳过合并。")
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(items), 0

    amap = _build_answer_map(ans_raw, items)
    added = 0
    for idx, obj in enumerate(items):
        qid = _get_qid_from_item(obj, idx + 1)
        if qid in amap:
            obj["answer"] = amap[qid]
            added += 1

    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(items), added

def cmd_merge(args: argparse.Namespace) -> None:
    gen_dir = Path(args.gen)
    ans_dir = Path(args.ans)
    if not gen_dir.exists():
        raise SystemExit(f"[ERROR] 生成目录不存在：{gen_dir}")
    if not ans_dir.exists():
        raise SystemExit(f"[ERROR] 答案目录不存在：{ans_dir}")

    out_dir = gen_dir if args.out == "same" else (gen_dir / "merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_q = 0
    total_added = 0
    for g_path in sorted(gen_dir.glob("*.json")):
        a_path = ans_dir / g_path.name
        out_path = out_dir / g_path.name
        q, a = merge_one_file(g_path, a_path, out_path)
        print(f"[OK] {g_path.name}: 题目 {q}，合并 answer {a}")
        total_q += q; total_added += a

    print(f"\n[SUMMARY] 总题目数：{total_q} | 成功加上 answer：{total_added}")
    if out_dir != gen_dir:
        print(f"[INFO] 输出目录：{out_dir}")

# =========================
# 统计脚本
# =========================

# 清理不可见空白用于“空白选项”判断
_WS_TRANSLATE_TABLE = {
    **{k: ord(v) for k, v in _UNICODE_SPACE_TO_SPACE.items()},
    **_ZERO_WIDTH_REMOVE,
}
def _normalize_ws(s: str) -> str:
    if s is None:
        return ""
    return s.translate(_WS_TRANSLATE_TABLE)

def _is_blank_option(text: str) -> bool:
    return _normalize_ws(text).strip() == ""

def cmd_stats_blanks(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] 未找到任何 .json 文件。")
        return

    rows: List[List[Union[str, int]]] = []
    total_q = 0
    total_blank = 0

    print("=== 空白选项统计 ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {p.name}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {p.name}: 根对象不是数组，跳过。")
            continue

        blanks: List[Tuple[int, int]] = []
        for idx, item in enumerate(data):
            opts = item.get("options", []) or []
            qnum = item.get("number") or item.get("question_id") or item.get("id") or (idx + 1)
            try:
                qnum = int(qnum)
            except Exception:
                pass
            for i, opt in enumerate(opts, start=1):
                if _is_blank_option(opt):
                    blanks.append((qnum, i))
                    rows.append([p.name, qnum, i])

        print(f"{p.name}: 题目 {len(data)}，空白选项 {len(blanks)}")
        if blanks:
            print("  明细: " + ", ".join([f"Q{q}-opt{i}" for q, i in blanks]))
        total_q += len(data)
        total_blank += len(blanks)

    csv_path = folder / "blank_options_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question_number", "option_index"])
        w.writerows(rows)

    print("\n=== 总计 ===")
    print(f"总题目数: {total_q}")
    print(f"空白选项总数: {total_blank}")
    print(f"已导出: {csv_path}")

def cmd_stats_over6(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] 未找到任何 .json 文件。")
        return

    rows: List[List[Union[str, int]]] = []
    total_q = 0
    total_over6 = 0

    print("=== 选项数 > 6 统计 ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {p.name}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {p.name}: 根对象不是数组，跳过。")
            continue

        overs: List[Tuple[int, int]] = []
        for idx, item in enumerate(data):
            opts = item.get("options", [])
            n = len(opts) if isinstance(opts, list) else 0
            if n > 6:
                qnum = item.get("number") or item.get("question_id") or item.get("id") or (idx + 1)
                try:
                    qnum = int(qnum)
                except Exception:
                    pass
                overs.append((qnum, n))
                rows.append([p.name, qnum, n])

        print(f"{p.name}: 题目 {len(data)}，>6 共有 {len(overs)}")
        if overs:
            print("  明细: " + ", ".join([f"Q{q}({n})" for q, n in overs]))
        total_q += len(data)
        total_over6 += len(overs)

    csv_path = folder / "over_six_options.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question_number", "option_count"])
        w.writerows(rows)

    print("\n=== 总计 ===")
    print(f"总题目数: {total_q}")
    print(f">6 选项题目总数: {total_over6}")
    print(f"已导出: {csv_path}")
def cmd_stats_under6(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] 未找到任何 .json 文件。")
        return

    rows: List[List[Union[str, int]]] = []
    total_q = 0
    total_under6 = 0

    print("=== 选项数 < 6 统计 ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {p.name}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {p.name}: 根对象不是数组，跳过。")
            continue

        unders: List[Tuple[int, int]] = []
        for idx, item in enumerate(data):
            opts = item.get("options", [])
            if isinstance(opts, list):
                if getattr(args, "effective", False):
                    # 按“有效选项”计数：忽略空白选项
                    n = sum(1 for o in opts if not _is_blank_option(o))
                else:
                    # 直接用数组长度
                    n = len(opts)
            else:
                n = 0

            if n < 6:
                qnum = item.get("number") or item.get("question_id") or item.get("id") or (idx + 1)
                try:
                    qnum = int(qnum)
                except Exception:
                    pass
                unders.append((qnum, n))
                rows.append([p.name, qnum, n])

        print(f"{p.name}: 题目 {len(data)}，<6 共有 {len(unders)}")
        if unders:
            print("  明细: " + ", ".join([f"Q{q}({n})" for q, n in unders]))
        total_q += len(data)
        total_under6 += len(unders)

    csv_path = folder / "under_six_options.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question_number", "option_count"])
        w.writerows(rows)

    print("\n=== 总计 ===")
    print(f"总题目数: {total_q}")
    print(f"<6 选项题目总数: {total_under6}")
    print(f"已导出: {csv_path}")

def cmd_concat(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] 未找到任何 .json 文件。"); return

    combined = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {p.name}: {e}")
            continue
        if not isinstance(data, list):
            print(f"[WARN] {p.name}: 根对象不是数组，跳过。")
            continue

        base = p.stem  # 源文件名（不含扩展名）
        for item in data:
            # 原题号（可能是 int 或 str）
            raw_no = item.get("number") or item.get("question_id") or item.get("id")
            # 统一成标签：文件名-题号
            item["number"] = f"{base}-{raw_no}"
            if args.keep_source and isinstance(item, dict):
                item["source_file"] = p.name
            combined.append(item)

    out_path = Path(args.out) if args.out else (folder / "all_questions.json")
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] 合并 {len(files)} 个文件，共 {len(combined)} 题 → {out_path}")

# =========================
# CLI
# =========================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qa_toolkit.py", description="Parse TXT -> JSON, merge answers, and run stats.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("parse", help="解析目录中的 .txt 为同名 .json")
    sp.add_argument("--src", required=True, help="包含 .txt 的目录")
    sp.add_argument("--confusables-policy", choices=["preserve", "ascii_approx", "visual_approx"],
                    default="preserve", help="σ 的替换策略（默认 preserve）")
    sp.set_defaults(func=cmd_parse)

    sm = sub.add_parser("merge", help="把 answers 目录里同名 .json 的 answer 合并回生成的 .json")
    sm.add_argument("--gen", required=True, help="已生成题目 JSON 的目录")
    sm.add_argument("--ans", required=True, help="答案 JSON 的目录（同名文件）")
    sm.add_argument("--out", choices=["same", "merged"], default="same",
                    help="输出位置：same=覆盖原目录；merged=写到 gen/merged")
    sm.set_defaults(func=cmd_merge)

    sb = sub.add_parser("stats-blanks", help="统计空白选项并导出 CSV")
    sb.add_argument("--dir", required=True, help="题目 JSON 目录")
    sb.set_defaults(func=cmd_stats_blanks)

    so = sub.add_parser("stats-over6", help="统计选项数 > 6 并导出 CSV")
    so.add_argument("--dir", required=True, help="题目 JSON 目录")
    so.set_defaults(func=cmd_stats_over6)

    su2 = sub.add_parser("stats-under6", help="统计选项数 < 6 并导出 CSV")
    su2.add_argument("--dir", required=True, help="题目 JSON 目录")
    su2.add_argument("--effective", action="store_true",
                    help="按有效选项计数（忽略空白项）；否则按原始 options 长度统计")
    su2.set_defaults(func=cmd_stats_under6)

    sc = sub.add_parser("concat", help="合并目录中所有 .json 为一个数组文件，并把 number 改为 文件名-题号")
    sc.add_argument("--dir", required=True, help="题目 JSON 目录")
    sc.add_argument("--out", help="输出文件路径（默认 <dir>/all_questions.json）")
    sc.add_argument("--keep-source", action="store_true", help="为每题添加 source_file 字段")
    sc.set_defaults(func=cmd_concat)


    return p

def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
