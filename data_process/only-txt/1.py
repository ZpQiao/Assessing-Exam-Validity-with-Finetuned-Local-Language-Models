#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA Toolkit (single-file)
Features:
  - parse: convert all .txt files in a directory to .json with the same name
           (per-question fields: language, number, context, question, options)
  - merge: merge answers from an answers directory back into the generated .json
           (matched by question id/number)
  - stats-blanks: count blank options (exports blank_options_report.csv)
  - stats-over6: count questions with more than 6 options (exports over_six_options.csv)

Usage examples:
  Parse:
    python qa_toolkit.py parse --src "path/to/txt_dir" [--confusables-policy preserve|ascii_approx|visual_approx]
  Merge answers:
    python qa_toolkit.py merge --gen "path/to/generated_jsons" --ans "path/to/answers_jsons" [--out same|merged]
  Stats for blank options:
    python qa_toolkit.py stats-blanks --dir "path/to/jsons"
  Stats for questions with > 6 options:
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
# Unicode cleaning & normalization
# =========================

# Map various weird spaces → normal space; remove zero-width chars; map various dash/minus → ASCII '-'
_UNICODE_SPACE_TO_SPACE = {
    0x00A0: ' ', 0x1680: ' ',
    **{cp: ' ' for cp in range(0x2000, 0x200B)},  # U+2000–U+200A (including U+2004)
    0x202F: ' ', 0x205F: ' ', 0x3000: ' ',
}
_ZERO_WIDTH_REMOVE = {
    0x180E: '', 0x200B: '', 0x200C: '', 0x200D: '',
    0x2060: '', 0xFEFF: '',
}
_DASHES_TO_MINUS = {
    0x2212: '-', 0x2010: '-', 0x2011: '-', 0x2012: '-', 0x2013: '-', 0x2014: '-', 0x2015: '-',
}

# Configurable strategy for sigma: 'preserve' (default, keep σ), 'ascii_approx' (σ→"sigma"), 'visual_approx' (σ→'o')
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
    # preserve: no extra replacement
    return table

def sanitize_text(text: str, translate_table: Dict[int, Union[int, str]]) -> str:
    """Normalize line breaks and apply the translate table; do not change '\n' itself."""
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.translate(translate_table)

# =========================
# Parse .txt → .json with the same name
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
    return "danish"  # fallback

def normalize_para_breaks(s: str) -> str:
    """Used only for context/question: collapse 2+ consecutive newlines into a single space; keep single newlines."""
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
    seq = 0         # fallback sequential number

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

        # number: prefer Question > Exercise > sequential index
        number = cur.get("q_no") or cur.get("ex_no") or (seq + 1)
        if cur.get("q_no") or cur.get("ex_no"):
            seq = int(number)
        else:
            seq += 1

        # options: keep internal newlines, only strip surrounding whitespace
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
        # Exercise header
        m_ex = ex_re.match(line)
        if m_ex:
            finalize()
            cur = {"context": "", "question": "", "options": [], "ex_no": int(m_ex.group(1)), "q_no": None}
            buffer, state = [], "after_ex"
            continue
        if not cur:
            continue

        # Question header
        m_q = q_re.match(line)
        if m_q:
            if buffer and state in ("reading_context", "after_ex"):
                flush_to("context")
            cur["q_no"] = int(m_q.group(1))
            buffer, state = [], "after_q"
            continue

        # Only recognize options after we have seen the question header
        # (to avoid treating tables/numbering as options)
        m_opt = OPTION_RE.match(line) if state in ("reading_options", "reading_question", "after_q") else None
        if m_opt:
            label = int(m_opt.group(1))
            text_after = m_opt.group(2)
            if state in ("reading_question", "after_q"):
                flush_to("question")
            state = "reading_options"
            cur["options"].append({"label": label, "text": text_after})
            continue

        # Blank line
        if line.strip() == "":
            if state == "after_ex":
                state = "reading_context"; continue
            if state == "after_q":
                state = "reading_question"; continue
            if state == "reading_options":
                # While reading options: blank line does not end options unless we already have 6;
                # otherwise treat it as a newline continuation for the previous option
                if len(cur["options"]) >= 6:
                    state = "reading_notes"; buffer = []
                else:
                    if cur["options"] and cur["options"][-1]["text"]:
                        cur["options"][-1]["text"] += "\n"
                continue
            if state in ("reading_context", "reading_question", "reading_notes"):
                buffer.append("")
            continue

        # Non-blank text route
        if state in ("reading_context", "after_ex"):
            buffer.append(line); state = "reading_context"
        elif state in ("reading_question", "after_q"):
            buffer.append(line); state = "reading_question"
        elif state == "reading_options":
            # Non-numbered line while in options: append to the latest option text (preserving newlines)
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
        raise SystemExit(f"[ERROR] Source directory does not exist: {src}")

    table = build_translate_table(args.confusables_policy)

    txt_paths = sorted(src.glob("*.txt"))
    if not txt_paths:
        print("[INFO] No .txt files found.")
        return

    total_q = 0
    for path in txt_paths:
        # Read and clean Unicode
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
# Merge answers
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
    Supported structures:
      - dict: {question_id: answer}
      - list[dict]: each item has question_id/number/id and answer
      - list[value]: same length as our_items, aligned by order
    """
    # dict: straight mapping
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

        # plain list and length matches: align by index
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
        print(f"[WARN] {gen_path.name}: Our JSON is not an array, skipping.")
        return 0, 0

    if not ans_path.exists():
        # Even without an answer file we still write out the original items (overwrite/output)
        out_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
        return len(items), 0

    ans_raw = json.loads(ans_path.read_text(encoding="utf-8"))
    if not isinstance(ans_raw, (dict, list)):
        print(f"[WARN] {ans_path.name}: Answer file has an unusual structure ({type(ans_raw).__name__}), skipping merge.")
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
        raise SystemExit(f"[ERROR] Generated JSON directory does not exist: {gen_dir}")
    if not ans_dir.exists():
        raise SystemExit(f"[ERROR] Answer JSON directory does not exist: {ans_dir}")

    out_dir = gen_dir if args.out == "same" else (gen_dir / "merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_q = 0
    total_added = 0
    for g_path in sorted(gen_dir.glob("*.json")):
        a_path = ans_dir / g_path.name
        out_path = out_dir / g_path.name
        q, a = merge_one_file(g_path, a_path, out_path)
        print(f"[OK] {g_path.name}: {q} questions, merged {a} answers")
        total_q += q; total_added += a

    print(f"\n[SUMMARY] Total questions: {total_q} | Answers merged: {total_added}")
    if out_dir != gen_dir:
        print(f"[INFO] Output directory: {out_dir}")

# =========================
# Stats helpers & commands
# =========================

# Clean invisible whitespace for "blank option" detection
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
        print("[INFO] No .json files found.")
        return

    rows: List[List[Union[str, int]]] = []
    total_q = 0
    total_blank = 0

    print("=== Blank options statistics ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {p.name}: Root object is not an array, skipping.")
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

        print(f"{p.name}: {len(data)} questions, {len(blanks)} blank options")
        if blanks:
            print("  Details: " + ", ".join([f"Q{q}-opt{i}" for q, i in blanks]))
        total_q += len(data)
        total_blank += len(blanks)

    csv_path = folder / "blank_options_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question_number", "option_index"])
        w.writerows(rows)

    print("\n=== Summary ===")
    print(f"Total questions: {total_q}")
    print(f"Total blank options: {total_blank}")
    print(f"Exported to: {csv_path}")

def cmd_stats_over6(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] No .json files found.")
        return

    rows: List[List[Union[str, int]]] = []
    total_q = 0
    total_over6 = 0

    print("=== Options count > 6 statistics ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {p.name}: Root object is not an array, skipping.")
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

        print(f"{p.name}: {len(data)} questions, {len(overs)} with > 6 options")
        if overs:
            print("  Details: " + ", ".join([f"Q{q}({n})" for q, n in overs]))
        total_q += len(data)
        total_over6 += len(overs)

    csv_path = folder / "over_six_options.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question_number", "option_count"])
        w.writerows(rows)

    print("\n=== Summary ===")
    print(f"Total questions: {total_q}")
    print(f"Total questions with > 6 options: {total_over6}")
    print(f"Exported to: {csv_path}")

def cmd_stats_under6(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] No .json files found.")
        return

    rows: List[List[Union[str, int]]] = []
    total_q = 0
    total_under6 = 0

    print("=== Options count < 6 statistics ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue

        if not isinstance(data, list):
            print(f"[WARN] {p.name}: Root object is not an array, skipping.")
            continue

        unders: List[Tuple[int, int]] = []
        for idx, item in enumerate(data):
            opts = item.get("options", [])
            if isinstance(opts, list):
                if getattr(args, "effective", False):
                    # Count "effective options": ignore blank options
                    n = sum(1 for o in opts if not _is_blank_option(o))
                else:
                    # Use raw length of options array
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

        print(f"{p.name}: {len(data)} questions, {len(unders)} with < 6 options")
        if unders:
            print("  Details: " + ", ".join([f"Q{q}({n})" for q, n in unders]))
        total_q += len(data)
        total_under6 += len(unders)

    csv_path = folder / "under_six_options.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "question_number", "option_count"])
        w.writerows(rows)

    print("\n=== Summary ===")
    print(f"Total questions: {total_q}")
    print(f"Total questions with < 6 options: {total_under6}")
    print(f"Exported to: {csv_path}")

def cmd_concat(args: argparse.Namespace) -> None:
    folder = Path(args.dir)
    files = sorted(folder.glob("*.json"))
    if not files:
        print("[INFO] No .json files found.")
        return

    combined = []
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Failed to read {p.name}: {e}")
            continue
        if not isinstance(data, list):
            print(f"[WARN] {p.name}: Root object is not an array, skipping.")
            continue

        base = p.stem  # source file name (without extension)
        for item in data:
            # original question number (may be int or str)
            raw_no = item.get("number") or item.get("question_id") or item.get("id")
            # normalize to label: <filename>-<number>
            item["number"] = f"{base}-{raw_no}"
            if args.keep_source and isinstance(item, dict):
                item["source_file"] = p.name
            combined.append(item)

    out_path = Path(args.out) if args.out else (folder / "all_questions.json")
    out_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] Concatenated {len(files)} files, {len(combined)} questions → {out_path}")

# =========================
# CLI
# =========================

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="qa_toolkit.py",
        description="Parse TXT -> JSON, merge answers, and run stats."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("parse", help="Parse all .txt files in a directory to .json with the same name.")
    sp.add_argument("--src", required=True, help="Directory containing .txt files.")
    sp.add_argument(
        "--confusables-policy",
        choices=["preserve", "ascii_approx", "visual_approx"],
        default="preserve",
        help="Replacement policy for Greek sigma characters (default: preserve)."
    )
    sp.set_defaults(func=cmd_parse)

    sm = sub.add_parser("merge", help="Merge answers from an answers directory back into generated JSON files.")
    sm.add_argument("--gen", required=True, help="Directory with generated question JSON files.")
    sm.add_argument("--ans", required=True, help="Directory with answer JSON files (same filenames).")
    sm.add_argument(
        "--out",
        choices=["same", "merged"],
        default="same",
        help="Where to write output: same = overwrite in gen dir; merged = write to gen/merged."
    )
    sm.set_defaults(func=cmd_merge)

    sb = sub.add_parser("stats-blanks", help="Count blank options and export CSV.")
    sb.add_argument("--dir", required=True, help="Directory containing question JSON files.")
    sb.set_defaults(func=cmd_stats_blanks)

    so = sub.add_parser("stats-over6", help="Count questions with > 6 options and export CSV.")
    so.add_argument("--dir", required=True, help="Directory containing question JSON files.")
    so.set_defaults(func=cmd_stats_over6)

    su2 = sub.add_parser("stats-under6", help="Count questions with < 6 options and export CSV.")
    su2.add_argument("--dir", required=True, help="Directory containing question JSON files.")
    su2.add_argument(
        "--effective",
        action="store_true",
        help="Count effective options only (ignore blank options); otherwise use raw options length."
    )
    su2.set_defaults(func=cmd_stats_under6)

    sc = sub.add_parser(
        "concat",
        help="Concatenate all .json files in a directory into one array file and rewrite number as <filename>-<question_no>."
    )
    sc.add_argument("--dir", required=True, help="Directory containing question JSON files.")
    sc.add_argument("--out", help="Output file path (default: <dir>/all_questions.json).")
    sc.add_argument("--keep-source", action="store_true", help="Add a source_file field to each question.")
    sc.set_defaults(func=cmd_concat)

    return p

def main():
    parser = build_cli()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
