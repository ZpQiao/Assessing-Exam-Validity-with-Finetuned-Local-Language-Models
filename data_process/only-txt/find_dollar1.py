# -*- coding: utf-8 -*-
"""
查找所有 .json 文件中 question 含有 "$1"（或 "$"+数字）的题目。
用法示例：
  只找 "$1":
    python find_dollar1.py --dir "path/to/jsons"
  找 "$" + 任意数字:
    python find_dollar1.py --dir "path/to/jsons" --any-digit
  同时导出 CSV:
    python find_dollar1.py --dir "path/to/jsons" --csv
"""
from pathlib import Path
import argparse, json, re, csv

def _qnum(item, idx):
    for k in ("number", "question_id", "id"):
        if k in item:
            try:
                return int(item[k])
            except Exception:
                return item[k]
    return idx + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="包含题目 JSON 的目录")
    ap.add_argument("--any-digit", action="store_true",
                    help="匹配 $ 后跟任意一位数字（默认只匹配 $1）")
    ap.add_argument("--csv", action="store_true", help="导出 CSV 明细")
    args = ap.parse_args()

    folder = Path(args.dir)
    pattern = re.compile(r"\$[0-9]" if args.any_digit else r"\$1")

    total = 0
    rows = []  # for CSV: file, question_number, question_text
    files = sorted(folder.glob("*.json"))
    if not files:
        print("未找到任何 .json 文件。")
        return

    print("=== question 含目标模式 的题目 ===")
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] 读取失败 {p.name}: {e}")
            continue
        if not isinstance(data, list):
            print(f"[WARN] {p.name}: 根对象不是数组，跳过。")
            continue

        found = []
        for idx, item in enumerate(data):
            qtext = str(item.get("question", ""))
            if pattern.search(qtext):
                qn = _qnum(item, idx)
                found.append((qn, qtext))
                rows.append([p.name, qn, qtext.replace("\n", " ")[:200]])  # CSV 里截断展示

        print(f"{p.name}: 匹配 {len(found)}")
        if found:
            # 列出题号；若想显示全文，把下面这行改成打印 qtext
            print("  明细: " + ", ".join([f"Q{q}" for q, _ in found]))
            total += len(found)

    print(f"\n总匹配题目数: {total}")

    if args.csv:
        csv_path = folder / ("questions_with_dollar" + ("_digit" if args.any_digit else "1") + ".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file", "question_number", "question_text_snippet"])
            w.writerows(rows)
        print(f"已导出: {csv_path}")

if __name__ == "__main__":
    main()
