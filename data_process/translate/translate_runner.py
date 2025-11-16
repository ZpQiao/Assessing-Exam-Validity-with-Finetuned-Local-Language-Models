#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, argparse
from typing import Dict, Any, Iterable, Set
from tqdm import tqdm
from openai import OpenAI

SCHEMA = {
  "name": "prob_mcq_translation",
  "schema": {
    "type":"object",
    "required":["canonical_id","lang_src","lang_tgt","context_tgt","question_tgt","options_tgt","answer_index"],
    "properties":{
      "canonical_id":{"type":"string"},
      "lang_src":{"type":"string","enum":["da","en"]},
      "lang_tgt":{"type":"string","enum":["da","en"]},
      "context_tgt":{"type":"string"},
      "question_tgt":{"type":"string"},
      "options_tgt":{"type":"array","items":{"type":"string"}},
      "answer_index":{"type":"integer"},
      "notes":{"type":"string"}
    }
  },
  "strict": True
}

SYSTEM_PROMPT = """You are a professional translator for math exam items.
Rules:
1) Copy ALL LaTeX segments ($...$ or \\[...\\]) EXACTLY as-is.
2) Do NOT alter any numbers/fractions/decimals; keep decimal comma if present.
3) Keep the options' order and count unchanged.
4) Translate only natural language; do not change meaning.
5) Output MUST follow the provided JSON schema strictly.
"""

def load_jsonl(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield json.loads(line)

def existing_cids(path)->Set[str]:
    if not os.path.exists(path): return set()
    ids=set()
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if not line: continue
            try:
                obj=json.loads(line); cid=obj.get("canonical_id")
                if cid: ids.add(cid)
            except Exception:
                continue
    return ids

def qc_pass(raw_path, clean_path, report_path, source_jobs):
    # strict QC: LaTeX segments identical; numbers identical up to decimal comma/dot; options count & index valid
    import re, json
    latex_re = re.compile(r"\$[^$]*\$|\\\[[^\]]*\\\]")
    num_re = re.compile(r"\d+(?:[.,]\d+)?")
    def latex_segments(s): return latex_re.findall(s or "")
    def nums_norm(s): return [x.replace(",",".") for x in num_re.findall(s or "")]

    src_map = {}
    for job in load_jsonl(source_jobs):
        src_map[job["canonical_id"]] = {
            "context": job.get("context",""),
            "question": job.get("question",""),
            "options": job.get("options",[])
        }

    ok=bad=0; issues=[]
    with open(clean_path,"w",encoding="utf-8") as out, open(raw_path,"r",encoding="utf-8") as rf:
        for line in rf:
            line=line.strip()
            if not line: continue
            try:
                rec=json.loads(line)
            except Exception:
                bad+=1; issues.append({"cid":None,"issue":"json_error"}); continue
            cid = rec.get("canonical_id")
            opts=rec.get("options_tgt",[]); ai=rec.get("answer_index",-1)
            if not isinstance(opts,list) or len(opts)==0 or not (0<=ai<len(opts)):
                bad+=1; issues.append({"cid":cid,"issue":"options_or_index"}); continue
            src = src_map.get(cid)
            if not src:
                rec["notes"]=(rec.get("notes","")+" [WARN:no_source]").strip()
                out.write(json.dumps(rec, ensure_ascii=False)+"\n"); ok+=1; continue
            src_lx = latex_segments(src["context"])+latex_segments(src["question"])+sum([latex_segments(o) for o in src["options"]],[])
            tgt_lx = latex_segments(rec.get("context_tgt",""))+latex_segments(rec.get("question_tgt",""))+sum([latex_segments(o) for o in opts],[])
            if src_lx != tgt_lx:
                bad+=1; issues.append({"cid":cid,"issue":"latex_mismatch"}); continue
            src_nums = nums_norm(src["context"])+nums_norm(src["question"]) + sum([nums_norm(o) for o in src["options"]],[])
            tgt_nums = nums_norm(rec.get("context_tgt",""))+nums_norm(rec.get("question_tgt","")) + sum([nums_norm(o) for o in opts],[])
            if sorted(src_nums) != sorted(tgt_nums):
                bad+=1; issues.append({"cid":cid,"issue":"numbers_mismatch"}); continue
            out.write(json.dumps(rec, ensure_ascii=False)+"\n"); ok+=1
    with open(report_path,"w",encoding="utf-8") as rep:
        json.dump({"ok":ok,"bad":bad,"issues":issues}, rep, ensure_ascii=False, indent=2)

def main():
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4.1-mini")
    ap.add_argument("--input", default="./to_translate_en.jsonl")
    ap.add_argument("--out-raw", default="./translated_en_raw.jsonl")
    ap.add_argument("--out-clean", default="./translated_en_clean.jsonl")
    ap.add_argument("--qc-report", default="./translation_qc_report.json")
    ap.add_argument("--tps", type=float, default=1.0)
    ap.add_argument("--rate-limit", type=int, default=0)
    args=ap.parse_args()

    api_key=os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Please set OPENAI_API_KEY"); sys.exit(1)
    client=OpenAI(api_key=api_key)

    done=existing_cids(args.out_raw)
    jobs=list(load_jsonl(args.input))
    todo=[j for j in jobs if j.get("canonical_id") not in done]

    print(f"Total jobs: {len(jobs)} | already done: {len(done)} | to run: {len(todo)}")
    if args.rate_limit and len(todo)>args.rate_limit:
        todo=todo[:args.rate_limit]; print(f"Applying rate limit -> {len(todo)}")

    with open(args.out_raw,"a",encoding="utf-8") as out:
        for job in tqdm(todo, desc="Translating", unit="job"):
            payload={
                "canonical_id": job["canonical_id"],
                "lang_src": job["lang_src"],
                "lang_tgt": job["lang_tgt"],
                "context_src": job.get("context",""),
                "question_src": job.get("question",""),
                "options_src": job.get("options",[]),
                "answer_index": job.get("answer_index",-1)
            }
            try:
                resp=client.responses.create(
                    model=args.model,
                    input=[{"role":"system","content":SYSTEM_PROMPT},{"role":"user","content":json.dumps(payload, ensure_ascii=False)}],
                    response_format={"type":"json_schema","json_schema":SCHEMA}
                )
                text=resp.output[0].content[0].text
                out.write(text+"\n"); out.flush()
            except Exception as e:
                out.write(json.dumps({"canonical_id": job.get("canonical_id"), "error": str(e)}, ensure_ascii=False)+"\n")
            if args.tps>0: time.sleep(1.0/args.tps)

    print("QC pass...")
    qc_pass(args.out_raw, args.out_clean, args.qc_report, args.input)
    print("Done. Clean ->", args.out_clean)

if __name__=="__main__":
    main()
