import json, time
from pathlib import Path
from openai import OpenAI
from tqdm.auto import tqdm
from collections import Counter

# 文件路径（按你的工程调整）
TEST_PATH = Path("probability_train_set.jsonl")      # 你之前切分出来的测试集
PRED_PATH = Path("preds_gpt5_zeroshot.jsonl")  # 预测输出（含解释，便于后续分析）

MODEL = "gpt-5"  

EXPL_SCHEMA = {
    "name": "prob_baseline_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "final_choice_index": {"type": "integer", "minimum": 0},
            "explanation_md": {"type":"string"},
        },
        "required": ["final_choice_index", "explanation_md"],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = """You are a careful probability/statistics teacher.
Given a multiple-choice item (context, question, options), choose the correct option index (0-based) and provide a concise explanation.
- Be precise and consistent with probability/statistics rules and common distributions.
- Use brief math notation where useful.
- Return ONLY JSON matching the schema."""

def load_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

def call_llm(payload, client: OpenAI, retry=4, cooldown=1.2):
    last_err = None
    for att in range(1, retry+1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role":"system","content": SYSTEM_PROMPT},
                    {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
                ],
                text={"format":{
                    "type":"json_schema",
                    "name":EXPL_SCHEMA["name"],
                    "schema":EXPL_SCHEMA["schema"],
                    "strict":EXPL_SCHEMA["strict"]
                }},
                temperature=1
            )
            out = getattr(resp, "output_text", None)
            if not out:
                chunks=[]
                for o in getattr(resp,"output",[]) or []:
                    for c in getattr(o,"content",[]) or []:
                        if getattr(c,"type","")=="output_text":
                            chunks.append(getattr(c,"text",""))
                out="".join(chunks)
            data = json.loads(out)
            # 兜底：确保字段存在
            if not isinstance(data.get("final_choice_index"), int):
                data["final_choice_index"] = 0
            data.setdefault("explanation_md","")
            return data
        except Exception as e:
            last_err = str(e)
            time.sleep(cooldown * att)
    raise RuntimeError(last_err or "LLM call failed")

def basic_view(item):
    ctx = item.get("english") or item.get("danish") or {}
    return {
        "context": ctx.get("context"),
        "question": ctx.get("question"),
        "options": ctx.get("options"),
    }

# 读取测试集
test_rows = load_jsonl(TEST_PATH)
print("Test size:", len(test_rows))

# 调用模型逐题作答并保存预测
client = OpenAI()
preds = []
with PRED_PATH.open("w", encoding="utf-8") as f:
    for item in tqdm(test_rows, desc="Zero-shot baseline"):
        payload = basic_view(item)
        result = call_llm(payload, client)
        rec = {
            "base_key": item.get("base_key"),
            "gold_answer_index": item.get("answer_index"),
            "pred_answer_index": result.get("final_choice_index"),
            "explanation": result.get("explanation_md"),
        }
        preds.append(rec)
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# 计算总体准确率 + 按选项位置的准确率
def accuracy(preds):
    ok = sum(1 for r in preds if isinstance(r.get("gold_answer_index"), int)
             and r.get("gold_answer_index") == r.get("pred_answer_index"))
    return ok / max(1, len(preds))

acc = accuracy(preds)

by_pos = {}
for r in preds:
    g = r.get("gold_answer_index")
    if not isinstance(g, int): continue
    by_pos.setdefault(g, {"n":0,"ok":0})
    by_pos[g]["n"] += 1
    by_pos[g]["ok"] += int(g == r.get("pred_answer_index"))

print(f"\nZero-shot baseline accuracy: {acc:.3%}")
print("Per-position accuracy:")
for pos in sorted(by_pos):
    n, ok = by_pos[pos]["n"], by_pos[pos]["ok"]
    print(f"  index {pos}: {ok}/{n} = {ok/max(1,n):.3%}")

dist_gold = Counter(r.get("gold_answer_index") for r in preds if r.get("gold_answer_index") is not None)
dist_pred = Counter(r.get("pred_answer_index") for r in preds if r.get("pred_answer_index") is not None)
print("\nGold position distribution:", dict(sorted(dist_gold.items())))
print("Pred position distribution:", dict(sorted(dist_pred.items())))
print("\nPredictions written to:", PRED_PATH)
