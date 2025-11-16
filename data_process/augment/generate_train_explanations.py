import json, time
from pathlib import Path
from openai import OpenAI
from tqdm.auto import tqdm

MODEL = "gpt-5"
client = OpenAI()

# Input: train.jsonl (generated in the previous pipeline)
train_path = Path("train.jsonl")
train_rows = [
    json.loads(l)
    for l in train_path.read_text(encoding="utf-8").splitlines()
    if l.strip()
]

# Output: train_expl.jsonl with explanations (supports resume)
out_path = Path("train_expl.jsonl")
done = {}
if out_path.exists():
    for l in out_path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        rec = json.loads(l)
        done[rec["base_key"]] = rec

EXPL_SCHEMA = {
    "name": "prob_expl_schema",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "explanation_md": {
                "type": "string",
                "description": "Concise step-by-step explanation in Markdown."
            },
            "key_steps": {
                "type": "array",
                "items": {"type": "string"}
            },
            "formulae": {
                "type": "array",
                "items": {"type": "string"}
            },
            "pitfalls": {
                "type": "array",
                "items": {"type": "string"}
            },
            "final_choice_index": {
                "type": "integer",
                "minimum": 0
            },
            "final_result_text": {
                "type": "string"
            }
        },
        "required": [
            "explanation_md",
            "key_steps",
            "formulae",
            "pitfalls",
            "final_choice_index",
            "final_result_text"
        ],
        "additionalProperties": False
    }
}

SYSTEM_PROMPT = """You are a probability teacher. Given a multiple-choice problem (context, question, options),
write a short, rigorous explanation and select the correct option index (0-based).
- Be precise; use symbols/LaTeX where helpful.
- Keep explanation <= 180 words if possible.
- Do NOT mention that an answer index was provided.
Return ONLY JSON that matches the schema."""

def call_llm(payload, retry=4, cooldown=1.2):
    last_err = None
    for att in range(1, retry + 1):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": EXPL_SCHEMA["name"],
                        "schema": EXPL_SCHEMA["schema"],
                        "strict": EXPL_SCHEMA["strict"]
                    }
                }
            )
            out = getattr(resp, "output_text", None)
            if not out:
                chunks = []
                for o in getattr(resp, "output", []) or []:
                    for c in getattr(o, "content", []) or []:
                        if getattr(c, "type", "") == "output_text":
                            chunks.append(getattr(c, "text", ""))
                out = "".join(chunks)
            return json.loads(out)
        except Exception as e:
            last_err = str(e)
            time.sleep(cooldown * att)
    raise RuntimeError(last_err or "LLM call failed")

def verify_and_fix(item, result):
    """
    If final_choice_index disagrees with the ground-truth answer_index,
    make one correction pass by sending the explanation back with the gold index.
    """
    gold = item.get("answer_index")
    pred = result.get("final_choice_index")
    if isinstance(gold, int) and isinstance(pred, int) and gold == pred:
        return result, False

    # Trigger correction: send back the previous explanation and the gold index,
    # and request a concise, consistent revision.
    ctx = item.get("english") or item.get("danish") or {}
    payload = {
        "context": ctx.get("context"),
        "question": ctx.get("question"),
        "options": ctx.get("options"),
        "knowledge_points": item.get("knowledge_points", []),
        "previous_explanation": result,
        "note": f"The correct option index is {gold}. Update explanation accordingly and ensure consistency."
    }
    resp = client.responses.create(
        model=MODEL,
        input=[
            {
                "role": "system",
                "content": "Revise the prior explanation to be correct and consistent with the correct option index, keeping it concise. Return JSON per schema."
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": EXPL_SCHEMA["name"],
                "schema": EXPL_SCHEMA["schema"],
                "strict": EXPL_SCHEMA["strict"]
            }
        }
    )
    out = getattr(resp, "output_text", None)
    if not out:
        chunks = []
        for o in getattr(resp, "output", []) or []:
            for c in getattr(o, "content", []) or []:
                if getattr(c, "type", "") == "output_text":
                    chunks.append(getattr(c, "text", ""))
        out = "".join(chunks)
    fixed = json.loads(out)
    return fixed, True

f = out_path.open("a", encoding="utf-8")
mismatches = 0

for item in tqdm(train_rows, desc="Generating explanations"):
    bk = item.get("base_key")
    if bk in done:
        continue

    ctx = item.get("english") or item.get("danish") or {}
    payload = {
        "context": ctx.get("context"),
        "question": ctx.get("question"),
        "options": ctx.get("options"),
        "knowledge_points": item.get("knowledge_points", [])
    }
    try:
        res = call_llm(payload)
        res, fixed = verify_and_fix(item, res)
        if fixed:
            mismatches += 1

        # Write back enriched record
        out_rec = dict(item)
        out_rec["explanation"] = res.get("explanation_md")
        out_rec["explanation_key_steps"] = res.get("key_steps", [])
        out_rec["explanation_formulae"] = res.get("formulae", [])
        out_rec["explanation_pitfalls"] = res.get("pitfalls", [])
        out_rec["final_choice_index"] = res.get("final_choice_index")
        out_rec["final_result_text"] = res.get("final_result_text")
        f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        f.flush()
    except Exception as e:
        # Fallback: write error info so this item can be retried later
        out_rec = dict(item)
        out_rec["_explanation_error"] = str(e)
        f.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
        f.flush()

f.close()
print("Done. mismatches corrected:", mismatches)
