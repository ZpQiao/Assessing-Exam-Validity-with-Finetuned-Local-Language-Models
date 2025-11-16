# augment_probability_mcq_bank.py
import os, sys, json, time, datetime, tempfile
from typing import Dict, List, Set
import openai
from openai import OpenAI

# ======== Config ========
# Default model; if gpt-5 is not available, you can set OPENAI_MODEL to gpt-4o / gpt-4o-mini
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TEMP = 1
INPUT_FILE = "train_expl.jsonl"
OUTPUT_FILE = "augmented_train.jsonl"
CHECKPOINT_FILE = OUTPUT_FILE + ".state.json"   # Checkpoint file for resuming progress
SLEEP_BETWEEN = 1                               # Delay between API calls (seconds)

client = OpenAI()  # Reads API key from environment variable OPENAI_API_KEY

# ======== Helpers ========
def fmt_options(options: List[str]) -> str:
    return "\n".join(f"{i+1}. {opt}" for i, opt in enumerate(options))

def load_jsonl(path: str) -> List[Dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    return items

def write_jsonl_line(path: str, obj: Dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def atomic_write_json(path: str, data: Dict):
    """
    Atomically write JSON to a file by first writing to a temp file and then replacing.
    This avoids partial writes if the process is interrupted.
    """
    fd, tmp = tempfile.mkstemp(prefix=".ckpt_", dir=os.path.dirname(path) or ".", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise

def load_checkpoint(path: str) -> Dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "processed_keys": [],
        "started_at": datetime.datetime.utcnow().isoformat() + "Z",
        "last_update": None
    }

def infer_processed_from_output(output_path: str) -> Set[str]:
    """
    If the checkpoint file is missing or empty, infer which base_keys
    have been fully processed by scanning the output JSONL file for
    entries with augmentation_type == "original".
    """
    done: Set[str] = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if obj.get("augmentation_type") == "original":
                    bk = obj.get("base_key") or obj.get("original_base_key")
                    if bk:
                        done.add(bk)
            except Exception:
                continue
    return done

def save_checkpoint_state(processed: Set[str], meta: Dict):
    atomic_write_json(CHECKPOINT_FILE, {
        "processed_keys": sorted(list(processed)),
        "started_at": meta.get("started_at"),
        "last_update": datetime.datetime.utcnow().isoformat() + "Z"
    })

def fmt_duration(seconds: float) -> str:
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"

def print_progress(i: int, total: int, done_new: int, remain_new: int, start_ts: float):
    elapsed = time.time() - start_ts
    pct = 100.0 * i / total if total else 100.0
    eta = "-"
    if done_new > 0:
        avg = elapsed / done_new
        eta = fmt_duration(avg * max(remain_new, 0))
    print(f"[{i}/{total}] {pct:6.2f}% | elapsed {fmt_duration(elapsed)} | ETA {eta}")

# ======== Prompts ========
REPHRASING_PROMPT = """You are a probability theory expert. Create exactly 5 REWRITTEN versions for the following MCQ.
Keep the same distribution type ({knowledge_points}).

Original (English)
Context: {context_en}
Question: {question_en}
Options:
{options_en}
Correct Answer Index: {answer_index}
Explanation: {explanation}

Requirements:
- Change scenario and adjust numbers (keep difficulty).
- Keep distribution type ({knowledge_points}); probabilities in [0,1].
- Mathematical formulas must be in LaTeX ($...$).
- Provide BOTH English and Danish.
- Output must strictly conform to the provided JSON schema (array of 5 objects).
"""

BACKWARD_PROMPT = """You are a probability theory expert. Create exactly 5 BACKWARD questions using different strategies.

Original (English)
Context: {context_en}
Question: {question_en}
Options:
{options_en}
Correct Answer Index: {answer_index}
Knowledge Points: {knowledge_points}
Explanation: {explanation}

Use distinct strategies (one each):
1) Given probability, find parameter
2) Given result, find sample size
3) Given success on k-th trial, find base probability
4) Given expected value, find parameter
5) Given variance, find parameter

Requirements:
- Unique solvable answer; same difficulty.
- Provide BOTH English and Danish; probabilities in [0,1]; LaTeX formulas.
- Output must strictly conform to the provided JSON schema (array of 5 objects).
"""

# ======== Structured Output Schemas (Responses API) ========
# Top-level object shape: {"items":[... exactly 5 ...]}
REPHRASE_ARRAY_SCHEMA = {
    "name": "rephrase_array",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "rewrite_id",
                        "english",
                        "danish",
                        "knowledge_points",
                        "explanation",
                        "explanation_key_steps",
                        "explanation_formulae",
                        "explanation_pitfalls",
                        "changes"
                    ],
                    "properties": {
                        "rewrite_id": {"type": "integer"},
                        "english": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["context", "question", "options", "answer_index"],
                            "properties": {
                                "context": {"type": "string"},
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "minItems": 6,
                                    "maxItems": 6,
                                    "items": {"type": "string"}
                                },
                                "answer_index": {"type": "integer"}
                            }
                        },
                        "danish": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["context", "question", "options", "answer_index"],
                            "properties": {
                                "context": {"type": "string"},
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "minItems": 6,
                                    "maxItems": 6,
                                    "items": {"type": "string"}
                                },
                                "answer_index": {"type": "integer"}
                            }
                        },
                        "knowledge_points": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "explanation": {"type": "string"},
                        "explanation_key_steps": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "explanation_formulae": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "explanation_pitfalls": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "changes": {"type": "string"}
                    }
                }
            }
        }
    }
}

BACKWARD_ARRAY_SCHEMA = {
    "name": "backward_array",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["items"],
        "properties": {
            "items": {
                "type": "array",
                "minItems": 5,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "backward_id",
                        "strategy",
                        "english",
                        "danish",
                        "knowledge_points",
                        "explanation",
                        "explanation_key_steps",
                        "explanation_formulae",
                        "explanation_pitfalls",
                        "relation"
                    ],
                    "properties": {
                        "backward_id": {"type": "integer"},
                        "strategy": {"type": "string"},
                        "english": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["context", "question", "options", "answer_index"],
                            "properties": {
                                "context": {"type": "string"},
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "minItems": 6,
                                    "maxItems": 6,
                                    "items": {"type": "string"}
                                },
                                "answer_index": {"type": "integer"}
                            }
                        },
                        "danish": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["context", "question", "options", "answer_index"],
                            "properties": {
                                "context": {"type": "string"},
                                "question": {"type": "string"},
                                "options": {
                                    "type": "array",
                                    "minItems": 6,
                                    "maxItems": 6,
                                    "items": {"type": "string"}
                                },
                                "answer_index": {"type": "integer"}
                            }
                        },
                        "knowledge_points": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "explanation": {"type": "string"},
                        "explanation_key_steps": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "explanation_formulae": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "explanation_pitfalls": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "relation": {"type": "string"}
                    }
                }
            }
        }
    }
}


# ======== Responses API caller (Structured Outputs) ========
def _to_text_format(schema_obj: dict) -> dict:
    """
    Transform a custom schema object into the format expected by the
    Responses API `text.format` field:
    Input schema_obj:
    {
      "name": "...",
      "schema": {...},
      "strict": True/False
    }
    Output:
    {
      "type": "json_schema",
      "name": ...,
      "schema": ...,
      "strict": ...
    }
    """
    return {
        "type": "json_schema",
        "name": schema_obj["name"],
        "schema": schema_obj["schema"],
        "strict": schema_obj.get("strict", False)
    }

def call_array_with_schema(prompt: str, schema_obj: dict, max_retries: int = 3):
    """
    Call the OpenAI Responses API, requesting output that strictly
    conforms to the given JSON schema. It retries up to max_retries
    times if parsing or validation fails, and returns the 'items' list.
    """
    last_err = None
    for _ in range(max_retries):
        try:
            resp = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-5"),
                temperature=1,
                input=prompt,
                text={"format": {
                    "type": "json_schema",
                    "name": schema_obj["name"],
                    "schema": schema_obj["schema"],
                    "strict": schema_obj.get("strict", True)
                }},
                max_output_tokens=40000,
                instructions=(
                    "You are an expert in probability theory and multilingual education. "
                    "Return ONLY valid JSON that matches the provided schema."
                )
            )
            # The top-level should be a JSON object.
            data = json.loads(resp.output_text)
            items = data.get("items", [])
            if not isinstance(items, list) or len(items) != 5:
                raise ValueError("Schema matched but 'items' is not an array of length 5.")
            return items
        except Exception as e:
            last_err = e
            time.sleep(3)
    raise RuntimeError(f"API failed after retries: {last_err}")

# ======== Build prompts + augment ========
def build_rephrase_prompt(q: Dict) -> str:
    return REPHRASING_PROMPT.format(
        context_en=q["english"]["context"],
        question_en=q["english"]["question"],
        options_en=fmt_options(q["english"]["options"]),
        answer_index=q["answer_index"],
        knowledge_points=", ".join(q["knowledge_points"]),
        explanation=q["explanation"]
    )

def build_backward_prompt(q: Dict) -> str:
    return BACKWARD_PROMPT.format(
        context_en=q["english"]["context"],
        question_en=q["english"]["question"],
        options_en=fmt_options(q["english"]["options"]),
        answer_index=q["answer_index"],
        knowledge_points=", ".join(q["knowledge_points"]),
        explanation=q["explanation"]
    )

def augment_one(q: Dict) -> Dict:
    """
    For a single question q, generate:
      - 5 rephrased variants
      - 5 backward (inverse) questions
    using the structured-output responses API.
    """
    rephrased = call_array_with_schema(build_rephrase_prompt(q), REPHRASE_ARRAY_SCHEMA)
    time.sleep(SLEEP_BETWEEN)
    backward = call_array_with_schema(build_backward_prompt(q), BACKWARD_ARRAY_SCHEMA)
    return {"base_key": q["base_key"], "original": q, "rephrased": rephrased, "backward": backward}

def append_full_question_results(out_path: str, item: Dict):
    """
    Append the original question, all rephrased questions, and all backward questions
    to the given JSONL output file.
    """
    # Original question
    orig = item["original"].copy()
    orig["augmentation_type"] = "original"
    write_jsonl_line(out_path, orig)
    
    # 5 rephrased questions
    for i, r in enumerate(item["rephrased"], 1):
        obj = {
            "base_key": f'{item["base_key"]}_rephrase_{i}',
            "augmentation_type": "rephrased",
            "original_base_key": item["base_key"],
            "danish": r["danish"],
            "english": r["english"],
            "answer_index": r["english"]["answer_index"],  # read from english block
            "knowledge_points": r.get("knowledge_points", []),
            "explanation": r.get("explanation", ""),
            "explanation_key_steps": r.get("explanation_key_steps", []),
            "explanation_formulae": r.get("explanation_formulae", []),
            "explanation_pitfalls": r.get("explanation_pitfalls", []),
            "changes": r.get("changes", "")
        }
        write_jsonl_line(out_path, obj)
    
    # 5 backward questions
    for i, b in enumerate(item["backward"], 1):
        obj = {
            "base_key": f'{item["base_key"]}_backward_{i}',
            "augmentation_type": "backward",
            "original_base_key": item["base_key"],
            "strategy": b["strategy"],
            "danish": b["danish"],
            "english": b["english"],
            "answer_index": b["english"]["answer_index"],  # read from english block
            "knowledge_points": b.get("knowledge_points", []),
            "explanation": b.get("explanation", ""),
            "explanation_key_steps": b.get("explanation_key_steps", []),
            "explanation_formulae": b.get("explanation_formulae", []),
            "explanation_pitfalls": b.get("explanation_pitfalls", []),
            "relation": b.get("relation", "")
        }
        write_jsonl_line(out_path, obj)

# ======== Main ========
if __name__ == "__main__":
    # Load question bank
    bank = load_jsonl(INPUT_FILE)
    total = len(bank)
    if total == 0:
        print("No items in input.")
        sys.exit(0)

    # Load checkpoint
    checkpoint_meta = load_checkpoint(CHECKPOINT_FILE)
    processed_keys: Set[str] = set(checkpoint_meta.get("processed_keys", []))
    if not processed_keys:
        inferred = infer_processed_from_output(OUTPUT_FILE)
        if inferred:
            processed_keys.update(inferred)
            save_checkpoint_state(processed_keys, checkpoint_meta)
            print(f"Recovered progress from output: {len(processed_keys)} items already done.")
    else:
        print(f"Loaded checkpoint: {len(processed_keys)} items already done.")

    already_done = len(processed_keys)
    remaining_new = max(total - already_done, 0)
    start_ts = time.time()
    newly_done = 0

    print(f"Start. Total: {total}, already done: {already_done}, to process: {remaining_new}.")
    print("-" * 60)

    for idx, q in enumerate(bank, 1):
        bk = q.get("base_key") or q.get("original_base_key")
        if bk in processed_keys:
            print(f"[SKIP] {bk}")
            print_progress(idx, total, newly_done, remaining_new - newly_done, start_ts)
            continue

        try:
            result = augment_one(q)
            append_full_question_results(OUTPUT_FILE, result)
            processed_keys.add(bk)
            newly_done += 1
            save_checkpoint_state(processed_keys, checkpoint_meta)
            print(f"[DONE] {bk} (+10)")
        except KeyError as e:
            print(f"[ERROR] {bk}: Missing key {e}")  # more explicit error message
            import traceback
            traceback.print_exc()  # print full stack trace
        except Exception as e:
            print(f"[ERROR] {bk}: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print_progress(idx, total, newly_done, remaining_new - newly_done, start_ts)
            time.sleep(SLEEP_BETWEEN)

    print("-" * 60)
    print(f"Complete. Originals: {total}, Rephrased: {total*5}, Backward: {total*5}.")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Checkpoint: {CHECKPOINT_FILE}")
