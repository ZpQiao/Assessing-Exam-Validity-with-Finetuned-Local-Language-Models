# Probability LLM Evaluation

Tools for fine-tuning and evaluating LLMs on probability theory multiple-choice questions.

---

## `fine_tuning.py` (QLoRA Training)

- **Input:** `--train_file`, `--val_file` = JSONL files with question-answer pairs.
- **Output:** `--output_dir` = trained LoRA adapter + checkpoints.
- Use for **4-bit QLoRA fine-tuning** of Qwen3-14B (8192 context, assistant-only loss).

Example:
```bash
python fine_tuning.py \
  --train_file probability_train.jsonl \
  --val_file probability_val.jsonl \
  --output_dir ./qwen3-qlora-output \
  --num_epochs 5 \
  --lora_r 256
```

Monitor training:
```bash
tensorboard --logdir ./qwen3-qlora-output
```

---

## `test_framework.py` (Full Evaluation Framework)

- **Input:** JSONL test data (loaded via `load_jsonl_data()`).
- **Output:** CSV results + comparison reports in `--output_dir`.
- Supports **RAG**, **checkpointing**, **multi-language** (English/Danish), and **model comparison**.

Configure models in script:
```python
model_configs = [
    {
        "name": "Qwen/Qwen3-14B",
        "local_path": "/root/models/qwen3-14b-q4",
        "use_quantization": True,
        "quantization_bits": 4,
        "temperature": 0.0,
    }
]
```

Run:
```bash
python test_framework.py
```

Features:
- Checkpoint every 10 questions (resume with `resume=True`)
- Ctrl+C safe interruption
- Detailed accuracy and error analysis

---

## `test_with_lora_updated.py` (LoRA Model Testing)

- **Input:** Test data JSONL + base model + LoRA adapter path.
- **Output:** CSV results with baseline vs fine-tuned comparison.
- Simplified tool for **evaluating LoRA fine-tuned models**.

Configure paths in script:
```python
TEST_DATA = "probability_test_set.jsonl"
BASE_MODEL = "/root/models/qwen3-14b-q4"
LORA_ADAPTER = "./qwen3-qlora-output/final_model"
```

Run:
```bash
python test_with_lora_updated.py
```

Outputs:
- `finetuned_<timestamp>.csv` = fine-tuned model results
- `comparison_report_<timestamp>.csv` = accuracy comparison

---

## Quick Workflow

1. **Prepare data** → JSONL format with `messages` field (see data format below)
2. **Fine-tune** → `python fine_tuning.py --train_file ... --val_file ...`
3. **Evaluate** → `python test_with_lora_updated.py` (set LoRA path in script)

## Data Format

Training data (JSONL):
```json
{
  "messages": [
    {"role": "user", "content": "Context: ...\n\nQuestion: ...\n\nOptions:\n1. ..."},
    {"role": "assistant", "content": "Reasoning...\n\n{\"answer\": 2}"}
  ]
}
```

Test data (JSONL):
```json
{
  "base_key": "2004_1_1",
  "english": {
    "context": "...",
    "question": "...",
    "options": ["...", "..."],
    "correct_answer": 2
  },
  "danish": { ... }
}
```

## Requirements

```bash
pip install torch transformers peft bitsandbytes datasets pandas tqdm tensorboard
```