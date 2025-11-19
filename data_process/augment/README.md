# Probability MCQ Augmentation

This folder contains two steps:

---

## 1. Generate explanations

Script: `generate_train_explanations.py`  
`train.jsonl` → `train_expl.jsonl`

- **Input:** `train.jsonl`  
  One MCQ per line, with fields like:
  - `base_key`
  - `english` / `danish` (context, question, options)
  - `answer_index`
  - `knowledge_points`

- **Output:** `train_expl.jsonl`  
  Same items, enriched with:
  - `explanation`
  - `explanation_key_steps`
  - `explanation_formulae`
  - `explanation_pitfalls`
  - `final_choice_index`
  - `final_result_text`

The script can resume from an existing `train_expl.jsonl` and will correct
the explanation once if the model’s chosen index disagrees with `answer_index`.

---

## 2. Augment the MCQ bank

Script: `augment_probability_mcq_bank.py`  
 `train_expl.jsonl` → `augmented_train.jsonl`

- **Input:** `train_expl.jsonl`  
  The explained MCQ bank from step 1.

- **Output:**
  - `augmented_train.jsonl`  
    For each `base_key`, writes:
    - 1 original item (`augmentation_type = "original"`)
    - 5 rewritten items (`augmentation_type = "rephrased"`)
    - 5 backward items (`augmentation_type = "backward"`)
  - `augmented_train.jsonl.state.json`  
    Checkpoint with processed `base_key`s (used to resume).

Total per original question: **1 (original) + 5 (rephrased) + 5 (backward) = 11** records in `augmented_train.jsonl`.
