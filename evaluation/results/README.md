# Data & Results

Dataset files and evaluation results for probability theory MCQ experiments.

---

## Dataset Files

**`probability_train_set.jsonl`**  
Training set with question-answer pairs.

**`probability_test_set.jsonl`**  
Test set for model evaluation.

**`probability_augmented_train_set.jsonl`**  
Training set with explanations (key steps, formulae, pitfalls) for RAG.

---

## Result Files

**`results_Qwen3-14B_base_en.csv`**  
Baseline model - English questions.

**`results_Qwen3-14B_base_da.csv`**  
Baseline model - Danish questions.

**`results_Qwen3-14B_finetuned_all.csv`**  
Fine-tuned model (QLoRA) - all languages.

**`results_Qwen3-14B_rag_eng.csv`**  
RAG-enhanced model - English questions.

**`results_Qwen3-14B_rag_dan.csv`**  
RAG-enhanced model - Danish questions.

---

## CSV Columns

- `base_key` - Question ID
- `language` - Question language
- `is_correct` - Correctness (True/False)
- `predicted_answer` - Model prediction
- `correct_answer` - Ground truth