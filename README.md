# Assessing Exam Validity with Finetuned Local Language Models


This repository contains the code accompanying the master’s thesis: 


>[`Assessing Exam Validity with Finetuned Local Language Models`](./Master_Thesis_Zhengping_Qiao.pdf)   
> Zhengping Qiao, Technical University of Denmark, 2025.


This repository evaluates local large language models on university-level probability theory exams.

The experiments are based on exams from DTU course [`02405 Probability Theory`](https://www.imm.dtu.dk/courses/02405/).

---

## Repository Structure

Every main folder has its own `README.md`.  
Each sub-README describes the **inputs**, **outputs**, and **main scripts** for that part of the project.

Use this top-level README as a map.  
Follow the links below for details and concrete usage.

```text
.                                           # Root of the project
├── Master_Thesis_Zhengping_Qiao.pdf        # Full master’s thesis
├── thesis_appendix_evaluation_results.zip  # Packed evaluation results used in the thesis
│
├── datasets/                               # Final datasets for training & evaluation
│   ├── probability_train_set.jsonl         # Base training set from processed exams
│   ├── probability_test_set.jsonl          # Bilingual test set for all experiments
│   └── probability_augmented_train_set.jsonl  # Training set with GPT-generated MCQs & explanations
│
├── data_process/                           # From raw PDFs to structured JSON & augmented data
│   ├── augment/                            # Synthetic MCQs and explanations
│   │   ├── augment_probability_mcq.py      # Generates MCQ variants from structured questions
│   │   └── generate_train_explanations.py  # Uses GPT to create step-by-step explanations
│   │
│   ├── only-txt/                           # Clean exam text + per-exam JSON
│   │   ├── qa_toolkit.py                   # Helper functions for Q–A structures
│   │   ├── exam_YYYY_MM_DD_{dansk,en}.txt  # Plain-text extraction per exam & language
│   │   ├── exam_YYYY_MM_DD_{dansk,en}.json # Parsed questions and answers per exam & language
│   │   ├── all_questions.json              # All extracted questions before splitting
│   │   └── merged/                         # Per-exam JSON aligned with final schema
│   │       └── exam_YYYY_MM_DD_{dansk,en}.json  # Ready-to-use exam-level JSON
│   │
│   ├── pdf_extraction/                     # OCR & LaTeX extraction from PDFs
│   │   ├── naugat_ocr.py                   # Nougat-based OCR for scientific PDFs
│   │   ├── mathpix_convert.py              # Mathpix wrapper for equation-heavy pages
│   │   └── Mathpix_Jupyter_Converter.ipynb # Notebook for interactive Mathpix conversion
│   │
│   └── translate/                          # Danish/English alignment utilities
│       ├── translate_runner.py             # Batch translation runner (external API)
│       └── translation_schema.json         # Translation metadata and fields
│
├── fine_tuning/                            # QLoRA fine-tuning of Qwen3-14B
│   ├── fine_tuning.py                      # Main QLoRA training script
│   ├── test_framework.py                   # Batch testing utilities
│   ├── test_with_lora_updated.py           # Inference for LoRA-adapted models
│   └── qwen3_qlora_model/                  # Adapters, checkpoints & logs
│       ├── mcq_history.jsonl               # Training-time MCQ interaction log
│       ├── checkpoint-*/                   # Intermediate adapter checkpoints
│       ├── final_model/                    # Final adapter used in experiments
│       └── runs/                           # TensorBoard event files
│
├── RAG/                                    # Retrieval-augmented generation experiments
│   ├── Qwen3-14B_trainset.csv              # Training set view for indexing
│   ├── Qwen3-14B_trainset_kp_accuracy.csv  # Knowledge-point accuracy for retrieval weights
│   ├── rag.py                              # Core RAG implementation & FAISS index handling
│   ├── test_framework.py                   # Shared evaluation helpers
│   └── test_main.py                        # Baseline vs RAG comparison entry script
│
├── evaluation/                             # Final evaluation scripts & result files
│   ├── gpt_baseline/                       # GPT-5 baseline evaluation
│   │   ├── gpt5_test.py                    # Queries GPT-5 on the test set
│   │   └── gpt-5_baseline_test.csv         # GPT-5 predictions and correctness
│   │
│   └── results/                            # Local model result CSVs
│       ├── results_Qwen3-14B_base_da.csv       # Baseline, Danish questions
│       ├── results_Qwen3-14B_base_en.csv       # Baseline, English questions
│       ├── results_Qwen3-14B_finetuned_all.csv # Fine-tuned, all questions
│       ├── results_Qwen3-14B_rag_dan.csv       # RAG, Danish questions
│       └── results_Qwen3-14B_rag_eng.csv       # RAG, English questions
│
└── raw_pdfs/                               # Original DTU 02405 exam & solution PDFs
    ├── exam_YYYY_MM_DD_{danish,english}.pdf    # Raw exam PDFs by date & language
    └── solution_YYYY_MM_DD_{danish,english}.pdf # Official solution PDFs
```

### Key entry points

- [`data_process/augment`](./data_process/augment/README.md) — Synthetic MCQs and explanations  
- [`data_process/only-txt`](./data_process/only-txt/README.md) — Cleaned exam text and per-exam JSON structure  
- [`data_process/pdf_extraction`](./data_process/pdf_extraction/README.md) — OCR and LaTeX extraction pipeline  
- [`data_process/translate`](./data_process/translate/README.md) — Danish/English alignment and translation schema  
- [`fine_tuning`](./fine_tuning/README.md) — QLoRA fine-tuning setup for Qwen3-14B  
- [`RAG`](./RAG/README.md) — Retrieval-augmented generation experiments and baselines  
- [`evaluation/gpt_baseline`](./evaluation/gpt_baseline/README.md) — GPT-5 baseline evaluation  
- [`evaluation/results`](./evaluation/results/README.md) — Final result CSVs used in the thesis

## Experimental results (summary)
The table below reports overall MCQ accuracy on the bilingual probability exam test set:
| Model / setup              | Type          | Overall accuracy (test set) |
|---------------------------|---------------|-----------------------------|
| GPT-5                     | API, zero-shot        |  96.08%                      |
| Qwen3-14B (baseline)      | Local, 4-bit, no RAG  |  76.96%                      |
| Qwen3-14B + pitfall RAG   | Local, 4-bit, with RAG|  82.84%                      |

Details, per-knowledge-point results, and significance tests are reported in the thesis and in `evaluation/results`.


## License

This project is released under the MIT License.

**Dataset Usage**: DTU Course 02405 examination materials are used for academic research purposes under fair use principles. Redistribution of the original exam PDFs may be subject to DTU regulations.

