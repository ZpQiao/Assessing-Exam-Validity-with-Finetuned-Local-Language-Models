# MCQ Translation

Script: `translate_runner.py`

Translate probability MCQs between Danish (`da`) and English (`en`) using an LLM, then filter outputs with a simple QC check (LaTeX and numbers must be preserved, options and `answer_index` must be valid).

- **Input** (`--input`, default: `to_translate_en.jsonl`):  
  JSONL with fields `canonical_id`, `lang_src`, `lang_tgt`, `context`, `question`, `options`, `answer_index`.

- **Outputs**  
  - `--out-raw` (default: `translated_en_raw.jsonl`) – all model responses.  
  - `--out-clean` (default: `translated_en_clean.jsonl`) – responses that pass QC.  
  - `--qc-report` (default: `translation_qc_report.json`) – small JSON summary of OK/bad items.

Typical run (using defaults):

```bash
python translate_runner.py
