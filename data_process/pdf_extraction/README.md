# PDF Extraction

Tools in this folder convert raw exam PDFs into text/Markdown for later
data processing.

---

## `naugat_ocr.py` (Nougat)

- **Input:** `--input` = one PDF file or a folder with PDFs (searched recursively).
- **Output:** `--output` = folder with one Markdown file per PDF: `<name>.md`.
- Use when you want **local OCR → Markdown**.

Example:
```bash
python naugat_ocr.py --input ../raw_pdfs --output ../only-txt
```
## `mathpix_convert.py` (Mathpix)
Subcommand pdf

- **Input:** PDFs (file or folder via --input).

Output: Rich formats per PDF, e.g. <name>.tex.zip, <name>.md in --output.

Subcommand texzip2txt

Input: *.tex.zip files (folder via --src).

Output: Plain text <name>.txt in --output (or next to the zip).

Mathpix_Jupyter_Converter.ipynb
Interactive notebook version of the Mathpix flow:
manually pick PDFs, send to Mathpix, inspect and download the results.