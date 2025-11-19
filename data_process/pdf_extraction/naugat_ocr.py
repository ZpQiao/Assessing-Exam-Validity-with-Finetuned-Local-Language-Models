#!/usr/bin/env python
"""
Simple CLI script for converting exam PDFs to Markdown using the Nougat model.

Usage:
    python nougat_ocr.py --input ../raw_pdfs --output ../only-txt

This script:
    - finds all *.pdf files under --input (recursively)
    - converts each PDF to page images (DPI=300)
    - runs Nougat (facebook/nougat-base) on each page
    - saves one Markdown file per PDF to --output
"""

from pathlib import Path
import argparse
from typing import List

import torch
from pdf2image import convert_from_path
from tqdm.auto import tqdm
from transformers import NougatProcessor, VisionEncoderDecoderModel


# ---------------------- Core functions ---------------------- #

def load_nougat(model_name: str = "facebook/nougat-base"):
    """Load Nougat processor and model, automatically choose device."""
    print(f"[info] Loading Nougat model: {model_name}")
    processor = NougatProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[info] Using device: {device}")
    model.to(device)
    model.eval()

    return processor, model, device


def find_pdfs(input_path: Path) -> List[Path]:
    """Return a sorted list of all PDF files under input_path (recursive)."""
    input_path = input_path.resolve()
    if input_path.is_file() and input_path.suffix.lower() == ".pdf":
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path is neither a PDF file nor a directory: {input_path}")

    pdfs = sorted(p for p in input_path.rglob("*.pdf") if p.is_file())
    return pdfs


def pdf_to_images(pdf_path: Path, dpi: int = 300, max_pages: int | None = None):
    """Convert PDF to a list of PIL images."""
    print(f"[info] Converting PDF to images: {pdf_path.name} (dpi={dpi})")
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    if max_pages is not None:
        pages = pages[:max_pages]
    print(f"[info] Total pages: {len(pages)}")
    return pages


def ocr_pages(pages, processor, model, device: str, max_new_tokens: int = 1024) -> List[str]:
    """Run Nougat OCR on each page and return markdown strings."""
    results: List[str] = []

    for img in tqdm(pages, desc="extracting", total=len(pages)):
        pixel_values = processor(images=img, return_tensors="pt").pixel_values.to(device)
        with torch.no_grad():
            out = model.generate(pixel_values, max_new_tokens=max_new_tokens)

        seq = processor.batch_decode(out, skip_special_tokens=True)[0]
        seq = processor.post_process_generation(seq, fix_markdown=True)
        results.append(seq)

    return results


def save_markdown(pages_md: List[str], out_path: Path):
    """Save all page markdown into a single .md file with simple separators."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n\n\n".join(
        f"<!-- Page {i+1} -->\n\n{md}" for i, md in enumerate(pages_md)
    )
    out_path.write_text(text, encoding="utf-8")
    print(f"[info] Saved Markdown to: {out_path}")


# ------------------------- CLI ------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Convert exam PDFs to Markdown using Nougat (simple version)."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a PDF file or a folder containing PDF files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output folder for Markdown files.",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=None,
        help="(Optional) Limit the number of pages per PDF (for quick tests).",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdfs = find_pdfs(input_path)
    if not pdfs:
        print("[info] No PDF files found; nothing to do.")
        return

    print(f"[info] Found {len(pdfs)} PDF file(s).")

    processor, model, device = load_nougat()

    for i, pdf in enumerate(pdfs, 1):
        try:
            print(f"\n[{i}/{len(pdfs)}] Processing {pdf.name}")
            pages = pdf_to_images(pdf, dpi=300, max_pages=args.max_pages)
            pages_md = ocr_pages(pages, processor, model, device, max_new_tokens=1024)

            out_path = output_dir / f"{pdf.stem}.md"
            save_markdown(pages_md, out_path)
        except Exception as e:  # noqa: BLE001
            print(f"[error] Failed to process {pdf}: {e}")


if __name__ == "__main__":
    main()
