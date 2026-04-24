"""Minimal pypdfium2-based parallel PDF extractor — mirrors the
src/preprocessing/extraction.py Phase-1-v2 fast path (heading promotion,
TOC stripping) but without the heavy docling/langchain/pydantic imports,
which cold-read for minutes on Lustre.

Usage:
    python tools/extract_pypdfium.py \
        --pdf data/chapters/textbook.pdf \
        --chunk_size 500 \
        --workers 4 \
        --out data/extracted_pypdfium_rtx6000/textbook--extracted_markdown.md
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


# Must start the line, be followed by a capitalized title, and not end in
# sentence punctuation (avoids matching cross-reference lines like "see 1.2").
_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+)+)\s+([A-Z][^\n]{2,80}?)[ \t]*$",
    re.MULTILINE,
)

# TOC entries look like "1.2 Database-System Applications 23" — ending in a page
# number. We strip these so they don't trigger a spurious `## 1.2 …` heading
# in the markdown.
_TOC_RE = re.compile(r"^\d+(?:\.\d+)+\s+[^\n]+?\s+\d+\s*$", re.MULTILINE)


def _postprocess_page_pypdfium(text: str) -> str:
    text = re.sub(r"^\s*Page\s+\d+\s*\n", "", text, count=1)
    text = _TOC_RE.sub("", text)
    text = _HEADING_RE.sub(lambda m: f"## {m.group(1)} {m.group(2)}", text)
    return text


def process_pdf_range_pypdfium(args: tuple) -> dict:
    import pypdfium2 as pdfium
    master_path, start, end = args
    pdf = pdfium.PdfDocument(master_path)
    parts: list[str] = []
    for i in range(start, end):
        raw = pdf[i].get_textpage().get_text_range()
        md = _postprocess_page_pypdfium(raw)
        parts.append(md)
        parts.append(f"\n\n--- Page {i + 1} ---\n\n")
    pdf.close()
    return {"offset": start, "markdown": "".join(parts)}


def plan_page_ranges(master_path: str, pages_per_chunk: int):
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(master_path)
    total = len(pdf)
    pdf.close()
    ranges = []
    for start in range(0, total, pages_per_chunk):
        ranges.append((master_path, start, min(start + pages_per_chunk, total)))
    return ranges, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--chunk_size", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    ranges, total = plan_page_ranges(args.pdf, args.chunk_size)
    print(f"Planned {len(ranges)} ranges over {total} pages, workers={args.workers}")

    t_map = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for r in ex.map(process_pdf_range_pypdfium, ranges):
            results.append(r)
    t_map_end = time.perf_counter()

    results.sort(key=lambda x: x["offset"])
    full = "".join(r["markdown"] for r in results)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(full)

    t_end = time.perf_counter()
    print(f"Extraction map phase   : {t_map_end - t_map:.2f}s")
    print(f"Total wall time        : {t_end - t_all:.2f}s")
    print(f"Pages                  : {total}")
    print(f"Output                 : {args.out}")
    print(f"Markdown size          : {len(full):,} chars")
    heading_count = sum(1 for ln in full.splitlines() if re.match(r"^## \d+(?:\.\d+)+", ln))
    print(f"Promoted ## headings   : {heading_count}")


if __name__ == "__main__":
    main()
