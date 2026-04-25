"""Fast-path PDF -> markdown extractor using pypdfium2.

Drops in for the docling-based extractor on plain-text textbooks where layout
analysis is unnecessary. pypdfium2 reads pages directly from the master PDF
(no map-phase split, no temp files), so wall-time is bounded by raw text
extraction throughput. Section headings are promoted to `## N.N …` so the
existing markdown section splitter still works downstream.
"""
from __future__ import annotations

import argparse
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Tuple


_HEADING_RE = re.compile(
    r"^(\d+(?:\.\d+)+)\s+([A-Z][^\n]{2,80}?)[ \t]*$",
    re.MULTILINE,
)
_TOC_RE = re.compile(r"^\d+(?:\.\d+)+\s+[^\n]+?\s+\d+\s*$", re.MULTILINE)


def _postprocess_page(text: str) -> str:
    text = re.sub(r"^\s*Page\s+\d+\s*\n", "", text, count=1)
    text = _TOC_RE.sub("", text)
    text = _HEADING_RE.sub(lambda m: f"## {m.group(1)} {m.group(2)}", text)
    return text


def process_pdf_range(args: Tuple[str, int, int]) -> dict:
    # Worker imports inside the function so the child process pays the load
    # cost once, not via pickle.
    import pypdfium2 as pdfium

    master_path, start, end = args
    pdf = pdfium.PdfDocument(master_path)
    parts: List[str] = []
    for i in range(start, end):
        raw = pdf[i].get_textpage().get_text_range()
        parts.append(_postprocess_page(raw))
        parts.append(f"\n\n--- Page {i + 1} ---\n\n")
    pdf.close()
    return {"offset": start, "markdown": "".join(parts)}


def plan_page_ranges(master_path: str, pages_per_chunk: int) -> Tuple[List[Tuple[str, int, int]], int]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(master_path)
    total = len(pdf)
    pdf.close()
    ranges = [
        (master_path, start, min(start + pages_per_chunk, total))
        for start in range(0, total, pages_per_chunk)
    ]
    return ranges, total


def extract_to_markdown(pdf_path: str, output_path: str, *, chunk_size: int = 500, workers: int = 4) -> dict:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    t_all = time.perf_counter()
    ranges, total = plan_page_ranges(pdf_path, chunk_size)

    t_map = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(process_pdf_range, ranges):
            results.append(r)
    t_map_end = time.perf_counter()

    results.sort(key=lambda x: x["offset"])
    full = "".join(r["markdown"] for r in results)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full)
    t_end = time.perf_counter()

    return {
        "pdf": pdf_path,
        "output": output_path,
        "pages": total,
        "ranges": len(ranges),
        "workers": workers,
        "map_seconds": round(t_map_end - t_map, 3),
        "wall_seconds": round(t_end - t_all, 3),
        "markdown_chars": len(full),
    }


def main():
    ap = argparse.ArgumentParser(description="pypdfium2-based fast PDF extractor.")
    ap.add_argument("--pdf", required=True, help="Path to input PDF.")
    ap.add_argument("--out", required=True, help="Path to output markdown file.")
    ap.add_argument("--chunk_size", type=int, default=500, help="Pages per worker range.")
    ap.add_argument("--workers", type=int, default=4, help="ProcessPoolExecutor workers.")
    args = ap.parse_args()

    stats = extract_to_markdown(
        args.pdf, args.out, chunk_size=args.chunk_size, workers=args.workers,
    )
    print(f"Pages              : {stats['pages']}")
    print(f"Ranges             : {stats['ranges']} (workers={stats['workers']})")
    print(f"Map-phase wall     : {stats['map_seconds']}s")
    print(f"Total wall         : {stats['wall_seconds']}s")
    print(f"Markdown size      : {stats['markdown_chars']:,} chars")
    print(f"Output             : {stats['output']}")


if __name__ == "__main__":
    main()
