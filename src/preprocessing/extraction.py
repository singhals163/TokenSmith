import argparse
import json
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

from pypdf import PdfReader, PdfWriter

from src.profiler import TimerBlock, print_profile_stats, timeit


def _import_docling():
    # Lazy import: docling pulls ~100 MB of shared libs and tokenizers; downstream
    # consumers (index_builder, pytest) that only import sibling helpers from this
    # module should not pay that cost.
    from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import (
        DocumentConverter,
        InputFormat,
        PdfFormatOption,
    )
    return PdfPipelineOptions, DocumentConverter, PdfFormatOption, InputFormat, DoclingParseV2DocumentBackend


@timeit(name="Extraction: extract_sections_from_markdown")
def extract_sections_from_markdown(
    file_path: str,
    exclusion_keywords: List[str] = None,
) -> List[Dict]:
    """Chunk a markdown file into sections delimited by `## N.N …` headings."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

    heading_pattern = r"(?=^## \d+(\.\d+)* .*)"
    numbering_pattern = re.compile(r"(\d+(?:\.\d+)*)")
    chunks = re.split(heading_pattern, content, flags=re.MULTILINE)

    sections = []
    if chunks[0].strip():
        sections.append({"heading": "Introduction", "content": chunks[0].strip()})

    for chunk in chunks[1:]:
        if not chunk or not chunk.strip():
            continue
        parts = chunk.split("\n", 1)
        heading = parts[0].strip().lstrip("#").strip()
        heading = f"Section {heading}"

        if exclusion_keywords and any(k.lower() in heading.lower() for k in exclusion_keywords):
            continue

        section_content = parts[1].strip() if len(parts) > 1 else ""
        if not section_content:
            continue
        section_content = preprocess_extracted_section(section_content)

        match = numbering_pattern.search(heading)
        if match:
            section_number = match.group(1)
            current_level = section_number.count(".") + 1
            try:
                chapter_num = int(section_number.split(".")[0])
            except ValueError:
                chapter_num = 0
        else:
            current_level = 1
            chapter_num = 0

        sections.append({
            "heading": heading,
            "content": section_content,
            "level": current_level,
            "chapter": chapter_num,
        })

    return sections


@timeit(name="Extraction: extract_index_with_range_expansion")
def extract_index_with_range_expansion(text_content):
    """Extract keywords and page numbers from the raw text of a book index."""
    text_content = re.sub(r"\\", "", text_content)
    text_content = re.sub(r"--- PAGE \d+ ---", "", text_content)
    text_content = re.sub(r"^\d+\s+Index\s*$", "", text_content, flags=re.MULTILINE)
    text_content = re.sub(r"^Index\s+\d+\s*$", "", text_content, flags=re.MULTILINE)

    pattern = re.compile(r"^(.*?),\s*([\d,\s\-]+?)(?=\n[A-Za-z]|\Z)", re.MULTILINE | re.DOTALL)
    index_data = {}

    for match in pattern.finditer(text_content):
        keyword = match.group(1).strip().replace("\n", " ")
        page_numbers_str = match.group(2).strip().replace("\n", " ")

        if keyword.lower() in ["mc", "graw", "hill", "education"]:
            continue

        pages = []
        for part in re.split(r",\s*", page_numbers_str):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                try:
                    start_str, end_str = part.split("-")
                    pages.extend(range(int(start_str), int(end_str) + 1))
                except ValueError:
                    pass
            else:
                try:
                    pages.append(int(part))
                except ValueError:
                    pass

        if keyword and pages:
            index_data.setdefault(keyword, []).extend(pages)

    return json.dumps(index_data, indent=2)


@timeit(name="Extraction: preprocess_extracted_section")
def preprocess_extracted_section(text: str) -> str:
    text = text.replace("\n", " ").replace("", " ").replace("**", "")
    return " ".join(text.split())


@timeit(name="Phase 1 Map: Split PDF (pypdf)")
def split_pdf(file_path: str, chunk_size: int = 50) -> List[dict]:
    """Split a PDF into page-range chunks written to a temp dir for worker consumption."""
    print(f"\n[Map Phase] Splitting '{Path(file_path).name}' into {chunk_size}-page chunks...")
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    temp_dir = tempfile.mkdtemp()
    chunks = []

    for start_page in range(0, total_pages, chunk_size):
        end_page = min(start_page + chunk_size, total_pages)
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        chunk_path = os.path.join(temp_dir, f"chunk_{start_page}_{end_page}.pdf")
        with open(chunk_path, "wb") as f:
            writer.write(f)
        chunks.append({"path": chunk_path, "start_page_offset": start_page, "end_page": end_page})

    print(f"[Map Phase] Successfully created {len(chunks)} chunks.")
    return chunks


@timeit(name="Phase 2 Execute: Docling Worker")
def process_pdf_chunk(chunk_info: dict) -> dict:
    """Worker: convert a single PDF chunk to markdown via docling."""
    (PdfPipelineOptions, DocumentConverter, PdfFormatOption, InputFormat,
     DoclingParseV2DocumentBackend) = _import_docling()
    input_file_path = chunk_info["path"]
    offset = chunk_info["start_page_offset"]
    end_page = chunk_info["end_page"]
    log_prefix = f"[Pages {offset}-{end_page}]"

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseV2DocumentBackend,
            )
        }
    )

    try:
        print(f"{log_prefix} Analyzing layout and structure...")
        result = converter.convert(Path(input_file_path))
        doc = result.document
    except Exception as e:
        print(f"{log_prefix} Error during conversion: {e}", file=sys.stderr)
        return {"offset": offset, "markdown": ""}

    num_pages_in_chunk = len(doc.pages)
    print(f"{log_prefix} Layout parsed. Serializing {num_pages_in_chunk} pages to Markdown...")

    output_parts = []
    for i in range(1, num_pages_in_chunk + 1):
        actual_page_num = offset + i
        print(f"{log_prefix} -> Exporting page {actual_page_num}...")
        output_parts.append(doc.export_to_markdown(page_no=i))
        output_parts.append(f"\n\n--- Page {actual_page_num} ---\n\n")

    print(f"{log_prefix} Worker finished successfully.")
    return {"offset": offset, "markdown": "".join(output_parts)}


def main():
    parser = argparse.ArgumentParser(description="Parallel PDF extraction (docling).")
    parser.add_argument("--workers", type=int, default=1, help="Number of CPU workers")
    parser.add_argument("--chunk_size", type=int, default=50, help="Pages per chunk")
    parser.add_argument("--out_dir", type=str, default="data/extracted", help="Output directory")
    parser.add_argument("--profile_out", type=str, default="profiling.txt", help="Profile-stats file")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    chapters_dir = project_root / "data/chapters"
    pdfs = sorted(chapters_dir.glob("*.pdf"))

    if not pdfs:
        print("ERROR: No PDFs found in data/chapters/. Please copy a PDF there first.", file=sys.stderr)
        sys.exit(1)

    out_dir = project_root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    markdown_files = []

    with TimerBlock("[Block] Total Parallel Extraction Pipeline"):
        for pdf_path in pdfs:
            pdf_name = pdf_path.stem
            output_md = out_dir / f"{pdf_name}--extracted_markdown.md"

            print(f"\n{'=' * 60}")
            print(f"STARTING EXTRACTION: {pdf_name}")
            print(f"Workers: {args.workers} | Chunk Size: {args.chunk_size} pages")
            print(f"Output Directory: {out_dir}")
            print(f"{'=' * 60}")

            chunks = split_pdf(str(pdf_path), chunk_size=args.chunk_size)

            results = []
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                for result in executor.map(process_pdf_chunk, chunks):
                    results.append(result)

            print("\n[Reduce Phase] Merging worker outputs and writing to disk...")
            results.sort(key=lambda x: x["offset"])
            full_document_markdown = "".join(res["markdown"] for res in results)

            try:
                with open(output_md, "w", encoding="utf-8") as f:
                    f.write(full_document_markdown)
                markdown_files.append(output_md)
            except Exception as e:
                print(f"Error writing to file {output_md}: {e}", file=sys.stderr)

            for chunk in chunks:
                try:
                    os.remove(chunk["path"])
                except OSError:
                    pass

    if markdown_files:
        print("\nStarting Section Extraction...")
        extracted_sections = extract_sections_from_markdown(str(markdown_files[0]))
        if extracted_sections:
            output_json = out_dir / "extracted_sections.json"
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(extracted_sections, f, indent=4, ensure_ascii=False)
            print(f"Extracted content saved to '{output_json}'")

    print(f"\nSaving profiling metrics to {args.profile_out}...")
    profile_path = out_dir / args.profile_out
    print_profile_stats(filepath=str(profile_path))
    print(f"Experiment complete. Results in {out_dir}")


if __name__ == "__main__":
    main()
