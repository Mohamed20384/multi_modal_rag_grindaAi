#!/usr/bin/env python3
"""
pdf_extract_with_citation_chunked.py

Extracts tables and images from PDFs with metadata (page number, bbox, file path).
Produces:
 - ./rag_data_last/pdf_tables/<pdfname>/table_pX_iY.csv
 - ./rag_data_last/pdf_images/<pdfname>/image_pX_iY.png
 - ./rag_data_last/pdf_metadata.jsonl (list of dicts with citation info + text chunks)
Also splits table text into chunks for RAG ingestion:
 - max 4000 chars per chunk
 - combine chunks < 2000 chars
"""

import os
import json
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pdfplumber
import fitz  # PyMuPDF

# ---------- CONFIG -----------
RAG_FILES_DIR = Path("./Data")
OUT_DIR = Path("./rag_data_last")
OUT_TABLES = OUT_DIR / "pdf_tables"
OUT_IMAGES = OUT_DIR / "pdf_images"
METADATA_PATH = OUT_DIR / "pdf_metadata.jsonl"

# chunking config
MAX_CHARS = 4000
NEW_AFTER_N_CHARS = 4000
COMBINE_UNDER_N_CHARS = 2000
# -----------------------------

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def df_to_csv_safe(df: pd.DataFrame, path: Path):
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def chunk_text(text: str):
    """
    Split text into chunks according to config
    """
    chunks = []
    current = ""
    for part in text.split("\n"):
        if len(current) + len(part) + 1 > MAX_CHARS:
            if len(current) >= NEW_AFTER_N_CHARS:
                chunks.append(current)
                current = ""
        current += part + "\n"
    if current:
        chunks.append(current)
    # combine small chunks
    merged = []
    buf = ""
    for c in chunks:
        if len(c) < COMBINE_UNDER_N_CHARS:
            buf += c
        else:
            if buf:
                merged.append(buf)
                buf = ""
            merged.append(c)
    if buf:
        merged.append(buf)
    return merged

def extract_tables(pdf_path: Path, out_dir: Path):
    """Extract tables with pdfplumber. Returns list of metadata dicts."""
    tables_meta = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            try:
                tables = page.extract_tables()
            except Exception:
                tables = []
            for i, table in enumerate(tables):
                df = pd.DataFrame(table)
                fname = f"{pdf_path.stem}__page{page_num}_table{i}.csv"
                out_path = out_dir / pdf_path.stem / fname
                df_to_csv_safe(df, out_path)
                # chunk text
                txt = "\n".join(df.astype(str).apply(lambda r: "\t".join(r), axis=1))
                chunks = chunk_text(txt)
                meta = {
                    "type": "table",
                    "pdf": str(pdf_path),
                    "page": page_num,
                    "bbox": page.bbox,
                    "rows": len(df),
                    "cols": len(df.columns),
                    "csv_path": str(out_path),
                    "chunks": chunks
                }
                tables_meta.append(meta)
    return tables_meta

def extract_images(pdf_path: Path, out_dir: Path):
    """Extract images with PyMuPDF. Returns list of metadata dicts."""
    images_meta = []
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        img_list = page.get_images(full=True)
        for i, img in enumerate(img_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            # Save file
            out_dir_page = out_dir / pdf_path.stem
            ensure_dir(out_dir_page)
            fname = f"{pdf_path.stem}__page{page_index+1}_img{i}.{image_ext}"
            out_path = out_dir_page / fname
            with open(out_path, "wb") as f:
                f.write(image_bytes)
            # Try to get bbox (rect) if referenced by page
            bbox = None
            for img_block in page.get_image_info(xrefs=[xref]):
                bbox = img_block.get("bbox")
            meta = {
                "type": "image",
                "pdf": str(pdf_path),
                "page": page_index+1,
                "bbox": bbox,
                "image_path": str(out_path),
            }
            images_meta.append(meta)
    return images_meta

def process_all_pdfs(rag_dir: Path, out_tables: Path, out_images: Path, metadata_path: Path):
    ensure_dir(out_tables)
    ensure_dir(out_images)
    if metadata_path.exists():
        metadata_path.unlink()

    all_meta = []
    pdf_files = list(rag_dir.glob("**/*.pdf"))
    for pdf in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            table_metas = extract_tables(pdf, out_tables)
            image_metas = extract_images(pdf, out_images)
            with open(metadata_path, "a", encoding="utf-8") as fo:
                for m in (table_metas + image_metas):
                    fo.write(json.dumps(m, ensure_ascii=False) + "\n")
                    all_meta.append(m)
        except Exception as e:
            print(f"[WARN] Failed to process {pdf}: {e}")
    print(f"Done. Extracted {len(all_meta)} items from {len(pdf_files)} PDFs")
    return all_meta

if __name__ == "__main__":
    ensure_dir(OUT_DIR)
    metas = process_all_pdfs(RAG_FILES_DIR, OUT_TABLES, OUT_IMAGES, METADATA_PATH)
