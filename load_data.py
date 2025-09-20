# import os
# import json
# from pathlib import Path
# from PIL import Image
# from transformers import BlipProcessor, BlipForConditionalGeneration

# # -------------------------
# # Model setup (BLIP captioning)
# # -------------------------
# CAPTION_MODEL = "Salesforce/blip-image-captioning-base"  # BLIP base
# processor = BlipProcessor.from_pretrained(CAPTION_MODEL)
# model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)

# # -------------------------
# # Paths
# # -------------------------
# BASE_DIR = Path("rag_data")
# OUTPUT_FILE = BASE_DIR / "unified_documents.jsonl"

# # Metadata sources
# METADATA_FILES = {
#     "excel": BASE_DIR / "excel_metadata.jsonl",
#     "pdf": BASE_DIR / "pdf_metadata.jsonl"
# }

# # -------------------------
# # Helpers
# # -------------------------
# def load_metadata():
#     """Load metadata from JSONL files into a dict keyed by 'source' field."""
#     metadata = {}
#     for source_type, file_path in METADATA_FILES.items():
#         if file_path.exists():
#             with open(file_path, "r", encoding="utf-8") as f:
#                 for line in f:
#                     try:
#                         entry = json.loads(line)
#                         key = entry.get("source")
#                         if key:
#                             metadata[key] = entry
#                     except json.JSONDecodeError:
#                         continue
#     return metadata

# def caption_image(image_path: Path) -> str:
#     """Generate a caption for an image using BLIP."""
#     try:
#         image = Image.open(image_path).convert("RGB")
#         inputs = processor(image, return_tensors="pt")
#         out = model.generate(**inputs, max_new_tokens=30)
#         caption = processor.decode(out[0], skip_special_tokens=True)
#         return caption
#     except Exception as e:
#         return f"[Error captioning image {image_path.name}: {str(e)}]"

# def process_tables(folder: Path, metadata: dict, source_type: str):
#     """Process all CSV tables inside subfolders and attach metadata."""
#     docs = []
#     if not folder.exists():
#         return docs

#     for subfolder in folder.iterdir():
#         if not subfolder.is_dir():
#             continue
#         for idx, csv_file in enumerate(subfolder.glob("*.csv")):
#             try:
#                 content = csv_file.read_text(encoding="utf-8")
#             except Exception:
#                 content = f"[Error reading CSV {csv_file.name}]"

#             source_key = f"{source_type}:{subfolder.name}/{csv_file.name}"
#             doc = {
#                 "id": f"{source_type}_table_{subfolder.name}_{idx+1}",
#                 "content": content,
#                 "source": str(csv_file),
#                 "type": "table",
#                 "metadata": metadata.get(source_key, {
#                     "sheet_name": subfolder.name,
#                     "source_file": csv_file.name
#                 })
#             }
#             docs.append(doc)
#     return docs

# def process_images(folder: Path, metadata: dict, source_type: str):
#     """Process all PNG images inside subfolders, generate captions, attach parent metadata."""
#     docs = []
#     if not folder.exists():
#         return docs

#     for subfolder in folder.iterdir():
#         if not subfolder.is_dir():
#             continue
#         for idx, img_file in enumerate(subfolder.glob("*.png")):
#             caption = caption_image(img_file)
#             # Print caption in real time
#             print(f"[{source_type.upper()} IMAGE] {img_file.name} → {caption}")

#             # Use parent folder metadata if exists, else minimal
#             parent_key = f"{source_type}:{subfolder.name}"
#             image_metadata = metadata.get(parent_key, {"sheet_name": subfolder.name})
#             image_metadata["source_image"] = img_file.name

#             doc = {
#                 "id": f"{source_type}_image_{subfolder.name}_{idx+1}",
#                 "content": caption,
#                 "source": str(img_file),
#                 "type": "image",
#                 "metadata": image_metadata
#             }
#             docs.append(doc)
#     return docs

# def build_unified_dataset():
#     """Main pipeline to unify Excel + PDF data into one JSONL dataset."""
#     metadata = load_metadata()
#     documents = []

#     # Excel
#     documents.extend(process_tables(BASE_DIR / "excel_tables", metadata, "excel"))
#     documents.extend(process_images(BASE_DIR / "excel_images", metadata, "excel"))

#     # PDF
#     documents.extend(process_tables(BASE_DIR / "pdf_tables", metadata, "pdf"))
#     documents.extend(process_images(BASE_DIR / "pdf_images", metadata, "pdf"))

#     # Save to JSONL
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#         for doc in documents:
#             f.write(json.dumps(doc, ensure_ascii=False) + "\n")

#     print(f"✅ Unified dataset built with {len(documents)} documents at {OUTPUT_FILE}")

# # -------------------------
# # Entry point
# # -------------------------
# if __name__ == "__main__":
#     build_unified_dataset()



import json
from pathlib import Path

# Path to your JSONL file
UNIFIED_FILE = Path("rag_data/unified_documents.jsonl")

with open(UNIFIED_FILE, "r", encoding="utf-8") as f:
    for line in f:
        try:
            doc = json.loads(line)
            # Pretty-print each document
            print(json.dumps(doc, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            continue
