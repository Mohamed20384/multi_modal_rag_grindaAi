import json
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# CONFIG
# =========================
INPUT_JSONL = "rag_data/unified_documents.jsonl"
OUTPUT_JSONL = "rag_data/chunked_documents.jsonl"

# Chunker config
chunk_size = 1000         # max characters per chunk
chunk_overlap = 200       # overlap between chunks

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
)

# =========================
# LOAD INPUT JSONL
# =========================
input_path = Path(INPUT_JSONL)
output_path = Path(OUTPUT_JSONL)

chunked_data = []

with input_path.open("r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        content = item.get("content", "")

        # Split content into chunks
        chunks = text_splitter.split_text(content)

        # Save each chunk as a new record
        for i, chunk_text in enumerate(chunks, start=1):
            chunked_item = {
                "id": f"{item.get('id')}_chunk{i}",
                "content": chunk_text,
                "source": item.get("source"),
                "type": item.get("type"),
                "metadata": item.get("metadata", {})
            }
            chunked_data.append(chunked_item)

# =========================
# SAVE CHUNKED JSONL
# =========================
with output_path.open("w", encoding="utf-8") as f:
    for item in chunked_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Chunking complete! Total chunks: {len(chunked_data)}")
print(f"Saved to {OUTPUT_JSONL}")
