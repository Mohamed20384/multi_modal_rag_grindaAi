# index_summaries_chroma.py
import os, json
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma

# ---------- config ----------
load_dotenv()
SUMMARY_FILE = Path("rag_data/summaries.jsonl")
PERSIST_DIR = "rag_data/chroma"
COLLECTION_NAME = "summaries"

# ---------- load summaries ----------
docs = []
with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        docs.append({
            "id": entry["id"],
            "text": entry["summary"],
            "metadata": entry.get("metadata", {})
        })

if not docs:
    raise SystemExit("No summaries found in rag_data/summaries.jsonl")

texts = [d["text"] for d in docs]
ids = [d["id"] for d in docs]
metadatas = [{"id": d["id"], **(d["metadata"] or {})} for d in docs]

# ---------- embeddings & index ----------
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise SystemExit("❌ GOOGLE_API_KEY not found in .env")

emb = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    api_key=google_api_key
)

db = Chroma.from_texts(
    texts=texts,
    embedding=emb,
    metadatas=metadatas,
    ids=ids,
    collection_name=COLLECTION_NAME,
    persist_directory=PERSIST_DIR
)

db.persist()
print(f"✅ Indexed {len(texts)} summaries into Chroma at {PERSIST_DIR} (collection: {COLLECTION_NAME})")
