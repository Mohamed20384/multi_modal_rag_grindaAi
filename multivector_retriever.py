import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# =========================
# CONFIG
# =========================
load_dotenv()
RAW_FILE = Path("rag_data/chunked_documents.jsonl")
SUMMARY_FILE = Path("rag_data/summaries.jsonl")

# Redis credentials
REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

REDIS_URL = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Chroma
CHROMA_DIR = "chroma_store"

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# =========================
# Redis Client + Store
# =========================
redis_client = get_client(REDIS_URL)
redis_store = RedisStore(client=redis_client, namespace="docs")

# =========================
# Chroma Vector Store
# =========================
chroma = Chroma(
    collection_name="summaries",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

# =========================
# Multi-Vector Retriever
# =========================
retriever = MultiVectorRetriever(
    vectorstore=chroma,
    docstore=redis_store,
    id_key="doc_id"
)

# =========================
# Load and store documents (skip images)
# =========================
if RAW_FILE.exists() and SUMMARY_FILE.exists():
    with open(RAW_FILE, "r", encoding="utf-8") as raw_f, open(SUMMARY_FILE, "r", encoding="utf-8") as sum_f:
        for raw_line, sum_line in zip(raw_f, sum_f):
            raw_doc = json.loads(raw_line)

            # Skip documents that are images
            if raw_doc.get("type") == "image":
                print(f"⚠️ Skipping image document: {raw_doc.get('id')}")
                continue

            sum_doc = json.loads(sum_line)
            doc_id = raw_doc["id"]

            # Store raw content in Redis
            redis_store.mset([(doc_id, json.dumps(raw_doc, ensure_ascii=False))])

            # Store summary in Chroma
            retriever.vectorstore.add_texts(
                texts=[sum_doc["summary"]],
                metadatas=[{"doc_id": doc_id}]
            )

            print(f"Stored document with doc_id: {doc_id}")

    print("✅ All documents stored in Redis and summaries stored in Chroma.")
else:
    if not RAW_FILE.exists():
        print(f"⚠️ File not found: {RAW_FILE}")
    if not SUMMARY_FILE.exists():
        print(f"⚠️ File not found: {SUMMARY_FILE}")
