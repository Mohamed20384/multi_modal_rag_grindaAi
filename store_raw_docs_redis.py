"""
Store unified documents into Redis Cloud (raw storage).
"""

import os
import json
from pathlib import Path
import test_redis
from dotenv import load_dotenv

# ---------------------
# Config
# ---------------------
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "redis-15760.c73.us-east-1-2.ec2.redns.redis-cloud.com")
REDIS_PORT = int(os.getenv("REDIS_PORT", 15760))
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "Mohamed")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "xMUduYuIxYEpcYcOka0cpRxWG06puTec")
REDIS_DB = int(os.getenv("REDIS_DB", 0))

UNIFIED_FILE = Path("rag_data/unified_documents.jsonl")

# ---------------------
# Connect
# ---------------------
r = test_redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    username=os.getenv("REDIS_USERNAME", "default"),
    password=REDIS_PASSWORD,
    db=REDIS_DB,
    decode_responses=True
)

# ---------------------
# Test connection
# ---------------------
print("🔌 Testing Redis Cloud connection...")
success = r.set("foo", "bar")
print("Set test key:", success)
print("Get test key:", r.get("foo"))

# ---------------------
# Store docs
# ---------------------
if UNIFIED_FILE.exists():
    with open(UNIFIED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            doc_id = doc["id"]

            # Convert all values to JSON strings (safe for nested dicts/lists)
            flat_doc = {k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else str(v) 
                        for k, v in doc.items()}

            r.hset(f"doc:{doc_id}", mapping=flat_doc)

    print(f"✅ Stored raw documents in Redis Cloud at {REDIS_HOST}:{REDIS_PORT}")
else:
    print(f"⚠️ File not found: {UNIFIED_FILE}")

