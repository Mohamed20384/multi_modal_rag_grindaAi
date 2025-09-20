import os
import json
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------
# Load environment variables
# -------------------
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# -------------------
# Paths
# -------------------
UNIFIED_FILE = Path("rag_data/chunked_documents.jsonl")
SUMMARY_FILE = Path("rag_data/summaries.jsonl")

# -------------------
# Initialize model
# -------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    api_key=google_api_key
)

# -------------------
# Summarization function
# -------------------
def summarize_doc(doc):
    prompt = f"""
    Summarize the following document in Korean in 2-3 sentences. 
    Focus on key meaning and data, not formatting.

    Content Type: {doc['type']}
    Content: {doc['content']}
    """
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"[Summarization failed: {str(e)}]"

# -------------------
# Process unified documents
# -------------------
def generate_summaries():
    if not UNIFIED_FILE.exists():
        raise FileNotFoundError(f"❌ Unified file not found at {UNIFIED_FILE}")

    with open(UNIFIED_FILE, "r", encoding="utf-8") as f, open(SUMMARY_FILE, "w", encoding="utf-8") as out:
        for line_num, line in enumerate(f, start=1):
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                print(f"⚠️ Skipping invalid JSON at line {line_num}")
                continue

            # Skip documents of type 'image'
            if doc.get("type", "").lower() == "image":
                print(f"⏭ Skipping image document at line {line_num}")
                continue

            summary = summarize_doc(doc)
            summary_entry = {
                "id": doc.get("id", f"doc_{line_num}"),
                "summary": summary,
                "metadata": doc.get("metadata", {})
            }

            out.write(json.dumps(summary_entry, ensure_ascii=False) + "\n")
            print(f"✅ Processed {summary_entry['id']}")

    print(f"\n🎉 Summaries saved to {SUMMARY_FILE}")

# -------------------
# Entry point
# -------------------
if __name__ == "__main__":
    generate_summaries()
