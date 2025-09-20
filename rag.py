import os
import json
import re
import math
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_community.storage import RedisStore
from langchain_community.utilities.redis import get_client
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.schema import Document

# =========================
# CONFIG
# =========================
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")
REDIS_USERNAME = os.getenv("REDIS_USERNAME", "default")
REDIS_DB = int(os.getenv("REDIS_DB", 0))
REDIS_URL = f"redis://{REDIS_USERNAME}:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

CHROMA_DIR = "chroma_store"
COLLECTION_NAME = "summaries"

# Embeddings used elsewhere (compressor uses the same instance)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# =========================
# Redis + Chroma Init
# =========================
redis_client = get_client(REDIS_URL)
redis_store = RedisStore(client=redis_client, namespace="docs")

chroma = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=None
)

multi_retriever = MultiVectorRetriever(
    vectorstore=chroma,
    docstore=redis_store,
    id_key="doc_id"
)

vector_retriever = chroma.as_retriever(search_kwargs={"k": 40})

# =========================
# Custom Keyword Retriever
# =========================
class SimpleKeywordRetriever:
    def __init__(self, redis_store, k=20):
        self.redis_store = redis_store
        self.k = k

    def get_relevant_documents(self, query: str):
        results = []
        pattern = re.compile(re.escape(query), re.IGNORECASE)

        keys = self.redis_store.client.keys(f"{self.redis_store.namespace}:*")
        for key in keys:
            raw_json = self.redis_store.client.get(key)
            if not raw_json:
                continue
            doc = json.loads(raw_json)
            content = doc.get("content", "")
            if pattern.search(content):
                results.append(Document(page_content=content, metadata={"doc_id": key.decode().split(":")[-1]}))
                if len(results) >= self.k:
                    break
        return results

keyword_retriever = SimpleKeywordRetriever(redis_store, k=40)

# =========================
# Hybrid Retriever
# =========================
class HybridRetriever:
    def __init__(self, vector_retriever, keyword_retriever):
        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever

    def get_relevant_documents(self, query):
        vector_docs = self.vector_retriever.get_relevant_documents(query)
        keyword_docs = self.keyword_retriever.get_relevant_documents(query)

        merged, seen_ids = [], set()
        for d in vector_docs + keyword_docs:
            doc_id = d.metadata.get("doc_id")
            if doc_id and doc_id not in seen_ids:
                merged.append(d)
                seen_ids.add(doc_id)
        return merged

hybrid_retriever = HybridRetriever(vector_retriever, keyword_retriever)

# =========================
# Manual Compression
# =========================
compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.6)

def compress_docs(query, docs):
    return compressor.compress_documents(docs, query=query)

# =========================
# Gemini
# =========================
gemini = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# =========================
# Advanced (non-invasive) RAG enhancements
# - Re-rank compressed docs using embeddings similarity
# - Extract concise snippet (best sentence) per doc
# - Compute a simple confidence score
# =========================

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Use your embeddings object to embed documents.
    This expects the GoogleGenerativeAIEmbeddings instance to support .embed_documents
    (if different in your environment, swap to the correct method).
    """
    if not texts:
        return []
    # embeddings.embed_documents should accept list[str] and return list[list[float]]
    vecs = embeddings.embed_documents(texts)
    return [np.array(v, dtype=np.float32) for v in vecs]

def embed_query(query: str) -> np.ndarray:
    # embeddings.embed_query returns a list/array; wrap as numpy
    qvec = embeddings.embed_query(query)
    return np.array(qvec, dtype=np.float32)

def get_best_snippet(text: str, query: str, max_len=400) -> str:
    """
    Find the best sentence matching query terms; fallback to first max_len chars.
    """
    # split into sentences (simple)
    sentences = re.split(r'(?<=[.!?。\n])\s+', text.strip())
    terms = re.findall(r'\w+', query)
    if terms:
        term_pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
        # rank sentences by number of term matches and sentence length proximity
        scored = []
        for s in sentences:
            matches = len(term_pattern.findall(s))
            if matches > 0:
                scored.append((matches, s.strip()))
        if scored:
            # choose sentence with highest matches (and not too long)
            scored.sort(key=lambda x: (-x[0], len(x[1])))
            best = scored[0][1]
            # truncate if necessary
            return best if len(best) <= max_len else best[:max_len].rsplit(' ', 1)[0] + "..."
    # fallback: return the beginning of the document
    plain = re.sub(r'\s+', ' ', text).strip()
    return plain[:max_len].rsplit(' ', 1)[0] + ("..." if len(plain) > max_len else "")

def rerank_and_prepare(query: str, docs: List[Document], top_n: int = 5):
    """
    Re-rank compressed documents by embedding similarity to the query,
    add snippet and score metadata, and return top_n documents.
    """
    if not docs:
        return []

    texts = [d.page_content for d in docs]
    # compute embeddings
    qvec = embed_query(query)
    doc_vecs = embed_texts(texts)

    scores = []
    for i, dv in enumerate(doc_vecs):
        sim = _cosine_sim(qvec, dv)
        scores.append((i, sim))

    # sort by similarity descending
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = []
    for idx, sim in scores[:top_n]:
        d = docs[idx]
        # keep original metadata, add score & snippet
        snippet = get_best_snippet(d.page_content, query)
        # copy doc (Document is a dataclass-like object) - create new Document preserving metadata
        metadata = dict(d.metadata or {})
        metadata["rerank_score"] = float(sim)
        metadata["snippet"] = snippet
        # preserve doc_id if it exists
        selected.append(Document(page_content=d.page_content, metadata=metadata))
    return selected

def compute_confidence_score(reranked_docs: List[Document]) -> float:
    """
    Simple confidence: average rerank_score normalized to [0,1], with weight for number of docs.
    """
    if not reranked_docs:
        return 0.0
    scores = [float(d.metadata.get("rerank_score", 0.0)) for d in reranked_docs]
    avg = float(np.mean(scores))
    # map avg (which is cosine similarity in [-1,1]) to [0,1]
    normalized = (avg + 1) / 2
    # weight by fraction of docs that have score > 0.1 (signal present)
    positive_frac = sum(1 for s in scores if s > 0.1) / len(scores)
    confidence = normalized * (0.6 + 0.4 * positive_frac)  # base 0.6 + up to 0.4
    return float(max(0.0, min(1.0, confidence)))

# =========================
# Query Function (keeps pipeline logic intact)
# =========================
def query_rag(query_text, top_k=20, rerank_top_k=5):
    # 1) retrieval (unchanged)
    retrieved_docs = hybrid_retriever.get_relevant_documents(query_text)

    # 2) compression (unchanged)
    compressed_docs = compress_docs(query_text, retrieved_docs)

    # 3) prepare raw docs and used_doc_ids (unchanged part kept, but we add re-ranking/snip)
    raw_docs, used_doc_ids = [], []
    for doc in compressed_docs:
        doc_id = doc.metadata.get("doc_id")
        if not doc_id:
            continue
        raw_json_list = redis_store.mget([doc_id])
        if raw_json_list and raw_json_list[0]:
            raw_doc = json.loads(raw_json_list[0])
            # keep original content inside Document for reranking/snippet stage
            # we attach full content into a Document instance for reranking step
            raw_docs.append(Document(page_content=raw_doc.get("content", ""), metadata={"doc_id": doc_id}))
            used_doc_ids.append(doc_id)

    if not raw_docs:
        return {"answer": "No relevant documents found.", "doc_ids": [], "confidence": 0.0, "used_snippets": []}

    # 4) rerank and extract snippets (new enhancement, does NOT replace compression/retrieval)
    reranked_docs = rerank_and_prepare(query_text, raw_docs, top_n=rerank_top_k)

    # 5) compute confidence
    confidence = compute_confidence_score(reranked_docs)

    # 6) build a concise context using snippets to reduce prompt noise
    context_parts = []
    used_snippets = []
    for i, d in enumerate(reranked_docs, start=1):
        doc_id = d.metadata.get("doc_id", f"doc_{i}")
        snippet = d.metadata.get("snippet", "")
        score = d.metadata.get("rerank_score", 0.0)
        # Each context includes a short snippet and an explicit [SOURCE: <id>] label
        part = f"[SOURCE: {doc_id} | score:{score:.4f}]\n{snippet}"
        context_parts.append(part)
        used_snippets.append({"doc_id": doc_id, "snippet": snippet, "score": float(score)})

    context_text = "\n\n---\n\n".join(context_parts[:top_k])

    # 7) improved system prompt but still strict: ask the model to cite sources and avoid hallucination
    system_prompt = (
        "You are a precise assistant. You must answer ONLY using the provided document snippets. "
        "For every factual claim, append the source tags in square brackets exactly as provided (e.g. [SOURCE: doc123]). "
        "Do not invent dates, numbers, or facts not present in documents."
    )

    user_prompt = f"Documents (snippets):\n{context_text}\n\nQuestion: {query_text}\n\nProvide a concise answer and cite sources."

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    # 8) call Gemini exactly like before
    response = gemini.invoke(messages)

    # 9) return extended info (answer, doc ids, confidence, snippets)
    return {
        "answer": response.content,
        "doc_ids": used_doc_ids,
        "confidence": confidence,
        "used_snippets": used_snippets
    }

# =========================
# RAGAS imports (unchanged)
# =========================
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas import evaluate
from datasets import Dataset
from ragas.llms import LangchainLLMWrapper

# =========================
# Evaluation Function (unchanged logic, uses query_rag results)
# =========================
def evaluate_batch(qa_pairs):
    """Evaluate RAG system using RAGAS metrics"""
    records = []

    for q, expected in qa_pairs:
        result = query_rag(q, top_k=5, rerank_top_k=5)
        row = {
            "question": q,
            "answer": result["answer"],
            "contexts": result["doc_ids"],  # doc IDs instead of raw text to save space
            "ground_truth": expected
        }
        records.append(row)

    df = pd.DataFrame(records)
    dataset = Dataset.from_pandas(df)

    evaluator_llm = LangchainLLMWrapper(gemini)

    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=evaluator_llm
    )

    df_results = results.to_pandas()
    df_results.to_csv("ragas_eval_results.csv", index=False, encoding="utf-8-sig")
    print("Evaluation complete. Saved to ragas_eval_results.csv")

    print("\nMetrics per example:")
    print(df_results[['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']])

    return df_results

# =========================
# QA Pairs for Evaluation (unchanged)
# =========================
qa_pairs = [
    ("이번 달 우리 회사 전체 매출은 얼마야?", "2025년 1월 삼광 Global 전체 매출은 335.4억원입니다. 이는 당초 사업계획(213.4억원) 대비 57% 초과 달성한 수치이며, 실행계획(307.8억원) 대비도 109% 달성한 성과입니다."),
    ("사업부별 매출 비중이 어떻게 되나요?", "2025년 1월 기준 사업부별 매출 비중은 다음과 같습니다:\n\n한국 사업부: 213.0억원 (39.7%)\n베트남 사업부: 38.6억원 (44.1%)\n인도 사업부: 미미한 수준\n윈테크: 미미한 수준\n\n한국과 베트남 사업부가 전체 매출의 약 84%를 차지하고 있습니다."),
    ("우리 회사 영업이익률은 몇 %야?", "2025년 1월 전사 영업이익률은 3%입니다. 영업이익은 8.97억원이며, 사업부별로는 한국 4%, 베트남 2%, 윈테크는 -7%의 영업이익률을 기록했습니다."),
    ("TAB S10 도장 공정 수율이 어떻게 되나요?", "TAB S10 제품의 도장 공정 수율은 평균 98%로 매우 양호합니다. 세부적으로 TAB S10 REAR BODY 도장은 98%, TAB S10 KNOB 도장은 99%의 수율을 보이고 있습니다."),
    ("최근 수율이 낮은 공정이 있나요?", "네, 몇 가지 주의가 필요한 공정이 있습니다:\n\nR47 ENCLOSURE, LOWER, BATTERY, LARGE 사출: 59%\nR47 ARM, FRONT RIGHT, UPPER 사출: 80%\nTab S10 FE FRONT BODY 사출: 87%\n\n이 공정들은 90% 미만의 수율로 개선이 필요합니다."),
    ("삼성 폴더블폰 부품(SM-F 시리즈) 생산 현황은?", "삼성 폴더블폰 부품 생산이 활발합니다:\n\nSM-F721U: FRONT DECO MAIN/SUB NC 공정 수율 96-97%\nSM-F731U: NC 공정 수율 97%, 조립 수율 100%\nSM-F741U: NC 공정 수율 95%, 레이저 공정 수율 99%\nSM-F936U: NC 및 조립 공정 모두 100% 수율 달성"),
    ("R47 시리즈 재고 현황이 어떻게 되나요?", "R47 시리즈 주요 품목 재고 현황:\n\nR47 ENCLOSURE, LOWER, BATTERY, LARGE 가공품: 568 EA (양품)\n기타 R47 부품들은 현재 재고가 없는 상태입니다.\n대부분 게이트 커팅 가공이나 사출 공정을 거치는 부품들입니다."),
    ("C18 제품군 재고가 있나요?", "C18 제품군은 모두 재고가 0인 상태입니다. CLAMSHELL COVER, ENCLOSURE 등 주요 부품들이 재고 소진 상태이므로 생산 계획 수립이 필요합니다."),
    ("우리 회사 매출원가율이 높은 이유가 뭐야?", "2025년 1월 전사 매출원가율은 92%로 매우 높습니다. 주요 원인은:\n\n매입비(원부자재+외주가공비): 67% - 가장 큰 비중\n노무비: 12%\n제조경비: 11%\n\n특히 베트남 사업부(94%)와 인도 사업부(92%)의 매출원가율이 높아 수익성 개선이 시급합니다."),
    ("실패비용이 얼마나 발생했나요?", "2025년 1월 전사 실패비용은 5.16억원(매출 대비 2%)입니다. 사업부별로는:\n\n한국: 0.23억원 (1%)\n베트남: 3.95억원 (2%) - 가장 높음\n인도: 0.48억원 (1%)\n윈테크: 0.50억원 (1%)\n\n베트남 사업부의 실패비용 절감이 필요합니다."),
    ("SMF741UB6 조립 작업 시 주의사항이 뭐야?", "SMF741UB6 FRONT DECO SUB 조립 작업표준서에 따른 주요 주의사항을 확인해야 합니다. 2024년 7월 8일에 조립 부분이 수정된 최신 버전을 참고하시기 바랍니다."),
    ("이번 달 생산성이 가장 좋은 공정은?", "다음 공정들이 100% 수율을 달성했습니다:\n\nSM-F936U NC 및 조립 공정\nC18 SHIM 가공 및 사출\nPA3 DECO 아노다이징, 샌딩, 버핑\n대부분의 조립(ASS'Y) 공정\n\n이들 공정은 벤치마킹 대상으로 삼을 수 있습니다.")
]

# =========================
# CLI Menu (unchanged UX)
# =========================
if __name__ == "__main__":
    print("\nChoose an option:")
    print("1 - Ask a Question")
    print("2 - Evaluate RAG with RAGAS")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        user_q = input("\nEnter your question: ")
        result = query_rag(user_q)
        print("\n✅ Answer from Gemini:")
        print(result["answer"])
        print(f"\n🔎 Confidence: {result['confidence']:.3f}")
        print("\n📄 Doc IDs used:")
        print(result["doc_ids"])
        print("\n✂️ Used snippets:")
        for s in result["used_snippets"]:
            print(f"- {s['doc_id']} (score={s['score']:.4f}): {s['snippet'][:200]}")

    elif choice == "2":
        print("\n🔎 Running RAGAS evaluation...")
        evaluate_batch(qa_pairs)

    else:
        print("❌ Invalid choice. Please select 1 or 2.")
