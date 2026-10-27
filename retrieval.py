"""Hybrid retrieval building block.

It uses BM25 when rank_bm25 is installed, an embedding provider when configured,
and a deterministic lexical fallback for offline demos. The interface is stable
enough to swap the in-process corpus for pgvector/Qdrant later.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Any, Iterable


@dataclass
class RankedDocument:
    document: dict[str, Any]
    bm25_score: float
    vector_score: float
    rerank_score: float


def tokenize(text: str) -> list[str]:
    text = text.lower()
    tokens = re.findall(r"[\u4e00-\u9fff]{2,}|[a-z0-9_]+", text)
    tokens.extend(text[i : i + 2] for i in range(len(text) - 1) if "\u4e00" <= text[i] <= "\u9fff" and "\u4e00" <= text[i + 1] <= "\u9fff")
    return tokens


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a) * sum(y * y for y in b))
    return dot / norm if norm else 0.0


def hybrid_search(query: str, documents: Iterable[dict[str, Any]], top_k: int = 5, query_vector: list[float] | None = None) -> list[RankedDocument]:
    docs = list(documents)
    query_tokens = tokenize(query)
    corpus = []
    for document in docs:
        text = " ".join(str(value) for key, value in document.items() if key in {"title", "content", "summary", "holding", "keywords", "relevance"})
        corpus.append(tokenize(text))
    try:
        from rank_bm25 import BM25Okapi
        bm25_engine = BM25Okapi(corpus)
        bm25_scores = list(bm25_engine.get_scores(query_tokens)) if corpus else []
        max_bm25 = max(bm25_scores, default=0.0)
    except Exception:
        bm25_scores = [0.0] * len(docs)
        max_bm25 = 0.0
    scored: list[RankedDocument] = []
    query_set = set(query_tokens)
    for index, document in enumerate(docs):
        text = " ".join(str(value) for key, value in document.items() if key in {"title", "content", "summary", "holding", "keywords", "relevance"})
        tokens = set(tokenize(text))
        lexical_overlap = len(query_set & tokens) / max(1, len(query_set))
        bm25 = bm25_scores[index] / max_bm25 if max_bm25 else lexical_overlap
        vector = cosine(query_vector or [], document.get("embedding", []))
        # Cross-encoder 适配点：真实服务可替换该分数；当前用标题/关键词覆盖度作为可解释 rerank。
        title_tokens = set(tokenize(str(document.get("title", document.get("case_title", "")))))
        title_boost = len(query_set & title_tokens) / max(1, len(query_set))
        rerank = 0.45 * bm25 + 0.35 * vector + 0.20 * title_boost if query_vector else 0.8 * bm25 + 0.2 * title_boost
        scored.append(RankedDocument(document, bm25, vector, rerank))
    return sorted(scored, key=lambda item: item.rerank_score, reverse=True)[:top_k]


def pgvector_schema_sql() -> str:
    dimension = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
    return f"CREATE EXTENSION IF NOT EXISTS vector; ALTER TABLE legal_documents ADD COLUMN IF NOT EXISTS embedding vector({dimension}); CREATE INDEX IF NOT EXISTS legal_documents_embedding_idx ON legal_documents USING ivfflat (embedding vector_cosine_ops);"


def pgvector_query_sql(limit: int = 5) -> str:
    return "SELECT document_id, title, content, 1 - (embedding <=> :query_vector) AS vector_score FROM legal_documents WHERE tenant_id = :tenant_id ORDER BY embedding <=> :query_vector LIMIT :limit"


def embed_text(text: str) -> list[float] | None:
    """调用 OpenAI 兼容 embedding 接口；未配置时返回 None，启用词法回退。"""
    api_key = os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com"), timeout=10, max_retries=0)
        result = client.embeddings.create(model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"), input=text)
        return result.data[0].embedding
    except Exception:
        return None
