import os
import re
import tempfile
from dataclasses import dataclass
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9$%.,/-]+", text.lower())


@dataclass
class RAGIndex:
    """Hybrid index: Google FAISS (semantic) + BM25 (keyword)."""
    db: FAISS
    chunks: List[Document]
    bm25: BM25Okapi

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


def _build_index_from_bytes(file_bytes: bytes, file_name: str) -> RAGIndex:
    is_pdf = file_name.lower().endswith(".pdf")
    suffix = ".pdf" if is_pdf else ".txt"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        if is_pdf:
            docs = PyPDFLoader(tmp_path).load()
        else:
            try:
                docs = TextLoader(tmp_path, encoding="utf-8").load()
            except UnicodeDecodeError:
                docs = TextLoader(tmp_path, encoding="latin-1").load()

        if not docs:
            raise ValueError("Document is empty or could not be parsed.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=60,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_documents(docs)
        if not chunks:
            raise ValueError("Document produced no text chunks after splitting.")

        # Gemini embedding models don't support batch embed_documents —
        # embed each chunk individually then build the FAISS index manually.
        embeddings = _get_embeddings()
        texts = [c.page_content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        vectors = [embeddings.embed_query(t) for t in texts]
        db = FAISS.from_embeddings(list(zip(texts, vectors)), embeddings, metadatas=metadatas)
        tokenized = [_tokenize(c.page_content) for c in chunks]
        bm25 = BM25Okapi(tokenized)
        return RAGIndex(db=db, chunks=chunks, bm25=bm25)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def build_or_get_index_from_session() -> RAGIndex:
    """Return cached RAGIndex or build it from session_state bytes."""
    import streamlit as st

    file_bytes = st.session_state.get("rag_file_bytes")
    file_name = st.session_state.get("rag_file_name", "document")
    file_size = st.session_state.get("rag_file_size", 0)

    if not file_bytes:
        raise ValueError("No document in session — upload a PDF or TXT first.")

    file_key = f"{file_name}_{file_size}"
    cache = st.session_state.get("_rag_cache", {})

    if cache.get("key") == file_key and cache.get("index") is not None:
        return cache["index"]

    index = _build_index_from_bytes(file_bytes, file_name)
    st.session_state["_rag_cache"] = {"key": file_key, "index": index}
    return index


def search(index: RAGIndex, query: str, k: int = 8) -> str:
    """
    Hybrid retrieval: BM25 (keyword) + FAISS (semantic) fused with RRF.
    BM25 catches exact values (amounts, codes, names).
    FAISS catches semantic matches (paraphrased questions).
    """
    if not index.chunks:
        return "The document index is empty."

    actual_k = min(k, index.total_chunks)

    # ── BM25 keyword search ───────────────────────────────────────────────────
    tokens = _tokenize(query)
    bm25_scores = index.bm25.get_scores(tokens)
    bm25_ranked = sorted(range(index.total_chunks), key=lambda i: bm25_scores[i], reverse=True)
    bm25_rank = {i: rank for rank, i in enumerate(bm25_ranked[:actual_k])}

    # ── FAISS semantic search ─────────────────────────────────────────────────
    sem_docs = index.db.similarity_search(query, k=actual_k)
    content_to_idx = {c.page_content: i for i, c in enumerate(index.chunks)}
    sem_rank = {}
    for rank, doc in enumerate(sem_docs):
        idx = content_to_idx.get(doc.page_content)
        if idx is not None:
            sem_rank[idx] = rank

    # ── Reciprocal Rank Fusion ────────────────────────────────────────────────
    K_RRF = 60
    all_idx = set(bm25_rank) | set(sem_rank)

    def rrf(i):
        b = bm25_rank.get(i, actual_k + 1)
        s = sem_rank.get(i, actual_k + 1)
        return 1 / (K_RRF + b) + 1 / (K_RRF + s)

    top = sorted(all_idx, key=rrf, reverse=True)[:actual_k]

    parts = []
    for rank, i in enumerate(top, 1):
        doc = index.chunks[i]
        page = doc.metadata.get("page")
        page_str = ""
        if page is not None and page != "":
            try:
                page_str = f", page {int(page) + 1}"
            except (TypeError, ValueError):
                pass
        parts.append(f"[Section {rank}{page_str}]\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts) if parts else "No relevant sections found for this query."


# ── Legacy RAG graph node ─────────────────────────────────────────────────────

def rag_retrieve(state: dict) -> dict:
    messages = state.get("messages", [])
    question = str(messages[-1].content) if messages else ""

    try:
        index = build_or_get_index_from_session()
    except ValueError as exc:
        raise ValueError("No document uploaded for RAG") from exc

    context = search(index, question, k=8)
    return {**state, "context": context, "rag_context_text": context}
