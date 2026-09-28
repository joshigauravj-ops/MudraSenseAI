from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _validated_index_path(path: str | None) -> Path:
    """Return a safe project-local index path and reject traversal outside the workspace."""
    if not path:
        raise ValueError("Index path is required")
    resolved = Path(path).expanduser().resolve()
    allowed_root = Path.cwd().resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise ValueError("Index path must stay within the project directory") from exc
    return resolved


def build_faiss_index(documents: list[dict[str, Any]], output_path: str = "faiss_index.index") -> "FAISSReadBackend":
    """Build a local FAISS index from a list of documents and return the backend."""
    safe_output = str(_validated_index_path(output_path))
    backend = FAISSReadBackend(index_path=safe_output)
    backend.build_index(documents)
    backend.save_index()
    return backend


class FAISSReadBackend:
    """Optional FAISS-backed vector store adapter.

    This keeps the project local-first while still providing a real vector-search
    path when the FAISS dependency is installed and an index is available.
    """

    def __init__(self, index_path: str | None = None, model_name: str | None = None) -> None:
        raw_path = index_path or os.getenv("MUDRASENSE_FAISS_INDEX_PATH")
        self.index_path = str(_validated_index_path(raw_path)) if raw_path else None
        self.model_name = model_name or os.getenv("MUDRASENSE_EMBEDDING_MODEL")
        self._documents: list[dict[str, Any]] = []
        self._vector_dim = 0
        self._vocabulary: list[str] = []
        self._index: Any | None = None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    @staticmethod
    def _coerce_document(document: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticker": str(document.get("ticker", "")).upper(),
            "title": str(document.get("title", "")).strip(),
            "content": str(document.get("content", "")).strip(),
            "source_url": str(document.get("source_url", "")).strip(),
            "publication_date": str(document.get("publication_date", "")).strip(),
            "document_type": str(document.get("document_type", "")).strip(),
            "chunk_id": str(document.get("chunk_id", document.get("title", ""))),
        }

    def _ensure_faiss(self) -> Any:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("FAISS backend is unavailable in this environment") from exc
        return faiss

    def _vectorize(self, text: str, vocabulary: list[str]) -> list[float]:
        counts = Counter(self._tokenize(text))
        vector = [0.0] * len(vocabulary)
        for token in vocabulary:
            vector[vocabulary.index(token)] = float(counts.get(token, 0))
        return vector

    def _build_vocab(self, documents: list[dict[str, Any]]) -> list[str]:
        tokens: set[str] = set()
        for document in documents:
            text = f"{document.get('ticker', '')} {document.get('title', '')} {document.get('content', '')} {document.get('document_type', '')}"
            tokens.update(self._tokenize(text))
        return sorted(tokens)

    def load_documents(self, documents: list[dict[str, Any]]) -> None:
        self._documents = [self._coerce_document(document) for document in documents]
        self._vocabulary = self._build_vocab(self._documents)
        self._vector_dim = len(self._vocabulary)

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        self.load_documents(documents)
        if not self._documents:
            self._index = None
            return

        faiss = self._ensure_faiss()
        vectors = []
        for document in self._documents:
            text = f"{document.get('ticker', '')} {document.get('title', '')} {document.get('content', '')} {document.get('document_type', '')}"
            vectors.append(self._vectorize(text, self._vocabulary))

        import numpy as np

        matrix = np.asarray(vectors, dtype="float32")
        index = faiss.IndexFlatL2(matrix.shape[1])
        index.add(matrix)
        self._index = index

    def load_index(self, path: str | None = None) -> None:
        index_path = path or self.index_path
        if not index_path:
            raise RuntimeError("MUDRASENSE_FAISS_INDEX_PATH is not configured")

        index_file = _validated_index_path(index_path)
        faiss = self._ensure_faiss()
        if not index_file.exists():
            raise RuntimeError(f"FAISS index file does not exist: {index_file}")

        self._index = faiss.read_index(str(index_file))
        metadata_file = index_file.with_suffix(".json")
        if metadata_file.exists():
            with metadata_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self._documents = payload.get("documents", [])
            self._vocabulary = payload.get("vocabulary", [])
            self._vector_dim = len(self._vocabulary)

    def save_index(self, path: str | None = None) -> None:
        index_path = path or self.index_path
        if not index_path:
            raise RuntimeError("MUDRASENSE_FAISS_INDEX_PATH is not configured")
        if self._index is None:
            raise RuntimeError("No FAISS index exists to save")

        index_file = _validated_index_path(index_path)
        faiss = self._ensure_faiss()
        faiss.write_index(self._index, str(index_file))
        metadata_file = index_file.with_suffix(".json")
        with metadata_file.open("w", encoding="utf-8") as handle:
            json.dump({"documents": self._documents, "vocabulary": self._vocabulary}, handle, ensure_ascii=True, indent=2)

    def search(self, ticker: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not ticker or not query:
            return []

        if self._index is None:
            raise RuntimeError("FAISS index is not initialized")

        if not self._vocabulary:
            return []

        faiss = self._ensure_faiss()
        import numpy as np

        query_vector = np.asarray([self._vectorize(f"{ticker} {query}", self._vocabulary)], dtype="float32")
        scores, indices = self._index.search(query_vector, min(top_k, len(self._documents)))
        hits: list[dict[str, Any]] = []
        for raw_score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._documents):
                continue
            document = self._documents[int(idx)]
            if str(document.get("ticker", "")).upper() != str(ticker).upper():
                continue
            hits.append({
                "ticker": document.get("ticker", ""),
                "title": document.get("title", ""),
                "content": document.get("content", ""),
                "source_url": document.get("source_url", ""),
                "publication_date": document.get("publication_date", ""),
                "document_type": document.get("document_type", ""),
                "chunk_id": document.get("chunk_id", ""),
                "score": float(raw_score),
            })
        return hits
