from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    ticker: str
    title: str
    content: str
    source_url: str
    publication_date: str
    document_type: str
    chunk_id: str
    score: float = 0.0


class LocalDocumentVectorStore:
    """Simple local vector-style store for ticker-scoped retrieval.

    This keeps the project local-first and avoids introducing a full vector DB
    dependency. The code uses lightweight keyword matching over normalized text and
    explicit ticker filtering so retrieval remains deterministic and auditable.
    """

    def __init__(self, documents: list[dict[str, Any]] | None = None) -> None:
        self._documents: list[DocumentChunk] = []
        for doc in documents or []:
            self.add_document(doc)

    def add_document(self, document: dict[str, Any]) -> None:
        self._documents.append(
            DocumentChunk(
                ticker=str(document.get("ticker", "")).upper(),
                title=str(document.get("title", "")).strip(),
                content=str(document.get("content", "")).strip(),
                source_url=str(document.get("source_url", "")).strip(),
                publication_date=str(document.get("publication_date", "")).strip(),
                document_type=str(document.get("document_type", "")).strip(),
                chunk_id=str(document.get("chunk_id", document.get("title", ""))),
            )
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.lower().replace("/", " ").split())

    @staticmethod
    def _score_document(ticker: str, query: str, chunk: DocumentChunk) -> float:
        if not chunk.ticker:
            return 0.0

        query_norm = LocalDocumentVectorStore._normalize(query)
        ticker_norm = LocalDocumentVectorStore._normalize(ticker)
        combo = f"{chunk.ticker} {chunk.title} {chunk.content} {chunk.document_type}"
        combo_norm = LocalDocumentVectorStore._normalize(combo)

        score = 0.0
        if ticker_norm and ticker_norm in combo_norm:
            score += 10.0
        if ticker_norm and ticker_norm == LocalDocumentVectorStore._normalize(chunk.ticker):
            score += 5.0

        for term in query_norm.split():
            if not term:
                continue
            if term in combo_norm:
                score += 1.5

        return score

    def search(self, ticker: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not ticker or not query:
            return []

        hits: list[tuple[float, DocumentChunk]] = []
        ticker_norm = self._normalize(ticker)
        for chunk in self._documents:
            if self._normalize(chunk.ticker) != ticker_norm:
                continue
            score = self._score_document(ticker, query, chunk)
            if score > 0:
                chunk.score = score
                hits.append((score, chunk))

        hits.sort(key=lambda item: item[0], reverse=True)
        results: list[dict[str, Any]] = []
        for _, chunk in hits[:top_k]:
            results.append(
                {
                    "ticker": chunk.ticker,
                    "title": chunk.title,
                    "content": chunk.content,
                    "source_url": chunk.source_url,
                    "publication_date": chunk.publication_date,
                    "document_type": chunk.document_type,
                    "chunk_id": chunk.chunk_id,
                    "score": round(float(chunk.score), 3),
                }
            )
        return results

    def save(self, path: str) -> None:
        payload = [
            {
                "ticker": chunk.ticker,
                "title": chunk.title,
                "content": chunk.content,
                "source_url": chunk.source_url,
                "publication_date": chunk.publication_date,
                "document_type": chunk.document_type,
                "chunk_id": chunk.chunk_id,
            }
            for chunk in self._documents
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    @classmethod
    def load(cls, path: str) -> "LocalDocumentVectorStore":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(payload)
