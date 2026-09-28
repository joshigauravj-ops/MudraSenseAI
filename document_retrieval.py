from __future__ import annotations

import os
from typing import Any, Protocol

from faiss_vector_store import FAISSReadBackend
from vector_search import LocalDocumentVectorStore


class ReadBackend(Protocol):
    """Optional external read backend contract for richer document retrieval."""

    def search(self, ticker: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        ...


class ExternalReadBackendAdapter:
    """Adapter contract for an external document-retrieval provider.

    Concrete providers such as OpenAI, Azure OpenAI, or a vector database can be
    plugged in here without changing the local fallback contract used by the rest
    of the project.
    """

    def __init__(self, provider_name: str) -> None:
        self.provider_name = provider_name

    def search(self, ticker: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        raise RuntimeError(
            f"{self.provider_name} backend is configured but no vector index or remote "
            "search implementation is attached yet. Falling back to the local store."
        )


class OpenAIReadBackendAdapter(ExternalReadBackendAdapter):
    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__("openai")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("MUDRASENSE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def search(self, ticker: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("OpenAI API key is not configured")
        raise RuntimeError(
            "OpenAI vector retrieval is not connected to a document index yet. "
            "Use the local fallback store until an index is configured."
        )


class AzureOpenAIReadBackendAdapter(ExternalReadBackendAdapter):
    def __init__(self, endpoint: str | None = None, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__("azure-openai")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.model = model or os.getenv("MUDRASENSE_AZURE_OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def search(self, ticker: str, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        if not self.endpoint or not self.api_key:
            raise RuntimeError("Azure OpenAI endpoint or API key is not configured")
        raise RuntimeError(
            "Azure OpenAI vector retrieval is not connected to a document index yet. "
            "Use the local fallback store until an index is configured."
        )


DEFAULT_DOCUMENTS: list[dict[str, Any]] = [
    {
        "ticker": "RELIANCE",
        "title": "Reliance Industries earnings update",
        "content": "Reliance Industries announces strong quarterly earnings and expansion in telecom and retail operations.",
        "source_url": "https://example.com/reliance-earnings",
        "publication_date": "2024-01-15",
        "document_type": "earnings",
        "chunk_id": "reliance-earnings-001",
    },
    {
        "ticker": "TCS",
        "title": "TCS corporate action update",
        "content": "TCS confirms a large digital transformation contract and related capital allocation update.",
        "source_url": "https://example.com/tcs-contract",
        "publication_date": "2024-01-20",
        "document_type": "corporate_action",
        "chunk_id": "tcs-contract-001",
    },
]


def build_retrieval_store(documents: list[dict[str, Any]] | None = None) -> LocalDocumentVectorStore:
    return LocalDocumentVectorStore(documents or DEFAULT_DOCUMENTS)


def build_external_read_backend() -> ReadBackend | None:
    """Create an optional external retrieval backend when configured.

    The project remains local-first by default. The backend is only used when a
    provider is explicitly enabled through environment variables.
    """
    mode = (os.getenv("MUDRASENSE_READ_BACKEND") or "local").lower()
    if mode in {"", "local", "none"}:
        return None
    if mode == "openai":
        return OpenAIReadBackendAdapter()
    if mode == "azure-openai":
        return AzureOpenAIReadBackendAdapter()
    if mode == "faiss":
        return FAISSReadBackend()
    return None


def retrieve_relevant_documents(
    ticker: str,
    query: str,
    *,
    documents: list[dict[str, Any]] | None = None,
    external_backend: ReadBackend | None = None,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Prefer an external read backend when available, but fall back to the local store.

    This keeps the project local-first and safe: paid or remote retrieval services may be
    enabled in production, but the local deterministic store continues to work as a fallback.
    """
    if external_backend is not None:
        try:
            hits = external_backend.search(ticker, query, top_k=top_k)
            if hits:
                return hits
        except Exception:
            pass

    store = build_retrieval_store(documents)
    return store.search(ticker, query, top_k=top_k)
