from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from faiss_vector_store import FAISSReadBackend, _validated_index_path


def _infer_ticker_from_path(path: str | Path) -> str:
    name = Path(path).stem.upper()
    parts = [part for part in re.split(r"[^A-Z0-9]+", name) if part]
    if not parts:
        return "UNKNOWN"
    return parts[0]


def _read_text_file(path: str | Path) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _chunk_text(text: str, chunk_size: int = 1200) -> list[str]:
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def ingest_documents_from_directory(directory: str | os.PathLike[str], output_path: str = "faiss_index.index") -> FAISSReadBackend:
    """Build a local FAISS index from text documents stored in a directory.

    Expected input is plain-text files, one document per file. Each file is mapped
    to a ticker inferred from its filename. This is intentionally simple and
    deterministic so it can be used as a production-safe ingestion bootstrap.
    """
    directory_path = Path(directory)
    if not directory_path.exists() or not directory_path.is_dir():
        raise FileNotFoundError(f"Document directory does not exist: {directory}")

    documents: list[dict[str, Any]] = []
    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in {".txt", ".md", ".csv", ".json"}:
            continue

        ticker = _infer_ticker_from_path(file_path)
        content = _read_text_file(file_path)
        for index, chunk in enumerate(_chunk_text(content)):
            documents.append(
                {
                    "ticker": ticker,
                    "title": f"{file_path.stem} chunk {index + 1}",
                    "content": chunk,
                    "source_url": f"file://{file_path.as_posix()}",
                    "publication_date": "",
                    "document_type": "filing",
                    "chunk_id": f"{ticker.lower()}-{file_path.stem}-{index + 1}",
                }
            )

    if not documents:
        raise ValueError(f"No readable documents were found in: {directory}")

    safe_output = str(_validated_index_path(output_path))
    backend = FAISSReadBackend(index_path=safe_output)
    backend.build_index(documents)
    backend.save_index()
    return backend


if __name__ == "__main__":
    import argparse
    import re

    parser = argparse.ArgumentParser(description="Build a FAISS index from local text documents.")
    parser.add_argument("directory", help="Directory containing text documents to index")
    parser.add_argument("--output", default="faiss_index.index", help="Output path for generated FAISS index")
    args = parser.parse_args()

    backend = ingest_documents_from_directory(args.directory, output_path=args.output)
    print(f"Built FAISS index at {backend.index_path} with {len(backend._documents)} documents")
