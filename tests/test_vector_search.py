import os
import unittest

from document_retrieval import retrieve_relevant_documents
from faiss_vector_store import build_faiss_index
from ingest_documents import ingest_documents_from_directory
from vector_search import LocalDocumentVectorStore


class FailingBackend:
    def search(self, ticker: str, query: str, top_k: int = 5):
        raise RuntimeError("paid read backend unavailable")


class VectorSearchTests(unittest.TestCase):
    def test_search_returns_relevant_document_for_ticker(self):
        store = LocalDocumentVectorStore(
            [
                {
                    "ticker": "RELIANCE",
                    "title": "Reliance Industries Q1 results",
                    "content": "Reliance Industries announces strong quarterly earnings and telecom subscriber expansion.",
                    "source_url": "https://example.com/reliance-earnings",
                    "publication_date": "2024-01-15",
                    "document_type": "earnings",
                    "chunk_id": "reliance-earnings-001",
                },
                {
                    "ticker": "TCS",
                    "title": "TCS order win",
                    "content": "TCS announces a new digital transformation contract in India.",
                    "source_url": "https://example.com/tcs-contract",
                    "publication_date": "2024-01-20",
                    "document_type": "corporate_action",
                    "chunk_id": "tcs-contract-001",
                },
            ]
        )

        hits = store.search("RELIANCE", "reliance quarterly earnings telecom")

        self.assertTrue(hits)
        self.assertEqual(hits[0]["ticker"], "RELIANCE")
        self.assertIn("earnings", hits[0]["content"].lower())
        self.assertGreater(hits[0]["score"], 0)

    def test_search_handles_missing_documents_gracefully(self):
        store = LocalDocumentVectorStore([])
        hits = store.search("SBIN", "sebi filing")
        self.assertEqual(hits, [])

    def test_retrieve_falls_back_to_local_store_when_external_backend_fails(self):
        documents = [
            {
                "ticker": "RELIANCE",
                "title": "Reliance Industries Q1 results",
                "content": "Reliance Industries announces strong quarterly earnings and telecom subscriber expansion.",
                "source_url": "https://example.com/reliance-earnings",
                "publication_date": "2024-01-15",
                "document_type": "earnings",
                "chunk_id": "reliance-earnings-001",
            }
        ]

        hits = retrieve_relevant_documents(
            "RELIANCE",
            "quarterly earnings telecom",
            documents=documents,
            external_backend=FailingBackend(),
        )

        self.assertTrue(hits)
        self.assertEqual(hits[0]["ticker"], "RELIANCE")
        self.assertIn("earnings", hits[0]["content"].lower())

    def test_faiss_backend_rejects_traversal_paths(self):
        with self.assertRaises(ValueError):
            __import__("faiss_vector_store").FAISSReadBackend(index_path="../tmp_faiss.index")

    def test_build_faiss_index_from_documents(self):
        index_name = "tmp_test_faiss.index"
        if os.path.exists(index_name):
            os.remove(index_name)
        if os.path.exists(index_name + ".json"):
            os.remove(index_name + ".json")

        backend = build_faiss_index(
            [
                {
                    "ticker": "RELIANCE",
                    "title": "Reliance Industries Q1 results",
                    "content": "Reliance Industries announces strong quarterly earnings and telecom subscriber expansion.",
                    "source_url": "https://example.com/reliance-earnings",
                    "publication_date": "2024-01-15",
                    "document_type": "earnings",
                    "chunk_id": "reliance-earnings-001",
                }
            ],
            output_path=index_name,
        )

        hits = backend.search("RELIANCE", "quarterly earnings", top_k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0]["ticker"], "RELIANCE")

        if os.path.exists(index_name):
            os.remove(index_name)
        if os.path.exists(index_name + ".json"):
            os.remove(index_name + ".json")

    def test_ingest_documents_from_directory(self):
        sample_dir = os.path.join(os.getcwd(), "tmp_ingest_docs")
        os.makedirs(sample_dir, exist_ok=True)
        sample_file = os.path.join(sample_dir, "reliance_earnings.txt")
        with open(sample_file, "w", encoding="utf-8") as handle:
            handle.write("Reliance Industries announces strong quarterly earnings and telecom expansion.")

        index_name = "tmp_ingest_faiss.index"
        for path in (index_name, index_name + ".json"):
            if os.path.exists(path):
                os.remove(path)

        try:
            backend = ingest_documents_from_directory(sample_dir, output_path=index_name)
            hits = backend.search("RELIANCE", "quarterly earnings", top_k=3)
            self.assertTrue(hits)
            self.assertEqual(hits[0]["ticker"], "RELIANCE")
        finally:
            if os.path.exists(sample_file):
                os.remove(sample_file)
            if os.path.exists(sample_dir):
                os.rmdir(sample_dir)
            for path in (index_name, index_name + ".json"):
                if os.path.exists(path):
                    os.remove(path)


if __name__ == "__main__":
    unittest.main()
