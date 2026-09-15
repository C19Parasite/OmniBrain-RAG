"""
Unit and Integration Tests for Hybrid Retrieval (Dense ChromaDB + Sparse BM25 + RRF).
"""
import unittest
import shutil
from pathlib import Path
from backend.app.rag.hybrid import BM25Index, reciprocal_rank_fusion
from backend.app.rag.vector_store import ChromaVectorStore
from backend.app.agents.search_agent import SearchAgent


class TestHybridSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_dir = Path("./tests/data/test_hybrid_chroma")
        cls.test_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        if cls.test_dir.exists():
            shutil.rmtree(cls.test_dir, ignore_errors=True)

    def setUp(self):
        self.bm25 = BM25Index()
        self.sample_chunks = [
            {
                "id": "c1",
                "text": "NVIDIA reported record revenue of $30.04B in Q2 FY2025, up 122% year-over-year.",
                "source_document": "nvda_q2.md",
                "page_number": 1,
                "chunk_type": "text",
                "section_title": "Financial Highlights",
                "doc_id": "doc_nvda"
            },
            {
                "id": "c2",
                "text": "Data Center compute revenue grew to $26.3B driven by Hopper GPU architecture demand.",
                "source_document": "nvda_q2.md",
                "page_number": 2,
                "chunk_type": "text",
                "section_title": "Segment Performance",
                "doc_id": "doc_nvda"
            },
            {
                "id": "c3",
                "text": "Microsoft Cloud revenue was $36.8B, up 21% year-over-year driven by Azure AI infrastructure.",
                "source_document": "msft_q4.md",
                "page_number": 1,
                "chunk_type": "text",
                "section_title": "Cloud Overview",
                "doc_id": "doc_msft"
            }
        ]
        self.bm25.add_chunks(self.sample_chunks)

    def test_financial_tokenization(self):
        """Prove tokenizer preserves currency figures, percentages, and tickers."""
        text = "NVIDIA generated $30.77B revenue, achieving 75.1% gross margin in FY2025."
        tokens = BM25Index.tokenize(text)
        self.assertIn("$30.77b", tokens)
        self.assertIn("75.1%", tokens)
        self.assertIn("fy2025", tokens)
        self.assertIn("nvidia", tokens)
        self.assertIn("revenue", tokens)

    def test_bm25_keyword_exact_retrieval(self):
        """Prove BM25 accurately ranks chunks containing exact numerical terms."""
        results = self.bm25.search("revenue $30.04B", top_k=2)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["id"], "c1")
        self.assertIn("bm25_score", results[0])
        self.assertGreater(results[0]["bm25_score"], 0.0)

    def test_bm25_document_filtering(self):
        """Prove BM25 respects doc_ids filter."""
        results = self.bm25.search("revenue", top_k=5, doc_ids=["doc_msft"])
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["doc_id"], "doc_msft")

    def test_reciprocal_rank_fusion_boosting(self):
        """Prove chunks present in both dense and sparse results receive fused rank boost."""
        dense_candidates = [
            {"id": "doc_a", "text": "Semantic match A", "similarity_score": 0.85},
            {"id": "doc_b", "text": "Semantic match B", "similarity_score": 0.72}
        ]
        sparse_candidates = [
            {"id": "doc_b", "text": "Semantic match B", "bm25_score": 4.5},
            {"id": "doc_c", "text": "Keyword match C", "bm25_score": 3.8}
        ]
        fused = reciprocal_rank_fusion(dense_candidates, sparse_candidates, top_k=3)
        self.assertEqual(len(fused), 3)
        # doc_b is in both dense (rank 2) and sparse (rank 1), so its reciprocal rank sum is highest:
        # doc_b: 0.5/(60+2) + 0.5/(60+1) = 0.00806 + 0.00819 = 0.01625
        # doc_a: 0.5/(60+1) = 0.00819
        self.assertEqual(fused[0]["id"], "doc_b")
        self.assertEqual(fused[0]["retrieval_method"], "hybrid")
        self.assertIn("rrf_score", fused[0])

    def test_vector_store_hybrid_integration(self):
        """Prove ChromaVectorStore supports hybrid search mode and updates BM25 index."""
        store = ChromaVectorStore(
            persist_dir=self.test_dir,
            collection_name="test_hybrid_col"
        )
        fake_embeddings = [
            [0.1] * 384,
            [0.2] * 384,
            [0.3] * 384
        ]
        store.add_chunks(self.sample_chunks, fake_embeddings)
        self.assertEqual(store.count(), 3)
        self.assertEqual(store.bm25_index.corpus_size, 3)

        # Hybrid search query
        query_vec = [0.1] * 384
        results = store.search_hybrid(
            query="Hopper GPU architecture",
            query_embedding=query_vec,
            top_k=2,
            mode="hybrid"
        )
        self.assertGreater(len(results), 0)
        self.assertIn("retrieval_method", results[0])

        # Test doc deletion keeps BM25 in sync
        deleted = store.delete_by_doc_id("doc_msft")
        self.assertEqual(deleted, 1)
        self.assertEqual(store.bm25_index.corpus_size, 2)

    def test_dense_threshold_excludes_low_similarity_top_k_fillers(self):
        """Top-k must not turn weak vector matches into evidence."""
        store = ChromaVectorStore(
            persist_dir=self.test_dir,
            collection_name="test_threshold_col"
        )
        chunks = [
            {**self.sample_chunks[0], "id": "threshold_relevant"},
            {**self.sample_chunks[1], "id": "threshold_irrelevant"},
        ]
        store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])

        results = store.search_chunks(
            query_embedding=[1.0, 0.0],
            top_k=2,
            min_similarity=0.75,
        )

        self.assertEqual([result["id"] for result in results], ["threshold_relevant"])


if __name__ == "__main__":
    unittest.main()
