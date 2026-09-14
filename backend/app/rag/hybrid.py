"""
Hybrid Retrieval Module for OmniBrain RAG.
Combines Dense Semantic Embeddings (ChromaDB) with Sparse Keyword Matching (Okapi BM25)
using Reciprocal Rank Fusion (RRF).
"""
import re
import math
from typing import List, Dict, Any, Optional, Tuple, Set


class BM25Index:
    """
    In-memory Okapi BM25 indexer optimized for financial filings.
    Tokenizes financial figures ($30.77B, 75.1%), ticker symbols, and accounting terms.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size: int = 0
        self.avg_doc_len: float = 0.0
        self.doc_lens: Dict[str, int] = {}
        # Term -> Document Frequency (number of documents containing term)
        self.doc_freqs: Dict[str, int] = {}
        # Document ID -> Term -> Term Frequency
        self.term_freqs: Dict[str, Dict[str, int]] = {}
        # Document ID -> Chunk object
        self.chunks_by_id: Dict[str, Dict[str, Any]] = {}
        # Term -> Precomputed Inverse Document Frequency
        self.idf: Dict[str, float] = {}

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """
        Tokenizes financial text into normalized terms while preserving:
        - Currency figures ($30.77B, $4.2M, $100K)
        - Percentages (75.1%, 12%)
        - Stock tickers and fiscal codes (NVDA, FY2025, 10-Q, Q2)
        - Alphanumeric financial tokens
        """
        if not text:
            return []
        text_clean = text.lower()
        tokens = re.findall(r'\$?\d+(?:\.\d+)?[bmk%]?|[a-z0-9]+(?:-[a-z0-9]+)*', text_clean)
        return [t for t in tokens if len(t) > 1 or t.isdigit()]

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Indexes multimodal chunks into BM25 dictionary."""
        if not chunks:
            return 0

        for c in chunks:
            cid = c.get("id")
            if not cid:
                continue
            text = c.get("text", "")
            tokens = self.tokenize(text)
            doc_len = len(tokens)

            # If chunk already existed, subtract old stats
            if cid in self.chunks_by_id:
                old_tf = self.term_freqs.get(cid, {})
                for term in old_tf:
                    self.doc_freqs[term] = max(0, self.doc_freqs.get(term, 1) - 1)
            else:
                self.corpus_size += 1

            self.chunks_by_id[cid] = c
            self.doc_lens[cid] = doc_len

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            self.term_freqs[cid] = tf

            for t in tf:
                self.doc_freqs[t] = self.doc_freqs.get(t, 0) + 1

        self._recompute_stats()
        return len(chunks)

    def _recompute_stats(self):
        """Recomputes avg document length and IDF scores across current corpus."""
        if self.corpus_size == 0:
            self.avg_doc_len = 0.0
            self.idf = {}
            return

        total_len = sum(self.doc_lens.values())
        self.avg_doc_len = total_len / self.corpus_size

        # Okapi BM25 IDF with smoothing: ln(1 + (N - n + 0.5) / (n + 0.5))
        self.idf = {}
        for term, df in self.doc_freqs.items():
            if df > 0:
                self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_ids: Optional[List[str]] = None,
        chunk_type_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Calculates BM25 scores for candidate chunks against user query.
        Applies optional document ID and chunk type filters.
        """
        query_tokens = self.tokenize(query)
        if not query_tokens or self.corpus_size == 0:
            return []

        doc_id_filter_set = set(doc_ids) if doc_ids else None
        scores: List[Tuple[str, float]] = []

        for cid, chunk in self.chunks_by_id.items():
            # Apply filters
            if doc_id_filter_set is not None:
                c_doc_id = chunk.get("doc_id", "")
                if c_doc_id not in doc_id_filter_set:
                    continue
            if chunk_type_filter and chunk.get("chunk_type") != chunk_type_filter:
                continue

            doc_len = self.doc_lens.get(cid, 0)
            tf_dict = self.term_freqs.get(cid, {})
            score = 0.0

            for q_term in query_tokens:
                if q_term not in tf_dict:
                    continue
                tf = tf_dict[q_term]
                idf = self.idf.get(q_term, 0.0)

                # Okapi BM25 formula:
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-6)))
                score += idf * (numerator / max(denominator, 1e-6))

            if score > 0.0:
                scores.append((cid, score))

        # Sort descending by BM25 score
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for cid, score in scores[:top_k]:
            chunk_copy = dict(self.chunks_by_id[cid])
            chunk_copy["bm25_score"] = round(score, 4)
            # Default similarity for purely sparse results
            chunk_copy["similarity_score"] = round(min(1.0, score / 10.0), 4)
            chunk_copy["retrieval_method"] = "bm25"
            results.append(chunk_copy)

        return results

    def delete_by_doc_id(self, doc_id: str) -> int:
        """Removes all chunks belonging to doc_id and recomputes stats."""
        cids_to_del = [
            cid for cid, c in self.chunks_by_id.items()
            if c.get("doc_id") == doc_id
        ]
        if not cids_to_del:
            return 0

        for cid in cids_to_del:
            tf = self.term_freqs.pop(cid, {})
            for term in tf:
                self.doc_freqs[term] = max(0, self.doc_freqs.get(term, 1) - 1)
            self.doc_lens.pop(cid, None)
            self.chunks_by_id.pop(cid, None)
            self.corpus_size = max(0, self.corpus_size - 1)

        self._recompute_stats()
        return len(cids_to_del)

    def clear(self):
        """Clears index."""
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens.clear()
        self.doc_freqs.clear()
        self.term_freqs.clear()
        self.chunks_by_id.clear()
        self.idf.clear()


def reciprocal_rank_fusion(
    dense_results: List[Dict[str, Any]],
    sparse_results: List[Dict[str, Any]],
    top_k: int = 5,
    k_rrf: int = 60,
    w_dense: float = 0.5,
    w_sparse: float = 0.5
) -> List[Dict[str, Any]]:
    """
    Fuses ranked lists from Dense Vector Search and Sparse BM25 Search
    using canonical Reciprocal Rank Fusion:
        RRF(d) = (w_dense / (k_rrf + rank_dense(d))) + (w_sparse / (k_rrf + rank_sparse(d)))
    """
    fused_scores: Dict[str, float] = {}
    fused_chunks: Dict[str, Dict[str, Any]] = {}
    found_in_dense: Set[str] = set()
    found_in_sparse: Set[str] = set()

    # Process Dense Search Ranks (1-indexed)
    for rank, chunk in enumerate(dense_results, start=1):
        cid = chunk.get("id") or f"dense_{rank}"
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (w_dense / (k_rrf + rank))
        fused_chunks[cid] = dict(chunk)
        found_in_dense.add(cid)

    # Process Sparse BM25 Search Ranks (1-indexed)
    for rank, chunk in enumerate(sparse_results, start=1):
        cid = chunk.get("id") or f"sparse_{rank}"
        fused_scores[cid] = fused_scores.get(cid, 0.0) + (w_sparse / (k_rrf + rank))
        if cid not in fused_chunks:
            fused_chunks[cid] = dict(chunk)
        else:
            if "bm25_score" in chunk:
                fused_chunks[cid]["bm25_score"] = chunk["bm25_score"]
        found_in_sparse.add(cid)

    # Determine tags and normalized similarity scores
    ranked_cids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    results = []
    max_rrf = max(fused_scores.values()) if fused_scores else 1.0

    for cid in ranked_cids[:top_k]:
        chunk = fused_chunks[cid]
        rrf_val = fused_scores[cid]
        chunk["rrf_score"] = round(rrf_val, 6)

        is_in_both = (cid in found_in_dense) and (cid in found_in_sparse)
        if is_in_both:
            chunk["retrieval_method"] = "hybrid"
            # Boosted similarity score reflects multi-modal confirmation
            dense_sim = chunk.get("similarity_score", 0.5)
            chunk["similarity_score"] = round(min(1.0, dense_sim * 1.15), 4)
        elif cid in found_in_dense:
            chunk["retrieval_method"] = "dense"
        else:
            chunk["retrieval_method"] = "bm25"
            if "similarity_score" not in chunk or chunk["similarity_score"] == 0.0:
                chunk["similarity_score"] = round(min(0.95, (rrf_val / max(max_rrf, 1e-6)) * 0.85), 4)

        results.append(chunk)

    return results
