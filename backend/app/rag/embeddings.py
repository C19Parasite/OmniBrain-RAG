"""Embedding providers used for semantic retrieval.

The provider is deliberately fail-closed: a failed remote request must never
silently change a semantic search into a token/hash search.
"""
import json
import math
import time
import urllib.error
import urllib.request
from typing import List, Optional

import numpy as np

from ..config import settings


class EmbeddingError(RuntimeError):
    """Raised when a usable semantic embedding cannot be produced."""


class TextEmbeddings:
    """Generate normalized embeddings from one stable provider/model space.

    Gemini uses its retrieval-specific task types, while preserving the same
    model and dimensionality for documents and queries. A local
    ``sentence-transformers`` model is an explicit, genuinely semantic option
    when no API provider is configured; it is never a hidden fallback.
    """

    def __init__(self, provider: Optional[str] = None, dimension: Optional[int] = None):
        self.provider = (provider or settings.EMBEDDING_PROVIDER).lower()
        self.dimension = dimension or settings.EMBEDDING_DIMENSION
        self.model = self._resolve_model()
        self._local_model = None

        if self.provider == "local" and self.dimension != 384:
            raise EmbeddingError(
                "The bundled local model all-MiniLM-L6-v2 produces 384 dimensions; "
                f"configured dimension is {self.dimension}."
            )

    def _resolve_model(self) -> str:
        if self.provider == "gemini":
            if not settings.GEMINI_API_KEY:
                raise EmbeddingError("GEMINI_API_KEY is required for EMBEDDING_PROVIDER=gemini.")
            return settings.GEMINI_EMBEDDING_MODEL
        if self.provider == "openai":
            if not settings.OPENAI_API_KEY:
                raise EmbeddingError("OPENAI_API_KEY is required for EMBEDDING_PROVIDER=openai.")
            return settings.OPENAI_EMBEDDING_MODEL
        if self.provider == "local":
            return settings.LOCAL_EMBEDDING_MODEL
        raise EmbeddingError(f"Unsupported embedding provider: {self.provider!r}.")

    @property
    def embedding_space(self) -> str:
        """Identifier persisted with Chroma to force re-indexing on changes."""
        return f"{self.provider}:{self.model}:{self.dimension}"

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text, task_type="RETRIEVAL_QUERY")

    def embed_text(self, text: str) -> List[float]:
        """Compatibility alias for callers that need a retrieval-query vector."""
        return self.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "gemini":
            return self._embed_gemini_documents(texts)
        return [self._embed(text, task_type="RETRIEVAL_DOCUMENT") for text in texts]

    def _embed_gemini_documents(self, texts: List[str]) -> List[List[float]]:
        """Use Gemini's batch endpoint for re-indexing without per-chunk calls."""
        if any(not text or not text.strip() for text in texts):
            raise EmbeddingError("Cannot embed empty document text.")
        values: List[List[float]] = []
        model_path = f"models/{self.model}"
        # Small batches respect provider token/rate quotas during an index migration.
        for start in range(0, len(texts), 10):
            batch = texts[start:start + 10]
            payload = {
                "requests": [
                    {
                        "model": model_path,
                        "content": {"parts": [{"text": text}]},
                        "taskType": "RETRIEVAL_DOCUMENT",
                        "outputDimensionality": self.dimension,
                    }
                    for text in batch
                ]
            }
            url = (
                "https://generativelanguage.googleapis.com/v1beta/"
                f"{model_path}:batchEmbedContents?key={settings.GEMINI_API_KEY}"
            )
            request = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            for attempt in range(3):
                try:
                    with urllib.request.urlopen(request, timeout=60) as response:
                        response_payload = json.loads(response.read().decode("utf-8"))
                    batch_values = [item["values"] for item in response_payload["embeddings"]]
                    break
                except urllib.error.HTTPError as exc:
                    if exc.code != 429 or attempt == 2:
                        detail = exc.read().decode("utf-8", errors="replace")[:500]
                        raise EmbeddingError(
                            f"Gemini document embedding batch failed ({exc.code}): {detail}"
                        ) from exc
                    time.sleep(2 ** attempt)
                except (urllib.error.URLError, KeyError, ValueError, OSError) as exc:
                    raise EmbeddingError(f"Gemini document embedding batch failed: {exc}") from exc
            if len(batch_values) != len(batch):
                raise EmbeddingError(
                    f"Gemini returned {len(batch_values)} vectors for {len(batch)} documents."
                )
            values.extend(self._validate_and_normalize(vector) for vector in batch_values)
        return values

    def _embed(self, text: str, task_type: str) -> List[float]:
        if not text or not text.strip():
            raise EmbeddingError("Cannot embed empty text.")
        try:
            if self.provider == "gemini":
                values = self._embed_gemini(text, task_type)
            elif self.provider == "openai":
                values = self._embed_openai(text)
            else:
                values = self._embed_local(text)
        except EmbeddingError:
            raise
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, ValueError, OSError) as exc:
            raise EmbeddingError(
                f"{self.provider} embedding request failed for {task_type}: {exc}"
            ) from exc
        return self._validate_and_normalize(values)

    def _embed_gemini(self, text: str, task_type: str) -> List[float]:
        model_path = f"models/{self.model}"
        payload = {
            "model": model_path,
            "content": {"parts": [{"text": text}]},
            "taskType": task_type,
            "outputDimensionality": self.dimension,
        }
        url = (
            "https://generativelanguage.googleapis.com/v1beta/"
            f"{model_path}:embedContent?key={settings.GEMINI_API_KEY}"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        return response_payload["embedding"]["values"]

    def _embed_openai(self, text: str) -> List[float]:
        payload = {"model": self.model, "input": text, "dimensions": self.dimension}
        request = urllib.request.Request(
            "https://api.openai.com/v1/embeddings",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {settings.OPENAI_API_KEY}",
            },
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
        return response_payload["data"][0]["embedding"]

    def _embed_local(self, text: str) -> List[float]:
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer(self.model)
            except Exception as exc:
                raise EmbeddingError(
                    f"Unable to load local semantic model {self.model!r}. "
                    "Install/cache it or configure Gemini/OpenAI embeddings."
                ) from exc
        return self._local_model.encode(text, normalize_embeddings=True).tolist()

    def _validate_and_normalize(self, values: List[float]) -> List[float]:
        if len(values) != self.dimension:
            raise EmbeddingError(
                f"{self.provider} returned {len(values)} dimensions; expected {self.dimension}."
            )
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise EmbeddingError(f"{self.provider} returned non-finite embedding values.")
        norm = float(np.linalg.norm(array))
        if math.isclose(norm, 0.0, abs_tol=1e-12):
            raise EmbeddingError(f"{self.provider} returned a zero embedding.")
        return (array / norm).tolist()
