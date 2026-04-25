import hashlib
import multiprocessing
import multiprocessing.pool
import os
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm


class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    model_path: str
    _embedding_dimension: Optional[int] = None

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        normalize: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        ...

    def get_sentence_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            v = self.encode(["test"])
            self._embedding_dimension = int(v.shape[1])
        return self._embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        return self.get_sentence_embedding_dimension()


# ---------------------------------------------------------------------------
# Backend A: llama.cpp GGUF
# ---------------------------------------------------------------------------

_worker_model = None
_worker_embedding_dim: int = 0


def _init_worker(model_path: str, n_ctx: int, n_threads: int):
    from llama_cpp import Llama
    global _worker_model, _worker_embedding_dim
    _worker_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=False,
        use_mmap=True,
    )
    test_emb = _worker_model.create_embedding("test")["data"][0]["embedding"]
    _worker_embedding_dim = len(test_emb)


def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    global _worker_model, _worker_embedding_dim
    if _worker_model is None:
        return []
    embeddings = []
    for text in texts:
        try:
            emb = _worker_model.create_embedding(text)["data"][0]["embedding"]
            embeddings.append(emb)
        except Exception:
            embeddings.append([0.0] * _worker_embedding_dim)
    return embeddings


class LlamaCppEmbedder(BaseEmbedder):
    backend_name = "llama_cpp"

    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None, n_gpu_layers: int = -1):
        from llama_cpp import Llama
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            embedding=True,
            verbose=True,
            use_mmap=True,
            n_gpu_layers=n_gpu_layers,
        )
        _ = self.embedding_dimension

    def get_sentence_embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            v = self.model.create_embedding("test")["data"][0]["embedding"]
            self._embedding_dimension = len(v)
        return self._embedding_dimension

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        normalize: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)

        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]

        embeddings: List[List[float]] = []
        num_batches = (len(sorted_texts) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(sorted_texts))
            batch_texts = sorted_texts[start_idx:end_idx]
            try:
                response = self.model.create_embedding(batch_texts)
                embeddings.extend(item["embedding"] for item in response["data"])
            except Exception as e:
                print(f"Error encoding batch: {e}")
                for _ in batch_texts:
                    embeddings.append([0.0] * self.embedding_dimension)

        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        ordered = [embeddings[i] for i in inverse_indices]

        vecs = np.array(ordered, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)
        return vecs

    def start_multi_process_pool(self, num_workers: int = None) -> multiprocessing.pool.Pool:
        workers = num_workers or max(1, multiprocessing.cpu_count() - 2)
        print(f"Creating {workers} worker processes...")
        return multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.model_path, self.n_ctx, 1),
        )

    def encode_multi_process(
        self, texts: List[str], pool: multiprocessing.pool.Pool, batch_size: int = 32,
    ) -> np.ndarray:
        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]
        chunks = [sorted_texts[i : i + batch_size] for i in range(0, len(sorted_texts), batch_size)]

        results = []
        for batch_result in tqdm(
            pool.imap(_encode_batch_worker, chunks),
            total=len(chunks),
            desc="Parallel Encoding",
        ):
            results.append(batch_result)

        flat = [e for b in results for e in b]
        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        ordered = [flat[i] for i in inverse_indices]
        return np.array(ordered, dtype=np.float32)

    @staticmethod
    def stop_multi_process_pool(pool: multiprocessing.pool.Pool):
        pool.close()
        pool.join()


# ---------------------------------------------------------------------------
# Backend B: GPT4All (Nomic via Embed4All); supports CPU, CUDA, kompute (Vulkan)
# ---------------------------------------------------------------------------


class GPT4AllEmbedder(BaseEmbedder):
    backend_name = "gpt4all"

    def __init__(self, model_path: str, **kwargs):
        from gpt4all import Embed4All
        self.model_path = model_path
        cache_dir = os.environ.get("GPT4ALL_MODEL_PATH") or str(Path.home() / ".cache" / "gpt4all")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        device = kwargs.get("device")
        kw = {"model_name": model_path, "model_path": cache_dir, "n_threads": kwargs.get("n_threads")}
        if device:
            kw["device"] = device
        self.model = Embed4All(**kw)
        self.device = device or "cpu"

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 16,
        normalize: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)

        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]

        embeddings: List[List[float]] = []
        num_batches = (len(sorted_texts) + batch_size - 1) // batch_size
        for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
            s = i * batch_size
            e = min((i + 1) * batch_size, len(sorted_texts))
            out = self.model.embed(sorted_texts[s:e])
            if isinstance(out, list) and out and isinstance(out[0], (int, float)):
                out = [out]
            embeddings.extend(out)

        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        ordered = [embeddings[i] for i in inverse_indices]

        vecs = np.array(ordered, dtype=np.float32)
        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)
        return vecs


# ---------------------------------------------------------------------------
# Backend C: HuggingFace sentence-transformers (Nomic on CUDA, MiniLM, ...)
# ---------------------------------------------------------------------------


class HFSentenceTransformerEmbedder(BaseEmbedder):
    backend_name = "sentence_transformers"

    def __init__(self, model_path: str, device: Optional[str] = None, trust_remote_code: bool = True, **kwargs):
        from sentence_transformers import SentenceTransformer as _STModel
        import torch

        self.model_path = model_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = _STModel(model_path, device=device, trust_remote_code=trust_remote_code)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = False,
        show_progress_bar: bool = False,
        **kwargs,
    ) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)

        indices = np.argsort([len(t) for t in texts])[::-1]
        sorted_texts = [texts[i] for i in indices]

        vecs_sorted = self.model.encode(
            sorted_texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        inverse_indices = np.empty_like(indices)
        inverse_indices[indices] = np.arange(len(indices))
        vecs = vecs_sorted[inverse_indices]

        if normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            vecs = vecs / np.where(norms == 0, 1e-12, norms)
        return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Factory + back-compat alias
# ---------------------------------------------------------------------------


def make_embedder(model_path: str, backend: str = "llama_cpp", **kwargs) -> BaseEmbedder:
    backend = (backend or "llama_cpp").lower()
    if backend == "llama_cpp":
        return LlamaCppEmbedder(model_path, **kwargs)
    if backend == "gpt4all":
        return GPT4AllEmbedder(model_path, **kwargs)
    if backend in ("sentence_transformers", "st", "hf"):
        return HFSentenceTransformerEmbedder(model_path, **kwargs)
    raise ValueError(f"Unknown embed backend: {backend}")


# Existing callers import `SentenceTransformer` and expect llama.cpp semantics;
# preserve that.
SentenceTransformer = LlamaCppEmbedder


# ---------------------------------------------------------------------------
# Persistent embedding cache
# ---------------------------------------------------------------------------


class EmbeddingCache:
    """SQLite-backed cache keyed on (model_hash, query_text)."""

    def __init__(self, cache_dir: str = "index/cache"):
        self.db_path = Path(cache_dir) / "embeddings.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT,
                    model_hash TEXT,
                    query_text TEXT,
                    embedding BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_hash, query_text)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")

    def get(self, model_path: str, query: str) -> Optional[np.ndarray]:
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE model_hash=? AND query_text=?",
                (model_hash, query),
            ).fetchone()
            if row:
                return np.frombuffer(row[0], dtype=np.float32)
        return None

    def set(self, model_path: str, query: str, embedding: np.ndarray):
        model_name = Path(model_path).stem
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        blob = embedding.astype(np.float32).tobytes()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (model_name, model_hash, query_text, embedding) VALUES (?,?,?,?)",
                (model_name, model_hash, query, blob),
            )


class CachedEmbedder:
    """Wraps any BaseEmbedder with an EmbeddingCache layer."""

    def __init__(self, model_path: str, backend: str = "llama_cpp", **kwargs):
        self.embedder = make_embedder(model_path, backend=backend, **kwargs)
        self.cache = EmbeddingCache()
        self.model_path = model_path

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        results: List[tuple] = []
        to_compute: List[str] = []
        to_compute_indices: List[int] = []

        for i, text in enumerate(texts):
            cached = self.cache.get(self.model_path, text)
            if cached is not None:
                results.append((i, cached))
            else:
                to_compute.append(text)
                to_compute_indices.append(i)

        if to_compute:
            computed = self.embedder.encode(to_compute, **kwargs)
            for idx, text, emb in zip(to_compute_indices, to_compute, computed):
                self.cache.set(self.model_path, text, emb)
                results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    def __getattr__(self, name):
        return getattr(self.embedder, name)
