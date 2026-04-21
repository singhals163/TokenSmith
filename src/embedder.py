import sqlite3
import hashlib
import multiprocessing
import multiprocessing.pool
import os
from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import List, Union, Optional

from tqdm import tqdm

from src.profiler import timeit, TimerBlock

# ================================================================
# Pluggable embedding backends (Phase 3)
# ================================================================

class BaseEmbedder(ABC):
    """All embedding backends expose encode() + get_sentence_embedding_dimension()."""

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


# ----------------------------------------------------------------
# Backend A: llama.cpp GGUF (original Qwen3-Embedding-4B path)
# ----------------------------------------------------------------

_worker_model = None
_worker_embedding_dim: int = 0


@timeit("Embedder [Worker]: Initialize Model")
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


@timeit("Embedder [Worker]: Encode Batch")
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

        with TimerBlock("Embedder [Main]: Load Model to Memory"):
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

        with TimerBlock("Embedder [Main]: Prepare and Sort Chunks"):
            indices = np.argsort([len(t) for t in texts])[::-1]
            sorted_texts = [texts[i] for i in indices]

        embeddings: List[List[float]] = []
        num_batches = (len(sorted_texts) + batch_size - 1) // batch_size
        with TimerBlock("Embedder [Main]: Sequential Batch Encoding"):
            for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(sorted_texts))
                batch_texts = sorted_texts[start_idx:end_idx]
                try:
                    with TimerBlock("Embedder [Main]: llama.cpp compute"):
                        response = self.model.create_embedding(batch_texts)
                    batch_embeddings = [item["embedding"] for item in response["data"]]
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"Error encoding batch: {e}")
                    for _ in batch_texts:
                        embeddings.append([0.0] * self.embedding_dimension)

        with TimerBlock("Embedder [Main]: Restore Order"):
            inverse_indices = np.empty_like(indices)
            inverse_indices[indices] = np.arange(len(indices))
            ordered = [embeddings[i] for i in inverse_indices]

        vecs = np.array(ordered, dtype=np.float32)
        if normalize:
            with TimerBlock("Embedder [Main]: L2 Normalization"):
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.where(norms == 0, 1e-12, norms)
        return vecs

    @timeit("Embedder [Pool]: Start Workers")
    def start_multi_process_pool(self, num_workers: int = None) -> multiprocessing.pool.Pool:
        workers = num_workers or max(1, multiprocessing.cpu_count() - 2)
        print(f"Creating {workers} worker processes...")
        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.model_path, self.n_ctx, 1),
        )
        return pool

    def encode_multi_process(
        self, texts: List[str], pool: multiprocessing.pool.Pool, batch_size: int = 32
    ) -> np.ndarray:
        with TimerBlock("Embedder [Pool]: Prepare and Sort Chunks"):
            indices = np.argsort([len(t) for t in texts])[::-1]
            sorted_texts = [texts[i] for i in indices]
            chunks = [sorted_texts[i : i + batch_size] for i in range(0, len(sorted_texts), batch_size)]

        results = []
        print(f"Dispatching {len(chunks)} batches to pool...")
        with TimerBlock("Embedder [Pool]: Map/Reduce Execution"):
            for batch_result in tqdm(
                pool.imap(_encode_batch_worker, chunks),
                total=len(chunks),
                desc="Parallel Encoding",
            ):
                results.append(batch_result)

        with TimerBlock("Embedder [Pool]: Restore Order"):
            flat = [e for b in results for e in b]
            inverse_indices = np.empty_like(indices)
            inverse_indices[indices] = np.arange(len(indices))
            ordered = [flat[i] for i in inverse_indices]
        return np.array(ordered, dtype=np.float32)

    @staticmethod
    def stop_multi_process_pool(pool: multiprocessing.pool.Pool):
        pool.close()
        pool.join()


# ----------------------------------------------------------------
# Backend B: GPT4All (Nomic via Embed4All)
# ----------------------------------------------------------------


class GPT4AllEmbedder(BaseEmbedder):
    backend_name = "gpt4all"

    def __init__(self, model_path: str, **kwargs):
        from gpt4all import Embed4All
        self.model_path = model_path
        cache_dir = os.environ.get("GPT4ALL_MODEL_PATH") or str(Path.home() / ".cache" / "gpt4all")
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with TimerBlock("Embedder [Main]: Load Model to Memory"):
            self.model = Embed4All(model_name=model_path, model_path=cache_dir, n_threads=kwargs.get("n_threads"))

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

        with TimerBlock("Embedder [Main]: Prepare and Sort Chunks"):
            indices = np.argsort([len(t) for t in texts])[::-1]
            sorted_texts = [texts[i] for i in indices]

        embeddings: List[List[float]] = []
        num_batches = (len(sorted_texts) + batch_size - 1) // batch_size
        with TimerBlock("Embedder [Main]: Sequential Batch Encoding"):
            for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
                s = i * batch_size
                e = min((i + 1) * batch_size, len(sorted_texts))
                batch_texts = sorted_texts[s:e]
                with TimerBlock("Embedder [Main]: gpt4all compute"):
                    out = self.model.embed(batch_texts)
                if isinstance(out, list) and out and isinstance(out[0], (int, float)):
                    out = [out]
                embeddings.extend(out)

        with TimerBlock("Embedder [Main]: Restore Order"):
            inverse_indices = np.empty_like(indices)
            inverse_indices[indices] = np.arange(len(indices))
            ordered = [embeddings[i] for i in inverse_indices]

        vecs = np.array(ordered, dtype=np.float32)
        if normalize:
            with TimerBlock("Embedder [Main]: L2 Normalization"):
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.where(norms == 0, 1e-12, norms)
        return vecs


# ----------------------------------------------------------------
# Backend C: HuggingFace sentence-transformers (Nomic on CUDA, MiniLM)
# ----------------------------------------------------------------


class HFSentenceTransformerEmbedder(BaseEmbedder):
    backend_name = "sentence_transformers"

    def __init__(self, model_path: str, device: Optional[str] = None, trust_remote_code: bool = True, **kwargs):
        from sentence_transformers import SentenceTransformer as _STModel
        import torch
        self.model_path = model_path
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        with TimerBlock("Embedder [Main]: Load Model to Memory"):
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

        with TimerBlock("Embedder [Main]: Prepare and Sort Chunks"):
            indices = np.argsort([len(t) for t in texts])[::-1]
            sorted_texts = [texts[i] for i in indices]

        with TimerBlock("Embedder [Main]: Sequential Batch Encoding"):
            with TimerBlock("Embedder [Main]: sentence-transformers compute"):
                vecs_sorted = self.model.encode(
                    sorted_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )

        with TimerBlock("Embedder [Main]: Restore Order"):
            inverse_indices = np.empty_like(indices)
            inverse_indices[indices] = np.arange(len(indices))
            vecs = vecs_sorted[inverse_indices]

        if normalize:
            with TimerBlock("Embedder [Main]: L2 Normalization"):
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.where(norms == 0, 1e-12, norms)
        return vecs.astype(np.float32)


# ----------------------------------------------------------------
# Factory
# ----------------------------------------------------------------


def make_embedder(model_path: str, backend: str = "llama_cpp", **kwargs) -> BaseEmbedder:
    backend = (backend or "llama_cpp").lower()
    if backend == "llama_cpp":
        return LlamaCppEmbedder(model_path, **kwargs)
    if backend == "gpt4all":
        return GPT4AllEmbedder(model_path, **kwargs)
    if backend in ("sentence_transformers", "st", "hf"):
        return HFSentenceTransformerEmbedder(model_path, **kwargs)
    raise ValueError(f"Unknown embed backend: {backend}")


# ----------------------------------------------------------------
# Backwards compatible alias (old imports still work)
# ----------------------------------------------------------------

# Existing code imports `from src.embedder import SentenceTransformer` and expects
# llama.cpp semantics. Keep that behavior.
SentenceTransformer = LlamaCppEmbedder


# ----------------------------------------------------------------
# Cache + CachedEmbedder
# ----------------------------------------------------------------


class EmbeddingCache:
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

    @timeit("Cache: Read SQLite")
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

    @timeit("Cache: Write SQLite")
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
    def __init__(self, model_path: str, backend: str = "llama_cpp", **kwargs):
        self.embedder = make_embedder(model_path, backend=backend, **kwargs)
        self.cache = EmbeddingCache()
        self.model_path = model_path

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]

        results = []
        to_compute: List[str] = []
        to_compute_indices: List[int] = []

        with TimerBlock("CachedEmbedder: Check Cache"):
            for i, text in enumerate(texts):
                cached = self.cache.get(self.model_path, text)
                if cached is not None:
                    results.append((i, cached))
                else:
                    to_compute.append(text)
                    to_compute_indices.append(i)

        if to_compute:
            computed = self.embedder.encode(to_compute, **kwargs)
            with TimerBlock("CachedEmbedder: Write Missing to Cache"):
                for idx, text, emb in zip(to_compute_indices, to_compute, computed):
                    self.cache.set(self.model_path, text, emb)
                    results.append((idx, emb))

        results.sort(key=lambda x: x[0])
        return np.array([emb for _, emb in results])

    def __getattr__(self, name):
        return getattr(self.embedder, name)
