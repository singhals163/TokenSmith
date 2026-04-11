import sqlite3
import hashlib
import multiprocessing
import multiprocessing.pool
import numpy as np
from pathlib import Path
from typing import List, Union, Optional
from llama_cpp import Llama
from tqdm import tqdm

# --- NEW: Import profiler tools ---
from src.profiler import timeit, TimerBlock

# Global variables for worker processes
_worker_model: Optional[Llama] = None
_worker_embedding_dim: int = 0

@timeit("Embedder [Worker]: Initialize Model")
def _init_worker(model_path: str, n_ctx: int, n_threads: int):
    """
    Initializes the model inside a worker process.
    """
    global _worker_model, _worker_embedding_dim

    _worker_model = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        embedding=True,
        verbose=False,
        use_mmap=True # Allows OS to share model weights across processes
    )
    
    # Cache dimension
    test_emb = _worker_model.create_embedding("test")['data'][0]['embedding']
    _worker_embedding_dim = len(test_emb)

@timeit("Embedder [Worker]: Encode Batch")
def _encode_batch_worker(texts: List[str]) -> List[List[float]]:
    """
    Encodes a batch of text using the worker's local model instance.
    """
    global _worker_model, _worker_embedding_dim
    if _worker_model is None:
        return []
        
    embeddings = []
    for text in texts:
        try:
            # Create embedding
            emb = _worker_model.create_embedding(text)['data'][0]['embedding']
            embeddings.append(emb)
        except Exception:
            # Return zero vector on failure
            embeddings.append([0.0] * _worker_embedding_dim)
            
    return embeddings

class SentenceTransformer:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: int = None):
        self.model_path = model_path
        self.n_ctx = n_ctx
        
        with TimerBlock("Embedder [Main]: Load Model to Memory"):
            self.model = Llama(
                model_path=model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                embedding=True,
                verbose=True,
                use_mmap=True,
                n_gpu_layers=-1 # use GPU if available
            )
        self._embedding_dimension = None
        
        _ = self.embedding_dimension

    @property
    def embedding_dimension(self) -> int:
        if self._embedding_dimension is None:
            test_embedding = self.model.create_embedding("test")['data'][0]['embedding']
            self._embedding_dimension = len(test_embedding)
        return self._embedding_dimension

    def encode(self, 
           texts: Union[str, List[str]], 
           batch_size: int = 16,  
           normalize: bool = False,
           show_progress_bar: bool = False,
           **kwargs) -> np.ndarray:

        if isinstance(texts, str):
            texts = [texts]
            
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, -1)
        
        embeddings = []
        num_batches = (len(texts) + batch_size - 1) // batch_size

        with TimerBlock("Embedder [Main]: Sequential Batch Encoding"):
            for i in tqdm(range(num_batches), desc="Encoding", disable=not show_progress_bar):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, len(texts))
                batch_texts = texts[start_idx:end_idx]
                
                try:
                    # Pass the entire LIST to the model at once.
                    with TimerBlock("Embedder [Main]: llama.cpp compute"):
                        response = self.model.create_embedding(batch_texts)
                    
                    batch_embeddings = [item['embedding'] for item in response['data']]
                    embeddings.extend(batch_embeddings)
                    
                except Exception as e:
                    print(f"Error encoding batch: {e}")
                    for _ in batch_texts:
                        embeddings.append([0.0] * self.embedding_dimension)
                    
        vecs = np.array(embeddings, dtype=np.float32)
        
        if normalize:
            with TimerBlock("Embedder [Main]: L2 Normalization"):
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / np.where(norms == 0, 1e-12, norms)
            
        return vecs

    def get_sentence_embedding_dimension(self) -> int:
        return self.embedding_dimension

    @timeit("Embedder [Pool]: Start Workers")
    def start_multi_process_pool(self, num_workers: int = None) -> multiprocessing.pool.Pool:
        if num_workers:
            workers = num_workers
        else:
            workers = max(1, multiprocessing.cpu_count() - 2)

        print(f"Creating {workers} worker processes...")
        worker_threads = 1
        
        pool = multiprocessing.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(self.model_path, self.n_ctx, worker_threads)
        )
        return pool

    def encode_multi_process(self, texts: List[str], pool: multiprocessing.pool.Pool, batch_size: int = 32) -> np.ndarray:
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
                desc="Parallel Encoding"
            ):
                results.append(batch_result)

        with TimerBlock("Embedder [Pool]: Restore Order"):
            flat_embeddings = [emb for batch in results for emb in batch]
            inverse_indices = np.empty_like(indices)
            inverse_indices[indices] = np.arange(len(indices))
            ordered_embeddings = [flat_embeddings[i] for i in inverse_indices]
        
        return np.array(ordered_embeddings, dtype=np.float32)

    @staticmethod
    def stop_multi_process_pool(pool: multiprocessing.pool.Pool):
        pool.close()
        pool.join()


class EmbeddingCache:
    def __init__(self, cache_dir: str = "index/cache"):
        self.db_path = Path(cache_dir) / "embeddings.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    model_name TEXT,
                    model_hash TEXT,
                    query_text TEXT,
                    embedding BLOB,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (model_hash, query_text)
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_name ON embeddings(model_name)")
    
    @timeit("Cache: Read SQLite")
    def get(self, model_path: str, query: str) -> Optional[np.ndarray]:
        model_hash = hashlib.md5(model_path.encode()).hexdigest()[:16]
        
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE model_hash=? AND query_text=?",
                (model_hash, query)
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
                (model_name, model_hash, query, blob)
            )


class CachedEmbedder:
    def __init__(self, model_path: str, **kwargs):
        self.embedder = SentenceTransformer(model_path, **kwargs)
        self.cache = EmbeddingCache()
        self.model_path = model_path
    
    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        
        results = []
        to_compute = []
        to_compute_indices = []
        
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