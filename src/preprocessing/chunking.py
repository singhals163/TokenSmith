import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional

# langchain_text_splitters pulls transformers (~100+ MB of cold Lustre reads)
# in via its tokenizer plumbing — skip the import until the splitter is
# actually used.

# --- NEW: Import profiler tools ---
from src.profiler import timeit

# -------------------------- Chunking Configs --------------------------

class ChunkConfig(ABC):
    @abstractmethod
    def validate(self):
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        pass

@dataclass
class SectionRecursiveConfig(ChunkConfig):
    recursive_chunk_size: int
    recursive_overlap: int
    
    def to_string(self) -> str:
        return f"chunk_mode=sections+recursive, chunk_size={self.recursive_chunk_size}, overlap={self.recursive_overlap}"

    def validate(self):
        assert self.recursive_chunk_size > 0, "recursive_chunk_size must be > 0"
        assert self.recursive_overlap >= 0, "recursive_overlap must be >= 0"

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def chunk(self, text: str) -> List[str]:
        pass
    
    @abstractmethod
    def artifact_folder_name(self) -> str:
        pass

class SectionRecursiveStrategy(ChunkStrategy):
    def __init__(self, config: SectionRecursiveConfig):
        self.config = config
        self.recursive_chunk_size = config.recursive_chunk_size
        self.recursive_overlap = config.recursive_overlap

    def name(self) -> str:
        return f"sections+recursive({self.recursive_chunk_size},{self.recursive_overlap})"

    def artifact_folder_name(self) -> str:
        return "sections"

    @timeit("Chunking: SectionRecursiveStrategy.chunk")
    def chunk(self, text: str) -> List[str]:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.recursive_chunk_size,
            chunk_overlap=self.recursive_overlap,
            separators=[". "]
        )
        return splitter.split_text(text)

# ----------------------------- Document Chunker ---------------------------------

class DocumentChunker:
    TABLE_RE = re.compile(r"<table>.*?</table>", re.DOTALL | re.IGNORECASE)

    def __init__(
        self,
        strategy: Optional[ChunkStrategy],
        keep_tables: bool = True
    ):
        self.strategy = strategy
        self.keep_tables = keep_tables

    def _extract_tables(self, text: str) -> Tuple[str, List[str]]:
        tables = self.TABLE_RE.findall(text)
        for i, t in enumerate(tables):
            text = text.replace(t, f"[TABLE_PLACEHOLDER_{i}]")
        return text, tables

    @staticmethod
    def _restore_tables(chunk: str, tables: List[str]) -> str:
        for i, t in enumerate(tables):
            ph = f"[TABLE_PLACEHOLDER_{i}]"
            if ph in chunk:
                chunk = chunk.replace(ph, t)
        return chunk

    @timeit("Chunking: DocumentChunker.chunk")
    def chunk(self, text: str) -> List[str]:
        if not text:
            return []
        work = text
        tables: List[str] = []
        if self.keep_tables:
            work, tables = self._extract_tables(work)

        if self.strategy is None:
            raise ValueError("No chunk strategy provided")
        else:
            chunks = self.strategy.chunk(work)

        if self.keep_tables and tables:
            chunks = [self._restore_tables(c, tables) for c in chunks]
        return chunks