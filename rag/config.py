import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv("local.secrets")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass
class Config:
    groq_api_key: str = field(default_factory=lambda: _env("GROQ_API_KEY", "test-key"))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "llama-3.3-70b-versatile"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    )
    corpus_dir: str = field(default_factory=lambda: _env("CORPUS_DIR", "corpus"))
    chroma_dir: str = field(default_factory=lambda: _env("CHROMA_DIR", "chroma_db"))
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "4")))
    retrieval_mode: str = field(default_factory=lambda: _env("RETRIEVAL_MODE", "similarity"))
    chunk_size: int = field(default_factory=lambda: int(_env("CHUNK_SIZE", "800")))
    chunk_overlap: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "100")))
    similarity_threshold: float = field(
        default_factory=lambda: float(_env("SIMILARITY_THRESHOLD", "0.3"))
    )
    max_answer_tokens: int = field(default_factory=lambda: int(_env("MAX_ANSWER_TOKENS", "400")))
    max_answer_words: int = field(default_factory=lambda: int(_env("MAX_ANSWER_WORDS", "200")))
    citation_score_ratio: float = field(
        default_factory=lambda: float(_env("CITATION_SCORE_RATIO", "0.9"))
    )
    max_citations: int = field(default_factory=lambda: int(_env("MAX_CITATIONS", "3")))
