from rag.config import Config


def test_config_defaults(monkeypatch):
    for key in [
        "GROQ_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL", "CORPUS_DIR",
        "CHROMA_DIR", "TOP_K", "RETRIEVAL_MODE", "CHUNK_SIZE",
        "CHUNK_OVERLAP", "SIMILARITY_THRESHOLD", "MAX_ANSWER_TOKENS",
        "MAX_ANSWER_WORDS",
    ]:
        monkeypatch.delenv(key, raising=False)

    config = Config()

    assert config.groq_api_key == "test-key"
    assert config.llm_model == "meta-llama/llama-4-scout-17b-16e-instruct"
    assert config.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert config.corpus_dir == "corpus"
    assert config.chroma_dir == "chroma_db"
    assert config.top_k == 4
    assert config.retrieval_mode == "similarity"
    assert config.chunk_size == 800
    assert config.chunk_overlap == 100
    assert config.similarity_threshold == 0.3
    assert config.max_answer_tokens == 400
    assert config.max_answer_words == 200


def test_config_reads_environment(monkeypatch):
    monkeypatch.setenv("TOP_K", "6")
    monkeypatch.setenv("RETRIEVAL_MODE", "mmr")

    config = Config()

    assert config.top_k == 6
    assert config.retrieval_mode == "mmr"
