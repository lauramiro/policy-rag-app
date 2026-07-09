from rag.config import Config
from rag.ingest import build_vectorstore
from rag.retriever import get_retriever


def test_get_retriever_returns_relevant_chunks(tmp_path):
    config = Config()
    vectorstore = build_vectorstore("tests/fixtures", str(tmp_path / "chroma_test"), config)

    retriever = get_retriever(vectorstore, config)
    results = retriever.invoke("How many vacation days do I get?")

    assert len(results) > 0
    assert results[0].metadata["doc_id"] == "sample-a"


def test_get_retriever_respects_top_k(tmp_path, monkeypatch):
    monkeypatch.setenv("TOP_K", "1")
    config = Config()
    vectorstore = build_vectorstore("tests/fixtures", str(tmp_path / "chroma_test"), config)

    retriever = get_retriever(vectorstore, config)
    results = retriever.invoke("policy")

    assert len(results) == 1
