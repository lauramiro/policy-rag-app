from rag.config import Config
from rag.ingest import build_vectorstore, load_corpus_documents


def test_load_corpus_documents_attaches_metadata():
    documents = load_corpus_documents("tests/fixtures")

    assert len(documents) >= 2
    doc_ids = {doc.metadata["doc_id"] for doc in documents}
    assert doc_ids == {"sample-a", "sample-b"}
    for doc in documents:
        assert "title" in doc.metadata
        assert "source" in doc.metadata


def test_build_vectorstore_indexes_all_chunks(tmp_path):
    config = Config()
    persist_dir = str(tmp_path / "chroma_test")

    vectorstore = build_vectorstore("tests/fixtures", persist_dir, config)

    assert vectorstore._collection.count() >= 2
    results = vectorstore.similarity_search("How many vacation days do I get?", k=1)
    assert results[0].metadata["doc_id"] == "sample-a"
