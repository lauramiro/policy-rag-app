from types import SimpleNamespace

from rag.chain import REFUSAL_MESSAGE, answer_question, make_retrieve_fn
from rag.config import Config
from rag.ingest import build_vectorstore


def test_in_corpus_question_is_grounded_with_citation(tmp_path):
    config = Config()
    vectorstore = build_vectorstore("tests/fixtures", str(tmp_path / "chroma_smoke"), config)
    retrieve_fn = make_retrieve_fn(vectorstore, config)
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(
        content="You accrue 15 vacation days per year, credited monthly."
    ))

    result = answer_question("How many vacation days do I accrue per year?", retrieve_fn, llm, config)

    assert "15 vacation days" in result["answer"]
    assert result["citations"]
    assert result["citations"][0]["doc_id"] == "sample-a"


def test_out_of_corpus_question_is_refused(tmp_path):
    config = Config()
    vectorstore = build_vectorstore("tests/fixtures", str(tmp_path / "chroma_smoke"), config)
    retrieve_fn = make_retrieve_fn(vectorstore, config)
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="should never be called"))

    result = answer_question("What is the airspeed velocity of an unladen swallow?", retrieve_fn, llm, config)

    assert result["answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []
