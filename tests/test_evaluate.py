from types import SimpleNamespace

from scripts.evaluate import (
    check_citation_accuracy,
    compute_latency_stats,
    judge_groundedness,
    run_evaluation,
    write_report,
)


def test_compute_latency_stats():
    stats = compute_latency_stats([0.1, 0.2, 0.3, 0.4, 0.5])

    assert stats["p50_ms"] == 300.0
    assert stats["p95_ms"] == 500.0


def _retrieved_doc(doc_id, text, score=0.9):
    from langchain_core.documents import Document
    return (Document(page_content=text, metadata={"doc_id": doc_id, "title": doc_id}), score)


def test_check_citation_accuracy_true_when_cited_passage_supports_answer():
    retrieved = [_retrieved_doc("pto-policy", "15 PTO days per year.")]
    citations = [{"doc_id": "pto-policy", "snippet": "15 PTO days per year."}]
    judge = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="SUPPORTS"))

    assert check_citation_accuracy(citations, retrieved, "You get 15 days.", judge) is True


def test_check_citation_accuracy_false_when_citation_not_retrieved():
    retrieved = [_retrieved_doc("pto-policy", "15 PTO days per year.")]
    citations = [{"doc_id": "unrelated-doc", "snippet": "something else"}]
    judge = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="SUPPORTS"))

    assert check_citation_accuracy(citations, retrieved, "You get 15 days.", judge) is False


def test_check_citation_accuracy_false_when_passage_does_not_support_answer():
    retrieved = [_retrieved_doc("holiday-schedule", "New Year's Day is a holiday.")]
    citations = [{"doc_id": "holiday-schedule", "snippet": "New Year's Day is a holiday."}]
    judge = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="IRRELEVANT"))

    assert check_citation_accuracy(citations, retrieved, "You get 15 PTO days.", judge) is False


def test_check_citation_accuracy_true_when_no_citations():
    judge = SimpleNamespace(
        invoke=lambda prompt: (_ for _ in ()).throw(AssertionError("judge should not be called"))
    )
    assert check_citation_accuracy([], [], "I can only answer about our policies.", judge) is True


def test_judge_groundedness_parses_supported_response():
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="SUPPORTED"))
    assert judge_groundedness("q", "a", "context", llm) is True


def test_judge_groundedness_parses_unsupported_response():
    llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="UNSUPPORTED"))
    assert judge_groundedness("q", "a", "context", llm) is False


def test_run_evaluation_and_write_report(tmp_path):
    questions = [
        {"id": "q1", "question": "How many PTO days?", "expected_doc_id": "pto-policy", "gold_answer": "15"},
    ]

    def retrieve_fn(question):
        from langchain_core.documents import Document
        doc = Document(page_content="15 PTO days per year.", metadata={"doc_id": "pto-policy", "title": "PTO"})
        return [(doc, 0.9)]

    answer_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="You get 15 PTO days per year."))
    judge_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="SUPPORTED"))

    from rag.config import Config
    report = run_evaluation(questions, retrieve_fn, answer_llm, judge_llm, Config())

    assert report["groundedness_pct"] == 100.0
    assert report["citation_accuracy_pct"] == 100.0
    assert "p50_ms" in report["latency"]

    report_path = tmp_path / "results.md"
    write_report(report, str(report_path))
    content = report_path.read_text(encoding="utf-8")
    assert "Groundedness" in content
    assert "q1" in content


def test_run_evaluation_calls_retrieve_fn_once_per_question():
    from rag.config import Config

    questions = [
        {"id": "q1", "question": "How many PTO days?", "expected_doc_id": "pto-policy", "gold_answer": "15"},
        {"id": "q2", "question": "What is the remote work policy?", "expected_doc_id": "remote-work-policy", "gold_answer": "n/a"},
    ]

    call_count = {"n": 0}

    def retrieve_fn(question):
        from langchain_core.documents import Document
        call_count["n"] += 1
        doc = Document(page_content="15 PTO days per year.", metadata={"doc_id": "pto-policy", "title": "PTO"})
        return [(doc, 0.9)]

    answer_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="You get 15 PTO days per year."))
    judge_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="SUPPORTED"))

    run_evaluation(questions, retrieve_fn, answer_llm, judge_llm, Config())

    assert call_count["n"] == len(questions)


def test_run_evaluation_with_empty_questions_returns_zeroed_report():
    from rag.config import Config

    def retrieve_fn(question):
        raise AssertionError("retrieve_fn should not be called for an empty question list")

    answer_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="unused"))
    judge_llm = SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content="unused"))

    report = run_evaluation([], retrieve_fn, answer_llm, judge_llm, Config())

    assert report["rows"] == []
    assert report["groundedness_pct"] == 0.0
    assert report["citation_accuracy_pct"] == 0.0
    assert report["latency"] == {"p50_ms": 0.0, "p95_ms": 0.0}
