from types import SimpleNamespace

from langchain_core.documents import Document

from rag.chain import REFUSAL_MESSAGE, answer_question


class StubLLM:
    def __init__(self, content):
        self.content = content
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return SimpleNamespace(content=self.content)


def _scored_doc(doc_id, title, text, score):
    metadata = {"doc_id": doc_id, "title": title, "source": f"{doc_id}.md"}
    return (Document(page_content=text, metadata=metadata), score)


def test_answer_question_refuses_when_nothing_relevant():
    llm = StubLLM("irrelevant")

    def retrieve_fn(question):
        return [_scored_doc("sample-a", "Sample Policy A", "unrelated text", 0.05)]

    from rag.config import Config
    result = answer_question("What is the meaning of life?", retrieve_fn, llm, Config())

    assert result["answer"] == REFUSAL_MESSAGE
    assert result["citations"] == []
    assert llm.calls == 0


def test_answer_question_builds_citations_from_retrieved_docs():
    llm = StubLLM("You accrue 15 vacation days per year, credited monthly.")

    def retrieve_fn(question):
        return [_scored_doc("sample-a", "Sample Policy A", "Employees accrue 15 vacation days per year.", 0.9)]

    from rag.config import Config
    result = answer_question("How many vacation days do I get?", retrieve_fn, llm, Config())

    assert "15 vacation days" in result["answer"]
    assert result["citations"] == [
        {
            "doc_id": "sample-a",
            "title": "Sample Policy A",
            "source": "sample-a.md",
            "snippet": "Employees accrue 15 vacation days per year.",
        }
    ]
    assert llm.calls == 1


def test_citations_dedupe_and_drop_weakly_relevant_sources():
    llm = StubLLM("PTO accrues at 1.25 days per month.")

    def retrieve_fn(question):
        return [
            _scored_doc("pto-policy", "PTO Policy", "PTO accrues 15 days per year.", 0.84),
            _scored_doc("pto-policy", "PTO Policy", "PTO accrues 15 days per year.", 0.84),
            _scored_doc("pto-policy", "PTO Policy", "Carryover is capped at 5 days.", 0.77),
            _scored_doc("expense-policy", "Expense Policy", "Submit within 30 days.", 0.50),
        ]

    from rag.config import Config
    result = answer_question("How does PTO accrue?", retrieve_fn, llm, Config())

    # Exact-duplicate chunk removed, same-doc lower chunk collapsed to one
    # citation, and the weakly-related other document dropped below the cutoff.
    assert len(result["citations"]) == 1
    assert result["citations"][0]["doc_id"] == "pto-policy"
    assert result["citations"][0]["snippet"].startswith("PTO accrues 15 days per year.")


def test_citations_keep_multiple_documents_scoring_near_the_top():
    llm = StubLLM("Both policies apply here.")

    def retrieve_fn(question):
        return [
            _scored_doc("policy-a", "Policy A", "Rule from A.", 0.80),
            _scored_doc("policy-b", "Policy B", "Rule from B.", 0.78),
        ]

    from rag.config import Config
    result = answer_question("What applies?", retrieve_fn, llm, Config())

    assert {c["doc_id"] for c in result["citations"]} == {"policy-a", "policy-b"}
