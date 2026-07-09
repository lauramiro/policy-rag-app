import json
import math
import statistics
import time
from pathlib import Path

from rag.chain import answer_question, get_llm, make_retrieve_fn
from rag.config import Config
from rag.ingest import load_or_build_vectorstore

JUDGE_PROMPT = """You are grading whether an ANSWER is fully supported by the \
CONTEXT below, with no invented facts. Reply with exactly one word: \
"SUPPORTED" or "UNSUPPORTED".

CONTEXT:
{context}

ANSWER:
{answer}
"""


def compute_latency_stats(latencies_seconds: list[float]) -> dict:
    if not latencies_seconds:
        return {"p50_ms": 0.0, "p95_ms": 0.0}

    ordered = sorted(latencies_seconds)
    p95_index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "p50_ms": round(statistics.median(ordered) * 1000, 1),
        "p95_ms": round(ordered[p95_index] * 1000, 1),
    }


def check_citation_accuracy(citations: list[dict], retrieved_doc_ids: set) -> bool:
    if not citations:
        return True
    return all(citation["doc_id"] in retrieved_doc_ids for citation in citations)


def judge_groundedness(question: str, answer: str, context: str, llm) -> bool:
    prompt = JUDGE_PROMPT.format(context=context, answer=answer)
    response = llm.invoke(prompt)
    return "UNSUPPORTED" not in response.content.strip().upper()


def run_evaluation(questions: list[dict], retrieve_fn, llm, judge_llm, config: Config) -> dict:
    rows = []
    latencies = []

    for item in questions:
        start = time.perf_counter()
        scored_docs = retrieve_fn(item["question"])
        relevant = [
            (doc, score) for doc, score in scored_docs if score >= config.similarity_threshold
        ]

        def _reuse_retrieval(_question, _docs=scored_docs):
            return _docs

        result = answer_question(item["question"], _reuse_retrieval, llm, config)
        latencies.append(time.perf_counter() - start)

        retrieved_doc_ids = {doc.metadata["doc_id"] for doc, _ in relevant}
        context = "\n\n".join(doc.page_content for doc, _ in relevant)
        grounded = (
            judge_groundedness(item["question"], result["answer"], context, judge_llm)
            if relevant else True
        )
        citation_ok = check_citation_accuracy(result["citations"], retrieved_doc_ids)

        rows.append({
            "id": item["id"],
            "question": item["question"],
            "answer": result["answer"],
            "expected_doc_id": item["expected_doc_id"],
            "grounded": grounded,
            "citation_ok": citation_ok,
        })

    if not rows:
        return {
            "rows": [],
            "groundedness_pct": 0.0,
            "citation_accuracy_pct": 0.0,
            "latency": compute_latency_stats([]),
        }

    groundedness_pct = round(100 * sum(r["grounded"] for r in rows) / len(rows), 1)
    citation_pct = round(100 * sum(r["citation_ok"] for r in rows) / len(rows), 1)

    return {
        "rows": rows,
        "groundedness_pct": groundedness_pct,
        "citation_accuracy_pct": citation_pct,
        "latency": compute_latency_stats(latencies),
    }


def write_report(report: dict, path: str) -> None:
    lines = [
        "# Evaluation Results",
        "",
        f"- Groundedness: {report['groundedness_pct']}%",
        f"- Citation accuracy: {report['citation_accuracy_pct']}%",
        f"- Latency p50: {report['latency']['p50_ms']} ms",
        f"- Latency p95: {report['latency']['p95_ms']} ms",
        "",
        "| ID | Question | Grounded | Citation OK | Expected Doc |",
        "|---|---|---|---|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['id']} | {row['question']} | {row['grounded']} | "
            f"{row['citation_ok']} | {row['expected_doc_id']} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main():
    config = Config()
    questions = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    vectorstore = load_or_build_vectorstore(config)
    retrieve_fn = make_retrieve_fn(vectorstore, config)
    llm = get_llm(config)

    report = run_evaluation(questions, retrieve_fn, llm, llm, config)
    write_report(report, "eval/results.md")

    print(f"Groundedness: {report['groundedness_pct']}%  Citation accuracy: {report['citation_accuracy_pct']}%")
    print(f"Latency p50: {report['latency']['p50_ms']}ms  p95: {report['latency']['p95_ms']}ms")


if __name__ == "__main__":
    main()
