import json
from pathlib import Path

import frontmatter


def _corpus_doc_ids():
    return {frontmatter.load(p).get("doc_id") for p in Path("corpus").glob("*.md")}


def test_eval_question_set_is_well_formed():
    data = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))

    assert 15 <= len(data) <= 30

    valid_ids = _corpus_doc_ids()
    seen_ids = set()
    for item in data:
        assert {"id", "question", "expected_doc_id", "gold_answer"} <= set(item.keys())
        assert item["expected_doc_id"] in valid_ids
        assert item["id"] not in seen_ids
        seen_ids.add(item["id"])


def test_eval_questions_cover_every_policy_topic():
    data = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
    covered = {item["expected_doc_id"] for item in data}

    assert covered == _corpus_doc_ids()
