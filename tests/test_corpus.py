from pathlib import Path

import frontmatter

EXPECTED_DOC_IDS = {
    "pto-policy", "remote-work-policy", "expense-policy", "security-policy",
    "code-of-conduct", "onboarding-guide", "benefits-overview", "travel-policy",
    "data-privacy-policy", "anti-harassment-policy", "holiday-schedule",
    "performance-review-policy", "equipment-offboarding-policy",
    "parental-leave-policy", "learning-development-policy",
    "acceptable-use-policy", "dei-policy", "workplace-safety-policy",
}


def test_corpus_files_have_required_frontmatter():
    paths = sorted(Path("corpus").glob("*.md"))
    assert len(paths) == 18

    seen_ids = set()
    for path in paths:
        post = frontmatter.load(path)
        doc_id = post.get("doc_id")
        title = post.get("title")
        assert doc_id, f"{path} missing doc_id"
        assert title, f"{path} missing title"
        assert doc_id not in seen_ids, f"duplicate doc_id {doc_id}"
        seen_ids.add(doc_id)
        assert len(post.content.strip()) > 400, f"{path} content too short"

    assert seen_ids == EXPECTED_DOC_IDS
