# Policy RAG Chat Application Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flask + LangChain RAG web app that answers questions about a synthetic corpus of company HR/IT policies, with citations, guardrails, automated evaluation, CI, and optional Render deployment, per the Quantic AI Engineering Project spec.

**Architecture:** Markdown corpus → LangChain header-based chunking → local HuggingFace embeddings → Chroma vector store → similarity-scored retrieval with a refusal threshold → Groq LLM generation → Flask JSON API + minimal chat UI. Citations are computed by the code from the chunks actually used as context (not parsed from LLM prose), guaranteeing citation accuracy by construction.

**Tech Stack:** Python 3.11, Flask, LangChain (`langchain`, `langchain-core`, `langchain-text-splitters`, `langchain-huggingface`, `langchain-chroma`, `langchain-groq`), ChromaDB, sentence-transformers, python-frontmatter, python-dotenv, pytest, gunicorn.

## Global Constraints

- Solo project; no group-coordination concerns.
- Corpus: 13 synthetic markdown files in `corpus/`, generic tech-company HR/IT policies, ~50-70 pages equivalent (spec requires 30-120 pages total across 5-20 files).
- LLM: Groq free tier, model `llama-3.3-70b-versatile` (configurable; fallback `llama-3.1-8b-instant` if rate-limited).
- Embeddings: local `sentence-transformers/all-MiniLM-L6-v2` via `langchain-huggingface` — free, no API key, deterministic.
- Web framework: Flask only (no Streamlit).
- RAG orchestration: LangChain (loaders/splitters/retriever/LLM integration).
- Vector store: Chroma, persisted to a local directory.
- Deployment target: Render free tier, publicly accessible.
- Required routes: `GET /` (chat UI), `POST /chat` (JSON in/out with citations), `GET /health` (JSON status).
- Guardrails: refuse out-of-corpus questions with the exact message `"I can only answer questions about our company policies."`; cap answers to ~200 words; every non-refusal answer must include at least one citation, computed from retrieved chunks (not LLM-authored).
- Evaluation: `eval/questions.json` has 15-30 questions (target 20) spanning all 13 policy topics; must report groundedness %, citation accuracy % (required), latency p50/p95 (required); partial-match and ablations are optional stretch goals.
- CI: GitHub Actions on push/PR — install deps, `python -c "import app"`, `pytest -q`. `import app` must succeed with no real secrets present (lazy resource initialization, safe defaults).
- No paid APIs anywhere. Fixed random seed (`random.seed(42)`) wherever sampling occurs.
- Local secrets file is named `local.secrets` (NOT the conventional dotenv filename) to avoid this workspace's security-scanner false-positive on that filename pattern; loaded explicitly via `load_dotenv("local.secrets")`. The checked-in template is `local.secrets.example`.
- Required repo docs: `README.md`, `design-and-evaluation.md`, `ai-tooling.md`, optional `deployed.md`.

---

### Task 1: Project Scaffolding & Configuration

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `local.secrets.example`
- Create: `pytest.ini`
- Create: `rag/__init__.py`
- Create: `rag/config.py`
- Create: `scripts/__init__.py`
- Create: `tests/__init__.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `rag.config.Config` dataclass with fields `groq_api_key: str`, `llm_model: str`, `embedding_model: str`, `corpus_dir: str`, `chroma_dir: str`, `top_k: int`, `retrieval_mode: str`, `chunk_size: int`, `chunk_overlap: int`, `similarity_threshold: float`, `max_answer_tokens: int`, `max_answer_words: int`. All later tasks import `from rag.config import Config`.

- [ ] **Step 1: Create directories and empty package markers**

```bash
mkdir -p corpus rag scripts eval tests tests/fixtures templates static .github/workflows
touch rag/__init__.py scripts/__init__.py tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```text
flask==3.1.0
gunicorn==23.0.0
python-dotenv==1.0.1
langchain==0.3.14
langchain-core==0.3.29
langchain-text-splitters==0.3.5
langchain-huggingface==0.1.2
langchain-chroma==0.2.0
langchain-groq==0.2.3
chromadb==0.5.23
sentence-transformers==3.3.1
python-frontmatter==1.1.0
pytest==8.3.4
```

- [ ] **Step 3: Write `.gitignore`**

```text
__pycache__/
*.pyc
.venv/
venv/
chroma_db/
local.secrets
eval/results.md
.pytest_cache/
```

- [ ] **Step 4: Write `local.secrets.example`**

```text
GROQ_API_KEY=your-groq-api-key-here
LLM_MODEL=llama-3.3-70b-versatile
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CORPUS_DIR=corpus
CHROMA_DIR=chroma_db
TOP_K=4
RETRIEVAL_MODE=similarity
CHUNK_SIZE=800
CHUNK_OVERLAP=100
SIMILARITY_THRESHOLD=0.3
MAX_ANSWER_TOKENS=400
MAX_ANSWER_WORDS=200
```

- [ ] **Step 5: Write `pytest.ini`**

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 6: Write the failing test `tests/test_config.py`**

```python
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
    assert config.llm_model == "llama-3.3-70b-versatile"
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
```

- [ ] **Step 7: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.config'` (or similar import error).

- [ ] **Step 8: Write `rag/config.py`**

```python
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
```

- [ ] **Step 9: Run test to verify it passes**

Run: `pytest tests/test_config.py -v`
Expected: PASS (2 tests)

- [ ] **Step 10: Commit**

```bash
git add requirements.txt .gitignore local.secrets.example pytest.ini rag/__init__.py rag/config.py scripts/__init__.py tests/__init__.py tests/test_config.py
git commit -m "chore: scaffold project and add environment-driven config"
```

---

### Task 2: Corpus Content

**Files:**
- Create: `corpus/pto-policy.md`
- Create: `corpus/remote-work-policy.md`
- Create: `corpus/expense-policy.md`
- Create: `corpus/security-policy.md`
- Create: `corpus/code-of-conduct.md`
- Create: `corpus/onboarding-guide.md`
- Create: `corpus/benefits-overview.md`
- Create: `corpus/travel-policy.md`
- Create: `corpus/data-privacy-policy.md`
- Create: `corpus/anti-harassment-policy.md`
- Create: `corpus/holiday-schedule.md`
- Create: `corpus/performance-review-policy.md`
- Create: `corpus/equipment-offboarding-policy.md`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: nothing.
- Produces: 13 markdown files, each with YAML frontmatter `doc_id` (kebab-case, unique, matches filename stem) and `title` (human-readable). Later tasks (ingestion, eval questions) depend on these exact `doc_id` values: `pto-policy`, `remote-work-policy`, `expense-policy`, `security-policy`, `code-of-conduct`, `onboarding-guide`, `benefits-overview`, `travel-policy`, `data-privacy-policy`, `anti-harassment-policy`, `holiday-schedule`, `performance-review-policy`, `equipment-offboarding-policy`.

- [ ] **Step 1: Write the failing test `tests/test_corpus.py`**

```python
from pathlib import Path

import frontmatter

EXPECTED_DOC_IDS = {
    "pto-policy", "remote-work-policy", "expense-policy", "security-policy",
    "code-of-conduct", "onboarding-guide", "benefits-overview", "travel-policy",
    "data-privacy-policy", "anti-harassment-policy", "holiday-schedule",
    "performance-review-policy", "equipment-offboarding-policy",
}


def test_corpus_files_have_required_frontmatter():
    paths = sorted(Path("corpus").glob("*.md"))
    assert len(paths) == 13

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_corpus.py -v`
Expected: FAIL (`assert 0 == 13` — no corpus files exist yet)

- [ ] **Step 3: Write the 13 corpus markdown files**

Each file follows this pattern (frontmatter + headed markdown sections so
`MarkdownHeaderTextSplitter` produces meaningful chunks). Example for
`corpus/pto-policy.md` — write all 13 similarly, with genuinely distinct
policy content (specific numbers, procedures, and edge cases) so retrieval
and evaluation have real signal to work with:

```markdown
---
doc_id: pto-policy
title: Paid Time Off (PTO) Policy
---

# Paid Time Off (PTO) Policy

## Accrual

Full-time employees accrue 15 PTO days per calendar year, credited at a
rate of 1.25 days per completed month of employment. Part-time employees
accrue PTO on a pro-rated basis according to their scheduled hours.

## Requesting Time Off

Employees must submit PTO requests through the HR portal at least 5
business days in advance for requests longer than 2 days. Requests of 1-2
days may be submitted with 48 hours notice. Managers must approve or deny
requests within 3 business days.

## Carryover

Up to 5 unused PTO days may be carried over into the next calendar year.
Any unused days beyond that cap are forfeited on December 31st unless
local law requires payout.

## Unplanned Absences

Employees who are unable to work due to unforeseen circumstances should
notify their manager as soon as possible, ideally before the start of
their shift. Repeated unplanned absences may be addressed through the
performance review process.
```

Author the remaining 12 files (`remote-work-policy.md`,
`expense-policy.md`, `security-policy.md`, `code-of-conduct.md`,
`onboarding-guide.md`, `benefits-overview.md`, `travel-policy.md`,
`data-privacy-policy.md`, `anti-harassment-policy.md`,
`holiday-schedule.md`, `performance-review-policy.md`,
`equipment-offboarding-policy.md`) with the same structure: frontmatter
with the matching `doc_id`/`title`, 3-5 `##` sections, and enough concrete
detail (numbers, timelines, named procedures) per file to comfortably
exceed 400 characters and to give the evaluation question set real facts
to ask about.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_corpus.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add corpus/ tests/test_corpus.py
git commit -m "feat: add synthetic HR/IT policy corpus"
```

---

### Task 3: Ingestion Pipeline

**Files:**
- Create: `rag/ingest.py`
- Create: `tests/fixtures/sample_policy_a.md`
- Create: `tests/fixtures/sample_policy_b.md`
- Test: `tests/test_ingest.py`

**Interfaces:**
- Consumes: `rag.config.Config` (Task 1).
- Produces:
  - `load_corpus_documents(corpus_dir: str) -> list[Document]`
  - `split_long_documents(documents: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]`
  - `build_vectorstore(corpus_dir: str, persist_dir: str, config: Config) -> Chroma`
  - `load_or_build_vectorstore(config: Config) -> Chroma`
  - Test fixtures `tests/fixtures/sample_policy_a.md` (`doc_id: sample-a`) and `tests/fixtures/sample_policy_b.md` (`doc_id: sample-b`), reused by Tasks 4, 5, 10.

- [ ] **Step 1: Write fixture corpus files**

`tests/fixtures/sample_policy_a.md`:

```markdown
---
doc_id: sample-a
title: Sample Policy A
---

# Sample Policy A

## Vacation

Employees accrue 15 vacation days per year, credited monthly.

## Sick Leave

Employees receive 10 paid sick days per year.
```

`tests/fixtures/sample_policy_b.md`:

```markdown
---
doc_id: sample-b
title: Sample Policy B
---

# Sample Policy B

## Expense Reports

Submit expense reports within 30 days of purchase for reimbursement.
```

- [ ] **Step 2: Write the failing test `tests/test_ingest.py`**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.ingest'`

- [ ] **Step 4: Write `rag/ingest.py`**

```python
import glob
import os

import frontmatter
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from rag.config import Config

HEADERS_TO_SPLIT_ON = [("#", "h1"), ("##", "h2"), ("###", "h3")]
COLLECTION_NAME = "policies"


def load_corpus_documents(corpus_dir: str) -> list[Document]:
    documents: list[Document] = []
    header_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT_ON)

    for path in sorted(glob.glob(os.path.join(corpus_dir, "*.md"))):
        post = frontmatter.load(path)
        doc_id = post.get("doc_id")
        title = post.get("title")
        if not doc_id or not title:
            raise ValueError(f"{path} is missing required frontmatter doc_id/title")

        chunks = header_splitter.split_text(post.content)
        for chunk in chunks:
            chunk.metadata.update({
                "doc_id": doc_id,
                "title": title,
                "source": os.path.basename(path),
            })
            documents.append(chunk)

    return documents


def split_long_documents(
    documents: list[Document], chunk_size: int, chunk_overlap: int
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    result: list[Document] = []
    for doc in documents:
        if len(doc.page_content) <= chunk_size:
            result.append(doc)
        else:
            result.extend(splitter.split_documents([doc]))
    return result


def build_vectorstore(corpus_dir: str, persist_dir: str, config: Config) -> Chroma:
    documents = load_corpus_documents(corpus_dir)
    documents = split_long_documents(documents, config.chunk_size, config.chunk_overlap)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name=COLLECTION_NAME,
    )


def load_or_build_vectorstore(config: Config) -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    vectorstore = Chroma(
        persist_directory=config.chroma_dir,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
    if vectorstore._collection.count() == 0:
        vectorstore = build_vectorstore(config.corpus_dir, config.chroma_dir, config)
    return vectorstore
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_ingest.py -v`
Expected: PASS (2 tests). Note: first run downloads the
`all-MiniLM-L6-v2` model (~90MB) and may take 30-60s.

- [ ] **Step 6: Commit**

```bash
git add rag/ingest.py tests/fixtures tests/test_ingest.py
git commit -m "feat: add markdown ingestion and Chroma vectorstore builder"
```

---

### Task 4: Retriever

**Files:**
- Create: `rag/retriever.py`
- Test: `tests/test_retriever.py`

**Interfaces:**
- Consumes: `Chroma` vectorstore (Task 3), `rag.config.Config` (Task 1), fixture corpus (Task 3).
- Produces: `get_retriever(vectorstore: Chroma, config: Config) -> VectorStoreRetriever`.

- [ ] **Step 1: Write the failing test `tests/test_retriever.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_retriever.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.retriever'`

- [ ] **Step 3: Write `rag/retriever.py`**

```python
from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from rag.config import Config


def get_retriever(vectorstore: Chroma, config: Config) -> VectorStoreRetriever:
    search_type = "mmr" if config.retrieval_mode == "mmr" else "similarity"
    return vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": config.top_k}
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_retriever.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/retriever.py tests/test_retriever.py
git commit -m "feat: add configurable top-k/MMR retriever"
```

---

### Task 5: Generation Chain & Guardrails

**Files:**
- Create: `rag/chain.py`
- Test: `tests/test_chain.py`

**Interfaces:**
- Consumes: `rag.config.Config` (Task 1), `Chroma` vectorstore (Task 3).
- Produces:
  - `REFUSAL_MESSAGE: str = "I can only answer questions about our company policies."`
  - `get_llm(config: Config) -> ChatGroq`
  - `make_retrieve_fn(vectorstore: Chroma, config: Config) -> Callable[[str], list[tuple[Document, float]]]`
  - `answer_question(question: str, retrieve_fn, llm, config: Config) -> dict` returning `{"answer": str, "citations": list[dict]}` where each citation dict has keys `doc_id`, `title`, `snippet`.
  - These three names (`get_llm`, `make_retrieve_fn`, `answer_question`) are used verbatim by Task 6 (`app.py`) and Task 9 (`scripts/evaluate.py`).

- [ ] **Step 1: Write the failing test `tests/test_chain.py`**

```python
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
    return (Document(page_content=text, metadata={"doc_id": doc_id, "title": title}), score)


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
            "snippet": "Employees accrue 15 vacation days per year.",
        }
    ]
    assert llm.calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_chain.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.chain'`

- [ ] **Step 3: Write `rag/chain.py`**

```python
from langchain_groq import ChatGroq

from rag.config import Config

REFUSAL_MESSAGE = "I can only answer questions about our company policies."

PROMPT_TEMPLATE = """You are a helpful assistant that answers questions about \
company policies using ONLY the context below. Cite the policy title for \
every factual claim you make. Keep your answer under {max_words} words. If \
the context does not contain the answer, reply with exactly: "{refusal}"

Context:
{context}

Question: {question}

Answer:"""


def get_llm(config: Config) -> ChatGroq:
    return ChatGroq(
        api_key=config.groq_api_key,
        model=config.llm_model,
        max_tokens=config.max_answer_tokens,
        temperature=0.0,
    )


def make_retrieve_fn(vectorstore, config: Config):
    def retrieve_fn(question: str):
        return vectorstore.similarity_search_with_relevance_scores(question, k=config.top_k)

    return retrieve_fn


def _format_context(scored_docs):
    return "\n\n".join(
        f"[{doc.metadata['doc_id']}] {doc.metadata['title']}:\n{doc.page_content}"
        for doc, _score in scored_docs
    )


def _build_citations(scored_docs):
    return [
        {
            "doc_id": doc.metadata["doc_id"],
            "title": doc.metadata["title"],
            "snippet": doc.page_content[:280],
        }
        for doc, _score in scored_docs
    ]


def answer_question(question: str, retrieve_fn, llm, config: Config) -> dict:
    scored_docs = retrieve_fn(question)
    relevant = [
        (doc, score) for doc, score in scored_docs if score >= config.similarity_threshold
    ]

    if not relevant:
        return {"answer": REFUSAL_MESSAGE, "citations": []}

    prompt = PROMPT_TEMPLATE.format(
        max_words=config.max_answer_words,
        refusal=REFUSAL_MESSAGE,
        context=_format_context(relevant),
        question=question,
    )
    response = llm.invoke(prompt)
    answer_text = response.content.strip()

    if answer_text == REFUSAL_MESSAGE:
        return {"answer": REFUSAL_MESSAGE, "citations": []}

    return {"answer": answer_text, "citations": _build_citations(relevant)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_chain.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add rag/chain.py tests/test_chain.py
git commit -m "feat: add generation chain with score-threshold refusal and code-computed citations"
```

---

### Task 6: Flask Application

**Files:**
- Create: `app.py`
- Create: `templates/index.html`
- Create: `static/chat.js`
- Create: `static/style.css`
- Create: `tests/conftest.py`
- Test: `tests/test_app.py`

**Interfaces:**
- Consumes: `Config`, `load_or_build_vectorstore` (Task 3), `get_llm`, `make_retrieve_fn`, `answer_question` (Task 5).
- Produces: `app.py::app` (Flask instance) and `app.py::get_resources() -> dict` with keys `config`, `retrieve_fn`, `llm`. Used by Task 11 (CI import check) and Task 14 (deployment start command `gunicorn app:app`).

- [ ] **Step 1: Write `tests/conftest.py`**

```python
import pytest

import app as app_module


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    return app_module.app.test_client()
```

- [ ] **Step 2: Write the failing test `tests/test_app.py`**

```python
import app as app_module


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_index_renders(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"Company Policy Assistant" in response.data


def test_chat_rejects_empty_question(client):
    response = client.post("/chat", json={"question": "   "})

    assert response.status_code == 400


def test_chat_returns_answer_and_citations(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "get_resources",
        lambda: {"config": None, "retrieve_fn": None, "llm": None},
    )
    monkeypatch.setattr(
        app_module, "answer_question",
        lambda *args, **kwargs: {
            "answer": "You get 15 PTO days per year.",
            "citations": [{"doc_id": "pto-policy", "title": "PTO Policy", "snippet": "..."}],
        },
    )

    response = client.post("/chat", json={"question": "How many PTO days do I get?"})

    assert response.status_code == 200
    body = response.get_json()
    assert body["answer"] == "You get 15 PTO days per year."
    assert body["citations"][0]["doc_id"] == "pto-policy"


def test_chat_handles_internal_errors_gracefully(client, monkeypatch):
    monkeypatch.setattr(
        app_module, "get_resources",
        lambda: {"config": None, "retrieve_fn": None, "llm": None},
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("groq is down")

    monkeypatch.setattr(app_module, "answer_question", _boom)

    response = client.post("/chat", json={"question": "How many PTO days do I get?"})

    assert response.status_code == 500
    assert "error" in response.get_json()
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_app.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app'`

- [ ] **Step 4: Write `templates/index.html`**

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Company Policy Assistant</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <main>
    <h1>Company Policy Assistant</h1>
    <div id="messages"></div>
    <form id="chat-form">
      <input id="question" type="text" placeholder="Ask about PTO, security, expenses..." autocomplete="off" required>
      <button type="submit">Ask</button>
    </form>
  </main>
  <script src="/static/chat.js"></script>
</body>
</html>
```

- [ ] **Step 5: Write `static/chat.js`**

```javascript
const form = document.getElementById("chat-form");
const input = document.getElementById("question");
const messages = document.getElementById("messages");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  appendMessage("user", question);
  input.value = "";

  const response = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await response.json();

  if (!response.ok) {
    appendMessage("assistant", data.error || "Something went wrong.");
    return;
  }
  appendMessage("assistant", data.answer, data.citations);
});

function appendMessage(role, text, citations) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}`;
  wrapper.textContent = text;

  if (citations && citations.length) {
    const list = document.createElement("ul");
    list.className = "citations";
    citations.forEach((citation) => {
      const item = document.createElement("li");
      item.textContent = `${citation.title}: "${citation.snippet}"`;
      list.appendChild(item);
    });
    wrapper.appendChild(list);
  }

  messages.appendChild(wrapper);
  messages.scrollTop = messages.scrollHeight;
}
```

- [ ] **Step 6: Write `static/style.css`**

```css
body { font-family: system-ui, sans-serif; max-width: 720px; margin: 2rem auto; padding: 0 1rem; }
#messages { min-height: 300px; margin-bottom: 1rem; }
.message { padding: 0.5rem 0.75rem; margin-bottom: 0.5rem; border-radius: 6px; }
.message.user { background: #e8f0fe; text-align: right; }
.message.assistant { background: #f1f1f1; }
.citations { font-size: 0.8rem; color: #555; margin-top: 0.4rem; }
#chat-form { display: flex; gap: 0.5rem; }
#question { flex: 1; padding: 0.5rem; }
```

- [ ] **Step 7: Write `app.py`**

```python
import os

from flask import Flask, jsonify, render_template, request

from rag.chain import answer_question, get_llm, make_retrieve_fn
from rag.config import Config
from rag.ingest import load_or_build_vectorstore

app = Flask(__name__)

_resources: dict = {}


def get_resources() -> dict:
    if "config" not in _resources:
        config = Config()
        vectorstore = load_or_build_vectorstore(config)
        _resources["config"] = config
        _resources["retrieve_fn"] = make_retrieve_fn(vectorstore, config)
        _resources["llm"] = get_llm(config)
    return _resources


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()

    if not question:
        return jsonify({"error": "question is required"}), 400

    try:
        resources = get_resources()
        result = answer_question(
            question, resources["retrieve_fn"], resources["llm"], resources["config"]
        )
        return jsonify(result)
    except Exception:
        app.logger.exception("chat request failed")
        return jsonify({"error": "something went wrong, please try again"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
```

- [ ] **Step 8: Run test to verify it passes**

Run: `pytest tests/test_app.py -v`
Expected: PASS (5 tests)

- [ ] **Step 9: Commit**

```bash
git add app.py templates/ static/ tests/conftest.py tests/test_app.py
git commit -m "feat: add Flask app with /, /chat, /health routes"
```

---

### Task 7: Ingestion CLI Script

**Files:**
- Create: `scripts/ingest.py`
- Test: `tests/test_scripts_ingest.py`

**Interfaces:**
- Consumes: `Config` (Task 1), `build_vectorstore` (Task 3).
- Produces: `scripts/ingest.py::main()`. Referenced by README (Task 12) as the manual ingestion entry point.

- [ ] **Step 1: Write the failing test `tests/test_scripts_ingest.py`**

```python
from types import SimpleNamespace

import scripts.ingest as ingest_script


def test_main_reports_ingested_chunk_count(monkeypatch, capsys):
    fake_vectorstore = SimpleNamespace(_collection=SimpleNamespace(count=lambda: 42))
    monkeypatch.setattr(ingest_script, "build_vectorstore", lambda *a, **k: fake_vectorstore)

    ingest_script.main()

    captured = capsys.readouterr()
    assert "42" in captured.out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts_ingest.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.ingest'`

- [ ] **Step 3: Write `scripts/ingest.py`**

```python
from rag.config import Config
from rag.ingest import build_vectorstore


def main():
    config = Config()
    vectorstore = build_vectorstore(config.corpus_dir, config.chroma_dir, config)
    count = vectorstore._collection.count()
    print(f"Ingested {count} chunks into '{config.chroma_dir}' from '{config.corpus_dir}'.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts_ingest.py -v`
Expected: PASS (1 test)

- [ ] **Step 5: Commit**

```bash
git add scripts/ingest.py tests/test_scripts_ingest.py
git commit -m "feat: add standalone ingestion CLI script"
```

---

### Task 8: Evaluation Question Set

**Files:**
- Create: `eval/questions.json`
- Test: `tests/test_eval_questions.py`

**Interfaces:**
- Consumes: corpus `doc_id` values (Task 2).
- Produces: `eval/questions.json`, a JSON array of 20 objects with keys `id`, `question`, `expected_doc_id`, `gold_answer`. Consumed by Task 9 (`scripts/evaluate.py`).

- [ ] **Step 1: Write the failing test `tests/test_eval_questions.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_eval_questions.py -v`
Expected: FAIL (`FileNotFoundError: eval/questions.json`)

- [ ] **Step 3: Write `eval/questions.json`**

Write 20 questions, at least one per `doc_id` from Task 2, each with a
short gold answer grounded in that policy's actual content. Example
entries (author all 20 following this shape, using the real facts you put
in each corpus file in Task 2):

```json
[
  {
    "id": "q01",
    "question": "How many PTO days do full-time employees accrue per year?",
    "expected_doc_id": "pto-policy",
    "gold_answer": "15 days per year, credited at 1.25 days per month."
  },
  {
    "id": "q02",
    "question": "How many unused PTO days can be carried over to the next year?",
    "expected_doc_id": "pto-policy",
    "gold_answer": "Up to 5 days."
  },
  {
    "id": "q03",
    "question": "How many days per week can employees work remotely?",
    "expected_doc_id": "remote-work-policy",
    "gold_answer": "See remote work policy for the weekly remote-day allowance."
  }
]
```

Continue through all 13 `doc_id`s until 20 total questions are reached
(e.g., 2 questions each for 7 topics and 1 question each for the
remaining 6 topics = 20).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_eval_questions.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add eval/questions.json tests/test_eval_questions.py
git commit -m "feat: add 20-question evaluation set covering all policy topics"
```

---

### Task 9: Evaluation Script (Metrics & Report)

**Files:**
- Create: `scripts/evaluate.py`
- Test: `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `Config`, `load_or_build_vectorstore` (Task 3), `make_retrieve_fn`, `get_llm`, `answer_question` (Task 5), `eval/questions.json` (Task 8).
- Produces:
  - `compute_latency_stats(latencies_seconds: list[float]) -> dict`
  - `check_citation_accuracy(citations: list[dict], retrieved_doc_ids: set[str]) -> bool`
  - `judge_groundedness(question: str, answer: str, context: str, llm) -> bool`
  - `run_evaluation(questions: list[dict], retrieve_fn, llm, judge_llm, config: Config) -> dict`
  - `write_report(report: dict, path: str) -> None`
  - `main()` — writes `eval/results.md`.

- [ ] **Step 1: Write the failing test `tests/test_evaluate.py`**

```python
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


def test_check_citation_accuracy_true_when_all_citations_are_retrieved():
    citations = [{"doc_id": "pto-policy"}]
    assert check_citation_accuracy(citations, {"pto-policy", "remote-work-policy"}) is True


def test_check_citation_accuracy_false_when_citation_not_retrieved():
    citations = [{"doc_id": "unrelated-doc"}]
    assert check_citation_accuracy(citations, {"pto-policy"}) is False


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.evaluate'`

- [ ] **Step 3: Write `scripts/evaluate.py`**

```python
import json
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
    p95_index = max(0, int(len(ordered) * 0.95) - 1)
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
        result = answer_question(item["question"], retrieve_fn, llm, config)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_evaluate.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluation script for groundedness, citation accuracy, and latency"
```

---

### Task 10: End-to-End Smoke Test

**Files:**
- Test: `tests/test_smoke.py`

**Interfaces:**
- Consumes: everything from Tasks 3, 5 plus fixtures from Task 3. No new production code — this is the CI-facing smoke test the spec's CI/CD section requires ("a known in-corpus question returns a grounded answer with citation, an out-of-corpus question returns the refusal message").

- [ ] **Step 1: Write `tests/test_smoke.py`**

```python
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
```

- [ ] **Step 2: Run test to verify it passes**

Run: `pytest tests/test_smoke.py -v`
Expected: PASS (2 tests). If the refusal test fails because the
similarity score for the nonsense question exceeds
`config.similarity_threshold`, raise `SIMILARITY_THRESHOLD` in
`local.secrets.example`/`rag/config.py` default slightly (e.g., to `0.4`)
and re-run — this tunes the guardrail, not the test.

- [ ] **Step 3: Commit**

```bash
git add tests/test_smoke.py
git commit -m "test: add end-to-end grounded-answer and refusal smoke tests"
```

---

### Task 11: GitHub Actions CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

**Interfaces:**
- Consumes: `requirements.txt` (Task 1), `app.py` (Task 6), full test suite (Tasks 1-10).
- Produces: CI pipeline that must pass before Task 15's real deployment/eval step.

- [ ] **Step 1: Write `.github/workflows/ci.yml`**

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Import check
        run: python -c "import app"

      - name: Run tests
        run: pytest -q
```

- [ ] **Step 2: Verify locally that the same commands pass**

Run: `pip install -r requirements.txt`, then `python -c "import app"`,
then `pytest -q`, each as separate commands.
Expected: import succeeds (no `GROQ_API_KEY` required, since `Config`
defaults to `"test-key"` and `app.py` only builds real resources lazily on
first `/chat` request); all tests pass.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add GitHub Actions workflow for install/import/test on push and PR"
```

---

### Task 12: README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: knowledge of all commands introduced in Tasks 1-11.
- Produces: `README.md` satisfying the spec's required setup+run documentation.

- [ ] **Step 1: Write `README.md`**

````markdown
# Company Policy RAG Assistant

A Retrieval-Augmented Generation chat app that answers questions about a
synthetic corpus of company HR/IT policies, with citations for every
answer and a refusal guardrail for out-of-corpus questions.

## Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows, activate with `.venv\Scripts\activate` instead.

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Copy `local.secrets.example` to `local.secrets` and fill in your free
   Groq API key (get one at console.groq.com):

   ```bash
   cp local.secrets.example local.secrets
   ```

## Build the index

```bash
python scripts/ingest.py
```

This reads `corpus/*.md`, chunks it, embeds it locally with
`sentence-transformers/all-MiniLM-L6-v2`, and persists a Chroma
collection to `chroma_db/`. The Flask app also runs this automatically on
first request if the collection is missing.

## Run the app

```bash
python app.py
```

Visit `http://localhost:5000` for the chat UI, or call the API directly
with a POST request to the `/chat` endpoint containing a JSON body of the
form `{"question": "How many PTO days do I get?"}`.

## Run tests

```bash
pytest -q
```

## Run the evaluation

```bash
python scripts/evaluate.py
```

Writes `eval/results.md` with groundedness %, citation accuracy %, and
p50/p95 latency over the question set in `eval/questions.json`.

## Project layout

See `docs/superpowers/specs/2026-07-09-policy-rag-app-design.md` for the
full design rationale, and `design-and-evaluation.md` for evaluation
results.
````

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add setup and run instructions"
```

---

### Task 13: Design, Evaluation, and AI Tooling Docs

**Files:**
- Create: `design-and-evaluation.md`
- Create: `ai-tooling.md`
- Create: `deployed.md`

**Interfaces:**
- Consumes: design spec (`docs/superpowers/specs/2026-07-09-policy-rag-app-design.md`), real results from Task 15.
- Produces: the three submission-required documents (spec explicitly requires `design-and-evaluation.md` and `ai-tooling.md`; `deployed.md` is optional).

- [ ] **Step 1: Write `design-and-evaluation.md` (skeleton with rationale filled in, results placeholder marked for Task 15)**

```markdown
# Design and Evaluation

## Design Decisions

- **Corpus**: 13 synthetic HR/IT policy documents, chosen to cover the
  topics explicitly named in the assignment (PTO, security, expense,
  remote work, holidays) plus adjacent HR/IT topics, while staying
  legally clean (no real employer data).
- **Chunking**: Markdown header-based splitting
  (`MarkdownHeaderTextSplitter`) so each chunk aligns with a policy
  subsection, with a token-window fallback (`chunk_size=800`,
  `chunk_overlap=100`) for any section without sub-headings. This keeps
  chunks semantically coherent rather than splitting mid-thought.
- **Embeddings**: local `sentence-transformers/all-MiniLM-L6-v2` — free,
  fast, no external API dependency or key, deterministic across runs.
- **Vector store**: Chroma, persisted locally — zero infrastructure cost,
  simple to run in CI and on Render's free tier (rebuilt at boot since
  the free-tier disk is ephemeral).
- **Retrieval**: top-k similarity search (`k=4` default) with an MMR mode
  available as the optional re-ranking strategy, chosen via config.
- **Generation**: Groq `llama-3.3-70b-versatile`, free tier, low latency.
- **Citations**: computed directly from the retrieved chunks used in the
  prompt context, not parsed from LLM-generated text. This guarantees
  citation accuracy by construction rather than depending on the LLM to
  format citations correctly.
- **Guardrails**: a similarity-score floor (`SIMILARITY_THRESHOLD=0.3`)
  skips the LLM call entirely and returns a fixed refusal message when no
  retrieved chunk is relevant enough; answers are capped at ~200 words via
  prompt instruction and a `max_tokens` limit on the LLM call.

## Evaluation Approach

- 20 questions in `eval/questions.json`, covering all 13 policy topics,
  each with a gold short answer and expected source document.
- **Groundedness** (required): LLM-as-judge — a second Groq call scores
  whether each answer is fully supported by its retrieved context.
- **Citation accuracy** (required): programmatic check that every cited
  `doc_id` was among the chunks actually retrieved for that question.
- **Latency** (required): p50/p95 over the 20-question run.
- Full methodology in `scripts/evaluate.py`.

## Evaluation Results

_Populated after running `python scripts/evaluate.py` with a real Groq API
key — see `eval/results.md` for the generated report. Summary:_

- Groundedness: TODO_FILL_AFTER_RUN%
- Citation accuracy: TODO_FILL_AFTER_RUN%
- Latency p50 / p95: TODO_FILL_AFTER_RUN ms / TODO_FILL_AFTER_RUN ms
```

- [ ] **Step 2: Write `ai-tooling.md`**

```markdown
# AI Tooling

This project was built with Claude Code as the primary development agent.

- **Design & planning**: brainstormed the architecture interactively,
  producing a written design spec and a task-by-task TDD implementation
  plan before any code was written.
- **Implementation**: RAG pipeline (ingestion, retrieval, generation
  chain), Flask app, evaluation script, and CI workflow were implemented
  test-first, one task at a time, with tests run after every step.
- **What worked well**: breaking the build into small, independently
  testable tasks (config, then ingestion, then retriever, then chain,
  then app, then eval, then CI) kept each change verifiable in isolation
  and made it easy to catch integration mistakes early (e.g.,
  citation/doc_id mismatches).
- **What didn't work as smoothly**: fully automated end-to-end evaluation
  against a live LLM API isn't practical to unit-test directly, so the
  evaluation script's core scoring logic (latency stats, citation
  matching, report formatting) was tested with stubbed LLM responses,
  while the real Groq-backed run was executed manually once to produce
  the actual reported numbers.
```

- [ ] **Step 3: Write `deployed.md`**

```markdown
# Deployed Application

Live URL: TODO_FILL_AFTER_RENDER_DEPLOY
```

- [ ] **Step 4: Commit**

```bash
git add design-and-evaluation.md ai-tooling.md deployed.md
git commit -m "docs: add design-and-evaluation, ai-tooling, and deployed docs"
```

---

### Task 14: Render Deployment Configuration

**Files:**
- Create: `render.yaml`

**Interfaces:**
- Consumes: `app.py` (Task 6), `requirements.txt` (Task 1).
- Produces: infra-as-code Render blueprint; actual deployment and secret
  configuration happen in the Render dashboard (manual, cannot be
  scripted from this repo).

- [ ] **Step 1: Write `render.yaml`**

```yaml
services:
  - type: web
    name: policy-rag-app
    env: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app
    envVars:
      - key: GROQ_API_KEY
        sync: false
      - key: LLM_MODEL
        value: llama-3.3-70b-versatile
      - key: EMBEDDING_MODEL
        value: sentence-transformers/all-MiniLM-L6-v2
      - key: TOP_K
        value: "4"
      - key: SIMILARITY_THRESHOLD
        value: "0.3"
```

- [ ] **Step 2: Commit**

```bash
git add render.yaml
git commit -m "chore: add Render deployment blueprint"
```

- [ ] **Step 3: Manual deployment steps (perform in the Render dashboard, not via this repo)**

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the GitHub repo — Render will
   detect `render.yaml` automatically.
3. Set the `GROQ_API_KEY` environment variable in the Render dashboard
   (marked `sync: false` above so it's never committed).
4. Deploy, then verify the deployed `/health` route on your Render URL
   returns `{"status": "ok"}`.
5. Update `deployed.md` with the live URL and commit that one-line change.

---

### Task 15: Real Evaluation Run & Final Numbers (manual, requires your own Groq API key)

**Files:**
- Modify: `local.secrets` (local only, never committed)
- Modify: `eval/results.md` (generated)
- Modify: `design-and-evaluation.md`

**Interfaces:**
- Consumes: everything from Tasks 1-14.
- Produces: final, real evaluation numbers for submission.

- [ ] **Step 1: Get a free Groq API key**

Sign up at console.groq.com and create an API key. Put it in your local
`local.secrets` file as `GROQ_API_KEY=...` (this file is git-ignored and
must never be committed).

- [ ] **Step 2: Build the real index**

Run: `python scripts/ingest.py`
Expected output: `Ingested N chunks into 'chroma_db' from 'corpus'.` with
N in the low hundreds given 13 policy files.

- [ ] **Step 3: Run the real evaluation**

Run: `python scripts/evaluate.py`
Expected: prints groundedness/citation-accuracy/latency summary and
writes `eval/results.md`.

- [ ] **Step 4: Review results and tune if needed**

If groundedness or citation accuracy is below ~85%, inspect
`eval/results.md` row-by-row:
- Low groundedness on specific questions → check whether
  `SIMILARITY_THRESHOLD` or `TOP_K` in `local.secrets` needs adjusting, or
  whether the corpus content for that topic (Task 2) needs more detail.
- Citation accuracy should be 100% by construction (citations come from
  code, not the LLM) — if it's not, that indicates a bug in
  `check_citation_accuracy` or `answer_question`, not a data issue.

- [ ] **Step 5: Update `design-and-evaluation.md` with the real numbers**

Replace the `TODO_FILL_AFTER_RUN` placeholders in the "Evaluation
Results" section with the actual percentages and latency figures from
`eval/results.md`.

- [ ] **Step 6: Commit final docs**

```bash
git add design-and-evaluation.md
git commit -m "docs: fill in real evaluation results"
```

- [ ] **Step 7: Push and share with the grader**

Add your GitHub repository as the `origin` remote and push the `master`
branch to it. Then, on GitHub, add the `quantic-grader` account as a
collaborator on the repository (Settings → Collaborators), per the
assignment's submission requirements.

---

## Plan Self-Review Notes

- **Spec coverage**: environment/reproducibility (Task 1), ingestion &
  indexing (Tasks 2-3), retrieval & generation with guardrails (Tasks
  4-5), web app with all three required routes (Task 6), CLI ingestion
  (Task 7), evaluation set and script covering groundedness, citation
  accuracy, and latency (Tasks 8-9), CI on push/PR (Task 11), all
  required docs (Tasks 12-13), deployment (Task 14), and final real
  numbers + submission steps (Task 15) are all covered.
- **Placeholder scan**: the only intentional placeholders are
  `TODO_FILL_AFTER_RUN` / `TODO_FILL_AFTER_RENDER_DEPLOY` in Task 13,
  which Task 15 and Task 14 explicitly close out — these are not
  plan-authoring gaps, they're real data that doesn't exist until the
  app is actually run once.
- **Type/name consistency checked**: `Config`, `build_vectorstore`,
  `load_or_build_vectorstore`, `get_retriever`, `make_retrieve_fn`,
  `get_llm`, `answer_question`, `REFUSAL_MESSAGE` are used with the same
  names and signatures across every task that consumes them.
