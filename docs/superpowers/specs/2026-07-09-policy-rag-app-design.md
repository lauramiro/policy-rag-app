# Policy RAG Chat Application — Design Spec

Date: 2026-07-09
Author: solo project (Quantic AI Engineering Project)

## Purpose

Build, evaluate, and (optionally) deploy a Retrieval-Augmented Generation (RAG)
web app that answers employee questions about a synthetic corpus of company
HR/IT policies, citing sources for every answer. Satisfies the Quantic AI
Engineering Project requirements (environment/reproducibility, ingestion,
RAG pipeline, web app, CI/CD, evaluation, design docs).

## Scope decisions (locked in during brainstorming)

- **Team**: solo.
- **Corpus**: synthetic, generic tech-company HR/IT policies (not a real org's
  private data), authored with AI assistance.
- **LLM provider**: Groq free tier (no paid keys required to run).
- **Embeddings**: local HuggingFace `sentence-transformers/all-MiniLM-L6-v2`
  (free, no API key, deterministic).
- **Web framework**: Flask.
- **RAG orchestration**: LangChain (loaders, splitter, retriever, prompt
  chaining, Groq chat model integration).
- **Deployment**: public deployment to Render free tier (optional per rubric,
  but included here).

## Corpus

13 synthetic markdown files in `corpus/`, each with YAML frontmatter
`doc_id` and `title` used later for citations:

1. PTO Policy
2. Remote Work Policy
3. Expense Reimbursement Policy
4. Acceptable Use & Security Policy
5. Code of Conduct
6. Onboarding Guide
7. Benefits Overview
8. Business Travel Policy
9. Data Privacy Policy
10. Anti-Harassment & Workplace Conduct Policy
11. Holiday Schedule Policy
12. Performance Review Policy
13. Equipment, BYOD & Offboarding Policy

Target ~50-70 pages of equivalent content across all files (within the
30-120 page corpus requirement).

## Architecture

```
corpus/*.md --> scripts/ingest.py --> Chroma (local vector store directory)
                                            |
users --> Flask app (/,/chat,/health) --> LangChain retriever (top-k, MMR optional)
                                            |
                                       Groq LLM (llama-3.3-70b-versatile)
                                            |
                                answer + citations (doc_id, title, snippet)
```

### Ingestion & Indexing
- Load each corpus file with a markdown loader, split via
  `MarkdownHeaderTextSplitter` (chunk by heading), falling back to a
  token-window splitter with ~50-token overlap for long sections without
  sub-headings.
- Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` via
  `langchain_huggingface`.
- Persist to a local Chroma collection on disk.
- `scripts/ingest.py` is the standalone entry point; the Flask app calls the
  same ingestion function at startup if the collection is missing/empty
  (required because Render's free-tier disk is ephemeral between deploys).
- Fixed random seed (e.g. `random.seed(42)`) used anywhere sampling occurs
  (currently only in evaluation question sampling, if applicable).

### Retrieval & Generation
- LangChain retriever, similarity search top-k, default `k=4`
  (configurable via a `TOP_K` setting).
- Optional re-ranking via Chroma's MMR search (`search_type="mmr"`),
  toggle via a `RETRIEVAL_MODE` setting.
- Chat model: Groq `llama-3.3-70b-versatile` (fallback `llama-3.1-8b-instant`
  if rate-limited) via `langchain_groq.ChatGroq`.
- Prompt template requirements:
  - Answer only using the provided context chunks.
  - Cite `doc_id` + title for every factual claim.
  - If retrieved context is empty or not relevant to the question, refuse
    with: "I can only answer questions about our company policies."
  - Output length capped (~200 words, enforced via prompt instruction and
    a max-tokens limit on the LLM call).
- Response schema: `{ answer: str, citations: [{doc_id, title, snippet}] }`.

### Guardrails
- Refusal behavior for out-of-corpus questions (prompt-enforced + a
  similarity-score floor: if top retrieved chunk score is below threshold,
  skip the LLM call and return the refusal message directly).
- Output length limit.
- Every answer must include at least one citation, or it must be a refusal.

## Web Application (Flask)

- `GET /` — chat UI: text box, message history, renders citations as
  linked snippets under each answer.
- `POST /chat` — body `{ "question": str }` → JSON response per schema
  above.
- `GET /health` — `{ "status": "ok" }`.
- Configuration is supplied via process environment variables (loaded from
  a local, git-ignored settings file for development): Groq API key,
  top-k, chunk size, chunk overlap, embedding model name, LLM model name,
  retrieval mode.

## Repository Layout

```
corpus/*.md
rag/
  ingest.py        # loader + splitter + embedder + Chroma writer
  retriever.py      # retriever construction
  chain.py          # prompt template + LLM call + guardrails
templates/index.html
static/chat.js, static/style.css
app.py               # Flask entry point, routes
scripts/ingest.py    # CLI wrapper around rag/ingest.py
scripts/evaluate.py  # runs eval/questions.json through the app, scores it
eval/questions.json  # 20 Q&A pairs with gold answer + expected doc_id
eval/results.md      # generated evaluation report
tests/test_app.py    # Flask route smoke tests
tests/test_rag.py    # ingestion + retrieval sanity tests
.github/workflows/ci.yml
requirements.txt
README.md
design-and-evaluation.md
ai-tooling.md
deployed.md          # optional, link to live Render URL
```

Note: local secrets/configuration (API keys, etc.) are kept in a
git-ignored local settings file and are never committed; a checked-in
example/template (with placeholder values only) documents which settings
are needed.

## Evaluation

- **Question set**: 20 questions in `eval/questions.json`, covering all 13
  policy topics, each with a short gold answer and the expected `doc_id`.
- **Groundedness** (required): LLM-as-judge — a Groq call compares each
  generated answer against its retrieved context and scores
  supported/unsupported; reported as % supported. Spot-checked manually
  against ~5 samples.
- **Citation accuracy** (required): programmatic check that every cited
  `doc_id` in an answer is among the chunks actually retrieved for that
  query, plus manual accuracy review of whether the cited passage truly
  supports the claim.
- **Exact/partial match** (optional): substring/keyword overlap between
  generated answer and gold answer.
- **Latency** (required): p50/p95 wall-clock time over 15 queries, run via
  `scripts/evaluate.py`, recorded in `eval/results.md`.
- **Ablations** (optional/stretch): compare k=2 vs k=4 vs k=6, and chunk
  size 300 vs 600 tokens.

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) on push/PR:
1. Checkout, setup Python.
2. `pip install -r requirements.txt`.
3. `python -c "import app"` (build/import check).
4. `pytest -q` — smoke tests: `/health` returns 200, ingestion produces a
   non-empty Chroma collection, a known in-corpus question returns a
   grounded answer with a citation, an out-of-corpus question returns the
   refusal message.
5. Optional: on success on `main`, `curl` a Render deploy hook URL stored
   as a GitHub Actions secret.

## Deployment

- Render free-tier web service.
- Build command: `pip install -r requirements.txt`.
- Start command: `gunicorn app:app` (ingestion runs lazily on first
  request/startup if the Chroma collection is empty).
- Secrets and model/embedding configuration are set directly in the
  Render dashboard's environment variable settings, never committed to
  the repo.
- `deployed.md` records the live URL once deployed.

## Error Handling

- Empty/whitespace-only question → 400 with a helpful message.
- Groq API failure/timeout → caught, returns a graceful error message to
  the user (no stack trace leaked), logged server-side.
- No retrieval results above threshold → refusal message, no LLM call.
- Malformed `/chat` request body → 400 JSON error.

## Documentation Deliverables

- `README.md` — setup (venv, requirements, local configuration), how to
  run ingestion, how to run the app locally, how to run tests/eval.
- `design-and-evaluation.md` — design rationale (embedding model, chunking,
  k, prompt format, vector store choice) + evaluation results summary.
- `ai-tooling.md` — which AI coding tools were used and how.
- `deployed.md` — optional live URL.

## Out of scope / explicitly not doing

- No re-ranking model beyond Chroma's built-in MMR (no external
  cross-encoder re-ranker) — kept simple per YAGNI.
- No multi-turn conversation memory — each question is answered
  independently (matches the assignment's single-turn `/chat` shape).
- No authentication/user accounts.
- No paid API usage; everything must run on free tiers.
