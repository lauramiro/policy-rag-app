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

   Use Python 3.11 or 3.12 (`tokenizers` has no prebuilt wheels for 3.13+).

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   On Windows, if the install fails with `WinError 206` (filename too
   long), PyTorch's nested files are exceeding the 260-character path
   limit: create the venv at a shorter path (e.g. `C:\venvs\rag`) or
   enable Windows long paths.

3. Copy `local.secrets.example` to `local.secrets` and fill in your free
   Groq API key (get one at console.groq.com):

   ```bash
   cp local.secrets.example local.secrets
   ```

## Build the index

```bash
python -m scripts.ingest
```

(Run as a module, not `python scripts/ingest.py` — the latter puts
`scripts/` rather than the project root on Python's import path, so the
sibling `rag` package won't be found.)

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
python -m scripts.evaluate
```

Writes `eval/results.md` with groundedness %, citation accuracy %, and
p50/p95 latency over the question set in `eval/questions.json`.

## Project layout

See `docs/superpowers/specs/2026-07-09-policy-rag-app-design.md` for the
full design rationale, and `design-and-evaluation.md` for evaluation
results.
