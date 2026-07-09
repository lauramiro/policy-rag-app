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
