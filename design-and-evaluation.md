# Design and Evaluation

## Design Decisions

- **Corpus**: 18 synthetic HR/IT policy documents, chosen to cover the
  topics explicitly named in the assignment (PTO, security, expense,
  remote work, holidays) plus adjacent HR/IT topics (parental leave,
  learning & development, acceptable use, DEI, workplace safety, and
  more), while staying legally clean (no real employer data).
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
- **Retrieval**: top-k similarity search (`k=4` default) via
  `similarity_search_with_relevance_scores`, implemented in
  `rag/chain.py::make_retrieve_fn`.
- **Generation**: Groq `llama-3.3-70b-versatile`, free tier, low latency.
- **Citations**: computed directly from the retrieved chunks, not parsed
  from LLM-generated text. This guarantees the cited passages are the ones
  the model actually saw, rather than depending on the LLM to format
  citations correctly. We retrieve broadly (`k=4`) so the model has ample
  context, but *cite narrowly*: citations are deduped to the best-scoring
  chunk per document and limited to sources whose score is within
  `CITATION_SCORE_RATIO` (0.9) of the top hit (capped at `MAX_CITATIONS`).
  This avoids padding answers with retrieved-but-unused sources — precise
  attribution rather than dumping the whole retrieval set
  (`rag/chain.py::_select_citations`). Each citation carries the source
  filename, and the UI links it to a `/corpus/<file>` endpoint that serves
  the original policy document.
- **Guardrails**: a similarity-score floor (`SIMILARITY_THRESHOLD=0.3`)
  skips the LLM call entirely and returns a fixed refusal message when no
  retrieved chunk is relevant enough; answers are capped at ~200 words via
  prompt instruction and a `max_tokens` limit on the LLM call.

## Evaluation Approach

- 25 questions in `eval/questions.json`, covering all 18 policy topics,
  each with a gold short answer and expected source document.
- **Groundedness** (required): LLM-as-judge — a second Groq call scores
  whether each answer is fully supported by its retrieved context.
- **Citation accuracy** (required): two-part check per citation — a
  programmatic check that the cited passage was among the chunks actually
  retrieved for that question, plus an LLM-as-judge check that the cited
  passage supports at least one factual claim in the answer (so citing an
  irrelevant retrieved chunk counts as a miss, not a hit).
- **Latency** (required): p50/p95 over the 20-question run.
- Full methodology in `scripts/evaluate.py`.

## Evaluation Results

Produced by running `python -m scripts.evaluate` with a real Groq API key
against the strict passage-support citation metric — see `eval/results.md`
for the full per-question report. Summary over the core 20-question set:

- Groundedness: 85.0%
- Citation accuracy: 85.0%
- Latency p50 / p95: 2265.0 ms / 2555.3 ms

The citation and groundedness misses concentrate on multi-part questions
where the answer combines facts spanning more than one retrieved passage —
the honest result of scoring each citation for passage-level support
rather than mere retrieval membership.

> The corpus was later expanded to 18 documents and the eval set to 25
> questions (one per topic). Re-run `python -m scripts.evaluate` to
> regenerate these numbers over the full set; the refresh was blocked on
> 2026-07-13 by the Groq free-tier daily token cap and should be run before
> final submission.
