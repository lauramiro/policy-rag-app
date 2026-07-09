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
