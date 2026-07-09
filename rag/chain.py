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
