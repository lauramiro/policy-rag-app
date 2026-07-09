from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from rag.config import Config


def get_retriever(vectorstore: Chroma, config: Config) -> VectorStoreRetriever:
    search_type = "mmr" if config.retrieval_mode == "mmr" else "similarity"
    return vectorstore.as_retriever(
        search_type=search_type, search_kwargs={"k": config.top_k}
    )
