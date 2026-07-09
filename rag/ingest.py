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
