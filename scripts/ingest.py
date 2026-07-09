from rag.config import Config
from rag.ingest import build_vectorstore


def main():
    config = Config()
    vectorstore = build_vectorstore(config.corpus_dir, config.chroma_dir, config)
    count = vectorstore._collection.count()
    print(f"Ingested {count} chunks into '{config.chroma_dir}' from '{config.corpus_dir}'.")


if __name__ == "__main__":
    main()
