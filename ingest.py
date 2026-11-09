# ingest.py  —  run with:  python ingest.py
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from sources import SOURCES

load_dotenv()  # reads OPENAI_API_KEY from .env for this local run

INDEX_DIR = "faiss_index"

def load_documents():
    docs = []
    for src in SOURCES:
        print(f"Loading: {src['title']}")
        if src["type"] == "pdf":
            loader = PyPDFLoader(src["url"])
        else:
            loader = WebBaseLoader(src["url"])
        loaded = loader.load()
        # Attach clean metadata so the app can cite the source later
        for d in loaded:
            d.metadata["title"] = src["title"]
            d.metadata["source"] = src["url"]
        docs.extend(loaded)
    return docs

def main():
    docs = load_documents()
    print(f"Loaded {len(docs)} raw documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    vectorstore.save_local(INDEX_DIR)
    print(f"Saved FAISS index to ./{INDEX_DIR}/")

if __name__ == "__main__":
    main()