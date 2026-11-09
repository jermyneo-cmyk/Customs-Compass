# rag.py
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"

@st.cache_resource(show_spinner="Loading Customs knowledge base...")
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # allow_dangerous_deserialization is required to load a local FAISS pickle
    # you created yourself. It is safe here because you built this index.
    return FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )

def retrieve(query, k=4):
    """Return the top-k most relevant chunks for a query."""
    vectorstore = load_vectorstore()
    return vectorstore.similarity_search(query, k=k)

def format_context(chunks):
    """Turn retrieved chunks into a numbered context block for the prompt."""
    blocks = []
    for i, c in enumerate(chunks, start=1):
        title = c.metadata.get("title", "Customs source")
        url = c.metadata.get("source", "")
        blocks.append(f"[{i}] {title} ({url})\n{c.page_content}")
    return "\n\n".join(blocks)

def answer_question(query, system_prompt, k=4):
    """Retrieve, then ask the LLM to answer grounded in the retrieved context.
    Returns (answer_text, source_chunks)."""
    chunks = retrieve(query, k=k)
    context = format_context(chunks)

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",
         "content": f"Context from official Singapore Customs sources:\n\n"
                    f"{context}\n\n"
                    f"Question: {query}\n\n"
                    f"Answer using only the context above. If the context does "
                    f"not contain the answer, say so and suggest contacting "
                    f"Singapore Customs."}
    ]
    response = llm.invoke(messages)
    return response.content, chunks