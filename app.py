import streamlit as st
from rag import answer_question

import os
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]


ASK_SYSTEM_PROMPT = (
    "You are a helpful assistant for Singapore Customs. Answer traders' "
    "questions clearly and concisely using only the provided context. "
    "Use plain language, avoid jargon where possible, and when you use a "
    "Customs term, briefly explain it. Do not invent requirements."
)

def render_ask_customs():
    st.header("Ask Customs")
    st.caption("Ask any question about import, export, transhipment, duties or GST, "
               "permits or registration.")

    query = st.text_input("Your question",
                          placeholder="e.g. Do I need a permit to import goods into Singapore?")

    if st.button("Ask", type="primary") and query:
        with st.spinner("Searching Customs resources..."):
            answer, sources = answer_question(query, ASK_SYSTEM_PROMPT, k=4)

        st.markdown(answer)

        # Show the underlying sources for verification
        st.divider()
        st.subheader("Sources")
        seen = set()
        for c in sources:
            title = c.metadata.get("title", "Customs source")
            url = c.metadata.get("source", "")
            if url in seen:
                continue
            seen.add(url)
            st.markdown(f"- [{title}]({url})")


JOURNEY_SYSTEM_PROMPT = (
    "You are a Singapore Customs guidance assistant. The trader will describe "
    "what they intend to do. Using ONLY the provided context, lay out the "
    "Customs requirements as a logical, numbered sequence of steps they should "
    "consider, in the order they would normally happen (for example: register "
    "an account, classify goods, check duties/GST, apply for the right permit, "
    "arrange payment). For each step give a one-line explanation. If the "
    "context does not cover part of their scenario, say so rather than guessing."
)

def render_trade_journey():
    st.header("Guide My Trade Journey")
    st.caption("Describe what you want to do, and we'll map out the Customs "
               "steps to consider.")

    scenario = st.text_area(
        "Describe your trading activity",
        placeholder="e.g. I'm a new business and want to start importing "
                    "cosmetics from South Korea to sell in Singapore.",
        height=120,
    )

    if st.button("Map my journey", type="primary") and scenario:
        with st.spinner("Building your journey..."):
            # Frame the scenario as a retrieval query
            query = f"Customs requirements and steps for: {scenario}"
            answer, sources = answer_question(query, JOURNEY_SYSTEM_PROMPT, k=6)

        st.markdown(answer)

        st.divider()
        st.subheader("Official Customs references")
        seen = set()
        for c in sources:
            title = c.metadata.get("title", "Customs source")
            url = c.metadata.get("source", "")
            if url in seen:
                continue
            seen.add(url)
            st.markdown(f"- [{title}]({url})")


from helper_functions.utility import check_password
from rag import answer_question   # used by the render functions below

st.set_page_config(page_title="Customs Compass", page_icon="🧭", layout="centered")

# --- Login gate: nothing below runs until the user signs in ---
if not check_password():
    st.stop()

# --- Paste render_ask_customs() and render_trade_journey() here ---
# (including their ASK_SYSTEM_PROMPT / JOURNEY_SYSTEM_PROMPT constants)

# --- Sidebar navigation ---
st.sidebar.title("🧭 Customs Compass")
st.sidebar.caption(f"Signed in as {st.session_state.get('username', '')}")


def render_about():
    st.header("About Customs Compass")
    st.markdown('''
### Project scope
Customs Compass is a web-based generative-AI assistant that helps aspiring and existing traders find and understand Singapore Customs information through natural-language questions. It uses Retrieval-Augmented Generation (RAG) to ground every answer in a curated set of official Customs resources.

### Objectives
- Give traders an intuitive, personalised way to navigate authoritative Customs information without needing to know Customs terminology.
- Reduce reliance on call centres and officers for routine enquiries that can be self-served.
- Keep answers grounded in official sources, with links for verification.

### Data sources
A curated knowledge base of official Singapore Customs webpages. Their content is split into chunks, embedded, and stored in a FAISS vector database, with each chunk keeping its source title and URL as metadata.

### Features
- **Ask Customs** — ask a question in plain language and get a grounded answer with source links.
- **Guide My Trade Journey** — describe your intended trading activity and get an ordered set of Customs steps to consider, each pointing to the official page.
''')


def render_methodology():
    st.header("Methodology")
    st.markdown('''
Customs Compass uses a Retrieval-Augmented Generation (RAG) pipeline that runs in two phases.

**Indexing (run once, offline via ingest.py):**
1. Load each curated Customs source.
2. Split the text into overlapping chunks (RecursiveCharacterTextSplitter, 1000 characters with 150 overlap).
3. Embed each chunk with OpenAI text-embedding-3-small.
4. Store the vectors in a FAISS index on disk, keeping each chunk's source title and URL.

**Querying (every user request):**
1. Embed the user's question or scenario with the same embedding model.
2. Retrieve the top-k most similar chunks from the FAISS index.
3. Inject those chunks into the prompt as context.
4. Generate a grounded answer with ChatOpenAI (gpt-4o-mini, temperature 0) and display the sources.

Access is restricted by a simple login with two hardcoded users. The OpenAI key and login passwords are kept out of the code using Streamlit secrets.
''')

    st.subheader("Use case 1: Ask Customs")
    st.graphviz_chart('''
digraph ask_customs {
    rankdir=TB;
    node [shape=box, style=rounded, fontname="Helvetica"];
    q [label="User enters a question"];
    e [label="Embed the question"];
    s [label="Retrieve top-k chunks from FAISS"];
    p [label="Build prompt with retrieved context"];
    l [label="LLM generates grounded answer"];
    d [label="Display answer and source links"];
    q -> e -> s -> p -> l -> d;
}
''')

    st.subheader("Use case 2: Guide My Trade Journey")
    st.graphviz_chart('''
digraph trade_journey {
    rankdir=TB;
    node [shape=box, style=rounded, fontname="Helvetica"];
    q [label="User describes trading scenario"];
    f [label="Frame scenario as a retrieval query"];
    s [label="Retrieve top-k chunks from FAISS"];
    p [label="Build prompt: organise into ordered steps"];
    l [label="LLM generates step-by-step journey"];
    d [label="Display ordered steps and official references"];
    q -> f -> s -> p -> l -> d;
}
''')

page = st.sidebar.radio(
    "Choose a page",
    ["About Us", "Methodology","Ask Customs", "Guide My Trade Journey"],
)

if page == "Ask Customs":
    render_ask_customs()
elif page == "Guide My Trade Journey":
    render_trade_journey()
elif page == "Methodology":
    render_methodology()
else:
    render_about()

