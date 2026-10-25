"""Streamlit app: today's digest (no key needed) and a chat that cites its sources.

    streamlit run app.py
"""

import json
from datetime import date, timedelta
from pathlib import Path

import streamlit as st

from rag_papers import store
from rag_papers.chat import ask
from rag_papers.embeddings import SentenceTransformerEmbedder
from rag_papers.ingest import DEFAULT_PATH
from rag_papers.llm import PROVIDERS, LLMError, OpenAICompatibleLLM
from rag_papers.retrieval import Index

DIGEST_DIR = Path("digests")
MAX_QUESTIONS_PER_SESSION = 20

st.set_page_config(page_title="AI papers assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading papers and building the search index...")
def load_index() -> Index:
    return Index(store.load(DEFAULT_PATH), SentenceTransformerEmbedder())


def latest_digest() -> dict | None:
    files = sorted(DIGEST_DIR.glob("*.json"))
    return json.loads(files[-1].read_text()) if files else None


def digest_tab() -> None:
    digest = latest_digest()
    if digest is None:
        st.info("No digest has been published yet.")
        return
    st.subheader(f"Digest for {digest['date']}")
    st.caption(
        f"Top {len(digest['papers'])} of {digest['candidates']} new cs.CL papers, "
        "ranked by similarity to the profile *RAG and information retrieval*."
    )
    for paper in digest["papers"]:
        st.markdown(f"#### [{paper['title']}]({paper['url']})")
        st.caption(", ".join(paper["authors"][:6]) + (" et al." if len(paper["authors"]) > 6 else ""))
        st.write(paper["summary"] or "_Summary unavailable today. Open the paper for the abstract._")


def chat_tab() -> None:
    with st.sidebar:
        st.header("Your API key")
        provider_id = st.selectbox("Provider", list(PROVIDERS), format_func=lambda p: PROVIDERS[p].name)
        provider = PROVIDERS[provider_id]
        model = st.text_input("Model", value=provider.default_model)
        # Kept only in this browser session's state: never written to disk or logged.
        st.text_input("API key", type="password", key="api_key", help=f"Get one at {provider.key_url}")
        st.caption("🔒 Your key stays in this session only. It is never stored or logged.")
        today = date.today()
        start, end = st.date_input("Papers listed between", value=(today - timedelta(days=7), today))

    st.session_state.setdefault("questions", 0)
    question = st.chat_input("e.g. What came out this week about agents?")
    if not question:
        return
    if not st.session_state.get("api_key"):
        st.warning("Enter your API key in the sidebar to use the chat.")
        return
    if st.session_state["questions"] >= MAX_QUESTIONS_PER_SESSION:
        st.warning("Question limit for this session reached. Reload the page to start a new session.")
        return
    st.session_state["questions"] += 1

    st.chat_message("user").write(question)
    llm = OpenAICompatibleLLM(provider, st.session_state["api_key"], model)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching and answering..."):
                answer = ask(question, load_index(), llm, start, end)
        except LLMError as exc:
            st.error(f"{exc}. Check your key, model name and free-tier quota.")
            return
        st.markdown(answer.text)
        v = answer.validation
        if answer.sources and not v.ok:
            problems = []
            if not v.has_citations:
                problems.append("no citations")
            if v.invalid_citations:
                problems.append("cites papers that were not retrieved: " + ", ".join(v.invalid_citations))
            if v.unsupported_numbers:
                problems.append("numbers not found in the sources: " + ", ".join(v.unsupported_numbers))
            st.warning("⚠️ Check this answer: " + "; ".join(problems))
        if answer.sources:
            st.markdown("**Sources**")
            for s in answer.sources:
                st.markdown(f"- [{s['arxiv_id']}]({s['url']}) {s['title']} ({s['listing_date']})")


digest, chat = st.tabs(["Today's digest", "Chat"])
with digest:
    digest_tab()
with chat:
    chat_tab()
