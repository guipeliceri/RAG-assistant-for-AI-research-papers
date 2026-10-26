"""Streamlit app: today's digest and the most-voted papers (no key needed), and a chat that cites its sources.

    streamlit run app.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

# Streamlit Community Cloud runs `streamlit run app.py` from the repo root without PYTHONPATH.
sys.path.insert(0, str(Path(__file__).parent / "src"))

import streamlit as st  # noqa: E402

from rag_papers import data  # noqa: E402
from rag_papers import popular  # noqa: E402
from rag_papers.chat import ask  # noqa: E402
from rag_papers.embeddings import OnnxEmbedder  # noqa: E402
from rag_papers.llm import PROVIDERS, LLMError, OpenAICompatibleLLM  # noqa: E402
from rag_papers.retrieval import Index  # noqa: E402

MAX_QUESTIONS_PER_SESSION = 20
# The daily job updates the Dataset once a day; re-check it every hour.
REFRESH_SECONDS = 3600

st.set_page_config(page_title="AI papers assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def embedder() -> OnnxEmbedder:
    return OnnxEmbedder()


@st.cache_resource(ttl=REFRESH_SECONDS, show_spinner=False)
def files() -> tuple[Path, Path, Path]:
    return data.fetch()


@st.cache_resource(ttl=REFRESH_SECONDS, show_spinner=False)
def load_index(papers_path: Path) -> Index:
    return Index(data.load_papers(papers_path), embedder())


def get_index() -> Index:
    with st.spinner("Waking up: loading papers and building the search index (about a minute after a cold start)..."):
        return load_index(files()[0])


def digest_tab() -> None:
    with st.spinner("Loading today's digest..."):
        digest = data.latest_digest(files()[1])
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


def most_voted_tab() -> None:
    with st.spinner("Loading votes..."):
        df = data.load_popular(files()[2])
    st.subheader("Most voted AI papers")
    st.caption(
        "Community upvotes on [Hugging Face Daily Papers](https://huggingface.co/papers), for every area "
        "of AI (not only cs.CL). arXiv itself publishes no view or rating counts. Votes are updated daily."
    )
    if df.empty:
        st.info("No votes collected yet. They appear after the next daily run.")
        return
    today = date.today()
    left, right = st.columns([3, 1])
    picked = left.date_input("Featured between", value=(today - timedelta(days=7), today), key="voted_range")
    n = right.number_input("How many", min_value=5, max_value=50, value=10, step=5)
    # While the user is mid-selection the widget returns a single date.
    start, end = (picked[0], picked[-1]) if isinstance(picked, (list, tuple)) and picked else (None, None)
    ranked = popular.top(df, start, end, int(n))
    if ranked.empty:
        st.info("No voted papers in this period.")
        return
    for rank, (_, paper) in enumerate(ranked.iterrows(), start=1):
        st.markdown(f"#### {rank}. [{paper['title']}]({paper['url']})")
        authors = list(paper["authors"])
        byline = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")
        org = f" · {paper['organization']}" if paper["organization"] else ""
        st.caption(
            f"▲ {paper['upvotes']} votes · 💬 {paper['comments']} · featured {paper['daily_date']}{org} · "
            f"[discussion]({paper['hf_url']})  \n{byline}"
        )
        if paper["summary"]:
            st.write(paper["summary"])


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
        picked = st.date_input("Papers listed between", value=(today - timedelta(days=7), today))
        # While the user is mid-selection the widget returns a single date.
        start, end = (picked[0], picked[-1]) if isinstance(picked, (list, tuple)) and picked else (None, None)

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
                answer = ask(question, get_index(), llm, start, end)
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


digest, voted, chat = st.tabs(["Today's digest", "Most voted", "Chat"])
with digest:
    digest_tab()
with voted:
    most_voted_tab()
with chat:
    chat_tab()
