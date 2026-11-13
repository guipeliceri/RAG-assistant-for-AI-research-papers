"""Streamlit app in three tabs: papers matching a profile, a chat that cites its sources, and the hype forecast.

    streamlit run app.py
"""

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Streamlit Community Cloud runs `streamlit run app.py` from the repo root without PYTHONPATH.
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st  # noqa: E402

from rag_papers import data  # noqa: E402
from rag_papers import popular  # noqa: E402
from rag_papers import profiles as profile_files  # noqa: E402
from rag_papers.chat import ask  # noqa: E402
from rag_papers.embeddings import OnnxEmbedder  # noqa: E402
from rag_papers.llm import PROVIDERS, LLMError, OpenAICompatibleLLM  # noqa: E402
from rag_papers.retrieval import Index  # noqa: E402
from rag_papers.ui import theme  # noqa: E402
from rag_papers.validate import citations  # noqa: E402

MAX_QUESTIONS_PER_SESSION = 20
# The daily job updates the Dataset once a day; re-check it every hour.
REFRESH_SECONDS = 3600

st.set_page_config(page_title="AI papers radar", page_icon="📡", layout="wide")
st.markdown(theme.CSS, unsafe_allow_html=True)


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


def profiles_tab() -> None:
    available = profile_files.load_all(ROOT / "profiles")
    if not available:
        st.info("No profiles yet. Add one by pull request in profiles/.")
        return
    left, right = st.columns([3, 2])
    profile = left.selectbox("Profile", available, format_func=lambda p: p.name, key="profile")
    results = data.profile_results(files()[1], profile.slug)
    if not results:
        legacy_digest()
        return
    day = right.selectbox("Listing", list(results), key="profile_day")
    result = results[day]
    current = result["profile_fingerprint"] == profile.fingerprint
    st.markdown(theme.stamp(_parse_time(result.get("generated_at"))), unsafe_allow_html=True)
    metrics_panel(profile)
    note = "" if current else " These results come from an earlier version of the profile."
    st.caption(
        f"{len(result['papers'])} of {result['candidates']} new papers match. "
        f"{result['classified']} were read by the classifier ({result['model']}); the rest are outside "
        f"the profile's categories ({', '.join(profile.categories)}) or far from its description.{note}"
    )
    if result.get("errors"):
        st.warning(f"{result['errors']} papers could not be classified today and are retried on the next run.")
    for paper in result["papers"]:
        st.markdown(
            theme.card(
                paper["title"],
                paper["url"],
                meta=f"{theme.byline(paper['authors'])} · {paper['arxiv_id']}",
                body=paper.get("reason") or "",
                tags=theme.chips(paper.get("categories", [])),
            ),
            unsafe_allow_html=True,
        )


def metrics_panel(profile) -> None:
    """Test metrics when they exist for this exact profile version; otherwise say it is not evaluated."""
    metrics = data.profile_metrics(profile.slug, ROOT / "eval" / "profiles")
    test = metrics.get("test")
    if not test or test.get("profile_fingerprint") != profile.fingerprint:
        dev = metrics.get("dev")
        label = "dev labels only" if dev and dev.get("profile_fingerprint") == profile.fingerprint else "not evaluated"
        st.markdown(theme.chips([label], "warn"), unsafe_allow_html=True)
        return
    c = test["classifier"]

    def tile(name, block):
        value = "n/a" if block["value"] is None else f"{block['value']:.2f}"
        ci = f"95% CI {block['ci95'][0]:.2f} to {block['ci95'][1]:.2f}" if block["ci95"] else ""
        return theme.metric(name, value, ci)

    base = test["always_yes"]["precision"]["value"]
    st.markdown(
        '<div class="metric-row">'
        + tile("Recall", c["recall"])
        + tile("Precision", c["precision"])
        + theme.metric("Labelled test papers", str(c["n"]), f"share relevant {base:.2f}" if base is not None else "")
        + "</div>",
        unsafe_allow_html=True,
    )


def legacy_digest() -> None:
    with st.spinner("Loading today's papers..."):
        digest = data.latest_digest(files()[1])
    if digest is None:
        st.info("No papers have been selected yet.")
        return
    st.markdown(
        theme.stamp(_parse_time(digest.get("generated_at")))
        + theme.chips(["legacy ranking: top 5 by similarity", "not yet evaluated as a classifier"], "warn"),
        unsafe_allow_html=True,
    )
    st.caption(
        f"Listing of {digest['date']}: top {len(digest['papers'])} of {digest['candidates']} new cs.CL papers. "
        "The classifier that returns every matching paper has not run for this profile yet."
    )
    for paper in digest["papers"]:
        st.markdown(
            theme.card(
                paper["title"],
                paper["url"],
                meta=theme.byline(paper["authors"]),
                body=paper["summary"] or "Summary unavailable today. Open the paper for the abstract.",
            ),
            unsafe_allow_html=True,
        )


def _parse_time(value) -> datetime | None:
    if value is None or (isinstance(value, float) and value != value):
        return None
    try:
        ts = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    except ValueError:
        return None
    # Timestamps without a zone are written in UTC by the daily job.
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def hype_tab() -> None:
    with st.spinner("Loading votes..."):
        df = data.load_popular(files()[2])
    st.markdown("### Hype forecast")
    st.caption(
        "The forecast is being built: an agent will read each new paper and bet on the week's most hyped ones, "
        "with a calibrated confidence. Until then this tab shows the answer key it will be scored against: "
        "the week's most upvoted papers on Hugging Face Daily Papers."
    )
    if df.empty:
        st.markdown(theme.stamp(None), unsafe_allow_html=True)
        st.info("No votes collected yet. They appear after the next daily run.")
        return
    st.markdown(theme.stamp(_parse_time(pd_max(df["fetched_at"]))), unsafe_allow_html=True)
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    ranked = popular.top(df, monday, today, 5)
    if ranked.empty:
        ranked = popular.top(df, today - timedelta(days=7), today, 5)
    for rank, (_, paper) in enumerate(ranked.iterrows(), start=1):
        org = f" · {paper['organization']}" if paper["organization"] else ""
        st.markdown(
            theme.card(
                paper["title"],
                paper["url"],
                rank=f"#{rank}",
                meta=f"{theme.byline(paper['authors'])}{org}",
                body=paper["summary"],
                tags=theme.chips([f"▲ {paper['upvotes']} votes", f"featured {paper['daily_date']}"]),
            ),
            unsafe_allow_html=True,
        )


def pd_max(series):
    values = [v for v in series if v is not None and v == v]
    return max(values) if values else None


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
            cited = set(citations(answer.text))
            used = [s for s in answer.sources if s["arxiv_id"] in cited]
            unused = [s for s in answer.sources if s["arxiv_id"] not in cited]
            if used:
                st.markdown("**Sources**")
                for s in used:
                    meta = f"{s['arxiv_id']} · listed {s['listing_date']}"
                    st.markdown(theme.card(s["title"], s["url"], meta=meta), unsafe_allow_html=True)
            with st.expander(f"Also retrieved but not used ({len(unused)})"):
                st.caption(f"Searched for: *{answer.search_query}*")
                for s in unused:
                    st.markdown(f"- [{s['arxiv_id']}]({s['url']}) {s['title']} ({s['listing_date']})")


st.markdown(
    theme.hero(
        "AI papers radar",
        "New AI papers matched to your interests, a chat that cites every claim, and a forecast of what will be hyped.",
    ),
    unsafe_allow_html=True,
)
profiles, chat, hype = st.tabs(["Profiles", "Ask the papers", "Hype forecast"])
with profiles:
    profiles_tab()
with chat:
    chat_tab()
with hype:
    hype_tab()
