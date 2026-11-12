"""Custom look for the Streamlit app: fonts, cards, chips and the header.

Every piece of text that comes from a paper goes through `html.escape` before it is placed in
HTML, because titles and abstracts are external content.
"""

import html
from collections.abc import Iterable
from datetime import datetime
from zoneinfo import ZoneInfo

BRASILIA = ZoneInfo("America/Sao_Paulo")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html, body, [class*="css"], .stMarkdown, .stTextInput, .stSelectbox { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 2rem; max-width: 1100px; }
.hero { padding: 1.4rem 1.6rem; border-radius: 16px; margin-bottom: 1.2rem;
        background: linear-gradient(135deg, #1B2340 0%, #20163A 55%, #0E1117 100%);
        border: 1px solid #2A3350; }
.hero h1 { font-size: 1.9rem; font-weight: 700; margin: 0 0 .3rem 0; letter-spacing: -0.02em; }
.hero p { color: #AEB6CC; margin: 0; font-size: .98rem; }
.stTabs [data-baseweb="tab-list"] { gap: .4rem; }
.stTabs [data-baseweb="tab"] { padding: .5rem 1rem; border-radius: 10px 10px 0 0; font-weight: 600; }
.card { border: 1px solid #252C3D; background: #131824; border-radius: 14px;
        padding: 1rem 1.2rem; margin-bottom: .8rem; transition: border-color .15s ease; }
.card:hover { border-color: #3A4970; }
.card .rank { font-family: 'JetBrains Mono', monospace; color: #7C9CFF; font-size: .85rem; }
.card h3 { font-size: 1.08rem; font-weight: 600; margin: .15rem 0 .35rem 0; line-height: 1.35; }
.card h3 a { color: #E6E9F2; text-decoration: none; }
.card h3 a:hover { color: #7C9CFF; }
.card .meta { color: #8B93A7; font-size: .84rem; margin-bottom: .5rem; }
.card .body { color: #C9CFDD; font-size: .94rem; line-height: 1.55; }
.chips { margin-top: .6rem; }
.chip { display: inline-block; font-size: .76rem; padding: .12rem .55rem; margin: 0 .3rem .3rem 0;
        border-radius: 999px; background: #1E2640; color: #AFC0FF; border: 1px solid #2C3760; }
.chip.warn { background: #3A2A12; color: #FFCB7A; border-color: #5C4317; }
.chip.ok { background: #12301F; color: #7FE0A6; border-color: #1E5233; }
.stamp { color: #8B93A7; font-size: .84rem; margin: -.2rem 0 1rem 0; }
.metric-row { display: flex; gap: .8rem; flex-wrap: wrap; margin: .4rem 0 1rem 0; }
.metric { flex: 1 1 140px; border: 1px solid #252C3D; border-radius: 12px; padding: .7rem .9rem; background: #131824; }
.metric .label { color: #8B93A7; font-size: .78rem; text-transform: uppercase; letter-spacing: .04em; }
.metric .value { font-size: 1.35rem; font-weight: 700; margin-top: .1rem; }
.metric .ci { color: #8B93A7; font-size: .78rem; }
</style>
"""


def e(text: object) -> str:
    return html.escape(str(text if text is not None else ""), quote=True)


def hero(title: str, subtitle: str) -> str:
    return f'<div class="hero"><h1>{e(title)}</h1><p>{e(subtitle)}</p></div>'


def byline(authors: Iterable[str], limit: int = 6) -> str:
    authors = list(authors)
    return ", ".join(authors[:limit]) + (" et al." if len(authors) > limit else "")


def chips(labels: Iterable[str], kind: str = "") -> str:
    cls = f"chip {kind}".strip()
    return "".join(f'<span class="{cls}">{e(label)}</span>' for label in labels if label)


def card(
    title: str,
    url: str,
    meta: str = "",
    body: str = "",
    rank: str = "",
    tags: str = "",
) -> str:
    """One paper as a card. `tags` is pre-built HTML from `chips()`; everything else is escaped here."""
    parts = ['<div class="card">']
    if rank:
        parts.append(f'<div class="rank">{e(rank)}</div>')
    parts.append(f'<h3><a href="{e(url)}" target="_blank" rel="noopener">{e(title)}</a></h3>')
    if meta:
        parts.append(f'<div class="meta">{e(meta)}</div>')
    if body:
        parts.append(f'<div class="body">{e(body)}</div>')
    if tags:
        parts.append(f'<div class="chips">{tags}</div>')
    parts.append("</div>")
    return "".join(parts)


def metric(label: str, value: str, ci: str = "") -> str:
    return (
        f'<div class="metric"><div class="label">{e(label)}</div>'
        f'<div class="value">{e(value)}</div><div class="ci">{e(ci)}</div></div>'
    )


def stamp(when: datetime | None, what: str = "Last updated") -> str:
    """'Last updated 12/11/2026 at 18:07 (Brasília)', or a note that nothing ran yet."""
    if when is None:
        return f'<div class="stamp">{e(what)}: not yet</div>'
    local = when.astimezone(BRASILIA)
    return f'<div class="stamp">{e(what)} {local:%d/%m/%Y} at {local:%H:%M} (Brasília time)</div>'
