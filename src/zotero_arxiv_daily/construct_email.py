# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import math

from .protocol import Paper


framework = """
<html>
<body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; line-height: 1.45;">
__CONTENT__
<hr/>
<p style="color:#777;font-size:13px;">
To unsubscribe, remove your email in your Github Action setting.
</p>
</body>
</html>
"""


def _esc(x) -> str:
    return html.escape(str(x or ""))


def get_empty_html() -> str:
    return """
<p style="font-size:16px;">No Papers Today. Take a Rest!</p>
"""


def _fmt_score(x) -> str:
    if x is None:
        return "N/A"
    try:
        return f"{float(x):.1f}/10"
    except Exception:
        return "N/A"


def get_stars(score: float | None) -> str:
    if score is None:
        return ""

    full_star = "⭐"
    low = 4.0
    high = 8.0

    if score <= low:
        return ""

    if score >= high:
        return full_star * 5

    interval = (high - low) / 5
    star_num = math.ceil((score - low) / interval)
    return full_star * max(1, min(5, star_num))


def _authors_text(authors: list[str]) -> str:
    author_list = [a for a in authors or []]
    num_authors = len(author_list)

    if num_authors == 0:
        return "Unknown authors"

    if num_authors <= 5:
        return ", ".join(author_list)

    return ", ".join(author_list[:3] + ["..."] + author_list[-2:])


def _affiliations_text(affiliations: list[str] | None) -> str:
    if affiliations is None:
        return "Unknown affiliation"

    if len(affiliations) == 0:
        return "Unknown affiliation"

    shown = affiliations[:5]
    text = ", ".join(shown)

    if len(affiliations) > 5:
        text += ", ..."

    return text


def get_block_html(p: Paper) -> str:
    title = _esc(p.title)
    authors = _esc(_authors_text(p.authors))
    affiliations = _esc(_affiliations_text(p.affiliations))
    tldr = _esc(p.tldr or "")

    score = _fmt_score(p.score)
    zotero_similarity = _fmt_score(getattr(p, "zotero_similarity", None))
    physics_depth = _fmt_score(getattr(p, "physics_depth", None))
    math_depth = _fmt_score(getattr(p, "math_depth", None))
    code_signal = _fmt_score(getattr(p, "code_reproducibility", None))
    noise_penalty = _fmt_score(getattr(p, "noise_penalty", None))

    stars = get_stars(p.score)

    paper_url = _esc(p.url)
    pdf_url = _esc(p.pdf_url or p.url)

    return f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0;">
  <h2 style="margin-top:0;font-size:18px;">
    <a href="{paper_url}" style="text-decoration:none;color:#1f4e79;">{title}</a>
  </h2>

  <p style="margin:4px 0;color:#333;"><b>Authors:</b> {authors}</p>
  <p style="margin:4px 0;color:#555;"><b>Affiliations:</b> {affiliations}</p>

  <p style="margin:8px 0;">
    <b>Final relevance:</b> {score} {stars}<br/>
    <b>Zotero similarity:</b> {zotero_similarity}<br/>
    <b>Physics depth:</b> {physics_depth}<br/>
    <b>Math depth:</b> {math_depth}<br/>
    <b>Code/reproducibility signal:</b> {code_signal}<br/>
    <b>Noise penalty:</b> {noise_penalty}
  </p>

  <p style="margin:8px 0;"><b>Summary:</b> {tldr}</p>

  <p style="margin:8px 0;">
    <a href="{paper_url}">arXiv page</a>
    &nbsp;|&nbsp;
    <a href="{pdf_url}">PDF</a>
  </p>
</div>
"""


def render_email(papers: list[Paper]) -> str:
    if len(papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    parts = [get_block_html(p) for p in papers]
    content = "\n".join(parts)
    return framework.replace("__CONTENT__", content)
