# -*- coding: utf-8 -*-
from __future__ import annotations

import html
import math
from typing import Optional

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


def get_empty_html():
    block_template = """
<p style="font-size:16px;">No Papers Today. Take a Rest!</p>
"""
    return block_template


def get_stars(score: float):
    """Return star HTML compatible with existing tests.

    Tests expect:
    - low score <= 6 -> empty string;
    - high score >= 8 -> five occurrences of "full-star";
    - mid score -> contains "star" and at least one full/half star.
    """
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'

    low = 6
    high = 8

    if score <= low:
        return ""

    if score >= high:
        return full_star * 5

    interval = (high - low) / 10
    star_num = math.ceil((score - low) / interval)

    full_star_num = int(star_num / 2)
    half_star_num = star_num - full_star_num * 2

    return "\n" + full_star * full_star_num + half_star * half_star_num + "\n"


def _fmt_score(x) -> str:
    if x is None:
        return "Unknown"

    try:
        return str(round(float(x), 1))
    except Exception:
        return "Unknown"


def _extra_score_html_from_paper(p: Paper) -> str:
    rows = []

    extra_fields = [
        ("Zotero similarity", getattr(p, "zotero_similarity", None)),
        ("Physics depth", getattr(p, "physics_depth", None)),
        ("Math depth", getattr(p, "math_depth", None)),
        ("Code/reproducibility", getattr(p, "code_reproducibility", None)),
        ("Noise penalty", getattr(p, "noise_penalty", None)),
    ]

    for name, value in extra_fields:
        if value is not None:
            rows.append(f"<br/><b>{_esc(name)}:</b> {_esc(_fmt_score(value))}")

    return "".join(rows)


def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    pdf_url: str,
    affiliations: Optional[str] = None,
    extra_scores_html: str = "",
):
    """Render one paper block.

    Keep the old public signature used by tests:
        get_block_html(title, authors, rate, tldr, pdf_url, affiliations)
    """
    safe_title = _esc(title)
    safe_authors = _esc(authors)
    safe_rate = _esc(rate)
    safe_tldr = _esc(tldr)
    safe_pdf_url = _esc(pdf_url)
    safe_affiliations = _esc(affiliations or "Unknown Affiliation")

    stars = ""

    try:
        stars = get_stars(float(rate))
    except Exception:
        stars = ""

    block_template = f"""
<div style="border:1px solid #ddd;border-radius:10px;padding:16px;margin:16px 0;">
  <h2 style="margin-top:0;font-size:18px;">{safe_title}</h2>

  <p><b>Authors:</b> {safe_authors}</p>
  <p><b>Affiliations:</b> {safe_affiliations}</p>

  <p>
    <b>Relevance:</b> {safe_rate} {stars}
    {extra_scores_html}
  </p>

  <p><b>TLDR:</b> {safe_tldr}</p>

  <p>
    <a href="{safe_pdf_url}">PDF</a>
  </p>
</div>
"""

    return block_template


def _authors_for_email(authors: list[str]) -> str:
    author_list = [a for a in authors or []]
    num_authors = len(author_list)

    if num_authors <= 5:
        return ", ".join(author_list)

    return ", ".join(author_list[:3] + ["..."] + author_list[-2:])


def _affiliations_for_email(affiliations: Optional[list[str]]) -> str:
    if affiliations is not None:
        aff = affiliations[:5]
        out = ", ".join(aff)

        if len(affiliations) > 5:
            out += ", ..."

        return out

    return "Unknown Affiliation"


def render_email(papers: list[Paper]) -> str:
    parts = []

    if len(papers) == 0:
        return framework.replace("__CONTENT__", get_empty_html())

    for p in papers:
        rate = _fmt_score(p.score)
        authors = _authors_for_email(p.authors)
        affiliations = _affiliations_for_email(p.affiliations)
        extra_scores_html = _extra_score_html_from_paper(p)

        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.tldr,
                p.pdf_url,
                affiliations,
                extra_scores_html=extra_scores_html,
            )
        )

    content = "\n" + "\n".join(parts) + "\n"
    return framework.replace("__CONTENT__", content)
