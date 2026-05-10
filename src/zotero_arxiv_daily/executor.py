# -*- coding: utf-8 -*-
from __future__ import annotations

import random
from datetime import datetime

from loguru import logger
from omegaconf import DictConfig, ListConfig
from openai import OpenAI
from pyzotero import zotero
from tqdm import tqdm

from .construct_email import render_email
from .protocol import CorpusPaper
from .reranker import get_reranker_cls
from .retriever import get_retriever_cls
from .utils import glob_match, send_email


def normalize_path_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str,
) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of glob patterns or null, "
            'for example ["2026/survey/**"]. Single strings are not supported.'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(
            f"config.zotero.{config_key} must contain only glob pattern strings."
        )

    return list(patterns)


def normalize_tag_patterns(
    patterns: list[str] | ListConfig | None,
    config_key: str,
) -> list[str] | None:
    if patterns is None:
        return None

    if not isinstance(patterns, (list, ListConfig)):
        raise TypeError(
            f"config.zotero.{config_key} must be a list of tag glob patterns or null, "
            'for example ["auto/field/stat-mech", "auto/method/replica"].'
        )

    if any(not isinstance(pattern, str) for pattern in patterns):
        raise TypeError(
            f"config.zotero.{config_key} must contain only tag pattern strings."
        )

    return list(patterns)


def _make_corpus_paper(
    title: str,
    abstract: str,
    added_date: datetime,
    paths: list[str],
    tags: list[str],
) -> CorpusPaper:
    """Create CorpusPaper while remaining compatible with older protocol.py.

    Newer protocol.py should define:
        tags: list[str] = field(default_factory=list)

    If not, dynamically attach tags so this executor still works.
    """
    try:
        return CorpusPaper(
            title=title,
            abstract=abstract,
            added_date=added_date,
            paths=paths,
            tags=tags,
        )
    except TypeError:
        paper = CorpusPaper(
            title=title,
            abstract=abstract,
            added_date=added_date,
            paths=paths,
        )
        setattr(paper, "tags", tags)
        return paper


class Executor:
    def __init__(self, config: DictConfig):
        self.config = config

        zotero_cfg = config.zotero

        self.include_path_patterns = normalize_path_patterns(
            zotero_cfg.get("include_path", None),
            "include_path",
        )
        self.ignore_path_patterns = normalize_path_patterns(
            zotero_cfg.get("ignore_path", None),
            "ignore_path",
        )

        self.include_tag_patterns = normalize_tag_patterns(
            zotero_cfg.get("include_tags", None),
            "include_tags",
        )
        self.ignore_tag_patterns = normalize_tag_patterns(
            zotero_cfg.get("ignore_tags", None),
            "ignore_tags",
        )

        self.retrievers = {
            source: get_retriever_cls(source)(config)
            for source in config.executor.source
        }

        self.reranker = get_reranker_cls(config.executor.reranker)(config)

        self.openai_client = OpenAI(
            api_key=config.llm.api.key,
            base_url=config.llm.api.base_url,
        )

    def fetch_zotero_corpus(self) -> list[CorpusPaper]:
        logger.info("Fetching zotero corpus")

        zot = zotero.Zotero(
            self.config.zotero.user_id,
            "user",
            self.config.zotero.api_key,
        )

        collections = zot.everything(zot.collections())
        collections = {c["key"]: c for c in collections}

        corpus = zot.everything(
            zot.items(itemType="conferencePaper || journalArticle || preprint")
        )

        corpus = [
            c for c in corpus
            if c.get("data", {}).get("abstractNote", "") != ""
        ]

        def get_collection_path(col_key: str) -> str:
            parent = collections[col_key]["data"]["parentCollection"]

            if parent:
                return (
                    get_collection_path(parent)
                    + "/"
                    + collections[col_key]["data"]["name"]
                )

            return collections[col_key]["data"]["name"]

        out: list[CorpusPaper] = []

        for c in corpus:
            data = c.get("data", {})

            paths = [
                get_collection_path(col)
                for col in data.get("collections", [])
                if col in collections
            ]

            tags = [
                t.get("tag", "")
                for t in data.get("tags", [])
                if t.get("tag")
            ]

            try:
                added_date = datetime.strptime(
                    data["dateAdded"],
                    "%Y-%m-%dT%H:%M:%SZ",
                )
            except Exception:
                added_date = datetime.utcnow()

            out.append(
                _make_corpus_paper(
                    title=data.get("title", ""),
                    abstract=data.get("abstractNote", ""),
                    added_date=added_date,
                    paths=paths,
                    tags=tags,
                )
            )

        logger.info(f"Fetched {len(out)} zotero papers")
        return out

    def _matches_any_path_pattern(
        self,
        paper: CorpusPaper,
        patterns: list[str] | None,
    ) -> bool:
        if not patterns:
            return False

        paths = getattr(paper, "paths", []) or []

        return any(
            glob_match(path, pattern)
            for path in paths
            for pattern in patterns
        )

    def _matches_any_tag_pattern(
        self,
        paper: CorpusPaper,
        patterns: list[str] | None,
    ) -> bool:
        if not patterns:
            return False

        tags = getattr(paper, "tags", []) or []

        return any(
            glob_match(tag, pattern)
            for tag in tags
            for pattern in patterns
        )

    def filter_corpus(self, corpus: list[CorpusPaper]) -> list[CorpusPaper]:
        """Filter Zotero corpus by collection paths and/or tags.

        Important test-compatibility detail:
        some tests construct Executor-like objects without calling __init__.
        Therefore all filter attributes must be read through getattr(...).
        """
        include_path_patterns = getattr(self, "include_path_patterns", None)
        ignore_path_patterns = getattr(self, "ignore_path_patterns", None)
        include_tag_patterns = getattr(self, "include_tag_patterns", None)
        ignore_tag_patterns = getattr(self, "ignore_tag_patterns", None)

        has_include_paths = bool(include_path_patterns)
        has_include_tags = bool(include_tag_patterns)
        has_ignore_paths = bool(ignore_path_patterns)
        has_ignore_tags = bool(ignore_tag_patterns)

        if has_include_paths or has_include_tags:
            logger.info(
                "Selecting zotero papers by include filters: "
                f"include_path={include_path_patterns}, "
                f"include_tags={include_tag_patterns}"
            )

            selected = []

            for c in corpus:
                path_ok = self._matches_any_path_pattern(c, include_path_patterns)
                tag_ok = self._matches_any_tag_pattern(c, include_tag_patterns)

                # If both include_path and include_tags are set, OR logic is used.
                if path_ok or tag_ok:
                    selected.append(c)

            corpus = selected

        if has_ignore_paths or has_ignore_tags:
            logger.info(
                "Excluding zotero papers by ignore filters: "
                f"ignore_path={ignore_path_patterns}, "
                f"ignore_tags={ignore_tag_patterns}"
            )

            filtered = []

            for c in corpus:
                path_bad = self._matches_any_path_pattern(c, ignore_path_patterns)
                tag_bad = self._matches_any_tag_pattern(c, ignore_tag_patterns)

                if not (path_bad or tag_bad):
                    filtered.append(c)

            corpus = filtered

        if (
            has_include_paths
            or has_include_tags
            or has_ignore_paths
            or has_ignore_tags
        ):
            samples = random.sample(corpus, min(5, len(corpus))) if corpus else []
            samples_text = "\n".join(
                [
                    c.title
                    + " | paths="
                    + ", ".join(getattr(c, "paths", []) or [])
                    + " | tags="
                    + ", ".join(getattr(c, "tags", []) or [])
                    for c in samples
                ]
            )
            logger.info(f"Selected {len(corpus)} zotero papers:\n{samples_text}\n...")

        return corpus

    def _theory_filter_enabled(self) -> bool:
        """Strict score filtering is opt-in to keep tests/upstream behaviour stable."""
        try:
            return bool(self.config.executor.get("theory_filter", False))
        except Exception:
            return False

    def _min_score(self) -> float:
        try:
            return float(self.config.executor.get("min_score", 2.0))
        except Exception:
            return 2.0

    def _selection_mode(self) -> str:
        try:
            return str(self.config.executor.get("selection_mode", "min_score"))
        except Exception:
            return "min_score"

    def _filter_by_theory_or_code(self, papers):
        min_score = self._min_score()
        before_filter = len(papers)

        theory_min_physics = float(
            self.config.executor.get("theory_min_physics", 5.0)
        )
        theory_min_math = float(
            self.config.executor.get("theory_min_math", 5.0)
        )
        code_min = float(
            self.config.executor.get("code_min", 5.0)
        )
        code_min_context = float(
            self.config.executor.get("code_min_context", 3.0)
        )
        code_max_noise = float(
            self.config.executor.get("code_max_noise", 4.0)
        )

        def keep(p):
            score = p.score or 0.0
            physics = getattr(p, "physics_depth", 0.0) or 0.0
            math = getattr(p, "math_depth", 0.0) or 0.0
            code = getattr(p, "code_reproducibility", 0.0) or 0.0
            noise = getattr(p, "noise_penalty", 0.0) or 0.0
            zotero_score = getattr(p, "zotero_similarity", 0.0) or 0.0

            theory_lane = (
                physics >= theory_min_physics
                and math >= theory_min_math
            )

            code_lane = (
                code >= code_min
                and max(physics, math, zotero_score) >= code_min_context
                and noise <= code_max_noise
            )

            return score >= min_score and (theory_lane or code_lane)

        kept = [p for p in papers if keep(p)]

        logger.info(
            f"Kept {len(kept)} / {before_filter} papers using theory_or_code "
            f"selection "
            f"(min_score={min_score:.2f}, "
            f"theory=physics>={theory_min_physics},math>={theory_min_math}, "
            f"code=code>={code_min},context>={code_min_context},noise<={code_max_noise})"
        )

        return kept

    def _filter_by_min_score(self, papers):
        min_score = self._min_score()
        before_filter = len(papers)

        kept = [
            p for p in papers
            if (p.score or 0.0) >= min_score
        ]

        logger.info(
            f"Kept {len(kept)} / {before_filter} papers "
            f"after min_score={min_score:.2f} filtering"
        )

        return kept

    def run(self):
        corpus = self.fetch_zotero_corpus()
        corpus = self.filter_corpus(corpus)

        if len(corpus) == 0:
            logger.error(
                f"No zotero papers found. Please check your zotero settings:\n"
                f"{self.config.zotero}"
            )
            return

        all_papers = []

        for source, retriever in self.retrievers.items():
            logger.info(f"Retrieving {source} papers...")

            papers = retriever.retrieve_papers()

            if len(papers) == 0:
                logger.info(f"No {source} papers found")
                continue

            logger.info(f"Retrieved {len(papers)} {source} papers")
            all_papers.extend(papers)

        logger.info(f"Total {len(all_papers)} papers retrieved from all sources")

        reranked_papers = []

        if len(all_papers) > 0:
            logger.info("Reranking papers...")

            reranked_papers = self.reranker.rerank(all_papers, corpus)

            if self._theory_filter_enabled():
                mode = self._selection_mode()

                if mode == "theory_or_code":
                    reranked_papers = self._filter_by_theory_or_code(reranked_papers)
                else:
                    reranked_papers = self._filter_by_min_score(reranked_papers)

                if len(reranked_papers) == 0 and not self.config.executor.send_empty:
                    logger.info(
                        "No papers passed the score threshold. No email will be sent."
                    )
                    return

            reranked_papers = reranked_papers[: self.config.executor.max_paper_num]

            logger.info("Generating TLDR and affiliations...")

            for p in tqdm(reranked_papers):
                p.generate_tldr(self.openai_client, self.config.llm)
                p.generate_affiliations(self.openai_client, self.config.llm)

        elif not self.config.executor.send_empty:
            logger.info("No new papers found. No email will be sent.")
            return

        logger.info("Sending email...")

        email_content = render_email(reranked_papers)
        send_email(self.config, email_content)

        logger.info("Email sent successfully")
