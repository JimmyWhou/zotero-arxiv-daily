# -*- coding: utf-8 -*-
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from omegaconf import DictConfig

from ..protocol import CorpusPaper, Paper


def _safe_text(*parts: str | None) -> str:
    return " ".join([p for p in parts if p]).lower()


PHYSICS_TERMS = {
    "phase transition": 1.4,
    "phase transitions": 1.4,
    "critical phenomena": 1.4,
    "critical phenomenon": 1.3,
    "critical point": 1.1,
    "criticality": 1.1,
    "universality": 1.2,
    "universal scaling": 1.2,
    "renormalization group": 1.5,
    "renormalisation group": 1.5,
    "rg flow": 1.2,
    "wilsonian": 1.2,
    "mean field": 1.0,
    "mean-field": 1.0,
    "spin glass": 1.5,
    "spin glasses": 1.5,
    "replica": 1.2,
    "replica symmetry breaking": 1.6,
    "cavity method": 1.3,
    "random matrix": 1.4,
    "random matrices": 1.4,
    "large deviations": 1.4,
    "large deviation principle": 1.5,
    "rate function": 1.2,
    "ising": 1.2,
    "conformal field theory": 1.5,
    "cft": 1.2,
    "integrable": 1.2,
    "integrability": 1.2,
    "random energy model": 1.5,
    "rem": 1.1,
    "grem": 1.1,
    "boltzmann machine": 1.1,
    "energy based model": 1.0,
    "scaling law": 1.0,
    "scaling laws": 1.0,
    "teacher student": 1.0,
    "teacher-student": 1.0,
}

MATH_TERMS = {
    "theorem": 1.2,
    "proof": 1.2,
    "lemma": 1.0,
    "proposition": 1.0,
    "corollary": 0.8,
    "derivation": 1.2,
    "derive": 0.8,
    "asymptotic": 1.1,
    "limit theorem": 1.2,
    "large deviation": 1.3,
    "rate function": 1.2,
    "variational principle": 1.3,
    "saddle point": 1.2,
    "saddle-point": 1.2,
    "mean-field equation": 1.0,
    "fixed point": 0.9,
    "scaling exponent": 1.1,
    "critical exponent": 1.2,
    "partition function": 1.1,
    "free energy": 1.1,
    "hamiltonian": 0.9,
    "rigorous": 1.1,
    "exact solution": 1.2,
    "solvable": 0.8,
    "closed form": 0.8,
    "bound": 0.7,
    "convergence": 0.7,
}

CODE_TERMS = {
    "github": 2.0,
    "gitlab": 1.6,
    "code is available": 2.0,
    "code available": 2.0,
    "implementation": 1.0,
    "open-source": 1.0,
    "open source": 1.0,
    "reproducible": 1.4,
    "reproducibility": 1.4,
    "dataset": 0.6,
    "benchmark": 0.4,
    "experiments": 0.6,
    "simulation": 0.7,
    "numerical": 0.7,
}

NOISE_TERMS = {
    "medical image": 2.0,
    "medical imaging": 2.0,
    "remote sensing": 2.0,
    "object detection": 1.6,
    "semantic segmentation": 1.8,
    "segmentation": 1.2,
    "prompt engineering": 1.8,
    "retrieval augmented generation": 1.5,
    "rag": 0.8,
    "chatbot": 1.5,
    "question answering": 1.2,
    "leaderboard": 1.2,
    "dataset paper": 1.5,
    "survey of": 0.8,
    "review of": 0.8,
}


def _weighted_keyword_score(text: str, weights: dict[str, float], cap: float = 10.0) -> float:
    score = 0.0
    for term, weight in weights.items():
        if term in text:
            score += weight
    return float(min(cap, score))


def _normalise_similarity(scores: np.ndarray) -> np.ndarray:
    """Map Zotero similarity scores to a stable 0--10 scale."""
    if len(scores) == 0:
        return scores

    arr = np.asarray(scores, dtype=float)

    # Existing score is roughly similarity * 10. Clip first.
    arr = np.clip(arr, 0.0, 10.0)

    # If all scores are nearly identical, keep them as-is.
    if float(arr.max() - arr.min()) < 1e-8:
        return arr

    # Mild min-max normalisation helps when embedding scores occupy a narrow band.
    arr = 10.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return np.clip(arr, 0.0, 10.0)


def _heuristic_components(paper: Paper) -> dict[str, float]:
    text = _safe_text(
        paper.title,
        paper.abstract,
        paper.full_text,
        paper.url,
        paper.pdf_url,
    )

    physics = _weighted_keyword_score(text, PHYSICS_TERMS, cap=10.0)
    math = _weighted_keyword_score(text, MATH_TERMS, cap=10.0)
    code = _weighted_keyword_score(text, CODE_TERMS, cap=10.0)
    noise = _weighted_keyword_score(text, NOISE_TERMS, cap=10.0)

    return {
        "physics_depth": physics,
        "math_depth": math,
        "code_reproducibility": code,
        "noise_penalty": noise,
    }


class BaseReranker(ABC):
    def __init__(self, config: DictConfig):
        self.config = config

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        """Hybrid reranking.

        Step 1:
            Compute the original Zotero-library similarity score.

        Step 2:
            Add a physics/math/code-aware heuristic score.

        Final score is still written to paper.score, so existing email rendering
        and executor logic remain compatible.
        """
        if not candidates:
            return []

        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)

        if len(corpus) == 0:
            zotero_scores = np.zeros(len(candidates), dtype=float)
        else:
            time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
            time_decay_weight = time_decay_weight / time_decay_weight.sum()

            sim = self.get_similarity_score(
                [c.abstract or c.title for c in candidates],
                [c.abstract or c.title for c in corpus],
            )

            assert sim.shape == (len(candidates), len(corpus))

            zotero_scores = (sim * time_decay_weight).sum(axis=1) * 10.0
            zotero_scores = _normalise_similarity(zotero_scores)

        for paper, zotero_score in zip(candidates, zotero_scores):
            components = _heuristic_components(paper)

            physics = components["physics_depth"]
            math = components["math_depth"]
            code = components["code_reproducibility"]
            noise = components["noise_penalty"]

            final_score = (
                0.30 * float(zotero_score)
                + 0.30 * physics
                + 0.25 * math
                + 0.15 * code
                - 0.25 * noise
            )

            final_score = float(np.clip(final_score, 0.0, 10.0))

            paper.zotero_similarity = float(zotero_score)
            paper.physics_depth = physics
            paper.math_depth = math
            paper.code_reproducibility = code
            paper.noise_penalty = noise
            paper.score = final_score

        candidates = sorted(candidates, key=lambda x: x.score or 0.0, reverse=True)
        return candidates

    @abstractmethod
    def get_similarity_score(self, s1: list[str], s2: list[str]) -> np.ndarray:
        raise NotImplementedError


registered_rerankers = {}


def register_reranker(name: str):
    def decorator(cls):
        registered_rerankers[name] = cls
        return cls

    return decorator


def get_reranker_cls(name: str) -> Type[BaseReranker]:
    if name not in registered_rerankers:
        raise ValueError(f"Reranker {name} not found")
    return registered_rerankers[name]
