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
    # Core statistical mechanics
    "statistical mechanics": 2.0,
    "statistical physics": 1.8,
    "partition function": 1.8,
    "free energy": 1.8,
    "gibbs measure": 1.8,
    "gibbs state": 1.5,
    "boltzmann distribution": 1.3,
    "hamiltonian": 1.0,
    "order parameter": 1.5,
    "correlation length": 1.5,
    "thermodynamic limit": 1.8,
    "infinite volume": 1.4,
    "finite size scaling": 1.6,
    "finite-size scaling": 1.6,

    # Phase transitions and criticality
    "phase transition": 2.0,
    "phase transitions": 2.0,
    "critical phenomena": 2.0,
    "critical phenomenon": 1.8,
    "critical point": 1.5,
    "criticality": 1.5,
    "critical exponent": 1.8,
    "critical exponents": 1.8,
    "scaling exponent": 1.5,
    "scaling exponents": 1.5,
    "universality": 1.8,
    "universality class": 1.8,
    "universal scaling": 1.6,

    # RG / mean field / saddle point
    "renormalization group": 2.0,
    "renormalisation group": 2.0,
    "renormalization group fixed point": 2.2,
    "renormalisation group fixed point": 2.2,
    "rg flow": 1.5,
    "wilsonian": 1.5,
    "functional rg": 1.5,
    "frg": 1.0,
    "mean field": 1.5,
    "mean-field": 1.5,
    "mean field theory": 1.6,
    "mean-field theory": 1.6,
    "mean-field limit": 1.5,
    "saddle point": 1.5,
    "saddle-point": 1.5,

    # Disordered systems / spin glasses / replica
    "spin glass": 2.2,
    "spin glasses": 2.2,
    "sherrington-kirkpatrick": 2.0,
    "sherrington kirkpatrick": 2.0,
    "sk model": 1.8,
    "replica": 1.6,
    "replica trick": 2.0,
    "replica method": 1.8,
    "replica symmetry breaking": 2.2,
    "rsb": 1.5,
    "cavity method": 1.8,
    "nishimori": 1.8,
    "parisi": 2.0,
    "overlap distribution": 1.8,
    "quenched disorder": 1.6,
    "quenched": 1.0,
    "annealed": 1.0,

    # Random matrices / large deviations / probability
    "random matrix": 2.0,
    "random matrices": 2.0,
    "random matrix theory": 2.2,
    "spectral statistics": 1.5,
    "eigenvalue statistics": 1.5,
    "large deviations": 2.0,
    "large deviation principle": 2.2,
    "rate function": 1.8,
    "varadhan": 1.6,
    "cramer": 1.2,
    "sanov": 1.2,
    "limiting distribution": 1.2,

    # Models
    "ising": 1.5,
    "ising model": 2.0,
    "potts model": 1.6,
    "random energy model": 2.2,
    "generalized random energy model": 2.2,
    "grem": 1.5,
    "rem": 1.5,
    "lee yang": 1.8,
    "yang lee": 1.8,
    "transfer matrix": 1.5,

    # CFT / integrability / mathematical physics
    "conformal field theory": 2.0,
    "cft": 1.5,
    "central charge": 1.6,
    "operator product expansion": 1.5,
    "ope": 1.0,
    "integrable model": 1.8,
    "integrability": 1.6,
    "bethe ansatz": 1.8,
    "yang-baxter": 1.5,
    "yang baxter": 1.5,

    # ML theory with stat-phys flavour
    "teacher student": 1.5,
    "teacher-student": 1.5,
    "teacher student model": 1.6,
    "boltzmann machine": 1.3,
    "restricted boltzmann machine": 1.4,
    "energy based model": 1.2,
    "energy-based model": 1.2,
    "scaling law": 1.2,
    "scaling laws": 1.2,
    "neural network theory": 1.2,
    "generalization theory": 1.0,
}


MATH_TERMS = {
    # Explicit mathematical structure
    "theorem": 1.7,
    "lemma": 1.2,
    "proposition": 1.2,
    "corollary": 1.0,
    "proof": 1.7,
    "rigorous": 1.6,
    "derive": 1.2,
    "derivation": 1.8,
    "we derive": 1.4,
    "exact formula": 1.6,
    "closed form": 1.3,
    "exact solution": 1.6,
    "solvable": 1.0,

    # Limits / asymptotics
    "asymptotic": 1.4,
    "asymptotic expansion": 1.6,
    "large n limit": 1.5,
    "thermodynamic limit": 1.8,
    "infinite volume": 1.4,
    "hydrodynamic limit": 1.4,
    "scaling limit": 1.6,
    "mean field limit": 1.5,
    "limiting distribution": 1.4,
    "convergence": 1.0,

    # Variational / saddle point / fixed point
    "variational formula": 1.6,
    "variational principle": 1.8,
    "saddle point": 1.5,
    "saddle-point": 1.5,
    "fixed point equation": 1.4,
    "self consistent equation": 1.3,
    "self-consistent equation": 1.3,
    "mean-field equation": 1.4,

    # Stat-mech mathematics
    "partition function": 1.5,
    "free energy": 1.5,
    "rate function": 1.6,
    "large deviation": 1.8,
    "large deviations": 1.8,
    "large deviation principle": 2.0,
    "legendre transform": 1.4,
    "laplace principle": 1.4,
    "varadhan": 1.5,

    # Probability / spectral / operator language
    "bound": 0.8,
    "concentration": 1.0,
    "martingale": 1.0,
    "spectral": 1.0,
    "eigenvalue": 1.0,
    "operator": 0.7,
}


CODE_TERMS = {
    "github": 2.0,
    "gitlab": 1.6,
    "code is available": 2.0,
    "code available": 2.0,
    "codebase": 1.6,
    "implementation": 1.0,
    "open-source": 1.0,
    "open source": 1.0,
    "reproducible": 1.4,
    "reproducibility": 1.4,
    "simulation": 0.8,
    "numerical": 0.8,
    "experiments": 0.6,
    "dataset": 0.4,
    "benchmark": 0.3,
}


NOISE_TERMS = {
    # Application-heavy areas
    "medical image": 2.5,
    "medical imaging": 2.5,
    "remote sensing": 2.5,
    "object detection": 2.2,
    "semantic segmentation": 2.4,
    "segmentation": 2.2,
    "gui": 2.2,
    "grounding": 1.6,
    "video generation": 2.4,
    "image generation": 2.0,
    "zero-shot": 1.0,

    # LLM/application engineering
    "prompt engineering": 2.5,
    "retrieval augmented generation": 2.0,
    "rag": 1.2,
    "chatbot": 2.0,
    "question answering": 1.6,
    "leaderboard": 2.2,
    "llm leaderboard": 2.8,
    "arena": 1.6,
    "benchmark": 1.8,
    "dataset": 1.5,
    "dataset paper": 2.0,

    # MoE engineering
    "mixture of experts": 2.0,
    "moe": 1.5,
    "expert routing": 1.5,
    "expert pool": 1.5,

    # Generic applied ML
    "supervised ml": 1.2,
    "classification accuracy": 1.0,
    "downstream task": 0.8,
    "instruction tuning": 1.2,
    "fine-tuning benchmark": 1.2,
}


def _weighted_keyword_score(text: str, weights: dict[str, float], cap: float = 10.0) -> float:
    score = 0.0
    for term, weight in weights.items():
        if term in text:
            score += weight
    return float(min(cap, score))


def _normalise_similarity(scores: np.ndarray) -> np.ndarray:
    """Map Zotero similarity scores to a stable 0--10 scale for theory-filter mode."""
    if len(scores) == 0:
        return scores

    arr = np.asarray(scores, dtype=float)
    arr = np.clip(arr, 0.0, 10.0)

    if float(arr.max() - arr.min()) < 1e-8:
        return arr

    arr = 10.0 * (arr - arr.min()) / (arr.max() - arr.min())
    return np.clip(arr, 0.0, 10.0)


def _heuristic_components(paper: Paper) -> dict[str, float]:
    text = _safe_text(
        getattr(paper, "title", None),
        getattr(paper, "abstract", None),
        getattr(paper, "full_text", None),
        getattr(paper, "url", None),
        getattr(paper, "pdf_url", None),
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

    def _theory_filter_enabled(self) -> bool:
        """Return whether strict theoretical-stat-phys scoring is enabled.

        Defaults to False to preserve upstream-compatible behaviour in tests.
        """
        try:
            if self.config is None:
                return False
            return bool(self.config.executor.get("theory_filter", False))
        except Exception:
            return False

    def rerank(self, candidates: list[Paper], corpus: list[CorpusPaper]) -> list[Paper]:
        """Rerank papers.

        Default behaviour:
            rank only by Zotero embedding similarity.

        When config.executor.theory_filter=true:
            use a theoretical-statistical-physics score based on
            physics depth, mathematical depth, code/reproducibility, and
            engineering/application noise.
        """
        if not candidates:
            return []

        corpus = sorted(corpus, key=lambda x: x.added_date, reverse=True)

        if len(corpus) == 0:
            raw_zotero_scores = np.zeros(len(candidates), dtype=float)
        else:
            time_decay_weight = 1 / (1 + np.log10(np.arange(len(corpus)) + 1))
            time_decay_weight = time_decay_weight / time_decay_weight.sum()

            sim = self.get_similarity_score(
                [c.abstract or c.title for c in candidates],
                [c.abstract or c.title for c in corpus],
            )

            assert sim.shape == (len(candidates), len(corpus))

            raw_zotero_scores = (sim * time_decay_weight).sum(axis=1) * 10.0

        theory_filter = self._theory_filter_enabled()

        if theory_filter:
            zotero_scores_for_theory = _normalise_similarity(raw_zotero_scores)
        else:
            zotero_scores_for_theory = raw_zotero_scores

        for paper, raw_score, theory_zotero_score in zip(
            candidates,
            raw_zotero_scores,
            zotero_scores_for_theory,
        ):
            components = _heuristic_components(paper)

            physics = components["physics_depth"]
            math = components["math_depth"]
            code = components["code_reproducibility"]
            noise = components["noise_penalty"]

            # Always attach diagnostics for email rendering.
            paper.zotero_similarity = float(theory_zotero_score)
            paper.physics_depth = physics
            paper.math_depth = math
            paper.code_reproducibility = code
            paper.noise_penalty = noise

            if not theory_filter:
                # Upstream-compatible path: score is pure raw Zotero similarity.
                paper.score = float(raw_score)
                continue

            # Theory-first scoring.
            # Code is useful only if the paper already has real physics/math signal.
            theory_gate = min(physics, math)

            if theory_gate >= 5.0:
                gated_code = code
            elif theory_gate >= 3.5:
                gated_code = 0.35 * code
            else:
                gated_code = 0.10 * code

            final_score = (
                0.15 * float(theory_zotero_score)
                + 0.45 * physics
                + 0.35 * math
                + 0.05 * gated_code
                - 0.80 * noise
            )

            # Penalise ML-engineering papers without real theoretical structure.
            if physics < 4.0 and math < 5.0:
                final_score -= 2.5

            # Do not let GitHub/code rescue an application paper.
            if physics < 3.5 and code > 5.0:
                final_score -= 2.0

            # Reward papers with both real physics and real mathematics.
            if physics >= 6.0 and math >= 6.0:
                final_score += 1.0

            if physics >= 7.0 and math >= 7.0:
                final_score += 0.8

            paper.score = float(np.clip(final_score, 0.0, 10.0))

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
