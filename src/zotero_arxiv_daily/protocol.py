# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, TypeVar

import tiktoken
from loguru import logger
from openai import OpenAI


RawPaperItem = TypeVar("RawPaperItem")


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    # Added by the hybrid reranker.
    zotero_similarity: Optional[float] = None
    physics_depth: Optional[float] = None
    math_depth: Optional[float] = None
    code_reproducibility: Optional[float] = None
    noise_penalty: Optional[float] = None

    def _score_line(self) -> str:
        parts = []

        if self.score is not None:
            parts.append(f"Final relevance score: {self.score:.1f}/10")

        if self.zotero_similarity is not None:
            parts.append(f"Zotero similarity: {self.zotero_similarity:.1f}/10")

        if self.physics_depth is not None:
            parts.append(f"Physics depth: {self.physics_depth:.1f}/10")

        if self.math_depth is not None:
            parts.append(f"Mathematical depth: {self.math_depth:.1f}/10")

        if self.code_reproducibility is not None:
            parts.append(f"Code/reproducibility signal: {self.code_reproducibility:.1f}/10")

        if self.noise_penalty is not None:
            parts.append(f"Engineering/application noise penalty: {self.noise_penalty:.1f}/10")

        return "\n".join(parts)

    def _generate_tldr_with_llm(self, openai_client: OpenAI, llm_params: dict) -> str:
        lang = llm_params.get("language", "English")

        prompt = f"""
You are a senior paper-reading assistant for theoretical physics, mathematical physics,
statistical mechanics, probability theory, and machine-learning theory.

Write ONE technically precise paragraph in {lang}.

Prioritise genuine physical and mathematical content over engineering novelty.
Explicitly evaluate whether the paper has:
1. physical depth: phase transitions, critical phenomena, universality, renormalisation group,
   mean-field theory, spin glasses, replicas, random matrix theory, large deviations,
   conformal field theory, integrable structures, stochastic processes, or scaling laws;
2. mathematical substance: derivations, equations, theorems, asymptotic regimes, rate functions,
   variational principles, saddle-point arguments, replica/cavity computations, or rigorous proofs;
3. reproducibility: code, data, experiments, simulations, or a clear computational pipeline;
4. relevance to a theoretical physicist working on statistical mechanics, critical phenomena,
   spin glasses, random matrices, and machine-learning theory.

End the paragraph with exactly these three judgements:
Relevance: high / medium / low.
Math depth: high / medium / low.
Code/reproducibility: yes / unclear / no.

Penalise purely engineering benchmark papers, dataset papers, prompt-engineering applications,
superficial LLM papers, medical-imaging papers, remote-sensing papers, and papers without
physical or mathematical mechanism.

Paper metadata follows.

Title:
{self.title}

Authors:
{", ".join(self.authors or [])}

URL:
{self.url}

PDF:
{self.pdf_url or "N/A"}

Hybrid reranker scores:
{self._score_line()}

Abstract:
{self.abstract}
"""

        if self.full_text:
            prompt += f"""

Preview of main content:
{self.full_text}
"""

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate summary. Neither full text nor abstract is provided."

        # Use gpt-4o tokenizer only for conservative truncation.
        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:5000]
        prompt = enc.decode(prompt_tokens)

        response = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise scientific-paper analyst for theoretical physics, "
                        "mathematical physics, statistical mechanics, probability, and "
                        "machine-learning theory. Avoid hype. Be concrete."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )

        tldr = response.choices[0].message.content
        return tldr

    def generate_tldr(self, openai_client: OpenAI, llm_params: dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client, llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract or "Failed to generate summary."
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        if self.full_text is None:
            return None

        prompt = (
            "Given the beginning of a paper, extract the affiliations of the authors "
            "in a python list format, sorted by the author order. If there is no "
            "affiliation found, return an empty list '[]':\n\n"
            f"{self.full_text}"
        )

        try:
            enc = tiktoken.encoding_for_model("gpt-4o")
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")

        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:2000]
        prompt = enc.decode(prompt_tokens)

        affiliations = openai_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an assistant who extracts affiliations of authors "
                        "from a paper. Return only a Python list of affiliations, "
                        "for example [\"Stanford University\", \"MIT\"]. "
                        "If no affiliation is found, return []."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            **llm_params.get("generation_kwargs", {}),
        )

        affiliations_text = affiliations.choices[0].message.content or "[]"
        match = re.search(r"\[.*?\]", affiliations_text, flags=re.DOTALL)

        if not match:
            return []

        affiliations_list = json.loads(match.group(0))
        affiliations_list = list(set(affiliations_list))
        affiliations_list = [str(a) for a in affiliations_list]
        return affiliations_list

    def generate_affiliations(
        self,
        openai_client: OpenAI,
        llm_params: dict,
    ) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client, llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None


@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]
