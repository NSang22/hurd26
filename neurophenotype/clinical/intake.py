"""Structured clinical intake models and feature engineering helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .hpo import HPOFlag, encode_hpo_terms, score_conditions


@dataclass
class PriorTestRecord:
    test_type: str
    result_class: str
    date: str | None = None
    genes_covered: list[str] = field(default_factory=list)
    summary: str = ""


@dataclass
class FamilyHistory:
    affected_relatives: int = 0
    suspected_inheritance: str = "unknown"
    consanguinity_reported: bool = False


@dataclass
class ClinicalIntake:
    hpo_terms: dict[str, HPOFlag | str] = field(default_factory=dict)
    onset_age_months: int | None = None
    developmental_regression: bool = False
    severity_score: float | None = None
    family_history: FamilyHistory = field(default_factory=FamilyHistory)
    prior_tests: list[PriorTestRecord] = field(default_factory=list)
    new_hpo_terms_since_last_test: int = 0
    notes: str = ""

    def hpo_feature_vector(self) -> np.ndarray:
        return encode_hpo_terms(self.hpo_terms)

    def hpo_scores(self) -> dict[str, dict[str, float]]:
        return score_conditions(self.hpo_terms)

    def reanalysis_trigger_score(self) -> float:
        """
        Simple, transparent rule-based reanalysis heuristic.

        Strongest triggers:
        - prior negative exome/panel
        - VUS or incomplete prior testing
        - new phenotypes since the last test
        """
        score = 0.0
        for record in self.prior_tests:
            result = record.result_class.strip().lower()
            test_type = record.test_type.strip().lower()

            if result == "negative":
                score += 0.35
            elif result == "vus":
                score += 0.25
            elif result == "incomplete_panel":
                score += 0.3

            if "exome" in test_type:
                score += 0.15
            elif "panel" in test_type:
                score += 0.1

        if self.new_hpo_terms_since_last_test > 0:
            score += min(0.1 * self.new_hpo_terms_since_last_test, 0.3)

        if self.developmental_regression:
            score += 0.1

        return round(min(score, 1.0), 4)

    def to_feature_vector(self) -> np.ndarray:
        """
        Build a fixed-length clinical feature vector for downstream fusion.

        Layout:
        - curated HPO encoding
        - onset age (normalized)
        - regression flag
        - severity score
        - affected relative count
        - inheritance flags (x-linked, dominant, recessive, de_novo)
        - prior testing flags (has testing, negative, vus, incomplete)
        - reanalysis trigger score
        """
        hpo = self.hpo_feature_vector()

        onset = 0.0 if self.onset_age_months is None else min(float(self.onset_age_months) / 120.0, 1.0)
        severity = 0.0 if self.severity_score is None else max(0.0, min(float(self.severity_score), 1.0))
        affected_relatives = min(float(self.family_history.affected_relatives), 3.0) / 3.0

        inheritance = self.family_history.suspected_inheritance.strip().lower()
        inheritance_flags = np.array(
            [
                1.0 if inheritance == "x_linked" else 0.0,
                1.0 if inheritance == "autosomal_dominant" else 0.0,
                1.0 if inheritance == "autosomal_recessive" else 0.0,
                1.0 if inheritance == "de_novo" else 0.0,
            ],
            dtype=np.float32,
        )

        has_testing = 1.0 if self.prior_tests else 0.0
        has_negative = 1.0 if any(r.result_class.lower() == "negative" for r in self.prior_tests) else 0.0
        has_vus = 1.0 if any(r.result_class.lower() == "vus" for r in self.prior_tests) else 0.0
        has_incomplete = 1.0 if any(r.result_class.lower() == "incomplete_panel" for r in self.prior_tests) else 0.0

        scalar_features = np.array(
            [
                onset,
                1.0 if self.developmental_regression else 0.0,
                severity,
                affected_relatives,
                has_testing,
                has_negative,
                has_vus,
                has_incomplete,
                self.reanalysis_trigger_score(),
            ],
            dtype=np.float32,
        )

        return np.concatenate([hpo, inheritance_flags, scalar_features]).astype(np.float32)

    @classmethod
    def from_pdf_summary(cls, summary: dict[str, Any]) -> "ClinicalIntake":
        """Convenience constructor for wiring parsed PDF output into intake."""
        prior_tests = []
        if summary.get("test_type") or summary.get("result_class"):
            prior_tests.append(
                PriorTestRecord(
                    test_type=summary.get("test_type", "unknown"),
                    result_class=summary.get("result_class", "unknown"),
                    date=summary.get("date"),
                    genes_covered=list(summary.get("genes_mentioned", [])),
                    summary=summary.get("summary", ""),
                )
            )
        return cls(prior_tests=prior_tests)
