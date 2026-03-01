"""
HPO term encoding and condition scoring utilities.

The current implementation is intentionally small and explicit:
- a curated, condition-relevant term list
- fixed status encoding for model features
- transparent rule-based match scoring per target condition
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping

import numpy as np


class HPOFlag(str, Enum):
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"
    NOT_ASSESSED = "not_assessed"


@dataclass(frozen=True)
class HPOTerm:
    term_id: str
    label: str


CURATED_HPO_TERMS: tuple[HPOTerm, ...] = (
    HPOTerm("HP:0002376", "Hand stereotypy"),
    HPOTerm("HP:0002384", "Irregular respiration"),
    HPOTerm("HP:0001263", "Global developmental delay"),
    HPOTerm("HP:0001250", "Seizures"),
    HPOTerm("HP:0002069", "Generalized tonic-clonic seizures"),
    HPOTerm("HP:0001344", "Absent speech"),
    HPOTerm("HP:0001257", "Spasticity"),
    HPOTerm("HP:0000717", "Autistic behavior"),
    HPOTerm("HP:0011344", "Severe speech impairment"),
)

CONDITION_HPO_WEIGHTS: dict[str, dict[str, float]] = {
    "rett_syndrome": {
        "HP:0002376": 3.0,
        "HP:0002384": 2.0,
        "HP:0001263": 1.5,
        "HP:0001344": 1.5,
        "HP:0011344": 1.0,
    },
    "dravet_syndrome": {
        "HP:0001250": 3.0,
        "HP:0002069": 2.5,
        "HP:0001263": 1.5,
        "HP:0002384": 0.5,
    },
    "angelman_syndrome": {
        "HP:0001344": 3.0,
        "HP:0001250": 2.0,
        "HP:0001257": 1.5,
        "HP:0001263": 1.5,
        "HP:0002376": 0.5,
    },
}

STATUS_ENCODING: dict[HPOFlag, float] = {
    HPOFlag.PRESENT: 1.0,
    HPOFlag.ABSENT: -1.0,
    HPOFlag.UNCERTAIN: 0.5,
    HPOFlag.NOT_ASSESSED: 0.0,
}


def _normalize_flag(value: str | HPOFlag) -> HPOFlag:
    if isinstance(value, HPOFlag):
        return value
    value = str(value).strip().lower()
    for flag in HPOFlag:
        if value == flag.value:
            return flag
    raise ValueError(f"Unsupported HPO flag: {value}")


def encode_hpo_terms(term_states: Mapping[str, str | HPOFlag]) -> np.ndarray:
    """
    Encode curated HPO terms into a fixed-length numeric vector.

    Each curated term contributes one scalar:
      present=1.0, absent=-1.0, uncertain=0.5, not_assessed=0.0
    """
    values = []
    for term in CURATED_HPO_TERMS:
        flag = _normalize_flag(term_states.get(term.term_id, HPOFlag.NOT_ASSESSED))
        values.append(STATUS_ENCODING[flag])
    return np.array(values, dtype=np.float32)


def score_conditions(term_states: Mapping[str, str | HPOFlag]) -> dict[str, dict[str, float]]:
    """
    Compute transparent HPO support/contradiction scores per condition.

    Scoring:
    - relevant present term contributes full positive weight
    - relevant uncertain term contributes half positive weight
    - relevant absent term contributes contradiction weight
    - not assessed terms are ignored
    """
    results: dict[str, dict[str, float]] = {}

    for condition, weights in CONDITION_HPO_WEIGHTS.items():
        support = 0.0
        contradiction = 0.0
        assessed_weight = 0.0

        for term_id, weight in weights.items():
            flag = _normalize_flag(term_states.get(term_id, HPOFlag.NOT_ASSESSED))

            if flag == HPOFlag.NOT_ASSESSED:
                continue

            assessed_weight += weight
            if flag == HPOFlag.PRESENT:
                support += weight
            elif flag == HPOFlag.UNCERTAIN:
                support += weight * 0.5
            elif flag == HPOFlag.ABSENT:
                contradiction += weight

        net = support - contradiction
        max_weight = max(sum(weights.values()), 1e-6)
        assessed_fraction = assessed_weight / max_weight
        normalized = (net / max_weight) if max_weight else 0.0

        results[condition] = {
            "support_score": round(support, 4),
            "contradiction_score": round(contradiction, 4),
            "net_score": round(net, 4),
            "normalized_score": round(normalized, 4),
            "assessed_fraction": round(assessed_fraction, 4),
        }

    return dict(
        sorted(
            results.items(),
            key=lambda item: item[1]["normalized_score"],
            reverse=True,
        )
    )


def suggest_next_hpo_terms(
    term_states: Mapping[str, str | HPOFlag],
    condition: str,
    limit: int = 3,
) -> list[str]:
    """Return the most informative not-yet-assessed terms for a condition."""
    weights = CONDITION_HPO_WEIGHTS.get(condition, {})
    term_lookup = {term.term_id: term.label for term in CURATED_HPO_TERMS}

    missing = []
    for term_id, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True):
        flag = _normalize_flag(term_states.get(term_id, HPOFlag.NOT_ASSESSED))
        if flag == HPOFlag.NOT_ASSESSED:
            missing.append(f"{term_id} ({term_lookup.get(term_id, term_id)})")
        if len(missing) >= limit:
            break
    return missing
