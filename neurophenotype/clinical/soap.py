"""Deterministic summary builders for patient- and clinician-facing output."""
from __future__ import annotations

from typing import Mapping

from .hpo import suggest_next_hpo_terms
from .intake import ClinicalIntake


CONDITION_DISPLAY = {
    "rett_syndrome": "Rett syndrome",
    "dravet_syndrome": "Dravet syndrome",
    "angelman_syndrome": "Angelman syndrome",
    "control": "neurotypical control profile",
}


def build_uncertainty_statement(
    intake: ClinicalIntake,
    top_condition: str,
    missing_modalities: list[str] | None = None,
) -> str:
    missing_hpo = suggest_next_hpo_terms(intake.hpo_terms, top_condition, limit=3)
    parts: list[str] = []

    if missing_hpo:
        parts.append("Confidence would increase with: " + ", ".join(missing_hpo))
    if missing_modalities:
        parts.append("Missing sensor data reduces confidence: " + ", ".join(missing_modalities))
    if intake.reanalysis_trigger_score() >= 0.5:
        parts.append("Prior testing history suggests reanalysis may be higher yield than a brand-new panel")

    if not parts:
        parts.append("Current recommendation is based on the available biosignal and structured phenotype data")
    return ". ".join(parts) + "."


def build_patient_summary(
    results: Mapping[str, Mapping[str, float | str]],
    intake: ClinicalIntake,
    missing_modalities: list[str] | None = None,
) -> str:
    top_condition, top_data = next(iter(results.items()))
    probability = float(top_data["probability"]) * 100.0
    label = CONDITION_DISPLAY.get(top_condition, top_condition)
    uncertainty = build_uncertainty_statement(intake, top_condition, missing_modalities)

    if top_condition == "control":
        return (
            f"The current signal pattern looks closest to a typical profile ({probability:.1f}% confidence). "
            f"This does not rule out a rare condition, but it suggests the next step should be guided by symptoms and prior testing. "
            f"{uncertainty}"
        )

    return (
        f"The combined pattern is most consistent with {label} ({probability:.1f}% confidence). "
        f"The recommended next step is targeted genomic follow-up rather than broad exploratory testing. "
        f"{uncertainty}"
    )


def build_clinician_note(
    results: Mapping[str, Mapping[str, float | str]],
    intake: ClinicalIntake,
    hpo_scores: Mapping[str, Mapping[str, float]],
    missing_modalities: list[str] | None = None,
) -> str:
    """Build a concise SOAP-style note from structured inputs."""
    top_condition, top_data = next(iter(results.items()))
    label = CONDITION_DISPLAY.get(top_condition, top_condition)
    hpo = hpo_scores.get(top_condition, {})
    uncertainty = build_uncertainty_statement(intake, top_condition, missing_modalities)

    if intake.reanalysis_trigger_score() >= 0.5:
        next_step = "Reanalyze prior sequencing before ordering a new broad test."
    else:
        next_step = f"Order: {top_data['recommended_panel']}"

    subjective = "Structured HPO intake completed; prior history incorporated."
    objective = (
        f"Top biosignal-classifier match: {label} ({float(top_data['probability']) * 100.0:.1f}%). "
        f"HPO support={hpo.get('support_score', 0.0):.2f}, contradiction={hpo.get('contradiction_score', 0.0):.2f}, "
        f"net={hpo.get('net_score', 0.0):.2f}."
    )
    assessment = (
        f"Combined phenotype is most aligned with {label}. "
        f"Estimated targeted-test savings: ${int(top_data['estimated_savings']):,} versus WES."
    )
    plan = f"{next_step} {uncertainty}"

    return "\n".join(
        [
            "S: " + subjective,
            "O: " + objective,
            "A: " + assessment,
            "P: " + plan,
        ]
    )
