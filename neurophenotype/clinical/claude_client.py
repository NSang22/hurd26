"""Minimal Claude wrapper for optional narrative enhancement.

This module is intentionally small:
- no external SDK dependency
- Anthropic Messages API via stdlib urllib
- deterministic fallback remains the default-safe path

Environment variables:
- ANTHROPIC_API_KEY: enables Claude calls when set
- NEUROPHENOTYPE_CLAUDE_MODEL: optional model override
"""
from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("NEUROPHENOTYPE_CLAUDE_MODEL", "claude-3-5-sonnet-latest")


def claude_available() -> bool:
    """Return True when Claude is configured for this environment."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def maybe_enhance_outputs(
    *,
    top_condition: str,
    classifier_results: dict[str, dict[str, Any]],
    hpo_scores: dict[str, dict[str, float]],
    deterministic_clinician_note: str,
    deterministic_uncertainty: str,
    reanalysis_score: float,
) -> dict[str, str]:
    """
    Optionally enhance deterministic outputs with Claude.

    If Claude is unavailable or fails, returns the original deterministic text.
    """
    fallback = {
        "clinician_note": deterministic_clinician_note,
        "uncertainty": deterministic_uncertainty,
        "reanalysis_explanation": "",
    }

    if not claude_available():
        return fallback

    prompt = _build_prompt(
        top_condition=top_condition,
        classifier_results=classifier_results,
        hpo_scores=hpo_scores,
        deterministic_clinician_note=deterministic_clinician_note,
        deterministic_uncertainty=deterministic_uncertainty,
        reanalysis_score=reanalysis_score,
    )

    schema_hint = (
        "Return strict JSON with keys: "
        '{"clinician_note": string, "uncertainty": string, "reanalysis_explanation": string}. '
        "Do not include markdown fences."
    )

    try:
        raw_text = _call_claude(system_prompt=schema_hint, user_prompt=prompt)
        parsed = json.loads(raw_text)
        clinician_note = str(parsed.get("clinician_note") or deterministic_clinician_note).strip()
        uncertainty = str(parsed.get("uncertainty") or deterministic_uncertainty).strip()
        reanalysis_explanation = str(parsed.get("reanalysis_explanation") or "").strip()

        return {
            "clinician_note": clinician_note or deterministic_clinician_note,
            "uncertainty": uncertainty or deterministic_uncertainty,
            "reanalysis_explanation": reanalysis_explanation,
        }
    except Exception:
        return fallback


def _call_claude(*, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 700,
        "system": system_prompt,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    }

    req = request.Request(
        ANTHROPIC_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        raise RuntimeError(f"Anthropic HTTP error: {exc.code}") from exc
    except error.URLError as exc:
        raise RuntimeError("Anthropic network error") from exc

    content = body.get("content") or []
    text_chunks = [block.get("text", "") for block in content if isinstance(block, dict)]
    result = "".join(text_chunks).strip()
    if not result:
        raise RuntimeError("Anthropic returned empty content")
    return result


def _build_prompt(
    *,
    top_condition: str,
    classifier_results: dict[str, dict[str, Any]],
    hpo_scores: dict[str, dict[str, float]],
    deterministic_clinician_note: str,
    deterministic_uncertainty: str,
    reanalysis_score: float,
) -> str:
    ordered = []
    for condition, data in classifier_results.items():
        ordered.append(
            {
                "condition": condition,
                "combined_probability": data.get("combined_probability"),
                "biosignal_probability": data.get("biosignal_probability"),
                "recommended_panel": data.get("recommended_panel"),
                "estimated_savings": data.get("estimated_savings"),
            }
        )

    return (
        "You are assisting a rare disease genomic triage demo.\n"
        "Rules:\n"
        "1. This is not a diagnosis.\n"
        "2. Keep all outputs concise and clinician-safe.\n"
        "3. Do not invent tests beyond the provided recommendation.\n"
        "4. If reanalysis is not relevant, leave reanalysis_explanation empty.\n\n"
        f"Top condition: {top_condition}\n"
        f"Reanalysis trigger score: {reanalysis_score:.2f}\n"
        f"Classifier results: {json.dumps(ordered)}\n"
        f"HPO scores: {json.dumps(hpo_scores)}\n\n"
        f"Deterministic clinician note:\n{deterministic_clinician_note}\n\n"
        f"Deterministic uncertainty statement:\n{deterministic_uncertainty}\n"
    )
