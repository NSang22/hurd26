"""
clinical/extractor.py — Claude-powered structured extraction for NeuroPhenotype.

Two document types:
  1. Prior genetic test results  → PriorTestRecord + genes + result class
  2. Referral / clinic letters   → HPO terms, onset, family history, clinical notes

Pipeline:
    PDF path or raw text
        ↓
    parse_pdf() + stitch_pages()   [pdf_parser.py]
        ↓
    summarize_genetic_test_report() for fast rule-based fields
        ↓
    Claude extraction for HPO terms, onset, family history, free text notes
        ↓
    ExtractionResult → pre-fills ClinicalIntake + paper document

Usage:
    from clinical.extractor import extract_clinical_document

    result = extract_clinical_document("report.pdf")
    # result.to_intake_dict() → pass to ClinicalIntake constructor
    # result.to_frontend_patch() → send to frontend to pre-fill paper

Environment:
    ANTHROPIC_API_KEY — enables Claude extraction. Falls back to rule-based only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib import error, request

from .pdf_parser import parse_pdf, stitch_pages, summarize_genetic_test_report


ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = os.environ.get("NEUROPHENOTYPE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
# HPO terms we care about — must match hpo.py CURATED_HPO_TERMS
CURATED_HPO_IDS = {
    "HP:0002376": "Hand stereotypy",
    "HP:0002384": "Irregular respiration",
    "HP:0001263": "Global developmental delay",
    "HP:0001250": "Seizures",
    "HP:0002069": "Generalized tonic-clonic seizures",
    "HP:0001344": "Absent speech",
    "HP:0001257": "Spasticity",
    "HP:0000717": "Autistic behavior",
    "HP:0011344": "Severe speech impairment",
}

TARGET_GENES = ["MECP2", "SCN1A", "UBE3A", "CDKL5", "FOXG1", "KCNQ2", "STXBP1", "PCDH19"]


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class ExtractionResult:
    """Structured output from document extraction."""

    # Document type detected
    doc_type: str = "unknown"          # "genetic_report" | "referral_note" | "clinic_letter"

    # Prior test fields (from genetic reports)
    test_type: str = "unknown"         # "exome" | "targeted_panel" | "methylation" | "array_cgh"
    result_class: str = "unknown"      # "negative" | "vus" | "positive" | "incomplete_panel"
    date: str | None = None
    genes_mentioned: list[str] = field(default_factory=list)
    genes_negative: list[str] = field(default_factory=list)    # explicitly reported negative
    genes_vus: list[str] = field(default_factory=list)         # VUS findings
    test_summary: str = ""

    # Clinical fields (from referral notes / clinic letters)
    hpo_states: dict[str, str] = field(default_factory=dict)   # HP:XXXXXXX -> "present"|"absent"|"uncertain"
    onset_age_months: int | None = None
    developmental_regression: bool = False
    severity_description: str = ""
    inheritance_pattern: str = "unknown"
    affected_relatives: int = 0
    new_hpo_terms_since_last_test: int = 0
    free_text_notes: str = ""          # raw extracted clinical narrative

    # Meta
    ocr_used: bool = False
    claude_used: bool = False
    confidence: str = "low"           # "low" | "medium" | "high"
    warnings: list[str] = field(default_factory=list)

    def to_intake_dict(self) -> dict[str, Any]:
        """Convert to dict compatible with server.py _build_clinical_intake()."""
        return {
            "hpo_states": self.hpo_states,
            "onset_age_months": self.onset_age_months or 0,
            "developmental_regression": self.developmental_regression,
            "severity_score": 0.5,     # not extractable from text reliably
            "affected_relatives": self.affected_relatives,
            "inheritance_pattern": self.inheritance_pattern,
            "prior_test_status": self._prior_test_status_key(),
            "new_hpo_terms_since_last_test": self.new_hpo_terms_since_last_test,
        }

    def to_frontend_patch(self) -> dict[str, Any]:
        """
        Structured patch for the frontend paper document pre-fill.
        Sent as JSON response from /api/parse_pdf.
        """
        return {
            "doc_type": self.doc_type,
            "hpo_states": self.hpo_states,
            "onset_age_months": self.onset_age_months,
            "developmental_regression": self.developmental_regression,
            "affected_relatives": self.affected_relatives,
            "inheritance_pattern": self.inheritance_pattern,
            "prior_test_status": self._prior_test_status_key(),
            "new_hpo_terms_since_last_test": self.new_hpo_terms_since_last_test,
            "genes_mentioned": self.genes_mentioned,
            "genes_negative": self.genes_negative,
            "genes_vus": self.genes_vus,
            "test_summary": self.test_summary,
            "free_text_notes": self.free_text_notes,
            "ocr_used": self.ocr_used,
            "claude_used": self.claude_used,
            "confidence": self.confidence,
            "warnings": self.warnings,
        }

    def _prior_test_status_key(self) -> str:
        if self.result_class == "negative":
            return "negative_exome" if self.test_type == "exome" else "negative_panel"
        elif self.result_class == "vus":
            return "vus"
        elif self.result_class == "incomplete_panel":
            return "incomplete_panel"
        return "none"


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_clinical_document(
    source: str | Path,
    doc_type: str = "auto",
) -> ExtractionResult:
    """
    Extract structured clinical fields from a PDF or raw text string.

    Args:
        source: Path to PDF file, or raw text string (for paste-in notes)
        doc_type: "auto" | "genetic_report" | "referral_note"
                  "auto" detects from content.

    Returns:
        ExtractionResult with all extractable fields populated.
    """
    # Handle raw text input (paste-in clinical notes)
    if isinstance(source, str) and not Path(source).exists():
        raw_text = source
        ocr_used = False
        rule_fields: dict[str, Any] = {}
    else:
        pages = parse_pdf(source)
        raw_text = stitch_pages(pages)
        ocr_used = any(p.ocr_used for p in pages)
        rule_fields = summarize_genetic_test_report(pages)

    if not raw_text.strip():
        result = ExtractionResult(ocr_used=ocr_used)
        result.warnings.append("Document appears empty or could not be read.")
        return result

    # Auto-detect document type
    if doc_type == "auto":
        doc_type = _detect_doc_type(raw_text)

    # Rule-based fields first (fast, deterministic, always runs)
    result = ExtractionResult(
        doc_type=doc_type,
        ocr_used=ocr_used,
        test_type=rule_fields.get("test_type", "unknown"),
        result_class=rule_fields.get("result_class", "unknown"),
        date=rule_fields.get("date"),
        genes_mentioned=rule_fields.get("genes_mentioned", []),
        test_summary=rule_fields.get("summary", ""),
    )

    # Claude enhancement (runs when API key available)
    if _claude_available():
        try:
            claude_result = _extract_with_claude(raw_text, doc_type)
            _merge_claude_result(result, claude_result)
            result.claude_used = True
            result.confidence = "high"
        except Exception as exc:
            result.warnings.append(f"Claude extraction failed, using rule-based only: {exc}")
            result.confidence = "low"
    else:
        result.confidence = "low"
        result.warnings.append("ANTHROPIC_API_KEY not set — rule-based extraction only.")

    return result


def extract_from_text(text: str) -> ExtractionResult:
    """Convenience wrapper for paste-in clinical notes (no PDF needed)."""
    return extract_clinical_document(text, doc_type="auto")


# ── Document type detection ───────────────────────────────────────────────────

def _detect_doc_type(text: str) -> str:
    text_lower = text.lower()
    genetic_signals = [
        "sequencing", "variant", "pathogenic", "vus", "exome",
        "panel", "gene report", "genomic", "allele", "mutation report",
        "laboratory report", "genetic testing", "methylation",
    ]
    referral_signals = [
        "dear dr", "referral", "clinic letter", "referred", "to whom it may concern",
        "presenting complaint", "history of", "examination", "assessment and plan",
        "discharge summary", "follow up", "follow-up",
    ]
    genetic_score = sum(1 for s in genetic_signals if s in text_lower)
    referral_score = sum(1 for s in referral_signals if s in text_lower)

    if genetic_score > referral_score:
        return "genetic_report"
    elif referral_score > 0:
        return "referral_note"
    return "clinical_note"


# ── Claude extraction ─────────────────────────────────────────────────────────

def _claude_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _extract_with_claude(text: str, doc_type: str) -> dict[str, Any]:
    """
    Send document text to Claude and get structured extraction.
    Returns a dict with all extractable fields.
    """
    truncated = text[:6000]  # stay well within token budget

    system_prompt = (
        "You are a clinical data extraction assistant specializing in rare pediatric neurological disorders. "
        "Extract structured information from clinical documents. "
        "Return ONLY valid JSON with no markdown fences, preamble, or explanation. "
        "If a field cannot be determined from the text, use null. "
        "Never invent information not present in the source text."
    )

    hpo_list = "\n".join(f"  {hpo_id}: {label}" for hpo_id, label in CURATED_HPO_IDS.items())
    gene_list = ", ".join(TARGET_GENES)

    if doc_type == "genetic_report":
        user_prompt = _genetic_report_prompt(truncated, hpo_list, gene_list)
    else:
        user_prompt = _clinical_note_prompt(truncated, hpo_list, gene_list)

    raw = _call_claude(system_prompt=system_prompt, user_prompt=user_prompt)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Strip any accidental markdown fences
        clean = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return json.loads(clean)


def _genetic_report_prompt(text: str, hpo_list: str, gene_list: str) -> str:
    return f"""Extract structured fields from this genetic test report.

Target genes of interest: {gene_list}

HPO terms to check for (return HP ID as key, status as value):
{hpo_list}

Return JSON with this exact schema:
{{
  "test_type": "exome" | "targeted_panel" | "methylation" | "array_cgh" | "unknown",
  "result_class": "negative" | "vus" | "positive" | "incomplete_panel" | "unknown",
  "date": "YYYY-MM-DD or null",
  "genes_tested": ["list of gene names explicitly mentioned as tested"],
  "genes_negative": ["genes explicitly reported as negative/normal"],
  "genes_vus": ["genes with variant of uncertain significance"],
  "genes_pathogenic": ["genes with pathogenic/likely pathogenic variants"],
  "hpo_states": {{
    "HP:0002376": "present" | "absent" | "uncertain" | "not_assessed",
    ... (include all 9 HPO IDs above, use not_assessed if not mentioned)
  }},
  "onset_age_months": integer or null,
  "developmental_regression": true | false | null,
  "inheritance_pattern": "x_linked" | "autosomal_dominant" | "autosomal_recessive" | "de_novo" | "unknown",
  "affected_relatives": integer or null,
  "clinical_summary": "1-2 sentence plain English summary of key findings",
  "reanalysis_recommended": true | false,
  "reanalysis_reason": "string or null — why reanalysis is/isn't indicated"
}}

Document text:
{text}"""


def _clinical_note_prompt(text: str, hpo_list: str, gene_list: str) -> str:
    return f"""Extract structured clinical fields from this referral letter or clinic note.
This is for a rare pediatric neurological disorder diagnostic workup (Rett syndrome, Dravet syndrome, Angelman syndrome).

HPO terms to identify (return HP ID as key, "present"/"absent"/"uncertain"/"not_assessed" as value):
{hpo_list}

Key genes if mentioned: {gene_list}

Return JSON with this exact schema:
{{
  "hpo_states": {{
    "HP:0002376": "present" | "absent" | "uncertain" | "not_assessed",
    ... (include all 9 HPO IDs, use not_assessed if term not mentioned at all,
        use absent if explicitly stated as absent/denied/no)
  }},
  "onset_age_months": integer or null,
  "developmental_regression": true | false | null,
  "severity_description": "mild" | "moderate" | "severe" | null,
  "inheritance_pattern": "x_linked" | "autosomal_dominant" | "autosomal_recessive" | "de_novo" | "unknown",
  "affected_relatives": integer or null,
  "prior_test_type": "exome" | "targeted_panel" | "methylation" | "array_cgh" | "none" | "unknown",
  "prior_test_result": "negative" | "vus" | "positive" | "incomplete_panel" | "none" | "unknown",
  "prior_test_date": "YYYY-MM-DD or null",
  "genes_mentioned": ["any target genes mentioned"],
  "new_hpo_terms_count": integer or null,
  "free_text_summary": "2-3 sentence clinical narrative extracted from the note",
  "urgent_flags": ["list any urgent clinical concerns mentioned, empty list if none"]
}}

Important extraction rules:
- "no seizures" or "denies seizures" → HP:0001250: "absent"
- "seizures" or "seizure disorder" → HP:0001250: "present"  
- "possible" or "query" or "?" before a finding → "uncertain"
- Family member findings do NOT count as patient findings
- Onset age: convert to months (e.g. "18 months" → 18, "2 years" → 24, "18 months old" → 18)

Document text:
{text}"""


def _merge_claude_result(result: ExtractionResult, claude: dict[str, Any]) -> None:
    """Merge Claude extraction output into the ExtractionResult, preferring Claude over rule-based."""
    if claude.get("hpo_states"):
        # Validate all keys are known HPO IDs
        valid_states = {}
        for hpo_id, state in claude["hpo_states"].items():
            if hpo_id in CURATED_HPO_IDS and state in ("present", "absent", "uncertain", "not_assessed"):
                valid_states[hpo_id] = state
        result.hpo_states = valid_states

    if claude.get("onset_age_months") is not None:
        result.onset_age_months = int(claude["onset_age_months"])

    if claude.get("developmental_regression") is not None:
        result.developmental_regression = bool(claude["developmental_regression"])

    if claude.get("inheritance_pattern"):
        result.inheritance_pattern = str(claude["inheritance_pattern"])

    if claude.get("affected_relatives") is not None:
        result.affected_relatives = int(claude["affected_relatives"])

    if claude.get("new_hpo_terms_count") is not None:
        result.new_hpo_terms_since_last_test = int(claude["new_hpo_terms_count"])

    # Genetic report specific fields
    if claude.get("test_type") and claude["test_type"] != "unknown":
        result.test_type = str(claude["test_type"])
    if claude.get("result_class") and claude["result_class"] != "unknown":
        result.result_class = str(claude["result_class"])
    if claude.get("date"):
        result.date = str(claude["date"])
    if claude.get("genes_tested"):
        result.genes_mentioned = list(claude["genes_tested"])
    if claude.get("genes_negative"):
        result.genes_negative = list(claude["genes_negative"])
    if claude.get("genes_vus"):
        result.genes_vus = list(claude["genes_vus"])

    # Prior test fields from clinical notes
    if claude.get("prior_test_type") and claude["prior_test_type"] not in ("none", "unknown"):
        result.test_type = str(claude["prior_test_type"])
    if claude.get("prior_test_result") and claude["prior_test_result"] not in ("none", "unknown"):
        result.result_class = str(claude["prior_test_result"])
    if claude.get("prior_test_date"):
        result.date = str(claude["prior_test_date"])

    # Narrative fields
    if claude.get("free_text_summary"):
        result.free_text_notes = str(claude["free_text_summary"])
    if claude.get("clinical_summary"):
        result.free_text_notes = str(claude["clinical_summary"])
    if claude.get("test_summary") or claude.get("clinical_summary"):
        result.test_summary = str(claude.get("test_summary") or claude.get("clinical_summary", ""))

    # Urgent flags → warnings
    urgent = claude.get("urgent_flags") or []
    if urgent:
        result.warnings.extend([f"⚠ Urgent: {flag}" for flag in urgent])


# ── Claude HTTP call (stdlib only, no SDK) ────────────────────────────────────

def _call_claude(*, system_prompt: str, user_prompt: str) -> str:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    payload = {
        "model": DEFAULT_MODEL,
        "max_tokens": 1200,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
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
        with request.urlopen(req, timeout=25) as response:
            body = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8")
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError("Anthropic network error") from exc

    content = body.get("content") or []
    text = "".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    if not text:
        raise RuntimeError("Claude returned empty content")
    return text