"""
NeuroPhenotype Dashboard
Run: streamlit run dashboard/app.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from classifier.model import NeuroPhenotypeClassifier
from clinical.hpo import CURATED_HPO_TERMS, HPOFlag
from clinical.intake import ClinicalIntake, FamilyHistory, PriorTestRecord
from clinical.soap import (
    build_clinician_note,
    build_patient_summary,
    build_uncertainty_statement,
)

MODEL_PATH = "classifier/model.pkl"

CONDITION_DISPLAY = {
    "rett_syndrome": "Rett Syndrome",
    "dravet_syndrome": "Dravet Syndrome",
    "angelman_syndrome": "Angelman Syndrome",
    "control": "Neurotypical Control",
    "reanalysis_triggered": "Reanalysis Trigger",
}

GENE_DISPLAY = {
    "rett_syndrome": "MECP2 (Xq28)",
    "dravet_syndrome": "SCN1A (2q24.3)",
    "angelman_syndrome": "UBE3A / chr15q11-q13",
    "control": "-",
    "reanalysis_triggered": "Prior testing review",
}

CONDITION_COLORS = {
    "rett_syndrome": "#e74c3c",
    "dravet_syndrome": "#e67e22",
    "angelman_syndrome": "#9b59b6",
    "control": "#27ae60",
    "reanalysis_triggered": "#2980b9",
}

PANELS = {
    "rett_syndrome": {
        "panel": "MECP2 Sequencing + Deletion/Duplication Analysis",
        "cost": 400,
        "wes_cost": 4500,
        "turnaround": "3-5 days",
        "rationale": "MECP2 mutations account for most classic Rett presentations, so targeted sequencing with del/dup is the highest-yield first test.",
    },
    "dravet_syndrome": {
        "panel": "SCN1A Sequencing + MLPA",
        "cost": 300,
        "wes_cost": 4800,
        "turnaround": "5-7 days",
        "rationale": "SCN1A variants explain most Dravet cases; MLPA adds deletion detection that sequencing alone can miss.",
    },
    "angelman_syndrome": {
        "panel": "Chromosome 15 Methylation Analysis + UBE3A Sequencing",
        "cost": 500,
        "wes_cost": 4300,
        "turnaround": "7-10 days",
        "rationale": "Methylation analysis detects deletion, UPD, and imprinting causes; UBE3A sequencing covers point variants.",
    },
    "control": {
        "panel": "No targeted panel indicated",
        "cost": 0,
        "wes_cost": 0,
        "turnaround": "-",
        "rationale": "Current biosignal profile appears within a neurotypical range. Follow symptoms and prior history if concern remains.",
    },
}

DEMO_PROFILES = {
    "rett_syndrome": np.array(
        [4.0, 6.5, 0.8, 0.6, 0.4, 0.7, 8.0, 5.5, 0.3, 0.5, 0.5, 0.3, 0.9, *([4.0] * 10)],
        dtype=np.float32,
    ),
    "dravet_syndrome": np.array(
        [5.0, 4.0, 1.2, 0.7, 0.4, 1.3, 3.5, 3.0, 1.2, 0.6, 0.55, 0.5, 0.6, *([5.0] * 10)],
        dtype=np.float32,
    ),
    "angelman_syndrome": np.array(
        [8.0, 3.0, 1.0, 0.5, 0.3, 2.8, 3.2, 2.5, 0.6, 0.7, 0.8, 0.4, 0.8, *([7.5] * 10)],
        dtype=np.float32,
    ),
    "control": np.array(
        [1.5, 1.8, 3.5, 1.2, 0.5, 0.8, 0.5, 10.0, 0.05, 0.35, 0.3, 0.15, 0.1, *([1.5] * 10)],
        dtype=np.float32,
    ),
}

BIOMARKER_LABELS = [
    "Delta Power",
    "Theta Power",
    "Alpha Power",
    "Beta Power",
    "Gamma Power",
    "Delta/Theta Ratio",
    "Theta/Alpha Ratio",
    "Dominant Frequency (Hz)",
    "Spike Rate",
    "Mean Coherence",
    "Frontal Coherence",
    "PAC Strength",
    "Background Slowing Score",
    *[f"Ch{i+1} Delta" for i in range(10)],
]

SENSOR_ICONS = {
    "EEG": "[EEG]",
    "HRV": "[HRV]",
    "GSR": "[GSR]",
    "Movement": "[Move]",
    "Speech": "[Speech]",
    "Keyboard/Mouse": "[KBM]",
    "rPPG": "[rPPG]",
}

HPO_FLAG_OPTIONS = [
    HPOFlag.NOT_ASSESSED.value,
    HPOFlag.PRESENT.value,
    HPOFlag.ABSENT.value,
    HPOFlag.UNCERTAIN.value,
]

PRIOR_TEST_OPTIONS = [
    "none",
    "negative_panel",
    "negative_exome",
    "vus",
    "incomplete_panel",
]


def _build_clinical_intake(
    hpo_states: dict[str, str],
    onset_age_months: int,
    developmental_regression: bool,
    severity_score: float,
    affected_relatives: int,
    inheritance_pattern: str,
    prior_test_status: str,
    new_hpo_terms_since_last_test: int,
) -> ClinicalIntake:
    prior_tests: list[PriorTestRecord] = []
    if prior_test_status == "negative_panel":
        prior_tests.append(PriorTestRecord("targeted_panel", "negative"))
    elif prior_test_status == "negative_exome":
        prior_tests.append(PriorTestRecord("exome", "negative"))
    elif prior_test_status == "vus":
        prior_tests.append(PriorTestRecord("targeted_panel", "vus"))
    elif prior_test_status == "incomplete_panel":
        prior_tests.append(PriorTestRecord("targeted_panel", "incomplete_panel"))

    return ClinicalIntake(
        hpo_terms=hpo_states,
        onset_age_months=onset_age_months if onset_age_months > 0 else None,
        developmental_regression=developmental_regression,
        severity_score=severity_score,
        family_history=FamilyHistory(
            affected_relatives=affected_relatives,
            suspected_inheritance=inheritance_pattern,
        ),
        prior_tests=prior_tests,
        new_hpo_terms_since_last_test=new_hpo_terms_since_last_test,
    )


def _synthesize_results(results: dict, hpo_scores: dict, reanalysis_score: float) -> dict:
    combined = {}
    for condition, data in results.items():
        base_prob = float(data["probability"])
        hpo_norm = float(hpo_scores.get(condition, {}).get("normalized_score", 0.0))
        hpo_assessed = float(hpo_scores.get(condition, {}).get("assessed_fraction", 0.0))
        hpo_component = max(0.0, hpo_norm) * 0.20 * max(hpo_assessed, 0.25)
        conflict_penalty = max(0.0, -hpo_norm) * 0.15
        combined_score = max(0.0, min(base_prob + hpo_component - conflict_penalty, 1.0))

        combined[condition] = {
            **data,
            "biosignal_probability": round(base_prob, 4),
            "hpo_normalized_score": round(hpo_norm, 4),
            "combined_probability": round(combined_score, 4),
        }

    if reanalysis_score >= 0.5:
        combined["reanalysis_triggered"] = {
            "probability": round(reanalysis_score, 4),
            "combined_probability": round(reanalysis_score, 4),
            "biosignal_probability": 0.0,
            "hpo_normalized_score": 0.0,
            "recommended_panel": "Reanalysis of prior sequencing + targeted follow-up",
            "estimated_cost": 200,
            "estimated_savings": 4000,
        }

    return dict(
        sorted(
            combined.items(),
            key=lambda item: item[1]["combined_probability"],
            reverse=True,
        )
    )


def _render_landing_state() -> None:
    st.markdown(
        """
        ### How It Works

        **Half 1:** A passive multimodal biosignal assessment captures EEG, HRV, GSR, movement, speech, rPPG, and optional fine-motor data.

        **Half 2:** A structured clinical intake captures HPO terms, prior testing, and family history.

        The system combines both streams into a **next best diagnostic step**:
        a targeted genomic panel recommendation, a reanalysis suggestion when prior testing is stale, and a clinician-facing summary.

        This is a triage tool, not a diagnosis.
        """
    )


st.set_page_config(
    page_title="NeuroPhenotype",
    page_icon="NP",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
.big-metric { font-size: 2rem; font-weight: 700; }
.condition-card {
    border-radius: 10px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
    border-left: 6px solid;
}
.disclaimer {
    font-size: 0.75rem;
    color: #888;
    font-style: italic;
}
</style>
""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## NeuroPhenotype")
    st.markdown("*Multimodal diagnostic copilot for rare neurological disorders*")
    st.divider()

    mode = st.radio(
        "Assessment Mode",
        ["Demo (Synthetic Profile)", "Live Pipeline"],
        index=0,
    )

    demo_condition = "control"
    if mode == "Demo (Synthetic Profile)":
        demo_condition = st.selectbox(
            "Simulate Patient Profile",
            options=list(DEMO_PROFILES.keys()),
            format_func=lambda x: CONDITION_DISPLAY[x],
        )
        st.info(
            "Demo mode uses literature-derived synthetic biomarker profiles. "
            "The clinical intake still changes the recommendation layer."
        )

    st.divider()
    st.markdown("**Active Sensors**")
    active_sensors = st.multiselect(
        "Modalities",
        options=list(SENSOR_ICONS.keys()),
        default=["EEG", "HRV", "GSR", "Movement", "Speech", "rPPG"],
    )
    for sensor in active_sensors:
        st.markdown(f"{SENSOR_ICONS[sensor]} {sensor}")

    st.divider()
    report_view = st.radio(
        "Output View",
        ["Both", "Clinician View", "Patient View"],
        index=0,
    )

    st.divider()
    st.markdown("**Clinical Intake**")
    with st.expander("HPO Term Checklist", expanded=False):
        hpo_states = {}
        for term in CURATED_HPO_TERMS:
            hpo_states[term.term_id] = st.selectbox(
                f"{term.term_id} - {term.label}",
                options=HPO_FLAG_OPTIONS,
                index=0,
                key=f"hpo_{term.term_id}",
            )

    onset_age_months = st.number_input("Age of Onset (months)", min_value=0, max_value=240, value=0, step=1)
    developmental_regression = st.checkbox("Developmental regression observed", value=False)
    severity_score = st.slider("Clinical severity", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    affected_relatives = st.number_input("Affected relatives", min_value=0, max_value=5, value=0, step=1)
    inheritance_pattern = st.selectbox(
        "Suspected inheritance",
        ["unknown", "x_linked", "autosomal_dominant", "autosomal_recessive", "de_novo"],
        index=0,
    )
    prior_test_status = st.selectbox(
        "Prior genetic testing",
        PRIOR_TEST_OPTIONS,
        index=0,
        format_func=lambda x: x.replace("_", " ").title(),
    )
    new_hpo_terms_since_last_test = st.number_input(
        "New phenotypes since last test",
        min_value=0,
        max_value=10,
        value=0,
        step=1,
    )

    clinical_intake = _build_clinical_intake(
        hpo_states=hpo_states,
        onset_age_months=int(onset_age_months),
        developmental_regression=developmental_regression,
        severity_score=float(severity_score),
        affected_relatives=int(affected_relatives),
        inheritance_pattern=inheritance_pattern,
        prior_test_status=prior_test_status,
        new_hpo_terms_since_last_test=int(new_hpo_terms_since_last_test),
    )

    st.divider()
    st.markdown(
        '<p class="disclaimer">For research use only. Not a clinical diagnostic tool.</p>',
        unsafe_allow_html=True,
    )

st.markdown("# NeuroPhenotype")
st.markdown("### Passive Biosignal Phenotyping + Clinical Record Integration")
st.caption("Combines live biosignals with structured phenotype data to recommend the next best diagnostic step.")

col_run, col_info = st.columns([1, 3])
with col_run:
    run_button = st.button("Run Assessment", type="primary", use_container_width=True)
with col_info:
    st.markdown(
        """
        Half 1: EEG, HRV, GSR, movement, speech, rPPG, optional fine motor.

        Half 2: HPO intake, prior test status, family history, and reanalysis logic.
        """
    )

st.divider()

if not run_button:
    _render_landing_state()
else:
    progress_bar = st.progress(0, text="Initializing pipeline...")
    stages = [
        (20, "Collecting or simulating biosignals..."),
        (45, "Running biosignal classifier..."),
        (70, "Scoring HPO and prior-testing context..."),
        (90, "Synthesizing next-step recommendation..."),
        (100, "Assessment complete."),
    ]
    for pct, msg in stages:
        time.sleep(0.2)
        progress_bar.progress(pct, text=msg)
    progress_bar.empty()

    clf = NeuroPhenotypeClassifier(MODEL_PATH)

    if mode == "Demo (Synthetic Profile)":
        features = DEMO_PROFILES[demo_condition].copy()
        rng = np.random.default_rng(42)
        features = features + rng.normal(0, 0.05, features.shape).astype(np.float32)
    else:
        try:
            from fusion.fusion import get_feature_vector

            with st.spinner("Running live pipeline..."):
                features = get_feature_vector("config.yaml")
        except Exception as exc:
            st.error(f"Live pipeline error: {exc}")
            st.stop()

    results = clf.predict(features)
    hpo_scores = clinical_intake.hpo_scores()
    reanalysis_score = clinical_intake.reanalysis_trigger_score()
    synthesized_results = _synthesize_results(results, hpo_scores, reanalysis_score)

    top_condition = next(iter(synthesized_results))
    top_data = synthesized_results[top_condition]
    panel_info = PANELS.get(top_condition, PANELS["control"])
    missing_modalities = [sensor for sensor in SENSOR_ICONS if sensor not in active_sensors]

    st.markdown("## Assessment Results")
    color = CONDITION_COLORS.get(top_condition, "#888")

    hero_col, savings_col = st.columns([2, 1])
    with hero_col:
        st.markdown(
            f"""
            <div class="condition-card" style="border-color: {color}; background: {color}18;">
                <div class="big-metric">{CONDITION_DISPLAY.get(top_condition, top_condition)}</div>
                <div style="font-size:1.1rem; margin-top:0.3rem;">
                    Gene Target: <strong>{GENE_DISPLAY.get(top_condition, "-")}</strong>
                </div>
                <div style="margin-top:0.5rem;">
                    Combined Confidence: <strong>{top_data['combined_probability'] * 100:.1f}%</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with savings_col:
        if top_condition == "reanalysis_triggered":
            st.metric("Recommended Action", "Reanalysis")
            st.metric("Projected Savings", f"${int(top_data['estimated_savings']):,}")
        elif panel_info["wes_cost"] > 0:
            savings = panel_info["wes_cost"] - panel_info["cost"]
            st.metric(
                "Recommended Panel Cost",
                f"${panel_info['cost']:,}",
                delta=f"-${savings:,} vs. WES",
                delta_color="inverse",
            )
            st.metric("Turnaround", panel_info["turnaround"])
        else:
            st.metric("Recommended Panel", "None indicated")

    st.markdown("### Combined Condition Priorities")
    for condition, data in synthesized_results.items():
        if condition == "control":
            continue

        label = CONDITION_DISPLAY.get(condition, condition)
        prob = float(data["combined_probability"])
        color = CONDITION_COLORS.get(condition, "#888")
        gene = GENE_DISPLAY.get(condition, "Clinical review")
        detail = ""
        if condition in hpo_scores:
            detail = (
                f"Biosignal {float(data['biosignal_probability']) * 100:.1f}% | "
                f"HPO score {float(data['hpo_normalized_score']):+.2f}"
            )

        st.markdown(
            f"""
            <div style="margin-bottom:0.6rem;">
                <div style="display:flex; justify-content:space-between; margin-bottom:2px;">
                    <span><strong>{label}</strong> - {gene}</span>
                    <span><strong>{prob * 100:.1f}%</strong></span>
                </div>
                <div style="font-size:0.85rem; color:#999; margin-bottom:2px;">{detail}</div>
                <div style="background:#333; border-radius:4px; height:20px;">
                    <div style="background:{color}; width:{prob * 100:.1f}%; height:100%;
                                border-radius:4px; transition:width 0.5s;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Recommended Next Step")
    pcol1, pcol2 = st.columns([2, 1])
    with pcol1:
        if top_condition == "reanalysis_triggered":
            st.warning(
                "**Reanalysis of prior sequencing + targeted follow-up**\n\n"
                "Prior negative or incomplete testing combined with new phenotype data makes reanalysis the highest-yield next step."
            )
        else:
            st.info(f"**{panel_info['panel']}**\n\n{panel_info['rationale']}")
    with pcol2:
        if top_condition == "reanalysis_triggered":
            st.markdown(
                """
                | | Value |
                |---|---|
                | **Estimated Cost** | $0-$200 |
                | Typical WES Repeat | $4,000+ |
                | **Savings** | **$4,000+** |
                """
            )
        elif panel_info["wes_cost"] > 0:
            st.markdown(
                f"""
                | | Cost |
                |---|---|
                | **Targeted Panel** | ${panel_info['cost']:,} |
                | Whole Exome Seq | ${panel_info['wes_cost']:,} |
                | **Savings** | **${panel_info['wes_cost'] - panel_info['cost']:,}** |
                """
            )

    uncertainty = build_uncertainty_statement(clinical_intake, top_condition, missing_modalities)
    patient_summary = build_patient_summary(synthesized_results, clinical_intake, missing_modalities)
    clinician_note = build_clinician_note(synthesized_results, clinical_intake, hpo_scores, missing_modalities)

    st.markdown("### Clinical Reasoning Layer")
    st.info(uncertainty)

    if report_view in ("Both", "Patient View"):
        st.markdown("#### Patient Summary")
        st.write(patient_summary)

    if report_view in ("Both", "Clinician View"):
        st.markdown("#### Clinician Note")
        st.code(clinician_note, language="text")

    with st.expander("Clinical Intake and HPO Evidence"):
        hpo_rows = [
            {
                "HPO Term": f"{term.term_id} - {term.label}",
                "Status": clinical_intake.hpo_terms.get(term.term_id, HPOFlag.NOT_ASSESSED.value),
            }
            for term in CURATED_HPO_TERMS
        ]
        st.dataframe(pd.DataFrame(hpo_rows), use_container_width=True, hide_index=True)

        score_rows = [
            {
                "Condition": CONDITION_DISPLAY.get(condition, condition),
                "Support": score["support_score"],
                "Contradiction": score["contradiction_score"],
                "Net": score["net_score"],
                "Normalized": score["normalized_score"],
                "Assessed": score["assessed_fraction"],
            }
            for condition, score in hpo_scores.items()
        ]
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)

        st.metric("Reanalysis Trigger Score", f"{reanalysis_score:.2f}")

    with st.expander("Biomarker Feature Breakdown"):
        feat_df = pd.DataFrame(
            {
                "Feature": BIOMARKER_LABELS[: len(features)],
                "Value": [f"{value:.3f}" for value in features],
            }
        )
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    with st.expander("Sensor Status"):
        scols = st.columns(4)
        for idx, (sensor, icon) in enumerate(SENSOR_ICONS.items()):
            status = "Active" if sensor in active_sensors else "Disabled"
            scols[idx % 4].markdown(f"{icon} **{sensor}**  \n{status}")

    st.divider()
    st.markdown(
        '<p class="disclaimer">Results are for research triage only and do not constitute a clinical diagnosis. '
        "Genomic testing should be ordered and interpreted by a qualified clinician.</p>",
        unsafe_allow_html=True,
    )
