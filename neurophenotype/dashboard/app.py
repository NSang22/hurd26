"""
Output Dashboard — Streamlit app
Shows: disorder match probabilities, recommended genomic panel, estimated cost savings
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fusion.fusion import get_feature_vector
from classifier.model import NeuroPhenotypeClassifier

MODEL_PATH = "classifier/model.pkl"
CONFIG_PATH = "config.yaml"

CONDITION_DISPLAY = {
    "rett_syndrome": "Rett Syndrome (MECP2)",
    "dravet_syndrome": "Dravet Syndrome (SCN1A)",
    "angelman_syndrome": "Angelman Syndrome (UBE3A / chr15q11-q13)",
}


def run_pipeline():
    with st.spinner("Collecting and processing biosignals..."):
        features = get_feature_vector(CONFIG_PATH)
    clf = NeuroPhenotypeClassifier(MODEL_PATH)
    return clf.predict(features)


st.set_page_config(page_title="NeuroPhenotype", layout="wide")
st.title("NeuroPhenotype")
st.caption("Multimodal Biomarker Platform for Genomic Diagnosis Prioritization")
st.warning("This tool is for clinical research only. It does not provide a diagnosis.")

if st.button("Run Assessment", type="primary"):
    results = run_pipeline()

    st.subheader("Disorder Match Probabilities")
    for condition, data in results.items():
        label = CONDITION_DISPLAY.get(condition, condition)
        prob = data["probability"]
        st.progress(prob, text=f"{label}: {prob * 100:.1f}%")

    top_condition, top_data = next(iter(results.items()))
    st.subheader("Recommended Genomic Panel")
    col1, col2, col3 = st.columns(3)
    col1.metric("Top Match", CONDITION_DISPLAY.get(top_condition, top_condition))
    col2.metric("Recommended Panel", top_data["recommended_panel"])
    col3.metric(
        "Estimated Cost",
        f"${top_data['estimated_cost']:,}",
        delta=f"-${top_data['estimated_savings']:,} vs. WES",
        delta_color="inverse",
    )
