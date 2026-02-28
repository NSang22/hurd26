"""
NeuroPhenotype — Entry Point

Runs the full pipeline:
  1. Collect + extract features from all active modalities (per config.yaml)
  2. Fuse into single feature vector
  3. Classify against trained model
  4. Print ranked results to stdout

For the interactive dashboard: streamlit run dashboard/app.py
"""
import sys
import yaml
from fusion.fusion import get_feature_vector
from classifier.model import NeuroPhenotypeClassifier

CONFIG_PATH = "config.yaml"
MODEL_PATH = "classifier/model.pkl"

CONDITION_DISPLAY = {
    "rett_syndrome": "Rett Syndrome      (MECP2)",
    "dravet_syndrome": "Dravet Syndrome    (SCN1A)",
    "angelman_syndrome": "Angelman Syndrome  (UBE3A)",
}


def main():
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    active = [k for k, v in config["modalities"].items() if v]
    print(f"Active modalities: {', '.join(active)}\n")

    print("Collecting biosignals...")
    features = get_feature_vector(CONFIG_PATH)
    print(f"Feature vector length: {len(features)}\n")

    clf = NeuroPhenotypeClassifier(MODEL_PATH)
    results = clf.predict(features)

    print("=" * 55)
    print("  DISORDER MATCH PROBABILITIES")
    print("=" * 55)
    for condition, data in results.items():
        label = CONDITION_DISPLAY.get(condition, condition)
        bar = "#" * int(data["probability"] * 30)
        print(f"  {label}: {data['probability']*100:5.1f}%  {bar}")

    print()
    top_condition, top_data = next(iter(results.items()))
    print("=" * 55)
    print("  RECOMMENDED GENOMIC PANEL")
    print("=" * 55)
    print(f"  Condition : {CONDITION_DISPLAY.get(top_condition, top_condition)}")
    print(f"  Panel     : {top_data['recommended_panel']}")
    print(f"  Est. cost : ${top_data['estimated_cost']:,}")
    print(f"  vs. WES   : saves ~${top_data['estimated_savings']:,}")
    print("=" * 55)
    print("\nNOTE: This tool is for research/triage only. Not a diagnosis.")


if __name__ == "__main__":
    main()
