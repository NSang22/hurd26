"""
Training script for the fused NeuroPhenotype classifier.

This version makes the model claim true:
- biosignal features
- structured HPO encodings
- prior-test / reanalysis flags

Run from neurophenotype/:
    python -m classifier.train
    python -m classifier.train --quick
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

TARGET_CONDITIONS = ["rett_syndrome", "dravet_syndrome", "angelman_syndrome"]

FEATURE_GROUPS = {
    "eeg": 23,
    "hrv": 4,
    "movement": 7,
    "speech": 6,
    "hpo": 9,
    "prior": 6,
}

GROUP_ORDER = list(FEATURE_GROUPS.keys())

EEG_FEATURES = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "gamma_power",
    "delta_theta_ratio",
    "theta_alpha_ratio",
    "dominant_freq",
    "spike_rate",
    "mean_coherence",
    "frontal_coherence",
    "pac_strength",
    "background_slowing",
    *[f"per_channel_delta_{i+1}" for i in range(10)],
]

HRV_FEATURES = ["sdnn", "rmssd", "pnn50", "lf_hf"]
MOVEMENT_FEATURES = [
    "stereotypy_score",
    "tremor_freq_hz",
    "movement_irregularity",
    "limb_rhythm_index",
    "imu_accel_std_x",
    "imu_accel_std_y",
    "imu_accel_std_z",
]
SPEECH_FEATURES = [
    "pitch_mean",
    "pitch_std",
    "speech_rhythm",
    "mean_pause_dur",
    "vocalization_rate",
    "mfcc_clarity",
]
HPO_FEATURES = [
    "hpo_hand_stereotypy",
    "hpo_irregular_respiration",
    "hpo_global_dev_delay",
    "hpo_seizures",
    "hpo_generalized_tonic_clonic",
    "hpo_absent_speech",
    "hpo_spasticity",
    "hpo_autistic_behavior",
    "hpo_severe_speech_impairment",
]
PRIOR_FEATURES = [
    "has_testing",
    "has_negative_test",
    "prior_exome",
    "has_vus",
    "has_incomplete_panel",
    "reanalysis_trigger_score",
]

FEATURE_NAMES = [
    *EEG_FEATURES,
    *HRV_FEATURES,
    *MOVEMENT_FEATURES,
    *SPEECH_FEATURES,
    *HPO_FEATURES,
    *PRIOR_FEATURES,
]


def feature_slices() -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for name, size in FEATURE_GROUPS.items():
        slices[name] = (start, start + size)
        start += size
    return slices


SYNTHETIC_PROFILES: dict[str, dict[str, dict[str, tuple[float, float]]]] = {
    "rett_syndrome": {
        "eeg": {
            "delta_power":        (4.0,  0.4),   # elevated but theta dominant
            "theta_power":        (7.5,  0.5),   # HIGH — primary Rett marker (central rhythmic theta)
            "alpha_power":        (0.5,  0.1),   # REDUCED
            "beta_power":         (0.4,  0.1),   # REDUCED
            "gamma_power":        (0.3,  0.08),  # REDUCED
            "delta_theta_ratio":  (0.55, 0.08),  # <1 — theta dominates (Rett vs Angelman key split)
            "theta_alpha_ratio":  (15.0, 1.5),   # VERY HIGH
            "dominant_freq":      (5.5,  0.5),   # theta range ~6Hz central theta
            "spike_rate":         (0.3,  0.08),  # LOW — not a seizure disorder primarily
            "mean_coherence":     (0.52, 0.06),
            "frontal_coherence":  (0.48, 0.06),  # moderate — NOT the frontal notching of Angelman
            "pac_strength":       (0.04, 0.01),  # REDUCED (MECP2 network disruption)
            "background_slowing": (0.92, 0.04),  # HIGH — correlates with severity
            "per_channel_delta":  (4.0,  0.4),
        },
        "hrv": {"values": ([18.0, 14.0, 5.0, 3.2], [3.0, 3.0, 2.0, 0.4])},   # low HRV, sympathetic dominance
        "movement": {"values": ([180.0, 1.6, 0.25, 0.85, 0.18, 0.17, 0.14], [20.0, 0.3, 0.06, 0.05, 0.04, 0.04, 0.03])},  # HIGH stereotypy
        "speech": {"values": ([90.0, 20.0, 0.30, 2.0, 0.15, 0.22], [20.0, 8.0, 0.08, 0.4, 0.06, 0.08])},  # low vocalization
        "hpo": {"values": ([0.95, 0.85, 0.90, 0.10, 0.05, 0.75, 0.05, 0.25, 0.80], [0.05, 0.08, 0.06, 0.15, 0.10, 0.10, 0.10, 0.15, 0.10])},
        "prior": {"values": ([0.55, 0.20, 0.10, 0.08, 0.12, 0.18], [0.15, 0.12, 0.10, 0.08, 0.10, 0.10])},
    },
    "dravet_syndrome": {
        "eeg": {
            "delta_power":        (3.0,  0.4),   # moderate
            "theta_power":        (4.5,  0.5),   # ELEVATED (Kim 2023 — theta higher in high-severity DS)
            "alpha_power":        (0.6,  0.1),   # REDUCED (key SCN1A discriminator, Neurology 2025)
            "beta_power":         (0.7,  0.1),
            "gamma_power":        (0.4,  0.08),  # reduced (parvalbumin+ interneuron deficit)
            "delta_theta_ratio":  (0.70, 0.08),  # <1 theta slightly dominant
            "theta_alpha_ratio":  (7.5,  0.8),   # HIGH (key SCN1A discriminator AUC 0.85)
            "dominant_freq":      (2.7,  0.2),   # 2-3.5Hz — Dravet hallmark spike-wave
            "spike_rate":         (22.0, 2.5),   # VERY HIGH — primary Dravet signature
            "mean_coherence":     (0.72, 0.06),  # elevated during discharge
            "frontal_coherence":  (0.65, 0.06),
            "pac_strength":       (0.06, 0.01),  # disrupted GABAergic
            "background_slowing": (0.55, 0.06),  # moderate
            "per_channel_delta":  (3.0,  0.4),
        },
        "hrv": {"values": ([28.0, 22.0, 12.0, 1.7], [4.0, 4.0, 3.0, 0.3])},
        "movement": {"values": ([4.0, 3.8, 0.62, 0.36, 0.28, 0.27, 0.24], [1.0, 0.5, 0.08, 0.07, 0.05, 0.05, 0.04])},
        "speech": {"values": ([200.0, 35.0, 0.52, 0.9, 0.40, 0.44], [25.0, 10.0, 0.08, 0.25, 0.08, 0.08])},
        "hpo": {"values": ([0.10, 0.10, 0.78, 0.94, 0.82, 0.05, 0.05, 0.10, 0.20], [0.15, 0.12, 0.10, 0.06, 0.08, 0.12, 0.10, 0.12, 0.15])},
        "prior": {"values": ([0.60, 0.24, 0.14, 0.10, 0.10, 0.20], [0.15, 0.12, 0.10, 0.08, 0.08, 0.10])},
    },
    "angelman_syndrome": {
        "eeg": {
            "delta_power":        (9.0,  0.6),   # VERY HIGH — pathognomonic (Ostrowski 2021)
            "theta_power":        (2.8,  0.4),   # moderate
            "alpha_power":        (0.3,  0.08),  # near-absent
            "beta_power":         (0.3,  0.08),  # REDUCED
            "gamma_power":        (0.3,  0.08),
            "delta_theta_ratio":  (3.2,  0.3),   # VERY HIGH — delta dominates (vs Rett theta dominant)
            "theta_alpha_ratio":  (9.5,  0.8),   # high (near-zero alpha inflates ratio)
            "dominant_freq":      (2.2,  0.2),   # ~2Hz delta
            "spike_rate":         (0.6,  0.12),  # moderate notched delta bursts
            "mean_coherence":     (0.62, 0.06),
            "frontal_coherence":  (0.92, 0.04),  # VERY HIGH — frontal notching signature
            "pac_strength":       (0.05, 0.01),
            "background_slowing": (0.94, 0.03),  # VERY HIGH
            "per_channel_delta":  (8.5,  0.6),   # VERY HIGH diffuse delta
        },
        "hrv": {"values": ([26.0, 20.0, 10.0, 1.9], [4.0, 4.0, 3.0, 0.3])},
        "movement": {"values": ([6.0, 2.1, 0.72, 0.58, 0.34, 0.31, 0.29], [1.5, 0.4, 0.08, 0.08, 0.05, 0.05, 0.04])},
        "speech": {"values": ([120.0, 8.0, 0.15, 2.8, 0.06, 0.14], [25.0, 4.0, 0.06, 0.4, 0.03, 0.06])},  # near-absent speech
        "hpo": {"values": ([0.15, 0.05, 0.85, 0.72, 0.10, 0.96, 0.72, 0.22, 0.88], [0.12, 0.10, 0.08, 0.12, 0.15, 0.05, 0.10, 0.15, 0.06])},
        "prior": {"values": ([0.58, 0.22, 0.10, 0.08, 0.14, 0.22], [0.15, 0.12, 0.08, 0.08, 0.10, 0.10])},
    },
}


def _sample_range(rng: np.random.Generator, means: list[float], stds: list[float], *, low: float | None = None, high: float | None = None) -> np.ndarray:
    arr = rng.normal(np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32))
    if low is not None or high is not None:
        lo = low if low is not None else -np.inf
        hi = high if high is not None else np.inf
        arr = np.clip(arr, lo, hi)
    return arr.astype(np.float32)


def _sample_eeg(profile: dict[str, tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    features = []
    base_keys = [
        "delta_power",
        "theta_power",
        "alpha_power",
        "beta_power",
        "gamma_power",
        "delta_theta_ratio",
        "theta_alpha_ratio",
        "dominant_freq",
        "spike_rate",
        "mean_coherence",
        "frontal_coherence",
        "pac_strength",
        "background_slowing",
    ]
    for key in base_keys:
        mean, std = profile[key]
        features.append(max(0.0, float(rng.normal(mean, std))))
    channel_mean, channel_std = profile["per_channel_delta"]
    for _ in range(10):
        features.append(max(0.0, float(rng.normal(channel_mean, channel_std))))
    return np.array(features, dtype=np.float32)


def generate_synthetic_sample(condition: str, rng: np.random.Generator) -> np.ndarray:
    profile = SYNTHETIC_PROFILES[condition]

    eeg = _sample_eeg(profile["eeg"], rng)
    hrv = _sample_range(rng, *profile["hrv"]["values"], low=0.0)
    movement = _sample_range(rng, *profile["movement"]["values"], low=0.0)
    speech = _sample_range(rng, *profile["speech"]["values"], low=0.0)
    hpo = _sample_range(rng, *profile["hpo"]["values"], low=-1.0, high=1.0)
    prior = _sample_range(rng, *profile["prior"]["values"], low=0.0, high=1.0)

    return np.concatenate([eeg, hrv, movement, speech, hpo, prior]).astype(np.float32)


def generate_synthetic_dataset(n_per_class: int = 300, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    labels: list[str] = []

    for condition in TARGET_CONDITIONS:
        print(f"  Generating {n_per_class} fused synthetic samples for [{condition}]")
        for _ in range(n_per_class):
            samples.append(generate_synthetic_sample(condition, rng))
            labels.append(condition)

    X = np.vstack(samples).astype(np.float32)
    y = np.array(labels)

    print(f"\nTotal: {len(y)} samples, {X.shape[1]} features, {len(set(y))} classes")
    print(f"Feature groups: {FEATURE_GROUPS}")
    return X, y


def train(output_path: str, quick: bool = False) -> None:
    n_per_class = 60 if quick else 320
    print(f"Generating fused synthetic training data ({n_per_class} samples/class)...")
    X, y = generate_synthetic_dataset(n_per_class=n_per_class)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = xgb.XGBClassifier(
        n_estimators=60 if quick else 320,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="mlogloss",
        random_state=42,
    )

    if not quick:
        min_class_count = int(np.bincount(y_encoded).min())
        n_splits = min(5, min_class_count)
        if n_splits >= 2:
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy")
            print(f"CV Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}")

    model.fit(X, y_encoded)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "label_encoder": label_encoder,
            "feature_names": FEATURE_NAMES,
            "feature_groups": FEATURE_GROUPS,
            "feature_slices": feature_slices(),
        },
        output_path,
    )

    meta = {
        "classes": list(label_encoder.classes_),
        "n_features": int(X.shape[1]),
        "n_training_samples": int(len(y)),
        "training_mode": "synthetic_fused",
        "quick": quick,
        "feature_groups": FEATURE_GROUPS,
        "feature_slices": feature_slices(),
        "feature_names": FEATURE_NAMES,
    }
    meta_path = Path(output_path).with_suffix(".meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)

    print(f"\nModel saved -> {output_path}")
    print(f"Classes: {list(label_encoder.classes_)}")
    print(f"Expected fused input width: {X.shape[1]}")


def scan() -> None:
    print("Fused feature schema:")
    start = 0
    for group, size in FEATURE_GROUPS.items():
        print(f"  {group:8s}  start={start:2d}  size={size:2d}")
        start += size
    print(f"\nTotal width: {start}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="classifier/model.pkl")
    parser.add_argument("--quick", action="store_true", help="Fewer trees and fewer samples")
    parser.add_argument("--scan", action="store_true", help="Print fused schema and exit")
    args = parser.parse_args()

    if args.scan:
        scan()
    else:
        train(args.out, quick=args.quick)
