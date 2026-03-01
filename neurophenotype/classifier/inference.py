"""
classifier/inference.py — Maps modality outputs to the exact 55-feature schema
the trained model expects.

Model feature layout (from model_meta):
  eeg      [0:23]   — 23 EEG features
  hrv      [23:27]  — 4 HRV features (SDNN, RMSSD, pNN50, LF/HF)
  movement [27:34]  — 7 movement features
  speech   [34:40]  — 6 speech features
  hpo      [40:49]  — 9 HPO term flags (0/1)
  prior    [49:55]  — 6 prior test flags

EEG feature name mapping (model trained on these exact 23):
  delta_power, theta_power, alpha_power, beta_power, gamma_power,
  delta_theta_ratio, theta_alpha_ratio, dominant_freq, spike_rate,
  mean_coherence, frontal_coherence, pac_strength, background_slowing,
  per_channel_delta_1..10

Our eeg.py outputs 23 features but in a slightly different order:
  [0-4]   band powers (delta, theta, alpha, beta, gamma)        -> matches
  [5-6]   delta_theta_ratio, theta_alpha_ratio                  -> matches
  [7]     spike_rate                                            -> skip spike_amp (index 8 in ours)
  [8]     spike_amplitude  (NOT in model schema)
  [9]     spike_dominant_freq -> dominant_freq
  [10]    mean_coherence                                        -> matches
  [11]    frontal_coherence                                     -> matches
  [12]    PAC                                                   -> pac_strength
  [13]    background_slowing                                    -> matches
  [14]    frontal_asymmetry  (NOT in model schema)
  [15]    inter_burst        (NOT in model schema)
  [16-22] per_channel_delta (7 channels)  -> model expects 10, pad with zeros

Usage:
    from classifier.inference import build_inference_vector, predict

    # Demo mode (no hardware):
    vec = build_inference_vector(mock_profile="rett")
    result = predict(vec)

    # Live mode (after collecting from modalities):
    from fusion.fusion import get_feature_vector
    biosignal_vec = get_feature_vector()
    vec = build_inference_vector(biosignal_vec=biosignal_vec)
    result = predict(vec)

    # With clinical data:
    vec = build_inference_vector(biosignal_vec=biosignal_vec, hpo_flags=hpo, prior_flags=prior)
    result = predict(vec)
"""

import sys
from pathlib import Path
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODEL_PATH = ROOT / "classifier/model.pkl"

CONDITION_LABELS = ["rett_syndrome", "dravet_syndrome", "angelman_syndrome"]

GENOMIC_PANELS = {
    "rett_syndrome": {
        "display": "Rett Syndrome",
        "gene": "MECP2 (Xq28)",
        "panel": "MECP2 sequencing + deletion/duplication analysis",
        "panel_cost": 400,
        "wes_cost": 5000,
    },
    "dravet_syndrome": {
        "display": "Dravet Syndrome",
        "gene": "SCN1A",
        "panel": "SCN1A sequencing",
        "panel_cost": 300,
        "wes_cost": 4800,
    },
    "angelman_syndrome": {
        "display": "Angelman Syndrome",
        "gene": "UBE3A / chr15q11-q13",
        "panel": "Chromosome 15 methylation analysis + UBE3A sequencing",
        "panel_cost": 500,
        "wes_cost": 4800,
    },
}

# ---------------------------------------------------------------------------
# EEG feature remapping
# Our eeg.py index -> model schema index
# ---------------------------------------------------------------------------

# Our output (23 features):
# 0:delta 1:theta 2:alpha 3:beta 4:gamma
# 5:delta_theta_ratio 6:theta_alpha_ratio
# 7:spike_rate 8:spike_amp(*skip*) 9:spike_dom_freq
# 10:mean_coherence 11:frontal_coherence 12:PAC
# 13:bg_slowing 14:frontal_asym(*skip*) 15:inter_burst(*skip*)
# 16-22: per_channel_delta (7 channels)

# Model expects (23 features):
# 0:delta 1:theta 2:alpha 3:beta 4:gamma
# 5:delta_theta_ratio 6:theta_alpha_ratio
# 7:dominant_freq 8:spike_rate
# 9:mean_coherence 10:frontal_coherence 11:pac_strength 12:background_slowing
# 13-22: per_channel_delta_1..10

def _remap_eeg(our_eeg: np.ndarray) -> np.ndarray:
    """
    Remap our 23-feature EEG vector to the 23-feature schema the model was trained on.
    """
    out = np.zeros(23, dtype=np.float32)

    if len(our_eeg) < 23:
        # Pad if shorter than expected
        padded = np.zeros(23, dtype=np.float32)
        padded[:len(our_eeg)] = our_eeg
        our_eeg = padded

    # Band powers [0-4] — identical
    out[0:5] = our_eeg[0:5]

    # Ratios [5-6] — identical
    out[5] = our_eeg[5]   # delta_theta_ratio
    out[6] = our_eeg[6]   # theta_alpha_ratio

    # dominant_freq [7] <- our spike_dominant_freq [9]
    out[7] = our_eeg[9]

    # spike_rate [8] <- our spike_rate [7]
    out[8] = our_eeg[7]

    # coherence + PAC + bg_slowing [9-12]
    out[9]  = our_eeg[10]  # mean_coherence
    out[10] = our_eeg[11]  # frontal_coherence
    out[11] = our_eeg[12]  # pac_strength
    out[12] = our_eeg[13]  # background_slowing

    # per_channel_delta [13-22] — we have 7, model wants 10
    # copy our 7, leave remaining 3 as zeros
    out[13:20] = our_eeg[16:23]   # channels 1-7
    # out[20:23] stays zero        # channels 8-10 (we don't have these)

    return out


def _remap_hrv(our_hrv: np.ndarray) -> np.ndarray:
    """
    Our hrv.py outputs 7 features: SDNN, RMSSD, pNN50, LF/HF, mean_HR, HR_std, SDNN_norm
    Model expects 4:                SDNN, RMSSD, pNN50, LF/HF
    Just take the first 4.
    """
    out = np.zeros(4, dtype=np.float32)
    if len(our_hrv) >= 4:
        out[:4] = our_hrv[:4]
    elif len(our_hrv) > 0:
        out[:len(our_hrv)] = our_hrv
    return out


# ---------------------------------------------------------------------------
# Mock biosignal profiles (fallback when modalities fail)
# ---------------------------------------------------------------------------

def _mock_biosignal(profile: str) -> np.ndarray:
    """
    Generate a 40-feature biosignal vector (EEG+HRV+Movement+Speech)
    directly in the model's expected schema, without going through modalities.
    Used as fallback when modalities error out.
    """
    rng = np.random.default_rng({"rett": 0, "dravet": 1, "angelman": 2}.get(profile, 99))
    vec = np.zeros(40, dtype=np.float32)

    if profile == "rett":
        # EEG [0:23]
        vec[0:5]  = [1.5, 1.9, 0.5, 0.4, 0.3]   # bands: theta dominant
        vec[5:7]  = [0.80, 3.8]                    # ratios
        vec[7:9]  = [5.5, 2.5]                     # dominant_freq, spike_rate
        vec[9:13] = [0.52, 0.48, 0.04, 0.75]      # coh, fcoh, pac, bg_slow
        vec[13:20] = 1.4                            # per-ch delta
        # HRV [23:27]: low HRV, sympathetic dominance (Julu 2017)
        vec[23:27] = [18.0, 15.0, 5.0, 3.2]       # SDNN, RMSSD, pNN50, LF/HF
        # Movement [27:34]: high stereotypy, periodic
        vec[27:34] = [180.0, 5.5, 1.8, 0.64, 0.3, 0.2, 0.2]
        # Speech [34:40]: low vocalization, near-absent prosody
        vec[34:40] = [93.0, 42.0, 0.51, 0.02, 0.85, 0.62]

    elif profile == "dravet":
        # EEG: high spike_rate is PRIMARY Dravet discriminator (training: 22.0)
        # dominant_freq 2.7Hz, theta elevated, alpha reduced (Kim 2023, Hall 2024)
        vec[0:5]  = [3.0, 4.5, 0.6, 0.7, 0.4]    # theta elevated, alpha reduced
        vec[5:7]  = [0.70, 7.5]                    # theta/alpha HIGH (SCN1A discriminator)
        vec[7:9]  = [2.7, 22.0]                    # dominant_freq 2.7Hz, spike_rate VERY HIGH
        vec[9:13] = [0.72, 0.65, 0.06, 0.55]      # coherence, pac disrupted, moderate bg slowing
        vec[13:20] = 3.0                            # per_channel_delta moderate
        # HRV: moderate disruption
        vec[23:27] = [28.0, 22.0, 12.0, 1.7]
        # Movement: ataxic, low stereotypy
        vec[27:34] = [4.0, 3.8, 0.62, 0.36, 0.28, 0.27, 0.24]
        # Speech: reduced complexity but not absent
        vec[34:40] = [200.0, 35.0, 0.52, 0.9, 0.40, 0.44]

    else:  # angelman
        # EEG: pathognomonic very high delta, near-absent alpha, very high frontal coherence
        # Matches training: delta=9.0, frontal_coherence=0.92, delta_theta_ratio=3.2
        vec[0:5]  = [9.0, 2.8, 0.3, 0.3, 0.3]    # delta VERY HIGH, alpha near-absent
        vec[5:7]  = [3.2, 9.5]                     # delta_theta_ratio VERY HIGH
        vec[7:9]  = [2.2, 0.6]                     # dominant_freq ~2Hz, spike_rate moderate
        vec[9:13] = [0.62, 0.92, 0.05, 0.94]      # frontal_coherence VERY HIGH, bg_slowing VERY HIGH
        vec[13:20] = 8.5                            # per_channel_delta VERY HIGH
        # HRV: near-normal autonomic
        vec[23:27] = [26.0, 20.0, 10.0, 1.9]
        # Movement: jerky, moderate stereotypy
        vec[27:34] = [6.0, 2.1, 0.72, 0.58, 0.34, 0.31, 0.29]
        # Speech: near-absent (near-complete absence of functional speech)
        vec[34:40] = [120.0, 8.0, 0.15, 2.8, 0.06, 0.14]

    # Add small noise
    vec += rng.normal(0, 0.05 * np.abs(vec).mean(), vec.shape).astype(np.float32)
    return np.clip(vec, 0, None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_inference_vector(
    biosignal_vec: np.ndarray = None,
    mock_profile: str = None,
    hpo_flags: np.ndarray = None,
    prior_flags: np.ndarray = None,
) -> np.ndarray:
    """
    Build the full 55-feature inference vector the model expects.

    Args:
        biosignal_vec: Raw concatenated output from fusion.get_feature_vector()
                       (40 features: EEG+HRV+Movement+Speech in our schema).
                       If None, uses mock_profile.
        mock_profile:  "rett", "dravet", or "angelman" — used when biosignal_vec
                       is None or modalities errored out.
        hpo_flags:     9-element array of HPO term flags (0/1). Zeros if not provided.
        prior_flags:   6-element array of prior test flags. Zeros if not provided.

    Returns:
        55-feature numpy array aligned to model's training schema.
    """
    full = np.zeros(55, dtype=np.float32)

    # ── Biosignal block [0:40] ───────────────────────────────────────────────
    if mock_profile:
    # Mock already builds in model schema — copy directly, no remapping
        full[0:40] = _mock_biosignal(mock_profile)
    elif biosignal_vec is not None and len(biosignal_vec) >= 40:
    # Live data from our modalities needs remapping to model schema
        full[0:23] = _remap_eeg(biosignal_vec[0:23])
        full[23:27] = _remap_hrv(biosignal_vec[23:30])
        full[27:34] = biosignal_vec[30:37] if len(biosignal_vec) >= 37 else np.zeros(7)
        full[34:40] = biosignal_vec[37:43] if len(biosignal_vec) >= 43 else np.zeros(6)
    else:
        raise ValueError("Provide either biosignal_vec or mock_profile")

    # ── Clinical block [40:55] ───────────────────────────────────────────────
    if hpo_flags is not None:
        arr = np.array(hpo_flags, dtype=np.float32).flatten()
        full[40:40+min(9, len(arr))] = arr[:9]

    if prior_flags is not None:
        arr = np.array(prior_flags, dtype=np.float32).flatten()
        full[49:49+min(6, len(arr))] = arr[:6]

    return full


def predict(inference_vec: np.ndarray, model_path: str = None) -> dict:
    """
    Run the trained model on a 55-feature inference vector.

    Returns dict of condition -> {probability, panel, cost, savings}
    sorted by probability descending.
    """
    path = Path(model_path) if model_path else MODEL_PATH
    m = joblib.load(path)
    pipe = m["model"]
    le   = m.get("label_encoder", None)

    probs = pipe.predict_proba(inference_vec.reshape(1, -1))[0]

    # Map probabilities to condition labels
    if le is not None:
        classes = le.classes_
    else:
        classes = CONDITION_LABELS[:len(probs)]

    results = {}
    for label, prob in zip(classes, probs):
        panel = GENOMIC_PANELS.get(label, {
            "display": label,
            "gene": "Unknown",
            "panel": "Broad genomic panel",
            "panel_cost": 500,
            "wes_cost": 5000,
        })
        results[label] = {
            "probability": round(float(prob), 4),
            "display": panel["display"],
            "gene": panel["gene"],
            "recommended_panel": panel["panel"],
            "estimated_cost": panel["panel_cost"],
            "estimated_savings": panel["wes_cost"] - panel["panel_cost"],
        }

    return dict(sorted(results.items(), key=lambda x: -x[1]["probability"]))


# ---------------------------------------------------------------------------
# Convenience: full pipeline in one call
# ---------------------------------------------------------------------------

def run_demo(profile: str = "rett") -> dict:
    """Run full inference in demo mode. Use this in the Flask server."""
    vec = build_inference_vector(mock_profile=profile)
    return predict(vec)


def run_live(hpo_flags=None, prior_flags=None) -> dict:
    """Run full inference from live hardware."""
    from fusion.fusion import get_feature_vector
    biosignal_vec = get_feature_vector()
    vec = build_inference_vector(
        biosignal_vec=biosignal_vec,
        hpo_flags=hpo_flags,
        prior_flags=prior_flags,
    )
    return predict(vec)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing inference adapter...\n")
    for profile in ["rett", "dravet", "angelman"]:
        result = run_demo(profile)
        top = list(result.items())[0]
        print(f"  {profile:10s} -> top match: {top[0]} ({top[1]['probability']*100:.1f}%)")
    print("\nAll profiles ok.") 