"""
NeuroPhenotype Flask dashboard server.

Run from neurophenotype/:
    python dashboard/server.py
"""
from __future__ import annotations
import os
import sys
import sqlite3
from typing import Any

import numpy as np
from flask import Flask, jsonify, request, send_from_directory, session
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from classifier.inference import build_inference_vector
from classifier.train import FEATURE_GROUPS, SYNTHETIC_PROFILES
from clinical.intake import ClinicalIntake, FamilyHistory, PriorTestRecord
from clinical.claude_client import maybe_enhance_outputs
from clinical.soap import (
    build_clinician_note,
    build_patient_summary,
    build_uncertainty_statement,
)
from classifier.train import FEATURE_GROUPS, SYNTHETIC_PROFILES

app = Flask(__name__, static_folder=os.path.dirname(__file__))
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "neurophenotype-hackathon-secret-2026")

GOOGLE_CLIENT_ID = "772107930801-trp0gb6shebkfok20d5glhrpeus2ngkv.apps.googleusercontent.com"

# ── SQLite user database ──────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")


def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    conn = _get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            password_hash TEXT,
            role TEXT NOT NULL DEFAULT 'patient',
            google_id TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    # Seed demo accounts if they don't exist
    demo_users = [
        ("doctor@neuro.com", "Dr. Clinician", generate_password_hash("NeuroPhenotype2026"), "doctor"),
        ("patient@family.com", "Patient Guardian", generate_password_hash("FamilyAccess2026"), "patient"),
    ]
    for email, name, pw_hash, role in demo_users:
        try:
            conn.execute(
                "INSERT INTO users (email, name, password_hash, role) VALUES (?, ?, ?, ?)",
                (email, name, pw_hash, role),
            )
        except sqlite3.IntegrityError:
            pass  # already seeded
    conn.commit()
    conn.close()


_init_db()

MODEL_PATH = "classifier/model.pkl"
CONFIG_PATH = "config.yaml"
SUPPORTED_SENSOR_ORDER = ["EEG", "HRV", "Movement", "Speech"]
SENSOR_TO_GROUP = {
    "EEG": "eeg",
    "HRV": "hrv",
    "Movement": "movement",
    "Speech": "speech",
}
BIOSIGNAL_GROUP_ORDER = ["eeg", "hrv", "movement", "speech"]

FALLBACK_RESULTS = {
    "rett_syndrome": {
        "rett_syndrome": {"probability": 0.731, "recommended_panel": "MECP2 Sequencing + Deletion/Duplication Analysis", "estimated_cost": 400, "estimated_savings": 4600},
        "dravet_syndrome": {"probability": 0.158, "recommended_panel": "SCN1A Sequencing + MLPA", "estimated_cost": 300, "estimated_savings": 4500},
        "angelman_syndrome": {"probability": 0.072, "recommended_panel": "Chr15 Methylation + UBE3A Sequencing", "estimated_cost": 500, "estimated_savings": 4300},
        "control": {"probability": 0.039, "recommended_panel": "No targeted panel indicated", "estimated_cost": 0, "estimated_savings": 0},
    },
    "dravet_syndrome": {
        "dravet_syndrome": {"probability": 0.762, "recommended_panel": "SCN1A Sequencing + MLPA", "estimated_cost": 300, "estimated_savings": 4500},
        "rett_syndrome": {"probability": 0.121, "recommended_panel": "MECP2 Sequencing + Deletion/Duplication Analysis", "estimated_cost": 400, "estimated_savings": 4600},
        "angelman_syndrome": {"probability": 0.083, "recommended_panel": "Chr15 Methylation + UBE3A Sequencing", "estimated_cost": 500, "estimated_savings": 4300},
        "control": {"probability": 0.034, "recommended_panel": "No targeted panel indicated", "estimated_cost": 0, "estimated_savings": 0},
    },
    "angelman_syndrome": {
        "angelman_syndrome": {"probability": 0.814, "recommended_panel": "Chr15 Methylation + UBE3A Sequencing", "estimated_cost": 500, "estimated_savings": 4300},
        "rett_syndrome": {"probability": 0.102, "recommended_panel": "MECP2 Sequencing + Deletion/Duplication Analysis", "estimated_cost": 400, "estimated_savings": 4600},
        "dravet_syndrome": {"probability": 0.058, "recommended_panel": "SCN1A Sequencing + MLPA", "estimated_cost": 300, "estimated_savings": 4500},
        "control": {"probability": 0.026, "recommended_panel": "No targeted panel indicated", "estimated_cost": 0, "estimated_savings": 0},
    },
    "control": {
        "control": {"probability": 0.748, "recommended_panel": "No targeted panel indicated", "estimated_cost": 0, "estimated_savings": 0},
        "rett_syndrome": {"probability": 0.119, "recommended_panel": "MECP2 Sequencing + Deletion/Duplication Analysis", "estimated_cost": 400, "estimated_savings": 4600},
        "dravet_syndrome": {"probability": 0.088, "recommended_panel": "SCN1A Sequencing + MLPA", "estimated_cost": 300, "estimated_savings": 4500},
        "angelman_syndrome": {"probability": 0.045, "recommended_panel": "Chr15 Methylation + UBE3A Sequencing", "estimated_cost": 500, "estimated_savings": 4300},
    },
}

PANEL_META = {
    "rett_syndrome": {
        "gene": "MECP2 (Xq28)",
        "turnaround": "3-5 days",
        "rationale": "Targeted MECP2 sequencing with deletion/duplication analysis is the highest-yield first test for classic Rett presentations.",
    },
    "dravet_syndrome": {
        "gene": "SCN1A (2q24.3)",
        "turnaround": "5-7 days",
        "rationale": "SCN1A plus copy-number follow-up is the most direct next step when the phenotype is Dravet-like.",
    },
    "angelman_syndrome": {
        "gene": "UBE3A / chr15q11-q13",
        "turnaround": "7-10 days",
        "rationale": "Methylation analysis detects the highest-yield Angelman mechanisms, with UBE3A sequencing covering point variants.",
    },
    "control": {
        "gene": "-",
        "turnaround": "-",
        "rationale": "Current signal pattern appears closest to a control profile. Continue workup based on symptoms and prior testing history.",
    },
    "reanalysis_triggered": {
        "gene": "Prior testing review",
        "turnaround": "2-3 weeks",
        "rationale": "New phenotype information plus prior negative or incomplete testing can make reanalysis higher yield than repeating broad sequencing.",
    },
}


def _biosignal_slices() -> dict[str, tuple[int, int]]:
    slices: dict[str, tuple[int, int]] = {}
    start = 0
    for group in BIOSIGNAL_GROUP_ORDER:
        size = FEATURE_GROUPS[group]
        slices[group] = (start, start + size)
        start += size
    return slices


BIOSIGNAL_SLICES = _biosignal_slices()


def _sample_range(rng: np.random.Generator, means: list[float], stds: list[float], *, low: float | None = None, high: float | None = None) -> np.ndarray:
    arr = rng.normal(np.asarray(means, dtype=np.float32), np.asarray(stds, dtype=np.float32))
    if low is not None or high is not None:
        lo = low if low is not None else -np.inf
        hi = high if high is not None else np.inf
        arr = np.clip(arr, lo, hi)
    return arr.astype(np.float32)


def _sample_eeg(profile: dict[str, tuple[float, float]], rng: np.random.Generator) -> np.ndarray:
    keys = [
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
    values = [max(0.0, float(rng.normal(*profile[key]))) for key in keys]
    channel_mean, channel_std = profile["per_channel_delta"]
    values.extend(max(0.0, float(rng.normal(channel_mean, channel_std))) for _ in range(10))
    return np.array(values, dtype=np.float32)


def _build_demo_biosignal_vector(profile_name: str) -> np.ndarray:
    profile = SYNTHETIC_PROFILES.get(profile_name, SYNTHETIC_PROFILES["control"])
    seed_map = {"rett_syndrome": 101, "dravet_syndrome": 102, "angelman_syndrome": 103, "control": 104}
    rng = np.random.default_rng(seed_map.get(profile_name, 199))

    eeg = _sample_eeg(profile["eeg"], rng)
    hrv = _sample_range(rng, *profile["hrv"]["values"], low=0.0)
    movement = _sample_range(rng, *profile["movement"]["values"], low=0.0)
    speech = _sample_range(rng, *profile["speech"]["values"], low=0.0)
    return np.concatenate([eeg, hrv, movement, speech]).astype(np.float32)


def _build_model_clinical_vector(intake: ClinicalIntake) -> np.ndarray:
    hpo = intake.hpo_feature_vector().astype(np.float32)

    has_testing = 1.0 if intake.prior_tests else 0.0
    has_negative = 1.0 if any(r.result_class.lower() == "negative" for r in intake.prior_tests) else 0.0
    prior_exome = 1.0 if any("exome" in r.test_type.lower() for r in intake.prior_tests) else 0.0
    has_vus = 1.0 if any(r.result_class.lower() == "vus" for r in intake.prior_tests) else 0.0
    has_incomplete = 1.0 if any(r.result_class.lower() == "incomplete_panel" for r in intake.prior_tests) else 0.0
    reanalysis = float(intake.reanalysis_trigger_score())

    prior = np.array(
        [has_testing, has_negative, prior_exome, has_vus, has_incomplete, reanalysis],
        dtype=np.float32,
    )
    return np.concatenate([hpo, prior]).astype(np.float32)


def _mask_biosignal_vector(vector: np.ndarray, active_sensors: list[str]) -> np.ndarray:
    masked = np.asarray(vector, dtype=np.float32).copy()
    active = set(active_sensors)

    for sensor, group in SENSOR_TO_GROUP.items():
        if sensor not in active:
            start, end = BIOSIGNAL_SLICES[group]
            if masked.size >= end:
                masked[start:end] = 0.0

    return masked


# ── In-memory store for individual modality features (for fusion) ──
_modality_feature_store: dict[str, list[float]] = {}


@app.route("/")
def index() -> Any:
    return send_from_directory(os.path.dirname(__file__), "landing.html")


@app.route("/dashboard")
def dashboard() -> Any:
    # Allow access if session has auth, OR if ?skip= param is present
    if not session.get("user_role") and "skip" not in request.args:
        return send_from_directory(os.path.dirname(__file__), "landing.html")
    return send_from_directory(os.path.dirname(__file__), "index.html")


# ── Auth API ──────────────────────────────────────────────────────────────────

@app.route("/api/auth/signup", methods=["POST"])
def auth_signup() -> Any:
    body = request.get_json() or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    name = (body.get("name") or "").strip()
    role = body.get("role", "patient")
    if role not in ("doctor", "patient"):
        role = "patient"
    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400
    if len(password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters."}), 400
    conn = _get_db()
    try:
        conn.execute(
            "INSERT INTO users (email, name, password_hash, role) VALUES (?, ?, ?, ?)",
            (email, name or email.split("@")[0], generate_password_hash(password), role),
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
        session["user_id"] = user["id"]
        session["user_email"] = user["email"]
        session["user_name"] = user["name"]
        session["user_role"] = user["role"]
        return jsonify({"status": "ok", "user": {"email": user["email"], "name": user["name"], "role": user["role"]}})
    except sqlite3.IntegrityError:
        return jsonify({"status": "error", "message": "An account with that email already exists."}), 409
    finally:
        conn.close()


@app.route("/api/auth/login", methods=["POST"])
def auth_login() -> Any:
    body = request.get_json() or {}
    email = (body.get("email") or "").strip().lower()
    password = body.get("password") or ""
    if not email or not password:
        return jsonify({"status": "error", "message": "Email and password are required."}), 400
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    if not user or not user["password_hash"] or not check_password_hash(user["password_hash"], password):
        return jsonify({"status": "error", "message": "Invalid email or password."}), 401
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["name"]
    session["user_role"] = user["role"]
    return jsonify({"status": "ok", "user": {"email": user["email"], "name": user["name"], "role": user["role"]}})


@app.route("/api/auth/google", methods=["POST"])
def auth_google() -> Any:
    """Verify a Google ID token, find-or-create the user, start session."""
    body = request.get_json() or {}
    credential = body.get("credential") or ""
    role = body.get("role", "patient")
    if role not in ("doctor", "patient"):
        role = "patient"
    if not credential:
        return jsonify({"status": "error", "message": "Missing Google credential."}), 400
    try:
        from google.oauth2 import id_token as google_id_token
        from google.auth.transport import requests as google_requests
        idinfo = google_id_token.verify_oauth2_token(
            credential, google_requests.Request(), GOOGLE_CLIENT_ID
        )
        google_id = idinfo["sub"]
        email = idinfo.get("email", "").lower()
        name = idinfo.get("name", email.split("@")[0])
    except Exception as exc:
        return jsonify({"status": "error", "message": f"Google token verification failed: {exc}"}), 401
    # Find or create user
    conn = _get_db()
    user = conn.execute("SELECT * FROM users WHERE google_id = ? OR email = ?", (google_id, email)).fetchone()
    if user:
        # Link google_id if not yet linked
        if not user["google_id"]:
            conn.execute("UPDATE users SET google_id = ? WHERE id = ?", (google_id, user["id"]))
            conn.commit()
    else:
        conn.execute(
            "INSERT INTO users (email, name, role, google_id) VALUES (?, ?, ?, ?)",
            (email, name, role, google_id),
        )
        conn.commit()
        user = conn.execute("SELECT * FROM users WHERE google_id = ?", (google_id,)).fetchone()
    conn.close()
    session["user_id"] = user["id"]
    session["user_email"] = user["email"]
    session["user_name"] = user["name"]
    session["user_role"] = user["role"]
    return jsonify({"status": "ok", "user": {"email": user["email"], "name": user["name"], "role": user["role"]}})


@app.route("/api/auth/me")
def auth_me() -> Any:
    if not session.get("user_role"):
        return jsonify({"status": "error", "message": "Not authenticated."}), 401
    return jsonify({"status": "ok", "user": {
        "email": session.get("user_email"),
        "name": session.get("user_name"),
        "role": session.get("user_role"),
    }})


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout() -> Any:
    session.clear()
    return jsonify({"status": "ok"})


@app.route("/speech-test")
def speech_test_page() -> Any:
    return send_from_directory(os.path.dirname(__file__), "speech_test.html")


@app.route("/movement-test")
def movement_test_page() -> Any:
    return send_from_directory(os.path.dirname(__file__), "movement_test.html")


@app.route("/api/test/speech", methods=["POST"])
def test_speech() -> Any:
    """Run the SpeechModality pipeline and return features.
    Pass {"mock": "rett"} (or dravet/angelman) to use synthetic data."""
    body = request.get_json() or {}
    task = body.get("task", "repeat_phrase")
    mock = body.get("mock")  # e.g. "rett", "dravet", "angelman"
    try:
        from modalities.speech import SpeechModality
        # skip_countdown=True  → client already showed the countdown
        # mic_delay=1.5        → give Windows time to release the mic
        #                        after the browser freed getUserMedia
        mod = SpeechModality(
            task=task,
            mock_profile=mock or None,
            mic_delay=0.0 if mock else 1.5,
            skip_countdown=True,
        )
        features = mod.run()
        feature_list = [float(v) for v in features]
        _modality_feature_store["speech"] = feature_list
        return jsonify({"status": "ok", "features": feature_list, "modality": "speech", "mock": bool(mock)})
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/test/movement", methods=["POST"])
def test_movement() -> Any:
    """Run the MovementModality pipeline and return features.
    Pass {"mock": "rett"} (or dravet/angelman) to use synthetic data."""
    body = request.get_json() or {}
    task = body.get("task", "open_close_fists")
    mock = body.get("mock")  # e.g. "rett", "dravet", "angelman"
    try:
        from modalities.movement import MovementModality
        # skip_imu=True  → don't waste 30s scanning for Arduino BLE
        # webcam_delay=1.5 → give Windows time to release the device
        #                     after the browser freed getUserMedia
        mod = MovementModality(
            task=task,
            mock_profile=mock or None,
            skip_imu=True,
            webcam_delay=0.0 if mock else 1.5,
        )
        features = mod.run()
        feature_list = [float(v) for v in features]
        _modality_feature_store["movement"] = feature_list
        return jsonify({"status": "ok", "features": feature_list, "modality": "movement", "mock": bool(mock)})
    except Exception as exc:
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


@app.route("/api/fuse-modality", methods=["POST"])
def fuse_modality() -> Any:
    """
    Accept features from a single modality test and attempt partial fusion.
    Stores the feature vector and, if we have enough modalities, runs the classifier.
    """
    body = request.get_json() or {}
    modality = body.get("modality", "")
    features = body.get("features", [])

    if modality and features:
        _modality_feature_store[modality] = [float(v) for v in features]

    # Try to build a fused vector from whatever we have stored
    try:
        from classifier.train import FEATURE_GROUPS
        from classifier.model import NeuroPhenotypeClassifier

        # Build a zero-padded vector in the canonical order: eeg, hrv, movement, speech
        total_dim = sum(FEATURE_GROUPS[g] for g in BIOSIGNAL_GROUP_ORDER)
        fused = np.zeros(total_dim, dtype=np.float32)

        filled_modalities = []
        offset = 0
        for group in BIOSIGNAL_GROUP_ORDER:
            dim = FEATURE_GROUPS[group]
            if group in _modality_feature_store:
                stored = _modality_feature_store[group]
                length = min(len(stored), dim)
                fused[offset:offset + length] = stored[:length]
                filled_modalities.append(group)
            offset += dim

        if not filled_modalities:
            return jsonify({
                "status": "stored",
                "message": "No modality features available yet.",
                "stored_modalities": list(_modality_feature_store.keys()),
            })

        clf = NeuroPhenotypeClassifier(MODEL_PATH)
        results = clf.predict(fused)

        # Find top condition
        top_condition = max(results, key=lambda k: results[k].get("probability", 0))
        top_prob = results[top_condition].get("probability", 0)

        return jsonify({
            "status": "ok",
            "top_condition": top_condition,
            "top_probability": top_prob,
            "results": {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv for kk, vv in v.items()} for k, v in results.items()},
            "filled_modalities": filled_modalities,
            "stored_modalities": list(_modality_feature_store.keys()),
        })

    except Exception as exc:
        import traceback; traceback.print_exc()
        # Even if classifier fails, we stored the features
        return jsonify({
            "status": "stored",
            "message": f"Features stored but classifier unavailable: {exc}",
            "stored_modalities": list(_modality_feature_store.keys()),
        })


@app.route("/api/modality-store", methods=["GET"])
def modality_store_status() -> Any:
    """Check which modalities have features stored for fusion."""
    return jsonify({
        "status": "ok",
        "stored_modalities": list(_modality_feature_store.keys()),
        "feature_counts": {k: len(v) for k, v in _modality_feature_store.items()},
    })


@app.route("/api/modality-store/clear", methods=["POST"])
def clear_modality_store() -> Any:
    """Clear all stored modality features."""
    _modality_feature_store.clear()
    return jsonify({"status": "ok", "message": "Feature store cleared."})


@app.route("/api/demo", methods=["POST"])
def demo() -> Any:
    body = request.get_json() or {}
    profile = body.get("profile", "rett_syndrome")
    intake = _build_clinical_intake(body.get("intake") or {})
    active_sensors = list(body["active_sensors"]) if "active_sensors" in body else list(SUPPORTED_SENSOR_ORDER)

    try:
        from classifier.model import NeuroPhenotypeClassifier

        clf = NeuroPhenotypeClassifier(MODEL_PATH)
        profile_key = str(profile).replace("_syndrome", "")
        features = build_inference_vector(
        mock_profile=profile_key,
        hpo_flags=_build_model_clinical_vector(intake)[:9],
        prior_flags=_build_model_clinical_vector(intake)[9:15],
)
        base_results = clf.predict(features)
        payload = _build_response_payload(base_results, intake, active_sensors, "demo")
        return jsonify({"status": "ok", **payload})
    except Exception:
        base_results = dict(FALLBACK_RESULTS.get(profile, FALLBACK_RESULTS["rett_syndrome"]))
        payload = _build_response_payload(base_results, intake, active_sensors, "demo_fallback")
        return jsonify({"status": "ok", **payload})


@app.route("/api/run", methods=["POST"])
def run_pipeline() -> Any:
    body = request.get_json() or {}
    intake = _build_clinical_intake(body.get("intake") or {})
    active_sensors = list(body["active_sensors"]) if "active_sensors" in body else list(SUPPORTED_SENSOR_ORDER)

    guided_tasks = body.get("guided_tasks", {})
    movement_task = guided_tasks.get("movement", "open_close_fists")
    speech_task = guided_tasks.get("speech", "repeat_phrase")

    try:
        from classifier.model import NeuroPhenotypeClassifier

        # ── Build feature vector from stored modal captures ──
        # The modals already ran /api/test/movement and /api/test/speech,
        # which stored features in _modality_feature_store.
        # We assemble a 40-feature biosignal vector from whatever is stored,
        # zero-padding modalities that weren't captured.
        total_biosignal_dim = sum(FEATURE_GROUPS[g] for g in BIOSIGNAL_GROUP_ORDER)
        features = np.zeros(total_biosignal_dim, dtype=np.float32)

        filled_groups: list[str] = []
        offset = 0
        for group in BIOSIGNAL_GROUP_ORDER:
            dim = FEATURE_GROUPS[group]
            if group in _modality_feature_store:
                stored = _modality_feature_store[group]
                length = min(len(stored), dim)
                features[offset:offset + length] = stored[:length]
                filled_groups.append(group)
            offset += dim

        # Mask out disabled sensors
        features = _mask_biosignal_vector(features, active_sensors)

        # ── Coverage calculation ──
        # How much real data do we have? Each active modality group counts
        # proportionally to its feature dimension.
        active_groups = {SENSOR_TO_GROUP[s] for s in active_sensors if s in SENSOR_TO_GROUP}
        filled_active = [g for g in filled_groups if g in active_groups]
        active_dim = sum(FEATURE_GROUPS[g] for g in active_groups) if active_groups else total_biosignal_dim
        filled_dim = sum(FEATURE_GROUPS[g] for g in filled_active)
        coverage = filled_dim / active_dim if active_dim > 0 else 0.0

        # Also count how many of the 4 biosignal groups have data
        n_modalities_active = len(active_groups)
        n_modalities_filled = len(filled_active)

        clf = NeuroPhenotypeClassifier(MODEL_PATH)
        clinical_vector = _build_model_clinical_vector(intake)
        base_results = clf.predict(features, clinical_vector=clinical_vector)

        # ── Confidence gating ──
        # With sparse input (≤1 modality, no HPO), the classifier doesn't
        # have enough signal for a reliable prediction.
        hpo_states = (body.get("intake") or {}).get("hpo_states", {})
        n_hpo_assessed = sum(1 for v in hpo_states.values() if str(v) not in ("not_assessed", ""))
        low_confidence = (n_modalities_filled <= 1 and n_hpo_assessed < 3)

        payload = _build_response_payload(base_results, intake, active_sensors, "live")
        payload["data_coverage"] = round(coverage, 3)
        payload["modalities_filled"] = filled_active
        payload["modalities_active"] = list(active_groups)
        payload["low_confidence"] = low_confidence
        if low_confidence:
            payload["coverage_warning"] = (
                f"Only {n_modalities_filled} of {n_modalities_active} selected "
                f"modality group{'s' if n_modalities_active != 1 else ''} provided data"
                f"{' and fewer than 3 HPO terms were assessed' if n_hpo_assessed < 3 else ''}. "
                f"Results are exploratory — not enough information for a reliable classification. "
                f"Enable additional modalities (EEG, HRV, Speech) and complete the HPO checklist "
                f"for a clinically meaningful assessment."
            )
        return jsonify({"status": "ok", **payload})
    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(exc)}), 500


def _build_clinical_intake(raw: dict[str, Any]) -> ClinicalIntake:
    prior_tests: list[PriorTestRecord] = []
    prior_test_status = str(raw.get("prior_test_status", "none"))

    if prior_test_status == "negative_panel":
        prior_tests.append(PriorTestRecord("targeted_panel", "negative"))
    elif prior_test_status == "negative_exome":
        prior_tests.append(PriorTestRecord("exome", "negative"))
    elif prior_test_status == "vus":
        prior_tests.append(PriorTestRecord("targeted_panel", "vus"))
    elif prior_test_status == "incomplete_panel":
        prior_tests.append(PriorTestRecord("targeted_panel", "incomplete_panel"))

    return ClinicalIntake(
        hpo_terms={str(k): str(v) for k, v in (raw.get("hpo_states") or {}).items()},
        onset_age_months=_to_int(raw.get("onset_age_months"), 0) or None,
        developmental_regression=bool(raw.get("developmental_regression", False)),
        severity_score=_to_float(raw.get("severity_score"), 0.5),
        family_history=FamilyHistory(
            affected_relatives=_to_int(raw.get("affected_relatives"), 0),
            suspected_inheritance=str(raw.get("inheritance_pattern", "unknown")),
        ),
        prior_tests=prior_tests,
        new_hpo_terms_since_last_test=_to_int(raw.get("new_hpo_terms_since_last_test"), 0),
    )


def _build_response_payload(
    base_results: dict[str, dict[str, Any]],
    intake: ClinicalIntake,
    active_sensors: list[str],
    mode: str,
) -> dict[str, Any]:
    _inject_meta(base_results)
    hpo_scores = intake.hpo_scores()
    reanalysis_score = intake.reanalysis_trigger_score()
    synthesized = _synthesize_results(base_results, hpo_scores, reanalysis_score)

    top_condition = next(iter(synthesized))
    missing_modalities = _missing_modalities(active_sensors)

    uncertainty = build_uncertainty_statement(intake, top_condition, missing_modalities)
    patient_summary = build_patient_summary(synthesized, intake, missing_modalities)
    clinician_note = build_clinician_note(synthesized, intake, hpo_scores, missing_modalities)
    enhanced = maybe_enhance_outputs(
        top_condition=top_condition,
        classifier_results=synthesized,
        hpo_scores=hpo_scores,
        deterministic_clinician_note=clinician_note,
        deterministic_uncertainty=uncertainty,
        reanalysis_score=reanalysis_score,
    )

    clinician_note = enhanced["clinician_note"]
    uncertainty = enhanced["uncertainty"]
    if top_condition == "reanalysis_triggered" and enhanced["reanalysis_explanation"]:
        synthesized["reanalysis_triggered"]["rationale"] = enhanced["reanalysis_explanation"]

    return {
        "mode": mode,
        "results": synthesized,
        "hpo_scores": hpo_scores,
        "reanalysis_score": reanalysis_score,
        "uncertainty": uncertainty,
        "patient_summary": patient_summary,
        "clinician_note": clinician_note,
        "top_condition": top_condition,
    }


def _inject_meta(results: dict[str, dict[str, Any]]) -> None:
    for condition, data in results.items():
        meta = PANEL_META.get(condition, PANEL_META["control"])
        data.update(meta)


def _synthesize_results(
    results: dict[str, dict[str, Any]],
    hpo_scores: dict[str, dict[str, float]],
    reanalysis_score: float,
) -> dict[str, dict[str, Any]]:
    combined: dict[str, dict[str, Any]] = {}

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
            **PANEL_META["reanalysis_triggered"],
        }

    return dict(
        sorted(
            combined.items(),
            key=lambda item: item[1].get("combined_probability", 0.0),
            reverse=True,
        )
    )


def _missing_modalities(active_sensors: list[str]) -> list[str]:
    canonical = SUPPORTED_SENSOR_ORDER
    active = set(active_sensors)
    return [sensor for sensor in canonical if sensor not in active]


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    print("\nNeuroPhenotype dashboard: http://localhost:5050\n")
    app.run(host = "0.0.0.0", port=5050, debug=True)
