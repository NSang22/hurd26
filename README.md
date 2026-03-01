# NeuroPhenotype

**Multimodal Diagnostic Copilot for Rare Neurological Disorders**

*Track: Genomic Diagnostics*

---

## What It Does

Rare neurological disorders like **Rett syndrome**, **Dravet syndrome**, and **Angelman syndrome** carry an average diagnostic delay of **5–7 years**. Children visit an average of **7 specialists** before receiving a genetic diagnosis. Whole exome sequencing costs $4,000–6,000 and takes weeks. Targeted gene panels cost $300–500 and take days — but clinicians don't know which to order.

**NeuroPhenotype tells clinicians which gene to sequence first — and why.**

It fuses two independent data streams into a single actionable recommendation:

| Half | What It Captures | How |
|------|-----------------|-----|
| **Half 1 — Biosignal Phenotyping** | Functional physiological evidence from 7 data streams | 20-minute passive assessment via EEG, HRV, GSR, movement, speech, rPPG, keyboard/mouse |
| **Half 2 — Clinical Record Integration** | Structured clinical knowledge from the existing chart | HPO terms, prior test history, family history, onset timeline, Claude-powered reasoning |

The output is **not a diagnosis**. It is the **next best diagnostic step**: a prioritized genomic panel recommendation with clinical justification, cost savings estimate, and a structured clinician note.

---

## Target Conditions

| Condition | Gene | Key Biosignal Signatures | Key HPO Terms |
|-----------|------|-------------------------|---------------|
| **Rett Syndrome** | MECP2 (Xq28) | EEG background slowing, delta/theta elevation, hand stereotypies, sympathovagal imbalance | Hand stereotypy, Irregular respiration, Global developmental delay |
| **Dravet Syndrome** | SCN1A (2q24.3) | Generalized spike-wave 2–3.5 Hz, high spike rate, autonomic dysregulation | Seizures, Generalized tonic-clonic, Global developmental delay |
| **Angelman Syndrome** | UBE3A / chr15q11-q13 | High-amplitude delta 2–4 Hz, frontal notching, near-absent speech, jerky movement | Absent speech, Seizures, Spasticity |

---

## Architecture

```
INPUT LAYER
├── Half 1: Passive Biosignals              Half 2: Clinical Record
│   ├── EEG (23 features) (Live)                   ├── HPO term checklist
│   ├── HRV - Apple Watch (7 features) (Live)    ├── Age of onset / severity
│   ├── GSR - Arduino BLE (4 features)      ├── Family history
│   ├── Movement - MediaPipe (7) (Live)        ├── Prior test history / PDF upload
│   ├── Speech - librosa (6 features) (Live)       └── Inheritance pattern
│   ├── rPPG - webcam (4 features)
│   └── Keyboard/Mouse (4 features)
│
├── FEATURE EXTRACTION → 55-dim vector
│
├── FUSION & CLASSIFICATION
│   └── XGBoost 4-class (Rett / Dravet / Angelman / Control)
│
├── CLAUDE REASONING LAYER
│   ├── Synthesized recommendation
│   ├── Next best diagnostic step
│   ├── Uncertainty statement ("confidence would increase with...")
│   ├── SOAP note (clinician view)
│   └── Plain English summary (patient/family view)
│
└── OUTPUT
    ├── Clinician View — SOAP note, genomic panel, cost, biomarker evidence
    └── Patient View — plain language summary, what happens next
```

---

## Genomic Output Mapping

| Top Match | Recommended Panel | Cost | Savings vs. WES | Turnaround |
|-----------|------------------|------|-----------------|------------|
| Rett Syndrome | MECP2 sequencing + del/dup analysis | ~$400 | ~$4,100 | 3–5 days |
| Dravet Syndrome | SCN1A sequencing + MLPA | ~$300 | ~$4,500 | 5–7 days |
| Angelman Syndrome | Chr15 methylation + UBE3A sequencing | ~$500 | ~$3,800 | 7–10 days |
| Reanalysis Triggered | Reanalysis of prior sequencing | ~$200 | ~$4,000+ | 2–3 weeks |

---

## Quick Start

### Prerequisites

- Python 3.10+
- (Optional) Arduino Nano 33 BLE Sense for IMU/GSR
- (Optional) OpenBCI Cyton or Muse for EEG
- (Optional) Apple Watch for HRV via BLE

### Setup

```bash
# Clone
git clone https://github.com/NSang22/hurd26.git
cd hurd26

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r neurophenotype/requirements.txt
```

### Run the Dashboard

```bash
cd neurophenotype
python dashboard/server.py
```

Open **http://localhost:5050** in your browser. You'll see the sign-in page with two roles:

| Role | Demo Credentials | Access |
|------|-----------------|--------|
| **Doctor** | `doctor@neuro.com` / `NeuroPhenotype2026` | Full clinician console — biosignal evidence, HPO scoring, SOAP notes, reanalysis triggers |
| **Patient** | `patient@family.com` / `FamilyAccess2026` | Simplified family-facing view — key outcome, next step, plain-language explanation |

You can also **Sign in with Google** or click **Skip (Demo)** to bypass auth for quick hackathon demos.

### Demo Mode vs. Clinical Mode

- **Demo Mode** — Select a synthetic profile (Rett / Dravet / Angelman / Control) and run the pipeline with literature-derived mock data. No hardware needed.
- **Clinical Mode** — Run the live biosignal pipeline with real sensor input. Toggle individual sensors on/off as available.

---

## Project Structure

```
neurophenotype/
├── modalities/                 # 7 modality implementations
│   ├── base.py                 # Abstract base class
│   ├── eeg.py                  # 23 features — OpenBCI/Muse/DIY circuit
│   ├── hrv.py                  # 7 features — Apple Watch BLE
│   ├── gsr.py                  # 4 features — Arduino BLE
│   ├── movement.py             # 7 features — MediaPipe + Arduino IMU
│   ├── speech.py               # 6 features — librosa, sounddevice
│   ├── rppg.py                 # 4 features — OpenCV webcam
│   └── keyboard_mouse.py       # 4 features — pynput
├── fusion/
│   └── fusion.py               # Config-driven modality loader + concatenation
├── classifier/
│   ├── model.py                # XGBoost wrapper with label encoder
│   ├── train.py                # Synthetic profile generation + training
│   └── inference.py            # Feature vector builder for inference
├── clinical/                   # Half 2 — Clinical Record Integration
│   ├── hpo.py                  # HPO term encoding, condition match scoring
│   ├── intake.py               # Clinical intake data model
│   ├── pdf_parser.py           # Claude API prior test PDF extraction
│   ├── soap.py                 # SOAP note + patient summary generation
│   └── claude_client.py        # Claude API client wrapper
├── dashboard/
│   ├── server.py               # Flask backend (port 5050) + auth
│   ├── landing.html            # Sign-in page (doctor/patient/Google/skip)
│   ├── index.html              # Main diagnostic dashboard
│   └── app.py                  # Streamlit fallback dashboard
├── arduino/
│   └── nano_ble_sense.ino      # IMU + GSR BLE streaming sketch
├── data/
│   ├── public/                 # Public EEG datasets
│   └── private/                # Private datasets (gitignored)
├── config.yaml                 # Toggle modalities on/off
├── main.py                     # CLI entry point
└── requirements.txt
```

---

## Hardware

| Component | Purpose | Required? |
|-----------|---------|-----------|
| **DIY EEG Circuit** | AD620AN instrumentation amp, TL084CN op-amps, notch/bandpass filters → laptop soundcard | Optional (demo mode works without) |
| **OpenBCI Cyton / Muse** | Clinical-grade EEG acquisition | Optional |
| **Apple Watch** | HRV and HR via HealthKit BLE | Optional |
| **Arduino Nano 33 BLE Sense** | Wearable IMU (LSM9DS1) + GSR circuit host + BLE streaming | Optional |
| **Laptop webcam** | rPPG cardiovascular, MediaPipe movement tracking | Included |
| **Laptop microphone** | Speech vocal biomarker capture | Included |

The pipeline is fully modular. If a sensor isn't available, set it to `false` in `config.yaml` — the fusion layer concatenates only active modality vectors and the classifier will handle the missing features.

---

## Dashboard Features

### Three-Column Layout
- **Left — Input Layer:** Sensor acquisition status, sensor toggles, HPO term checklist, clinical intake (onset, family history, prior testing)
- **Center — Fusion + Classification:** Live biosignal capture modals (movement via webcam, speech via mic), combined condition priorities, live clinical document with inline editing
- **Right — Recommendation Layer:** Top match, recommended genomic panel, confidence score, estimated savings, turnaround, HPO condition scores, uncertainty notes

### Live Clinical Document
After biosignal results return, the center column displays an interactive clinical paper document:
- HPO terms render as clickable styled spans (confirmed / absent / uncertain / unassessed)
- Onset, family history, and prior testing fields are editable via inline popovers
- Changes trigger live probability recalculation without a server round-trip
- Typewriter animation effect for text updates

### Authentication
- Username/password with SQLite backend (auto-created on first run)
- Google Sign-In (OAuth 2.0)
- Pre-seeded demo accounts for both doctor and patient roles
- **Skip (Demo)** button for hackathon presentations

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **XGBoost late fusion** | Concatenated 55-dim feature vector. Standard, defensible, interpretable. Not claiming deep multimodal fusion. |
| **Literature-derived synthetic training** | Profiles encode published qEEG biomarkers (Sidorov 2017, Ostrowski 2021, Hall 2024). Standard for rare disease proof-of-concept where IRB approval and data sharing take months. |
| **Claude reasoning layer** | Synthesizes classifier output + clinical context into structured notes. Framed as AI-assisted documentation, not validated clinical NLP. |
| **Reanalysis trigger** | Rule-based: prior negative test + new phenotypes since testing → recommend reanalysis over new sequencing. Appropriate and defensible. |
| **Graceful degradation** | Missing modalities handled with zero-masking and explicit confidence reduction. HPO-only pathway available for low-resource settings. |

---

## Equity Considerations

- Pipeline continues with any subset of sensors — explicit confidence degradation flagged to the user
- HPO-only path available when all hardware is unavailable (low-resource deployment)
- Explicit low-confidence flagging rather than silent unreliable output
- Uncertainty surfaced for atypical presentations (late-onset, male Rett, etc.)
- Patient view designed for families navigating the diagnostic journey — plain language, no jargon

---

## Supporting Literature

**EEG — Rett (MECP2):** Roche et al. 2019, *J. Neurodevelopmental Disorders*; Portnova et al. 2022, *J. Personalized Medicine*

**EEG — Dravet (SCN1A):** Hall et al. 2024, *Ann. Child Neurology Society*; Specchio et al. 2015, *ScienceDirect*

**EEG — Angelman (UBE3A):** Sidorov et al. 2017, *J. Neurodevelopmental Disorders*; Ostrowski et al. 2021, *Ann. Clinical and Translational Neurology*; Elber et al. 2022, *Brain Communications*

**HRV / Autonomic:** Julu et al. 2017, *ScienceDirect*; Singh et al. 2024, *MDPI*

**Movement / Stereotypy:** Temudo et al. 2007, *Neurology*; Dy et al. 2017, *Movement Disorders*; STOPme Project 2025, *ScienceDirect*

**Keystroke Dynamics:** Alfalahi et al. 2022, *Scientific Reports*; Gajos et al. 2020, *Movement Disorders*

---

## Important Disclaimer

This system is a **research prototype** for diagnostic triage. It does **not** provide a clinical diagnosis. All outputs are probabilistic recommendations intended to prioritize the next diagnostic step. Results should be interpreted by a qualified clinician in the context of the full clinical picture.
