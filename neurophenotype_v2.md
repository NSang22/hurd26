# NeuroPhenotype
## Multimodal Diagnostic Copilot for Rare Neurological Disorders

**Track: Genomic Diagnostics**

---

## Elevator Pitch

Rare neurological disorders like Rett syndrome, Dravet syndrome, and Angelman syndrome carry an average diagnostic delay of 5–7 years. Clinicians face two compounding problems: they lack functional physiological evidence to narrow genomic testing, and the clinical record they do have is fragmented, unstructured, and hard to act on.

NeuroPhenotype solves both problems in a single platform with two tightly integrated halves.

**Half 1 — Passive Biosignal Phenotyping:** A 20-minute passive assessment capturing 7 physiological and behavioral data streams (EEG, HRV, GSR, movement, speech, rPPG, fine motor) that maps live signals to the functional consequences of specific genetic mutations.

**Half 2 — Clinical Record Integration:** A structured clinical intake that ingests HPO terms, prior test history, and family history — then uses a Claude-powered reasoning layer to synthesize existing clinical knowledge with live biosignal findings into an actionable next-step recommendation and clinician-facing SOAP note.

The output is not a diagnosis. It is the **next best diagnostic step**: a prioritized genomic panel recommendation with clinical justification, cost savings estimate, and a structured note the referring physician can act on immediately.

**We're not diagnosing. We're telling clinicians which gene to sequence first — and why.**

---

## The Problem

- Average diagnostic delay for rare neurological disorders: **5–7 years**
- Children visit an average of **7 specialists** before receiving a genetic diagnosis
- Whole exome sequencing costs **$4,000–6,000** and takes weeks
- Targeted gene panels cost **$300–500** and take days — but clinicians don't know which to order
- Clinical records are fragmented: HPO terms buried in notes, prior negative tests poorly documented, family history inconsistently captured
- EEG and physiological data are routinely collected but sit underutilized

**NeuroPhenotype bridges the gap between the data clinicians already have and the physiological signal they've never had access to.**

---

## Target Conditions

| Condition | Gene | Key Biosignal Signatures | Key HPO Terms |
|---|---|---|---|
| Rett Syndrome | MECP2 (Xq28) | EEG background slowing, delta/theta elevation, hand stereotypies, sympathovagal imbalance | HP:0002376 (Hand stereotypy), HP:0002384 (Irregular respiration), HP:0001263 (Global dev delay) |
| Dravet Syndrome | SCN1A (2q24.3) | Generalized spike-wave discharges 2–3.5 Hz, high spike rate, autonomic dysregulation | HP:0001250 (Seizures), HP:0002069 (Generalized tonic-clonic), HP:0001263 (Global dev delay) |
| Angelman Syndrome | UBE3A / chr15q11-q13 | High-amplitude delta 2–4 Hz with frontal notching, near-absent speech, jerky movement | HP:0001344 (Absent speech), HP:0001250 (Seizures), HP:0001257 (Spasticity) |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                           INPUT LAYER                                │
│                                                                      │
│   HALF 1: Passive Biosignals          HALF 2: Clinical Record        │
│   ┌──────────────────────────┐        ┌──────────────────────────┐   │
│   │ EEG (neural signatures)  │        │ HPO term checklist        │   │
│   │ Apple Watch HRV          │        │ Age / onset timeline      │   │
│   │ GSR (galvanic skin)      │        │ Family history            │   │
│   │ Movement (MediaPipe+IMU) │        │ Prior test PDF upload     │   │
│   │ Speech (librosa)         │        │ VUS / negative panels     │   │
│   │ rPPG (webcam)            │        │ Inheritance pattern       │   │
│   │ Keyboard/Mouse dynamics  │        └──────────────────────────┘   │
│   └──────────────────────────┘                                       │
└──────────────────┬────────────────────────────┬──────────────────────┘
                   │                            │
     ┌─────────────▼──────────┐    ┌────────────▼────────────────┐
     │  FEATURE EXTRACTION    │    │  CLINICAL FEATURE ENCODING  │
     │  55-dim vector         │    │  HPO presence/absence/uncert │
     │  (23 EEG + 7 HRV +    │    │  Prior test flags            │
     │   4 GSR + 7 movement + │    │  Onset/severity scores       │
     │   6 speech + 4 rPPG +  │    │  Reanalysis trigger score    │
     │   4 keyboard/mouse)    │    └────────────┬────────────────┘
     └─────────────┬──────────┘                 │
                   └──────────────┬─────────────┘
                                  │
                   ┌──────────────▼──────────────────┐
                   │      FUSION & CLASSIFICATION     │
                   │  Combined biosignal + clinical   │
                   │  feature vector → XGBoost        │
                   │  4-class: Rett / Dravet /        │
                   │  Angelman / Control              │
                   │  Output: P(condition) per class  │
                   └──────────────┬──────────────────┘
                                  │
                   ┌──────────────▼──────────────────┐
                   │      CLAUDE REASONING LAYER      │
                   │  Synthesizes classifier output   │
                   │  + clinical context              │
                   │  → Next best diagnostic step     │
                   │  → Uncertainty explanation       │
                   │  → SOAP note generation          │
                   └──────────────┬──────────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
   ┌──────────▼──────────┐               ┌────────────▼────────────┐
   │    PATIENT VIEW      │               │    CLINICIAN VIEW        │
   │  Plain English       │               │  Structured SOAP note    │
   │  summary of findings │               │  Genomic panel + cost    │
   │  What happens next   │               │  Supporting biomarker    │
   │  (parent-readable)   │               │  evidence + citations    │
   └──────────────────────┘               │  "What would change this"│
                                          │  Uncertainty statement   │
                                          └──────────────────────────┘
```

---

## Half 1: Passive Biosignal Phenotyping

### Data Modalities & Feature Vectors

**1. EEG — Neural Signatures (23 features)**
- Hardware: DIY circuit (AD620AN instrumentation amp + notch/bandpass filters) or OpenBCI/Muse
- Features: Band power (delta, theta, alpha, beta, gamma), delta/theta ratio, theta/alpha ratio, dominant frequency, spike rate, mean coherence, frontal coherence, PAC strength, background slowing score, per-channel delta (10 channels)
- Genomic link: MECP2 → theta elevation + background slowing. SCN1A → spike-wave at 2–3.5 Hz + high spike rate. UBE3A → high-amplitude delta with frontal coherence signature.

**2. HRV / Heart Rate — Autonomic Nervous System (7 features)**
- Hardware: Apple Watch (HealthKit BLE)
- Features: SDNN, RMSSD, pNN50, LF/HF ratio, mean HR, HR std, normalized SDNN
- Genomic link: MECP2 mutations disrupt brainstem autonomic centers — Rett patients show reduced global HRV and sympathetic predominance. Parameters correlate with mutation subtype severity.

**3. GSR — Galvanic Skin Response (4 features)**
- Hardware: Arduino Nano 33 BLE Sense + custom two-electrode voltage divider circuit
- Features: SCL mean, SCL std, SCR count, SCR amplitude
- Genomic link: Paired with HRV for full sympathovagal profile. MECP2 and SCN1A mutations both disrupt autonomic circuits.

**4. Movement — Motor Phenotyping (7 features)**
- Hardware: Laptop webcam (MediaPipe Hands/Pose) + Arduino IMU wrist-worn
- Features: Stereotypy score (midline crossings/min), tremor frequency, sample entropy, limb rhythm index, IMU accel std (x, y, z)
- Genomic link: Hand stereotypies are pathognomonic for MECP2 mutations. Angelman shows jerky excitable motor patterns. Dravet shows ataxic features.

**5. Speech — Vocal Biomarkers (6 features)**
- Hardware: Laptop microphone (librosa, sounddevice)
- Features: Pitch mean/std, speech rhythm, mean pause duration, vocalization rate, MFCC clarity
- Genomic link: MECP2 → language regression (~80% of Rett patients nonverbal). UBE3A → near-complete speech absence.

**6. rPPG — Cardiovascular via Webcam (4 features)**
- Hardware: Laptop webcam (OpenCV-based rPPG)
- Features: Heart rate, respiratory rate, pulse waveform morphology, HRV proxy
- Genomic link: Supplements Apple Watch data; validates HR and extracts respiratory coupling relevant to MECP2 cardiorespiratory dysregulation.

**7. Keyboard & Mouse Dynamics — Fine Motor (4 features)**
- Hardware: Standard keyboard/mouse (pynput passive logging)
- Features: Key hold time mean/std, inter-keystroke interval mean/std
- Genomic link: Fine motor dysfunction is transdiagnostic. Keystroke dynamics validated as digital biomarkers for neuropsychiatric fine motor decline.

**Total feature vector: 55 dimensions**

### Classifier
- XGBoost on concatenated 55-dim feature vector (late fusion — standard and defensible)
- 4-class output: Rett / Dravet / Angelman / Control
- Training: Literature-derived synthetic profiles encoding published qEEG biomarkers per condition (Sidorov 2017, Ostrowski 2021, Roche 2019, Hall 2024)

---

## Half 2: Clinical Record Integration

### Clinical Intake Inputs

- **HPO Terms:** Structured checklist of condition-relevant Human Phenotype Ontology terms. Each flagged as: Present / Absent / Uncertain / Not assessed. Explicit absence is clinically meaningful — "no seizures" is different from "seizures not mentioned."
- **Age of Onset:** Symptom onset age, developmental regression timeline, progression pattern
- **Family History:** Affected relatives, suspected inheritance pattern (X-linked, autosomal dominant/recessive, de novo)
- **Prior Test History:** Test type (targeted panel / exome / array CGH / methylation), date, result class (negative / VUS / incomplete panel), specific genes or regions tested
- **Prior Test PDF Upload:** Uploaded result documents parsed by Claude to extract structured findings — genes tested, variants identified, interpretation summary

### Clinical Feature Encoding
- HPO terms encoded as ternary vectors (present / absent / unknown) per condition
- HPO-condition match score computed: which conditions are supported vs. contradicted by current HPO profile
- Prior test flags: has_had_testing, test_type, result_class, genes_covered
- **Reanalysis trigger score:** Prior negative test + new phenotypes emerged since testing → explicit reanalysis recommendation rather than new test order

### Claude Reasoning Layer
Takes classifier output + clinical intake and generates:

1. **Synthesized recommendation** — weighted combination of biosignal probability and HPO match score, with explicit flagging when the two conflict
2. **Next best diagnostic step** — one of: order targeted panel / reanalyze prior sequencing / refer to specialty clinic / collect additional phenotype data / urgent red flag
3. **Uncertainty statement** — "Confidence would increase with: [specific data]" and "This recommendation would change if: [alternative scenario]"
4. **Patient View** — plain English summary written for parents: what we found, what it means, what happens next
5. **Clinician View** — structured SOAP note with supporting biomarker evidence, panel recommendation, cost comparison, literature citations

### Equity Considerations
- System handles missing modalities gracefully — pipeline continues on remaining features with explicit confidence degradation
- HPO-only path available when all hardware is unavailable (low-resource deployment mode)
- Explicit low-confidence flagging rather than silent unreliable output
- Uncertainty surfaced for atypical presentations (e.g., late-onset, male Rett)

---

## Genomic Output Mapping

| Top Match | Recommended Panel | Cost | vs. WES | Turnaround |
|---|---|---|---|---|
| Rett Syndrome | MECP2 sequencing + deletion/duplication analysis | ~$400 | Save ~$4,100 | 3–5 days |
| Dravet Syndrome | SCN1A sequencing + MLPA | ~$300 | Save ~$4,500 | 5–7 days |
| Angelman Syndrome | Chr15 methylation analysis + UBE3A sequencing | ~$500 | Save ~$3,800 | 7–10 days |
| Reanalysis Triggered | Reanalysis of prior sequencing + targeted follow-up | ~$0–200 | Save ~$4,000+ | 2–3 weeks |

---

## Hardware

- **DIY EEG Circuit:** AD620AN instrumentation amp, TL084CN op-amps, passive notch + bandpass filters, 3.5mm audio input to laptop soundcard
- **Apple Watch:** HRV and HR via HealthKit BLE streaming
- **Arduino Nano 33 BLE Sense:** Wearable IMU (LSM9DS1 accel/gyro) + GSR circuit host + BLE wireless streaming
- **Laptop webcam + microphone:** rPPG (OpenCV), MediaPipe movement, speech (librosa)

---

## Software Stack

```
neurophenotype/
├── modalities/             # 7 modality implementations
│   ├── eeg.py              # 23 features
│   ├── hrv.py              # 7 features, Apple Watch BLE
│   ├── gsr.py              # 4 features, Arduino BLE
│   ├── movement.py         # 7 features, MediaPipe + IMU
│   ├── speech.py           # 6 features, librosa
│   ├── rppg.py             # 4 features, OpenCV
│   └── keyboard_mouse.py   # 4 features, pynput
├── fusion/fusion.py        # Config-driven modality loader + feature concatenation
├── classifier/
│   ├── model.py            # XGBoost wrapper with label encoder
│   └── train.py            # Synthetic data generation + training
├── clinical/               # Half 2
│   ├── hpo.py              # HPO term encoding, condition match scoring
│   ├── intake.py           # Clinical intake data model
│   ├── pdf_parser.py       # Claude API prior test PDF extraction
│   └── soap.py             # Claude API SOAP note + next-step generation
├── dashboard/app.py        # Streamlit dashboard (demo / live / patient / clinician views)
├── arduino/                # Arduino sketches for IMU + GSR BLE streaming
├── config.yaml             # Toggle modalities on/off
└── main.py
```

---

## Track Alignment

| Judge Criterion | How NeuroPhenotype Addresses It |
|---|---|
| Ingest variable inputs | 7 live biosignal streams + HPO terms + clinical notes + prior test PDFs |
| Handle uncertainty transparently | Confidence scores, conflicting evidence flags, "what would change this" on every output |
| Demonstrate equity | Graceful degradation when sensors unavailable; HPO-only path for low-resource settings; explicit low-confidence flagging |
| Next best diagnostic step | Output is a ranked action (order panel / reanalyze / refer / collect phenotype), not just probabilities |
| Reanalysis trigger | Prior negative test + new HPO phenotypes → explicit reanalysis recommendation |
| Prove validity | Synthetic profiles derived from published qEEG literature; modality ablation; confidence calibration |

---

## Team Split

| Role | Responsibilities |
|---|---|
| Hardware | EEG circuit build, GSR circuit, signal streaming to laptop, Arduino BLE |
| Hardware / Software | Webcam rPPG, MediaPipe movement, microphone speech pipeline, Arduino IMU |
| Software — Half 1 | EEG preprocessing + feature extraction, HRV + GSR features, fusion layer, classifier training |
| Software — Half 2 | HPO intake panel, Claude API integration (PDF parsing + SOAP note), clinical feature encoding |
| Software — Full Stack | Dashboard (Streamlit), demo flow, pitch deck |

---

## One-Line Judge Framing

**For physicians:** "We sit upstream of genomic testing. We give you the functional phenotype evidence to order the right $400 test instead of a $5,000 fishing expedition."

**For parents:** "No parent should spend 5 years not knowing why their child is suffering. We built the tool to end that wait."

**For industry:** "Every sensor maps to a biological system disrupted by a specific genetic mutation. We measure the functional consequences of broken genes — and combine it with everything the clinical record already knows."

**For the track:** "Other tools help clinicians make sense of data they already have. We generate data they've never had access to — and integrate both streams into a single actionable recommendation."

---

## Supporting Literature

### EEG Biomarkers

**Rett Syndrome (MECP2)**
- Roche et al. (2019). *Journal of Neurodevelopmental Disorders.* — Delta/theta elevation post-regression correlates with cognitive severity.
- Portnova et al. (2022). *Journal of Personalized Medicine.* — Background slowing correlates with MECP2 abnormalities and disease progression.
- Frontiers in Integrative Neuroscience (2025). — EEG abnormalities quantifiable across Rett, CDKL5, MECP2 duplication, Angelman syndromes.

**Dravet Syndrome (SCN1A)**
- Hall et al. (2024). *Annals of the Child Neurology Society.* — Delta power and PAC as EEG biomarkers for Dravet syndrome.
- Cheah et al. (PMC). — SCN1A mutations cause EEG slowing via impaired GABAergic parvalbumin interneuron function.
- Specchio et al. (2015). *ScienceDirect.* — Generalized 2–3.5 Hz spike-wave discharges as Dravet EEG signature.

**Angelman Syndrome (UBE3A)**
- Sidorov et al. (2017). *Journal of Neurodevelopmental Disorders.* — Delta rhythmicity is a reliable EEG biomarker; present during wakefulness and sleep.
- Ostrowski et al. (2021). *Annals of Clinical and Translational Neurology.* — Notched/polyphasic delta (2–4 Hz) is the most characteristic Angelman EEG feature.
- Elber et al. (2022). *Brain Communications.* — Longitudinal delta power model detects UBE3A treatment effects.

### HRV / Autonomic
- Frontiers in Neuroscience (2023). — Reduced global HRV and sympathovagal shift in Rett syndrome; correlates with MECP2 genotype.
- Julu et al. (2017). *ScienceDirect.* — Sympathovagal imbalance with sympathetic overactivity in MECP2-positive patients.
- Singh et al. (2024). *MDPI.* — Wearable HRV monitoring in Rett syndrome twins reveals autonomic profiles correlating with clinical severity.

### Movement / Stereotypy
- Temudo et al. (2007). *Neurology.* — Midline hand stereotypies highly indicative of MECP2 mutation in 83 Rett patients.
- Dy et al. (2017). *Movement Disorders.* — Hand stereotypies are pathognomonic for MECP2 and a primary Rett diagnostic criterion.
- STOPme Project (2025). *ScienceDirect.* — Automated detection of motor stereotypies in Rett using IMUs; validates wearable approach.

### Keystroke / Mouse Dynamics
- Alfalahi et al. (2022). *Scientific Reports.* — Keystroke dynamics as digital biomarkers for fine motor decline across neuropsychiatric disorders.
- Gajos et al. (2020). *Movement Disorders.* — Mouse dynamics capture ataxia and parkinsonism including rare genetic disorders.

---

## Key Honest Caveats (Q&A Prep)

- Classifier trained on literature-derived synthetic profiles, not clinical patient data. Standard for proof-of-concept in rare disease — IRB approval and data sharing agreements take months. Profiles encode published qEEG biomarkers, not arbitrary values.
- XGBoost on concatenated feature vector = late fusion. Defensible and standard. Don't claim deep multimodal fusion.
- HRV and movement literature is strongest for Rett. EEG is the primary discriminating signal for Dravet and Angelman.
- SOAP note is Claude API output — not a validated clinical NLP pipeline. Frame as "AI-assisted structured documentation."
- Reanalysis trigger is rule-based on prior test flags, not a learned model. Appropriate and defensible.
- Never say "diagnose." Always "triage genomic workup," "prioritize next diagnostic step," or "recommend panel."
