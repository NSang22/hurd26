# NeuroPhenotype
## Multimodal Biomarker Platform for Genomic Diagnosis Prioritization in Rare Neurological Disorders

---

## Elevator Pitch

Rare neurological disorders like Rett syndrome, Dravet syndrome, and Angelman syndrome carry an average diagnostic delay of 5–7 years. Genomic sequencing is the gold standard but clinicians lack tools to know which panels to prioritize. NeuroPhenotype is a passive, non-invasive multimodal phenotyping platform that simultaneously captures six physiological and behavioral data streams and fuses them into a unified classifier that outputs a ranked probability match to known rare neurological disorder genotypes — alongside a prioritized genomic testing panel recommendation.

We're not diagnosing. We're telling clinicians which genes to sequence first.

**Track: Genomic Diagnostics**

---

## The Problem

- Average diagnostic delay for rare neurological disorders: **5–7 years**
- Whole exome sequencing: **$4,000–6,000**, takes weeks
- Targeted gene panels: **$300–500**, takes days
- Clinicians order broad tests because they lack functional evidence to narrow down
- EEG and physiological data are routinely collected in these patients but sit underutilized

**NeuroPhenotype provides the functional evidence to narrow the genomic search space before expensive sequencing happens.**

---

## Target Conditions

| Condition | Gene | Key Biomarker Signatures |
|---|---|---|
| Rett Syndrome | MECP2 (Xq28) | EEG background slowing, delta/theta elevation, hand stereotypies, sympathovagal imbalance (HRV), language regression |
| Dravet Syndrome | SCN1A | Generalized spike-wave discharges (2–3.5 Hz), delta/theta power reduction, autonomic dysregulation |
| Angelman Syndrome | UBE3A / chr15q11-q13 | High-amplitude delta waves (2–4 Hz) with frontal notching, near-absent speech, jerky movement pattern |

---

## Data Modalities

### 1. EEG — Neural Signatures
- **Hardware:** OpenBCI or Muse headset
- **Features:** Band power (delta, theta, alpha, beta, gamma), spike detection, inter-electrode coherence, phase-amplitude coupling
- **Genomic link:** MECP2 mutations produce sensorimotor rhythm attenuation and delta/theta slowing. SCN1A haploinsufficiency reduces inhibitory interneuron firing, altering low-frequency oscillations. UBE3A loss produces pathognomonic high-amplitude delta with frontal notching.

### 2. HRV / Heart Rate — Autonomic Nervous System
- **Hardware:** Apple Watch (HealthKit API)
- **Features:** SDNN, RMSSD, pNN50, LF/HF ratio (sympathovagal balance)
- **Genomic link:** MECP2 mutations disrupt brainstem autonomic centers — Rett patients show reduced global HRV and sympathetic predominance with vagal withdrawal. HRV parameters correlate with MECP2 mutation subtype severity.

### 3. rPPG — Cardiovascular via Webcam
- **Hardware:** Laptop webcam
- **Library:** OpenCV-based rPPG (e.g. pyVHR or similar)
- **Features:** Heart rate, respiratory rate, pulse waveform morphology
- **Genomic link:** Supplements Apple Watch autonomic data; validates HR signal and extracts respiratory coupling metrics relevant to MECP2 cardiorespiratory dysregulation

### 4. GSR — Galvanic Skin Response (Sympathetic Reactivity)
- **Hardware:** Arduino Nano 33 BLE Sense + custom GSR circuit (two finger electrodes, voltage divider, analog input)
- **Features:** Skin conductance level (SCL), skin conductance response (SCR) peaks, sympathetic reactivity
- **Genomic link:** Paired with HRV to compute full sympathovagal profile. MECP2 and SCN1A mutations both disrupt autonomic circuits — GSR captures sympathetic arm while HRV captures parasympathetic arm.

### 5. Movement Analysis — Motor Phenotyping
- **Hardware:** Laptop webcam (MediaPipe) + Arduino Nano 33 BLE Sense IMU (wearable)
- **Libraries:** MediaPipe Pose / Hands, Arduino IMU streaming via BLE
- **Features:** Hand stereotypy detection (midline repetitive movements), tremor frequency, movement irregularity, upper limb rhythm
- **Genomic link:** Hand stereotypies are a primary diagnostic criterion for Rett syndrome and are pathognomonic for MECP2 mutations. Angelman syndrome presents with characteristic jerky, excitable motor patterns. Dravet shows ataxic features. The Arduino IMU worn on the wrist provides wearable tremor and stereotypy quantification.

### 6. Speech Analysis — Vocal Biomarkers
- **Hardware:** Laptop microphone
- **Library:** librosa (pitch, rhythm, pause patterns, formants), ElevenLabs API (optional transcription/prosody)
- **Features:** Pitch variability, speech rhythm, pause duration, phoneme clarity, vocalization rate
- **Genomic link:** MECP2 mutations cause language regression — ~80% of Rett patients are nonverbal or have severely reduced speech. UBE3A loss (Angelman) produces near-complete absence of functional speech. Vocal biomarkers reflect cortical and neurodevelopmental dysfunction specific to these mutations.

### 7. Keyboard & Mouse Dynamics — Fine Motor
- **Hardware:** Standard keyboard and mouse (passively logged)
- **Features:** Key hold time, inter-keystroke interval, mouse velocity, positional accuracy, error rate
- **Genomic link:** Fine motor dysfunction is a transdiagnostic marker across neurological conditions. Keystroke dynamics have been validated as digital biomarkers for neuropsychiatric fine motor decline. Provides passive baseline motor assessment.

---

## Hardware Kit

### Primary Hardware
- **OpenBCI Cyton or Muse** — EEG acquisition and streaming
- **Apple Watch** — HRV and HR via HealthKit
- **Arduino Nano 33 BLE Sense** — Wearable IMU (movement/tremor) + GSR circuit host
- **OV7675 Camera + TinyML Shield** — Optional on-device edge inference, secondary camera
- **Laptop webcam + microphone** — rPPG, MediaPipe, speech

### Arduino Nano 33 BLE Sense Onboard Sensors Being Used
- **IMU (LSM9DS1):** Accelerometer + gyroscope for wrist-worn motor analysis
- **BLE:** Wireless streaming to central pipeline
- **Analog input:** GSR electrode circuit

### GSR Circuit (Build Time: ~2 hours)
```
Finger Electrode 1 → 3.3V
Finger Electrode 2 → Resistor (100kΩ) → GND
                   → Analog Pin A0 (Arduino)
```
Signal requires low-pass filtering (cutoff ~5 Hz) to remove motion artifacts.

---

## Software Architecture

```
┌─────────────────────────────────────────────────────┐
│                  DATA ACQUISITION LAYER              │
│  EEG Stream │ Apple Watch │ Webcam │ Arduino BLE     │
│  Microphone │ Keyboard/Mouse Logger                  │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               FEATURE EXTRACTION LAYER               │
│  EEG: Band power, coherence, spike detection        │
│  HRV: SDNN, RMSSD, LF/HF ratio                     │
│  GSR: SCL, SCR peaks, sympathovagal balance         │
│  rPPG: HR, respiratory rate                         │
│  Movement: Stereotypy detection, tremor freq        │
│  Speech: Pitch variability, rhythm, pause patterns  │
│  Motor: Keystroke intervals, mouse dynamics         │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│               FUSION & CLASSIFICATION LAYER          │
│  Concatenated feature vector → XGBoost classifier  │
│  Trained on public datasets + HackRare private data │
│  Output: Probability scores per disorder            │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│                   OUTPUT DASHBOARD                   │
│  Disorder match probabilities                       │
│  Recommended genomic panel per top match            │
│  Estimated cost savings vs. whole exome             │
└─────────────────────────────────────────────────────┘
```

---

## Modular Design Principles

**The pipeline is fully modular.** Adding a new input modality should require writing one new file and editing one config value — nothing else.

### Abstract Base Class

Every modality inherits from a common base:

```python
# modalities/base.py
from abc import ABC, abstractmethod
import numpy as np

class BaseModality(ABC):
    @abstractmethod
    def collect(self) -> any:
        """Collect raw data from hardware/software source"""
        pass

    @abstractmethod
    def preprocess(self, raw_data: any) -> any:
        """Clean, filter, epoch raw data"""
        pass

    @abstractmethod
    def extract_features(self, processed_data: any) -> np.ndarray:
        """Return fixed-length 1D feature vector"""
        pass

    def run(self) -> np.ndarray:
        raw = self.collect()
        processed = self.preprocess(raw)
        return self.extract_features(processed)
```

### Repo Structure

```
neurophenotype/
├── modalities/
│   ├── base.py              ← abstract base class
│   ├── eeg.py
│   ├── hrv.py
│   ├── gsr.py
│   ├── rppg.py
│   ├── movement.py
│   ├── speech.py
│   └── keyboard_mouse.py
├── fusion/
│   └── fusion.py            ← loads active modalities, concatenates vectors
├── classifier/
│   ├── model.py             ← XGBoost wrapper, train/predict
│   └── train.py             ← training script on datasets
├── dashboard/
│   └── app.py               ← output UI (streamlit or similar)
├── arduino/
│   └── nano_ble_sense.ino   ← IMU + GSR streaming sketch
├── data/
│   ├── public/              ← public EEG datasets
│   └── private/             ← HackRare private datasets (gitignored)
├── config.yaml              ← toggle modalities on/off
├── main.py                  ← entry point
└── PLANNING.md              ← this file
```

### config.yaml — Toggle Modalities Without Code Changes

```yaml
modalities:
  eeg: true
  hrv: true
  gsr: true
  rppg: true
  movement: true
  speech: true
  keyboard_mouse: true

classifier:
  model: xgboost
  target_conditions:
    - rett_syndrome
    - dravet_syndrome
    - angelman_syndrome

output:
  show_genomic_panels: true
  show_cost_savings: true
```

If a sensor isn't working day-of, flip it to `false`. The fusion layer reads the config and concatenates only active modality vectors. The classifier retrains or loads a checkpoint on whatever features are available.

### Fusion Layer

```python
# fusion/fusion.py
import yaml
import numpy as np
from modalities import eeg, hrv, gsr, rppg, movement, speech, keyboard_mouse

MODALITY_MAP = {
    "eeg": eeg.EEGModality,
    "hrv": hrv.HRVModality,
    "gsr": gsr.GSRModality,
    "rppg": rppg.RPPGModality,
    "movement": movement.MovementModality,
    "speech": speech.SpeechModality,
    "keyboard_mouse": keyboard_mouse.KeyboardMouseModality,
}

def get_feature_vector(config_path="config.yaml") -> np.ndarray:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    vectors = []
    for name, active in config["modalities"].items():
        if active:
            modality = MODALITY_MAP[name]()
            vectors.append(modality.run())
    
    return np.concatenate(vectors)
```

### Adding a New Modality (e.g. pupillometry)

1. Create `modalities/pupillometry.py` implementing `BaseModality`
2. Add `"pupillometry": PupillometryModality` to `MODALITY_MAP` in `fusion.py`
3. Add `pupillometry: true` to `config.yaml`

That's it. Nothing else changes.

---

## Genomic Output Mapping

| Classifier Output | Recommended Genomic Panel | Targeted Cost | vs. WES |
|---|---|---|---|
| Rett Syndrome (high prob) | MECP2 sequencing + deletion/duplication analysis | ~$400 | Save ~$4,000 |
| Dravet Syndrome (high prob) | SCN1A sequencing | ~$300 | Save ~$4,500 |
| Angelman Syndrome (high prob) | Chromosome 15 methylation analysis + UBE3A sequencing | ~$500 | Save ~$3,800 |

---

## Team Split

| Role | Responsibilities |
|---|---|
| Hardware Person 1 | EEG setup and acquisition, GSR circuit build, signal streaming to laptop |
| Hardware Person 2 | Webcam rPPG setup, MediaPipe movement tracking, microphone audio pipeline, Arduino IMU BLE streaming |
| Software Person 1 | EEG preprocessing (filtering, artifact rejection, epoching), EEG + GSR + HRV feature extraction |
| Software Person 2 | rPPG, speech, MediaPipe, keyboard/mouse feature extraction — all lighter modalities |
| Software Person 3 | Dataset exploration (9am Saturday priority), feature fusion, classifier training, output dashboard, pitch deck |

---

## Saturday Morning Priority (9am)

- **Person 3 on private datasets immediately** — identify which conditions are represented before building classifier
- Everyone else builds pipeline skeleton on public datasets in parallel
- Swap in private data once conditions are confirmed
- Hardware people should have clean streaming signals by early Saturday afternoon

---

## Datasets

### Public
- PhysioNet EEG datasets
- Published Rett / Angelman / Dravet EEG studies (see citations)
- Any available open-access pediatric EEG repositories

### Private (HackRare-provided)
- To be explored at 9am Saturday — scope classifier to represented conditions

---

## Supporting Literature

### EEG Biomarkers

**Rett Syndrome (MECP2)**
- Roche et al. (2019). *Journal of Neurodevelopmental Disorders.* — EEG spectral power as marker of cortical function and disease severity in Rett syndrome; delta/theta elevation post-regression correlates with cognitive severity.
- Portnova et al. (2022). *Journal of Personalized Medicine.* — Resting EEG parameters correlate with MECP2 abnormalities and disease progression; generalized background slowing is a key marker.
- Frontiers in Integrative Neuroscience (2025). — EEG abnormalities quantifiable in Rett, CDKL5, MECP2 duplication, and Angelman syndromes; features correlate with progression and severity.

**Dravet Syndrome (SCN1A)**
- Hall et al. (2024). *Annals of the Child Neurology Society.* — Delta power and phase-amplitude coupling as EEG biomarkers for Dravet syndrome; SCN1A haploinsufficiency limits inhibitory interneuron firing, reducing low-frequency oscillations.
- Cheah et al. (PMC). — SCN1A mutations cause EEG slowing and altered theta/gamma oscillations due to impaired GABAergic parvalbumin-positive interneuron function.
- Specchio et al. (2015). *ScienceDirect.* — EEG features of SCN1A-positive Dravet syndrome patients: generalized 2–3.5 Hz spike-wave discharges, myoclonic seizure EEG correlates.

**Angelman Syndrome (UBE3A)**
- Sidorov et al. (2017). *Journal of Neurodevelopmental Disorders.* — Delta rhythmicity is a reliable EEG biomarker in Angelman syndrome; increased delta power present during wakefulness and sleep, generalized across neocortex.
- Ostrowski et al. (2021). *Annals of Clinical and Translational Neurology.* — Diffuse, frequent bursts of notched or polyphasic delta activity (2–4 Hz) are the most characteristic EEG feature of Angelman syndrome across all genotypic classes.
- Elber et al. (2022). *Brain Communications.* — Longitudinal delta power model detects treatment effects and correlates with UBE3A expression.

**ML + EEG for Rare Genetic Disorders**
- MDPI Sensors (2025). — ML applied to EEG in Angelman syndrome (SVR on delta power) correlates strongly with clinical severity; Fragile X ML model achieves AUC 0.96 distinguishing patients from controls.

### HRV / Autonomic

- Frontiers in Neuroscience (2023). — Comprehensive review: reduced global HRV and sympathovagal shift toward sympathetic predominance in Rett syndrome; HRV parameters correlate with MECP2 genotype.
- Julu et al. (2017). *ScienceDirect.* — Both parasympathetic and sympathetic HRV significantly reduced in MECP2 mutation-positive Rett patients vs. controls; significant sympathovagal imbalance with sympathetic overactivity.
- Singh et al. (2024). *MDPI.* — Wearable HRV monitoring in Rett syndrome twins with identical MECP2 mutations reveals discordant autonomic profiles correlating with clinical severity.

### Movement / Stereotypy

- Temudo et al. (2007). *Neurology.* — Hand stereotypies in 83 Rett patients; combination of midline stereotypies without hand gaze + bruxism highly indicative of MECP2 mutation.
- Dy et al. (2017). *Movement Disorders.* — Hand stereotypies are a primary diagnostic criterion for Rett syndrome; continuous, mainly midline, may precede loss of hand function.
- STOPme Project (2025). *ScienceDirect.* — Automated detection of motor stereotypies in Rett syndrome using inertial measurement units; directly validates wearable IMU approach.

### Keystroke / Mouse Dynamics

- Alfalahi et al. (2022). *Scientific Reports.* — Systematic review and meta-analysis: keystroke dynamics are feasible digital biomarkers for fine motor decline across neuropsychiatric disorders including Parkinson's, MCI, and bipolar disorder.
- Gajos et al. (2020). *Movement Disorders.* — Computer mouse use captures ataxia and parkinsonism including spinocerebellar ataxia (a rare genetic disorder); position and speed features most discriminative.
- Giancardo et al. (2016). *Scientific Reports.* — Keyboard interaction as indicator of early Parkinson's disease; key hold time as surrogate motor biomarker.

---

## One-Line Judge Framing

**For physicians:** "We sit upstream of genomic testing. We're a functional phenotyping filter that makes genomic diagnosis faster and cheaper."

**For parents of patients:** "No parent should spend 5 years not knowing why their child is suffering while doctors guess which gene to sequence. We built the tool to end that wait."

**For industry:** "Every sensor maps to a biological system disrupted by a specific genetic mutation. We're measuring the functional consequences of broken genes."

---

## Key Honest Caveats (Know These for Q&A)

- Classifier is XGBoost on concatenated feature vectors — not deep multimodal fusion. Frame fusion as "late fusion" which is defensible and standard.
- HRV and movement literature is strongest for Rett. For Dravet and Angelman, EEG is the primary classifier signal.
- Most keystroke/mouse literature is Parkinson's and MS — frame as "extending a validated methodology to rare genetic neurological disorders."
- You are not diagnosing. You are triaging genomic workup. Never say "diagnose" in the demo or pitch.