"""
Speech Modality — Vocal Biomarkers
Hardware: Laptop microphone
Libraries: librosa, sounddevice (or pyaudio), numpy, scipy

FEATURE VECTOR (length 6):
  [0] pitch_mean       — mean F0 in Hz (reduced in regression/nonverbal)
  [1] pitch_std        — pitch variability (prosodic range; reduced in Rett, Angelman)
  [2] speech_rhythm    — tempo regularity score (0–1)
  [3] mean_pause_dur   — mean silence duration in seconds
  [4] vocalization_rate — voiced frames per second (proxy for speech density)
  [5] mfcc_clarity     — MFCC spectral spread (phoneme clarity proxy; reduced when nonverbal)

Genomic relevance:
  MECP2 (Rett):     Language regression — ~80% of Rett patients are nonverbal or severely
                    reduced. Characteristic: low vocalization_rate, absent pitch variability,
                    long mean_pause_dur. Goodspeed 2023, MECP2 Disorders GeneReviews.
  UBE3A (Angelman): Near-complete absence of functional speech. vocalization_rate ~0,
                    pitch_std ~0 (no prosody), mfcc_clarity very low.
  SCN1A (Dravet):   Language delay, reduced complexity. Less severe than Rett/Angelman.

Requirements:
    pip install librosa sounddevice numpy scipy
    (fallback: pip install pyaudio instead of sounddevice)
"""

import numpy as np
import io

try:
    from .base import BaseModality
except ImportError:
    from base import BaseModality

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

SAMPLE_RATE     = 22050    # Hz — standard librosa SR
COLLECTION_SECONDS = 30    # 30s captures enough speech for stable features
FEATURE_DIM     = 6


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _extract_pitch_features(y: np.ndarray, sr: int) -> tuple:
    """
    Extract F0 mean and std using librosa pyin (probabilistic YIN).
    Returns (pitch_mean, pitch_std) in Hz.
    Nonverbal/low-speech patients will have pitch_mean ~0, pitch_std ~0.
    """
    if not LIBROSA_AVAILABLE:
        return 0.0, 0.0

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),   # ~65 Hz — below fundamental speech
            fmax=librosa.note_to_hz("C7"),   # ~2093 Hz — above normal speech
            sr=sr,
            fill_na=0.0,
        )
        voiced_f0 = f0[voiced_flag == 1] if voiced_flag is not None else f0[f0 > 0]
        if len(voiced_f0) < 5:
            return 0.0, 0.0
        return float(np.mean(voiced_f0)), float(np.std(voiced_f0))
    except Exception:
        return 0.0, 0.0


def _extract_rhythm_features(y: np.ndarray, sr: int) -> tuple:
    """
    Compute speech tempo regularity and pause statistics.
    Returns (rhythm_regularity, mean_pause_dur_s, vocalization_rate).

    rhythm_regularity: 1 = perfectly regular onset pattern, 0 = arrhythmic/absent
    mean_pause_dur: average gap between voiced segments
    vocalization_rate: fraction of frames with voiced activity
    """
    if not LIBROSA_AVAILABLE:
        return 0.0, 0.0, 0.0

    try:
        # Voice Activity Detection via RMS energy
        frame_len = int(sr * 0.025)   # 25ms frames
        hop_len   = int(sr * 0.010)   # 10ms hop
        rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]

        # Threshold: 15th percentile of non-silent frames
        threshold = np.percentile(rms[rms > 0], 15) if np.any(rms > 0) else 0.01
        voiced_frames = (rms > threshold).astype(int)

        vocalization_rate = float(voiced_frames.mean())

        # Pause segmentation
        transitions = np.diff(voiced_frames)
        speech_ends   = np.where(transitions == -1)[0]
        speech_starts = np.where(transitions == 1)[0]

        hop_duration = hop_len / sr
        if len(speech_ends) > 0 and len(speech_starts) > 0:
            # Pair ends with following starts
            pause_durations = []
            for end in speech_ends:
                next_starts = speech_starts[speech_starts > end]
                if len(next_starts) > 0:
                    pause_dur = (next_starts[0] - end) * hop_duration
                    if pause_dur < 10.0:   # exclude very long silences (not speech)
                        pause_durations.append(pause_dur)
            mean_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
        else:
            mean_pause = 0.0

        # Rhythm regularity: onset regularity via autocorrelation of voiced signal
        if len(speech_starts) > 3:
            inter_onset_intervals = np.diff(speech_starts) * hop_duration
            ioi_cv = np.std(inter_onset_intervals) / (np.mean(inter_onset_intervals) + 1e-8)
            rhythm = float(1.0 / (1.0 + ioi_cv))   # 1 = perfectly regular, 0 = totally irregular
        else:
            rhythm = 0.0

        return rhythm, mean_pause, vocalization_rate

    except Exception:
        return 0.0, 0.0, 0.0


def _extract_mfcc_clarity(y: np.ndarray, sr: int) -> float:
    """
    MFCC spectral spread as a phoneme clarity proxy.
    Nonverbal patients with noise/cry vocalizations have high spread (chaotic spectrum).
    Clear speech has organized MFCC structure.

    Returns normalized spectral entropy — low = clear/organized, high = noisy/absent.
    Inverted so higher value = better clarity.
    """
    if not LIBROSA_AVAILABLE:
        return 0.0

    try:
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=512)
        # Mean absolute energy per coefficient (spread measure)
        mfcc_energy = np.abs(mfcc).mean(axis=1)
        # Normalize
        mfcc_energy /= (mfcc_energy.sum() + 1e-8)
        # Shannon entropy — low = energy concentrated in few coefficients (clear speech)
        spec_entropy = -np.sum(mfcc_energy * np.log(mfcc_energy + 1e-8))
        # Normalize to 0–1, invert so high = clear
        max_entropy = np.log(13)
        clarity = float(1.0 - spec_entropy / max_entropy)
        return max(0.0, clarity)
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Audio collection
# ---------------------------------------------------------------------------

def _collect_audio(duration: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Record audio from default microphone. Returns float32 array."""
    if SOUNDDEVICE_AVAILABLE:
        print(f"[Speech] Recording {duration}s from microphone...")
        audio = sd.rec(
            int(duration * sr),
            samplerate=sr,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        return audio.flatten()
    else:
        # Fallback: pyaudio
        try:
            import pyaudio
            pa = pyaudio.PyAudio()
            stream = pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=sr,
                input=True,
                frames_per_buffer=1024,
            )
            frames = []
            n_chunks = int(sr / 1024 * duration)
            for _ in range(n_chunks):
                data = stream.read(1024, exception_on_overflow=False)
                frames.append(np.frombuffer(data, dtype=np.float32))
            stream.stop_stream()
            stream.close()
            pa.terminate()
            return np.concatenate(frames)
        except Exception as e:
            raise RuntimeError(f"No audio backend available (sounddevice or pyaudio): {e}")


# ---------------------------------------------------------------------------
# Modality class
# ---------------------------------------------------------------------------

class SpeechModality(BaseModality):
    """
    Records audio from microphone and extracts vocal biomarkers.
    Features reflect cortical and neurodevelopmental dysfunction
    associated with MECP2, UBE3A, and SCN1A mutations.
    """
    FEATURE_DIM = FEATURE_DIM

    def __init__(self, duration: int = COLLECTION_SECONDS, mock_profile: str = None):
        self.duration = duration
        self.mock_profile = mock_profile

    def collect(self) -> np.ndarray:
        if self.mock_profile:
            return self._mock_audio(self.mock_profile)
        if not LIBROSA_AVAILABLE:
            raise ImportError("pip install librosa sounddevice")
        return _collect_audio(self.duration, SAMPLE_RATE)

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        Normalize amplitude, trim leading/trailing silence.
        """
        if not LIBROSA_AVAILABLE or len(raw_data) == 0:
            return raw_data

        y = raw_data.astype(np.float32)

        # Normalize to -1..1
        max_amp = np.abs(y).max()
        if max_amp > 1e-6:
            y = y / max_amp

        # Trim long leading/trailing silence (keep up to 0.5s padding)
        try:
            y_trimmed, _ = librosa.effects.trim(y, top_db=30, frame_length=512, hop_length=128)
            if len(y_trimmed) > SAMPLE_RATE * 2:
                return y_trimmed
        except Exception:
            pass

        return y

    def extract_features(self, processed_data: np.ndarray) -> np.ndarray:
        y = processed_data
        sr = SAMPLE_RATE

        if len(y) < sr:
            print("[Speech] Warning: less than 1s of audio — returning zeros")
            return np.zeros(FEATURE_DIM, dtype=np.float32)

        pitch_mean, pitch_std = _extract_pitch_features(y, sr)
        rhythm, mean_pause, vocalization_rate = _extract_rhythm_features(y, sr)
        clarity = _extract_mfcc_clarity(y, sr)

        features = np.array(
            [pitch_mean, pitch_std, rhythm, mean_pause, vocalization_rate, clarity],
            dtype=np.float32,
        )
        print(f"[Speech] pitch={pitch_mean:.1f}Hz (std={pitch_std:.1f})  "
              f"rhythm={rhythm:.3f}  pause={mean_pause:.2f}s  "
              f"voc_rate={vocalization_rate:.3f}  clarity={clarity:.3f}")
        return features

    # ── Mock profiles ────────────────────────────────────────────────────────

    def _mock_audio(self, profile: str) -> np.ndarray:
        """
        Generate synthetic audio matching known speech phenotypes.
        Returns array that produces clinically plausible features.
        """
        rng = np.random.default_rng({"rett": 30, "dravet": 31, "angelman": 32}.get(profile, 0))
        sr = SAMPLE_RATE
        n = self.duration * sr
        t = np.linspace(0, self.duration, n)

        if profile == "rett":
            # Low vocalization: mostly silence with occasional low-pitch vocalizations
            # Corresponds to language regression (MECP2 GeneReviews: ~80% nonverbal)
            y = rng.normal(0, 0.01, n)   # baseline noise
            # Add occasional short vocalizations (not real speech)
            for onset in [3.0, 8.0, 15.0, 22.0]:
                start = int(onset * sr)
                length = int(0.5 * sr)
                if start + length < n:
                    y[start:start+length] += 0.4 * np.sin(2 * np.pi * 180 * t[start:start+length])

        elif profile == "dravet":
            # Reduced speech complexity: some speech, reduced prosody
            y = rng.normal(0, 0.02, n)
            for onset in np.arange(2.0, self.duration - 2, 3.0):
                start = int(onset * sr)
                length = int(1.0 * sr)
                if start + length < n:
                    freq = 200 + rng.normal(0, 20)
                    y[start:start+length] += 0.3 * np.sin(2 * np.pi * freq * t[start:start+length])

        else:  # angelman
            # Near-absent speech: only noise/laughter-like vocalizations, no language
            y = rng.normal(0, 0.015, n)
            # Happy affect vocalizations (higher pitch, irregular)
            for onset in [2.0, 7.0, 14.0, 21.0, 27.0]:
                start = int(onset * sr)
                length = int(0.3 * sr)
                if start + length < n:
                    y[start:start+length] += 0.35 * np.sin(2 * np.pi * 350 * t[start:start+length])

        return y.astype(np.float32)


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--mock", choices=["rett", "dravet", "angelman"])
    parser.add_argument("--file", type=str, help="Load audio from .wav file instead of mic")
    args = parser.parse_args()

    if args.file:
        if LIBROSA_AVAILABLE:
            y, sr = librosa.load(args.file, sr=SAMPLE_RATE)
            m = SpeechModality()
            m.mock_profile = None
            features = m.extract_features(m.preprocess(y))
        else:
            print("librosa required for file loading")
            features = np.zeros(FEATURE_DIM)
    else:
        m = SpeechModality(duration=args.duration, mock_profile=args.mock)
        features = m.run()

    labels = ["pitch_mean_hz", "pitch_std_hz", "speech_rhythm",
              "mean_pause_s", "vocalization_rate", "mfcc_clarity"]
    print("\n── Speech Feature Vector ──")
    for label, val in zip(labels, features):
        print(f"  {label:20s}: {val:.4f}")
    print(f"\nTotal features: {len(features)} (expected {FEATURE_DIM})")