"""
Keyboard & Mouse Modality — Fine Motor Dynamics
Hardware: Standard keyboard and mouse (passively logged via pynput)

FEATURE VECTOR (length 7):
  [0] mean_hold_time    — mean key press duration (ms); elevated in motor slowing
  [1] hold_time_std     — variability in hold time; elevated in tremor/dysmetria
  [2] mean_iki          — mean inter-keystroke interval (ms); slowed in motor dysfunction
  [3] iki_std           — IKI variability; high = irregular fine motor control
  [4] mouse_velocity    — mean mouse speed (px/s); reduced in ataxia/bradykinesia
  [5] mouse_vel_std     — velocity variability; high = tremor/jerky movement
  [6] error_rate        — backspace fraction of all keypresses (motor error proxy)

Genomic relevance:
  Fine motor dysfunction is transdiagnostic across neurological conditions.
  Literature basis: Alfalahi et al. (2022) Sci Reports — systematic review confirming
  keystroke dynamics as digital biomarkers for neuropsychiatric fine motor decline.
  Gajos et al. (2020) Movement Disorders — mouse dynamics capture ataxia and
  parkinsonism including spinocerebellar ataxia (rare genetic disorder).
  Giancardo et al. (2016) Sci Reports — key hold time as surrogate motor biomarker.

  For this project: extended from Parkinson's/MS validation to rare genetic
  neurological disorders (Rett, Dravet, Angelman) as a passive transdiagnostic
  fine motor assessment layer.

Requirements:
    pip install pynput numpy
"""

import time
import threading
import numpy as np

try:
    from .base import BaseModality
except ImportError:
    from base import BaseModality

try:
    from pynput import keyboard, mouse
    PYNPUT_AVAILABLE = True
except ImportError:
    PYNPUT_AVAILABLE = False

COLLECTION_SECONDS = 60
FEATURE_DIM = 7


class KeyboardMouseModality(BaseModality):
    FEATURE_DIM = FEATURE_DIM

    def __init__(self, duration: int = COLLECTION_SECONDS, mock_profile: str = None):
        self.duration = duration
        self.mock_profile = mock_profile

    def collect(self) -> dict:
        if self.mock_profile:
            return self._mock_data(self.mock_profile)
        if not PYNPUT_AVAILABLE:
            raise ImportError("pip install pynput")
        return self._collect_live()

    def preprocess(self, raw: dict) -> dict:
        hold_times = np.array(raw["hold_times"], dtype=np.float32)
        ikis       = np.array(raw["ikis"],       dtype=np.float32)
        velocities = np.array(raw["velocities"], dtype=np.float32)
        n_keys     = int(raw["n_keys"])
        n_errors   = int(raw["n_errors"])

        # Remove physiological outliers
        if len(hold_times) > 2:
            hold_times = hold_times[(hold_times > 10) & (hold_times < 2000)]
        if len(ikis) > 2:
            ikis = ikis[(ikis > 10) & (ikis < 5000)]
        if len(velocities) > 2:
            velocities = velocities[velocities < np.percentile(velocities, 99)]

        return {
            "hold_times": hold_times,
            "ikis":       ikis,
            "velocities": velocities,
            "n_keys":     n_keys,
            "n_errors":   n_errors,
        }

    def extract_features(self, processed: dict) -> np.ndarray:
        ht  = processed["hold_times"]
        iki = processed["ikis"]
        vel = processed["velocities"]
        n_keys   = processed["n_keys"]
        n_errors = processed["n_errors"]

        mean_hold = float(np.mean(ht))   if len(ht)  > 0 else 0.0
        std_hold  = float(np.std(ht))    if len(ht)  > 1 else 0.0
        mean_iki  = float(np.mean(iki))  if len(iki) > 0 else 0.0
        std_iki   = float(np.std(iki))   if len(iki) > 1 else 0.0
        mean_vel  = float(np.mean(vel))  if len(vel) > 0 else 0.0
        std_vel   = float(np.std(vel))   if len(vel) > 1 else 0.0
        error_rate = float(n_errors / n_keys) if n_keys > 0 else 0.0

        print(f"[KB/Mouse] hold={mean_hold:.1f}ms  IKI={mean_iki:.1f}ms  "
              f"vel={mean_vel:.1f}px/s  errors={error_rate:.3f}")

        return np.array(
            [mean_hold, std_hold, mean_iki, std_iki, mean_vel, std_vel, error_rate],
            dtype=np.float32,
        )

    # ── Live collection ──────────────────────────────────────────────────────

    def _collect_live(self) -> dict:
        hold_times = []
        ikis       = []
        velocities = []
        n_keys     = [0]
        n_errors   = [0]

        press_times   = {}    # key → press timestamp
        last_release  = [None]
        last_mouse_pos = [None]
        last_mouse_t   = [None]

        def on_press(key):
            t = time.time() * 1000   # ms
            k = str(key)
            press_times[k] = t
            n_keys[0] += 1
            if key == keyboard.Key.backspace:
                n_errors[0] += 1

        def on_release(key):
            t = time.time() * 1000
            k = str(key)
            if k in press_times:
                hold = t - press_times.pop(k)
                hold_times.append(hold)
                if last_release[0] is not None:
                    ikis.append(t - last_release[0])
                last_release[0] = t

        def on_move(x, y):
            t = time.time() * 1000
            if last_mouse_pos[0] is not None and last_mouse_t[0] is not None:
                dx = x - last_mouse_pos[0][0]
                dy = y - last_mouse_pos[0][1]
                dt = t - last_mouse_t[0]
                if dt > 0:
                    vel = np.sqrt(dx**2 + dy**2) / dt * 1000  # px/s
                    velocities.append(vel)
            last_mouse_pos[0] = (x, y)
            last_mouse_t[0] = t

        kb_listener    = keyboard.Listener(on_press=on_press, on_release=on_release)
        mouse_listener = mouse.Listener(on_move=on_move)

        kb_listener.start()
        mouse_listener.start()
        print(f"[KB/Mouse] Logging for {self.duration}s — use keyboard and mouse normally...")
        time.sleep(self.duration)
        kb_listener.stop()
        mouse_listener.stop()

        print(f"[KB/Mouse] {n_keys[0]} keys, {len(hold_times)} hold times, "
              f"{len(velocities)} velocity samples")
        return {
            "hold_times": hold_times,
            "ikis":       ikis,
            "velocities": velocities,
            "n_keys":     n_keys[0],
            "n_errors":   n_errors[0],
        }

    # ── Mock profiles ────────────────────────────────────────────────────────

    def _mock_data(self, profile: str) -> dict:
        """
        Synthetic typing/mouse profiles per condition.
        Rett:     Very slow, high hold time, high error rate (severe motor impairment)
        Dravet:   Moderate slowing, elevated IKI variability (ataxic features)
        Angelman: Significant impairment; coarse motor preserved but fine motor poor
        Normal:   Typical healthy typing dynamics
        """
        rng = np.random.default_rng({"rett": 40, "dravet": 41,
                                     "angelman": 42, "normal": 43}.get(profile, 0))
        n = 80  # simulated keystrokes

        if profile == "rett":
            hold  = rng.normal(180, 60, n).clip(50, 600)
            iki   = rng.normal(450, 180, n).clip(100, 2000)
            vel   = rng.normal(120, 80, n * 5).clip(0, 500)
            n_keys, n_err = n, int(n * 0.18)
        elif profile == "dravet":
            hold  = rng.normal(130, 55, n).clip(30, 500)
            iki   = rng.normal(320, 160, n).clip(80, 1500)
            vel   = rng.normal(200, 120, n * 5).clip(0, 700)
            n_keys, n_err = n, int(n * 0.10)
        elif profile == "angelman":
            hold  = rng.normal(160, 70, n).clip(40, 600)
            iki   = rng.normal(400, 200, n).clip(80, 2000)
            vel   = rng.normal(150, 110, n * 5).clip(0, 600)
            n_keys, n_err = n, int(n * 0.14)
        else:  # normal
            hold  = rng.normal(90, 25, n).clip(20, 300)
            iki   = rng.normal(180, 60, n).clip(50, 800)
            vel   = rng.normal(400, 150, n * 5).clip(0, 1200)
            n_keys, n_err = n, int(n * 0.04)

        return {
            "hold_times": hold.tolist(),
            "ikis":       iki.tolist(),
            "velocities": vel.tolist(),
            "n_keys":     n_keys,
            "n_errors":   n_err,
        }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=20)
    parser.add_argument("--mock", choices=["rett", "dravet", "angelman", "normal"])
    args = parser.parse_args()

    m = KeyboardMouseModality(duration=args.duration, mock_profile=args.mock)
    features = m.run()
    labels = ["mean_hold_ms", "hold_std_ms", "mean_iki_ms", "iki_std_ms",
              "mouse_vel_px/s", "vel_std_px/s", "error_rate"]
    print("\n── Keyboard/Mouse Feature Vector ──")
    for label, val in zip(labels, features):
        print(f"  {label:18s}: {val:.4f}")