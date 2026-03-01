"""
rPPG modality using the open-rppg package.

Supports:
- Live webcam capture via OpenCV
- Offline video-file processing

Feature vector:
- Heart rate (BPM)
- Breathing rate (breaths per minute)
- Pulse amplitude (BVP percentile range)
- Pulse regularity (inverse coefficient of variation of beat intervals)

Requirements:
    pip install open-rppg opencv-python scipy numpy
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import find_peaks

try:
    from .base import BaseModality
except ImportError:
    from base import BaseModality

DEFAULT_CAMERA_INDEX = 0
DEFAULT_COLLECTION_SECONDS = 20.0
DEFAULT_TARGET_FPS = 30.0
MIN_CAPTURE_FRAMES = 90


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _compute_bvp_features(signal: Any, timestamps: Any) -> tuple[float, float]:
    """Estimate waveform amplitude and beat-to-beat regularity from BVP."""
    bvp = np.asarray(signal, dtype=np.float32).reshape(-1)
    ts = np.asarray(timestamps, dtype=np.float64).reshape(-1)

    if bvp.size == 0:
        return 0.0, 0.0

    amplitude = float(np.percentile(bvp, 95) - np.percentile(bvp, 5))

    if ts.size != bvp.size or bvp.size < 3:
        return amplitude, 0.0

    prominence = max(float(np.std(bvp)) * 0.1, 1e-6)
    peaks, _ = find_peaks(bvp, prominence=prominence)
    if peaks.size < 3:
        return amplitude, 0.0

    intervals = np.diff(ts[peaks])
    intervals = intervals[intervals > 0]
    if intervals.size < 2:
        return amplitude, 0.0

    mean_interval = float(np.mean(intervals))
    if mean_interval <= 0:
        return amplitude, 0.0

    interval_std = float(np.std(intervals, ddof=1))
    cv = max(interval_std / mean_interval, 0.0)
    regularity = float(1.0 / (1.0 + cv))
    return amplitude, regularity


class RPPGModality(BaseModality):
    """Wrap open-rppg behind the project's common modality interface."""

    FEATURE_DIM = 4

    def __init__(
        self,
        source: int | str = DEFAULT_CAMERA_INDEX,
        duration: float = DEFAULT_COLLECTION_SECONDS,
        target_fps: float = DEFAULT_TARGET_FPS,
        model_name: str | None = None,
    ):
        self.source = source
        self.duration = duration
        self.target_fps = target_fps
        self.model_name = model_name
        self._model = None

    def collect(self) -> dict[str, Any]:
        """
        Collect either:
        - a path to an existing video file, or
        - a short RGB frame tensor captured from a webcam.
        """
        if isinstance(self.source, (str, Path)):
            path = Path(self.source)
            if not path.exists():
                raise FileNotFoundError(f"[rPPG] Video file not found: {path}")
            return {"mode": "video", "path": str(path)}

        try:
            import cv2
        except ImportError as exc:
            raise ImportError(
                "opencv-python is required for webcam rPPG capture. "
                "Install it with `pip install opencv-python`."
            ) from exc

        cap = cv2.VideoCapture(int(self.source))
        if not cap.isOpened():
            raise RuntimeError(f"[rPPG] Could not open webcam source: {self.source}")

        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames: list[np.ndarray] = []
        start = time.monotonic()

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                if time.monotonic() - start >= self.duration:
                    break
        finally:
            cap.release()

        elapsed = max(time.monotonic() - start, 1e-6)
        if len(frames) < MIN_CAPTURE_FRAMES:
            raise RuntimeError(
                f"[rPPG] Only captured {len(frames)} frames. "
                f"Capture at least {MIN_CAPTURE_FRAMES} frames for stable inference."
            )

        fps = native_fps if native_fps > 1.0 else len(frames) / elapsed

        if self.target_fps > 0 and fps > self.target_fps * 1.5:
            step = max(int(round(fps / self.target_fps)), 1)
            frames = frames[::step]
            fps = fps / step

        print(f"[rPPG] Captured {len(frames)} frames in {elapsed:.1f}s ({fps:.1f} FPS)")
        return {"mode": "tensor", "frames": frames, "fps": float(fps)}

    def preprocess(self, raw_data: dict[str, Any]) -> dict[str, Any]:
        """Convert captured webcam frames into the tensor format expected by open-rppg."""
        if raw_data["mode"] == "video":
            return raw_data

        tensor = np.stack(raw_data["frames"]).astype(np.uint8, copy=False)
        return {"mode": "tensor", "tensor": tensor, "fps": raw_data["fps"]}

    def extract_features(self, processed_data: dict[str, Any]) -> np.ndarray:
        model = self._get_model()

        if processed_data["mode"] == "video":
            results = model.process_video(processed_data["path"])
        else:
            results = model.process_video_tensor(processed_data["tensor"], fps=processed_data["fps"])

        hrv = results.get("hrv") or {}
        heart_rate = _safe_float(results.get("hr"))
        breathing_rate = _safe_float(hrv.get("breathingrate", results.get("breathingrate")))
        sqi = _safe_float(results.get("SQI"))

        pulse_amplitude = 0.0
        pulse_regularity = 0.0
        try:
            bvp, timestamps = model.bvp()
            pulse_amplitude, pulse_regularity = _compute_bvp_features(bvp, timestamps)
        except Exception:
            pass

        print(
            f"[rPPG] HR: {heart_rate:.1f} BPM  |  Resp: {breathing_rate:.1f} bpm  |  "
            f"SQI: {sqi:.2f}  |  Amp: {pulse_amplitude:.4f}  |  Reg: {pulse_regularity:.3f}"
        )

        return np.array(
            [heart_rate, breathing_rate, pulse_amplitude, pulse_regularity],
            dtype=np.float32,
        )

    def _get_model(self):
        if self._model is not None:
            return self._model

        try:
            import rppg
        except ImportError as exc:
            raise ImportError(
                "open-rppg is not installed. Install it with `pip install open-rppg`."
            ) from exc

        self._model = rppg.Model(self.model_name) if self.model_name else rppg.Model()
        return self._model
