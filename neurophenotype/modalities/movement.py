"""
Movement Modality — Motor Phenotyping
Hardware: Laptop webcam (MediaPipe Hands/Pose) + Arduino Nano 33 BLE Sense IMU (wrist-worn)
Libraries: mediapipe, opencv-python, bleak, numpy, scipy

FEATURE VECTOR (length 7):
  [0] stereotypy_score     — midline crossing rate (Rett MECP2 hallmark)
  [1] tremor_freq_hz       — dominant tremor frequency from IMU (3–8 Hz typical)
  [2] movement_irregularity — sample entropy of wrist trajectory
  [3] limb_rhythm_index    — periodicity score of upper limb motion
  [4] imu_accel_std_x      — IMU acceleration variability x-axis
  [5] imu_accel_std_y      — IMU acceleration variability y-axis
  [6] imu_accel_std_z      — IMU acceleration variability z-axis

Genomic relevance:
  MECP2 (Rett):    Hand stereotypies are a PRIMARY diagnostic criterion. Midline
                   repetitive hand-wringing movements are pathognomonic for MECP2
                   mutations. Temudo et al. (2007): present in 100% of Rett patients.
                   STOPme Project (2025): wearable IMU directly validates this approach.
  UBE3A (Angelman): Characteristic jerky, excitable motor pattern. Wide-based gait.
                    High limb_rhythm_index with high irregularity.
  SCN1A (Dravet):   Ataxic features during interictal periods. Motor slowing.

Requirements:
    pip install mediapipe opencv-python bleak scipy numpy
"""

import asyncio
import struct
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import welch, find_peaks
from scipy.stats import entropy as scipy_entropy

try:
    from .base import BaseModality
except ImportError:
    from base import BaseModality

try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    MEDIAPIPE_AVAILABLE = True
    USE_SOLUTIONS_API = hasattr(mp, "solutions")
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    USE_SOLUTIONS_API = False

HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
HAND_MODEL_PATH = Path(__file__).parent.parent / "models" / "hand_landmarker.task"


def _ensure_hand_model() -> bool:
    """Download hand_landmarker.task if not already present. Returns True if ready."""
    if HAND_MODEL_PATH.exists():
        return True
    try:
        import urllib.request
        HAND_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print("[Movement] Downloading hand_landmarker.task model (~8 MB)...")
        urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        print("[Movement] Model downloaded.")
        return True
    except Exception as e:
        print(f"[Movement] Could not download hand model: {e}")
        return False

try:
    from bleak import BleakScanner, BleakClient
    BLEAK_AVAILABLE = True
except ImportError:
    BLEAK_AVAILABLE = False

# Arduino BLE (must match nano_ble_sense.ino)
ARDUINO_DEVICE_NAME  = "NeuroPhenotype"
ARDUINO_SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
IMU_CHAR_UUID        = "12345678-1234-1234-1234-123456789abf"  # separate from GSR char

IMU_SAMPLE_RATE_HZ  = 50
COLLECTION_SECONDS  = 10
WEBCAM_FPS          = 30
FEATURE_DIM         = 7

TASK_CONFIG = {
    "open_close_fists": {
        "prompt": "Open and close both fists repeatedly.",
        "duration": 10,
    },
    "hands_to_midline": {
        "prompt": "Bring both hands together to the midline of your body repeatedly.",
        "duration": 12,
    },
    "wrist_rotations": {
        "prompt": "Rotate your wrists alternately in large circles.",
        "duration": 10,
    },
}


# ---------------------------------------------------------------------------
# Sample entropy (movement irregularity)
# ---------------------------------------------------------------------------

def _sample_entropy(x: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
    """
    Compute sample entropy of time series x.
    m = template length, r = tolerance (r_scale * std).
    Lower = more regular (Rett stereotypies are highly regular).
    Higher = more irregular/chaotic (Angelman jerky movements).
    """
    n = len(x)
    if n < 10:
        return 0.0
    r = r_scale * np.std(x)
    if r < 1e-8:
        return 0.0

    def _count_matches(template_len):
        count = 0
        for i in range(n - template_len):
            for j in range(i + 1, n - template_len):
                if np.max(np.abs(x[i:i+template_len] - x[j:j+template_len])) < r:
                    count += 1
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B == 0 or A == 0:
        return 0.0
    return float(-np.log(A / B))


def _fast_sample_entropy(x: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
    """Faster approximate sample entropy using binning."""
    n = len(x)
    if n < 20:
        return 0.0
    r = r_scale * (np.std(x) + 1e-8)
    # Subsample for speed
    if n > 200:
        x = x[::n//200]
        n = len(x)
    return _sample_entropy(x, m, r_scale)


# ---------------------------------------------------------------------------
# Stereotypy detection (midline hand crossing)
# ---------------------------------------------------------------------------

def _compute_stereotypy_score(
    hand_positions: list,   # list of (x, y) normalized coords per frame
    frame_width: int,
    fps: float,
) -> float:
    """
    Detect midline crossing rate — hallmark of Rett hand stereotypies.
    Midline = center 20% of frame width (0.4–0.6 normalized x).

    Returns crossings per minute. Rett patients show continuous midline
    repetitive movements (Dy et al. 2017, Movement Disorders).
    """
    if len(hand_positions) < fps * 2:
        return 0.0

    positions = np.array(hand_positions)
    x_coords = positions[:, 0]  # normalized 0–1

    # Low-pass filter to remove jitter
    from scipy.signal import savgol_filter
    if len(x_coords) > 11:
        x_smooth = savgol_filter(x_coords, window_length=11, polyorder=2)
    else:
        x_smooth = x_coords

    # Count midline crossings (x passes through 0.4–0.6 range)
    midline_lo, midline_hi = 0.4, 0.6
    in_midline = (x_smooth >= midline_lo) & (x_smooth <= midline_hi)

    # Count entries into midline zone
    entries = np.diff(in_midline.astype(int))
    crossing_count = float(np.sum(entries == 1))

    duration_min = len(hand_positions) / fps / 60.0
    return crossing_count / (duration_min + 1e-8)


# ---------------------------------------------------------------------------
# IMU tremor analysis
# ---------------------------------------------------------------------------

def _compute_tremor_features(
    imu_data: np.ndarray,   # (n_samples, 3) — x, y, z accelerometer
    fs: float,
) -> tuple:
    """
    Compute dominant tremor frequency and per-axis acceleration variability.
    Tremor bands: 3–8 Hz (pathological), 8–12 Hz (physiological).

    Returns (tremor_freq_hz, std_x, std_y, std_z)
    """
    if imu_data.shape[0] < fs * 2:
        return 0.0, 0.0, 0.0, 0.0

    # Use magnitude for dominant frequency
    magnitude = np.sqrt(np.sum(imu_data ** 2, axis=1))
    magnitude -= magnitude.mean()   # detrend

    nperseg = min(len(magnitude), int(fs * 4))
    freqs, psd = welch(magnitude, fs=fs, nperseg=nperseg)

    # Focus on tremor band 2–12 Hz
    tremor_mask = (freqs >= 2.0) & (freqs <= 12.0)
    if tremor_mask.sum() > 0:
        tremor_psd = psd[tremor_mask]
        tremor_freqs = freqs[tremor_mask]
        dominant_idx = np.argmax(tremor_psd)
        tremor_freq = float(tremor_freqs[dominant_idx])
    else:
        tremor_freq = 0.0

    std_x = float(np.std(imu_data[:, 0]))
    std_y = float(np.std(imu_data[:, 1]))
    std_z = float(np.std(imu_data[:, 2]))

    return tremor_freq, std_x, std_y, std_z


# ---------------------------------------------------------------------------
# Limb rhythm index
# ---------------------------------------------------------------------------

def _compute_limb_rhythm(trajectory: np.ndarray, fps: float) -> float:
    """
    Measure periodicity of hand trajectory.
    High rhythm + low entropy → Rett stereotypy (highly periodic)
    High rhythm + high entropy → Angelman excitable jerky movements
    Returns 0–1 score (1 = strongly periodic).
    """
    if len(trajectory) < fps * 2:
        return 0.0

    # PSD of trajectory magnitude
    mag = np.sqrt(np.sum(trajectory ** 2, axis=1)) if trajectory.ndim == 2 else trajectory
    mag -= mag.mean()

    nperseg = min(len(mag), int(fps * 4))
    freqs, psd = welch(mag, fs=fps, nperseg=nperseg)

    if psd.sum() < 1e-8:
        return 0.0

    # Ratio of peak power to total power (periodicity index)
    peak_power = psd.max()
    total_power = psd.sum()
    return float(peak_power / total_power)


# ---------------------------------------------------------------------------
# Arduino IMU BLE collection
# ---------------------------------------------------------------------------

async def _collect_imu_ble(duration: float) -> np.ndarray:
    """
    Connect to Arduino NeuroPhenotype peripheral and stream IMU data.
    Arduino sends 3 floats (ax, ay, az) per packet, little-endian.
    Returns (n_samples, 3) array.
    """
    if not BLEAK_AVAILABLE:
        return np.zeros((0, 3))

    imu_buffer = []

    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        devices = await BleakScanner.discover(timeout=8.0, return_adv=True)
        device = None
        for _, (d, _) in devices.items():
            if d.name and d.name.startswith(ARDUINO_DEVICE_NAME):
                device = d
                break

        if not device:
            print(f"[Movement] Arduino not found (attempt {attempt+1}/{MAX_RETRIES}) — IMU features will be zero")
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(3)
            continue

        try:
            async with BleakClient(device.address) as client:
                print(f"[Movement] Arduino connected. Streaming IMU for {duration}s...")

                def on_imu(sender, data):
                    if len(data) >= 12:
                        ax, ay, az = struct.unpack_from("<fff", data)
                        imu_buffer.append([ax, ay, az])

                await client.start_notify(IMU_CHAR_UUID, on_imu)
                await asyncio.sleep(duration)
                await client.stop_notify(IMU_CHAR_UUID)
                break  # done

        except Exception as e:
            print(f"[Movement] Arduino BLE error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"[Movement] Retrying in 3s... ({attempt+1}/{MAX_RETRIES})")
                await asyncio.sleep(3)

    if imu_buffer:
        return np.array(imu_buffer, dtype=np.float32)
    return np.zeros((0, 3))


# ---------------------------------------------------------------------------
# Webcam MediaPipe collection (runs in thread)
# ---------------------------------------------------------------------------

def _collect_mediapipe(duration: float, hand_positions_out: list, pose_positions_out: list, task_prompt: str = ""):
    """
    Run MediaPipe Hands in thread, collecting hand wrist positions.
    Uses solutions API (MediaPipe <0.10) when available, else tasks API (0.10+).
    Appends (x, y) tuples to hand_positions_out.
    """
    if not MEDIAPIPE_AVAILABLE:
        print("[Movement] MediaPipe not available — landmark features will be zero")
        return

    if task_prompt:
        print(f"\n[Movement] {task_prompt}")
        for i in range(3, 0, -1):
            print(f"[Movement] Starting in {i}...")
            time.sleep(1)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Movement] Webcam not available")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    fps = cap.get(cv2.CAP_PROP_FPS) or WEBCAM_FPS
    start_time = time.time()

    if USE_SOLUTIONS_API:
        mp_hands = mp.solutions.hands
        mp_pose  = mp.solutions.pose

        with mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands, mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
        ) as pose:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                hand_result = hands.process(rgb)
                if hand_result.multi_hand_landmarks:
                    for hand_landmarks in hand_result.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[0]
                        hand_positions_out.append((wrist.x, wrist.y))

                pose_result = pose.process(rgb)
                if pose_result.pose_landmarks:
                    lw = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]
                    rw = pose_result.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]
                    pose_positions_out.append(((lw.x + rw.x) / 2, (lw.y + rw.y) / 2))
    else:
        # MediaPipe 0.10+ tasks API
        if not _ensure_hand_model():
            print("[Movement] Hand model unavailable — landmark features will be zero")
            cap.release()
            return

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
        )
        with mp_vision.HandLandmarker.create_from_options(options) as detector:
            while time.time() - start_time < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp_ms = int((time.time() - start_time) * 1000)
                result = detector.detect_for_video(mp_image, timestamp_ms)
                if result.hand_landmarks:
                    for hand_lms in result.hand_landmarks:
                        wrist = hand_lms[0]  # index 0 = wrist
                        hand_positions_out.append((wrist.x, wrist.y))

    cap.release()


# ---------------------------------------------------------------------------
# Modality class
# ---------------------------------------------------------------------------

class MovementModality(BaseModality):
    """
    Simultaneously captures MediaPipe hand landmarks (webcam) and
    IMU accelerometer data (Arduino BLE wrist sensor) to extract
    motor phenotype features.
    """
    FEATURE_DIM = FEATURE_DIM

    def __init__(self, mock_profile: str = None, task: str = "open_close_fists",
                 skip_imu: bool = False, webcam_delay: float = 0.0):
        task_cfg = TASK_CONFIG.get(task, TASK_CONFIG["open_close_fists"])
        self.duration = task_cfg["duration"]
        self._task_prompt = task_cfg["prompt"]
        self.mock_profile = mock_profile
        self.skip_imu = skip_imu
        self.webcam_delay = webcam_delay

    def collect(self) -> dict:
        if self.mock_profile:
            return self._mock_data(self.mock_profile)

        # Optional delay to let the OS release the webcam device (Windows
        # exclusive-lock issue when the browser just freed getUserMedia).
        if self.webcam_delay > 0:
            print(f"[Movement] Waiting {self.webcam_delay}s for device release...")
            time.sleep(self.webcam_delay)

        hand_positions = []
        pose_positions = []

        # Run MediaPipe in background thread, IMU in async
        mp_thread = threading.Thread(
            target=_collect_mediapipe,
            args=(self.duration, hand_positions, pose_positions, self._task_prompt),
            daemon=True,
        )
        mp_thread.start()

        # IMU via BLE (async) — skip when called from web API to avoid
        # 30-second BLE scan hangups on machines without an Arduino.
        imu_data = np.zeros((0, 3))
        if not self.skip_imu:
            try:
                imu_data = asyncio.run(_collect_imu_ble(self.duration))
            except Exception as e:
                print(f"[Movement] IMU collection error: {e}")

        # Wait for MediaPipe thread with a timeout to prevent infinite hang
        timeout = self.duration + 15  # generous buffer
        mp_thread.join(timeout=timeout)
        if mp_thread.is_alive():
            print("[Movement] WARNING: MediaPipe thread timed out")

        print(f"[Movement] Hand positions: {len(hand_positions)}  |  IMU samples: {len(imu_data)}")
        return {
            "hand_positions": hand_positions,
            "pose_positions": pose_positions,
            "imu_data": imu_data,
            "fps": WEBCAM_FPS,
        }

    def preprocess(self, raw_data: dict) -> dict:
        """Smooth trajectories and filter IMU noise."""
        hand_pos = raw_data["hand_positions"]
        imu_data = raw_data["imu_data"]

        # Convert hand positions to numpy
        if len(hand_pos) > 0:
            hand_arr = np.array(hand_pos, dtype=np.float32)
        else:
            hand_arr = np.zeros((0, 2), dtype=np.float32)

        # High-pass filter IMU at 0.5 Hz to remove gravity component
        if len(imu_data) > IMU_SAMPLE_RATE_HZ * 2:
            from scipy.signal import butter, filtfilt
            b, a = butter(2, 0.5 / (IMU_SAMPLE_RATE_HZ / 2), btype="high")
            imu_filtered = filtfilt(b, a, imu_data, axis=0)
        else:
            imu_filtered = imu_data

        return {
            "hand_arr": hand_arr,
            "imu_data": imu_filtered,
            "fps": raw_data["fps"],
        }

    def extract_features(self, processed_data: dict) -> np.ndarray:
        hand_arr = processed_data["hand_arr"]
        imu_data = processed_data["imu_data"]
        fps = processed_data["fps"]

        # Stereotypy score (midline crossings/min)
        if len(hand_arr) > 0:
            stereotypy = _compute_stereotypy_score(
                hand_arr.tolist(), frame_width=640, fps=fps
            )
        else:
            stereotypy = 0.0

        # Tremor features from IMU
        if len(imu_data) > IMU_SAMPLE_RATE_HZ * 2:
            tremor_freq, std_x, std_y, std_z = _compute_tremor_features(
                imu_data, fs=IMU_SAMPLE_RATE_HZ
            )
        else:
            tremor_freq, std_x, std_y, std_z = 0.0, 0.0, 0.0, 0.0

        # Movement irregularity (sample entropy of hand trajectory)
        if len(hand_arr) > 20:
            traj_mag = np.sqrt(np.sum(np.diff(hand_arr, axis=0) ** 2, axis=1))
            irregularity = _fast_sample_entropy(traj_mag)
        else:
            irregularity = 0.0

        # Limb rhythm index
        if len(hand_arr) > int(fps * 2):
            rhythm = _compute_limb_rhythm(hand_arr, fps)
        elif len(imu_data) > IMU_SAMPLE_RATE_HZ * 2:
            rhythm = _compute_limb_rhythm(imu_data, IMU_SAMPLE_RATE_HZ)
        else:
            rhythm = 0.0

        features = np.array(
            [stereotypy, tremor_freq, irregularity, rhythm, std_x, std_y, std_z],
            dtype=np.float32,
        )
        print(f"[Movement] stereotypy={stereotypy:.2f}/min  tremor={tremor_freq:.1f}Hz  "
              f"irregularity={irregularity:.3f}  rhythm={rhythm:.3f}")
        return features

    # ── Mock profiles ────────────────────────────────────────────────────────

    def _mock_data(self, profile: str) -> dict:
        """Generate synthetic movement data for demo/training."""
        rng = np.random.default_rng({"rett": 20, "dravet": 21, "angelman": 22}.get(profile, 0))
        n_imu = COLLECTION_SECONDS * IMU_SAMPLE_RATE_HZ
        t = np.linspace(0, COLLECTION_SECONDS, n_imu)

        if profile == "rett":
            # Highly periodic midline hand stereotypies, low tremor amplitude, high rhythm
            ax = 0.3 * np.sin(2 * np.pi * 1.5 * t) + rng.normal(0, 0.05, n_imu)
            ay = 0.2 * np.sin(2 * np.pi * 1.5 * t + 0.5) + rng.normal(0, 0.05, n_imu)
            az = 0.1 * np.sin(2 * np.pi * 1.5 * t + 1.0) + rng.normal(0, 0.05, n_imu)
            # Hand positions: strong midline oscillation
            n_frames = COLLECTION_SECONDS * WEBCAM_FPS
            t_frames = np.linspace(0, COLLECTION_SECONDS, n_frames)
            hx = 0.5 + 0.15 * np.sin(2 * np.pi * 1.5 * t_frames) + rng.normal(0, 0.02, n_frames)
            hy = 0.5 + 0.05 * np.sin(2 * np.pi * 1.5 * t_frames) + rng.normal(0, 0.02, n_frames)
            hand_positions = list(zip(hx.tolist(), hy.tolist()))

        elif profile == "dravet":
            # Ataxic: irregular, lower amplitude, less periodic
            ax = rng.normal(0, 0.15, n_imu) + 0.1 * np.sin(2 * np.pi * 3.0 * t)
            ay = rng.normal(0, 0.12, n_imu)
            az = rng.normal(0, 0.10, n_imu)
            n_frames = COLLECTION_SECONDS * WEBCAM_FPS
            t_frames = np.linspace(0, COLLECTION_SECONDS, n_frames)
            hx = 0.5 + rng.normal(0, 0.08, n_frames)
            hy = 0.5 + rng.normal(0, 0.06, n_frames)
            hand_positions = list(zip(hx.tolist(), hy.tolist()))

        else:  # angelman
            # Jerky, excitable: high amplitude, somewhat periodic but irregular
            ax = 0.4 * np.sin(2 * np.pi * 2.0 * t) + rng.normal(0, 0.2, n_imu)
            ay = 0.3 * np.sin(2 * np.pi * 2.0 * t + 1.0) + rng.normal(0, 0.2, n_imu)
            az = rng.normal(0, 0.15, n_imu)
            n_frames = COLLECTION_SECONDS * WEBCAM_FPS
            t_frames = np.linspace(0, COLLECTION_SECONDS, n_frames)
            hx = 0.5 + 0.1 * np.sin(2 * np.pi * 2.0 * t_frames) + rng.normal(0, 0.07, n_frames)
            hy = 0.5 + rng.normal(0, 0.07, n_frames)
            hand_positions = list(zip(hx.tolist(), hy.tolist()))

        imu_data = np.stack([ax, ay, az], axis=1).astype(np.float32)
        return {
            "hand_positions": hand_positions,
            "pose_positions": [],
            "imu_data": imu_data,
            "fps": float(WEBCAM_FPS),
        }


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=15)
    parser.add_argument("--mock", choices=["rett", "dravet", "angelman"])
    args = parser.parse_args()

    m = MovementModality(duration=args.duration, mock_profile=args.mock)
    features = m.run()
    labels = ["stereotypy/min", "tremor_hz", "irregularity", "rhythm",
              "imu_std_x", "imu_std_y", "imu_std_z"]
    print("\n── Movement Feature Vector ──")
    for label, val in zip(labels, features):
        print(f"  {label:18s}: {val:.4f}")