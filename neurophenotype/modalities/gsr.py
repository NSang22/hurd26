"""
GSR Modality — Galvanic Skin Response (Sympathetic Reactivity)
Hardware: Arduino Nano 33 BLE Sense + custom GSR circuit
          3.3V → Electrode1 → skin → Electrode2 → 100kΩ → GND → A0

The Arduino streams a single float (voltage at A0) at 50 Hz over BLE.
We convert voltage to conductance (µS), decompose into tonic (SCL)
and phasic (SCR) components, and extract 4 features.

Features: SCL mean (µS), SCL std (µS), SCR count, SCR amplitude mean (µS)

Requirements:
    pip install bleak scipy numpy
"""
import asyncio
import struct
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

try:
    from .base import BaseModality
except ImportError:
    from base import BaseModality

# Arduino BLE identifiers (must match nano_ble_sense.ino)
ARDUINO_DEVICE_NAME = "NeuroPhenotype"
ARDUINO_SERVICE_UUID = "12345678-1234-1234-1234-123456789abc"
GSR_CHAR_UUID = "12345678-1234-1234-1234-123456789abe"

SAMPLE_RATE_HZ = 50       # Arduino sends at 50 Hz
COLLECTION_SECONDS = 30   # collect 60s for stable SCL baseline
VCC = 3.3                 # Arduino supply voltage
R_FIXED = 100_000         # 100 kΩ voltage divider resistor


# ── Signal processing helpers ──────────────────────────────────────────────────

def _voltage_to_conductance_us(voltage: np.ndarray) -> np.ndarray:
    """
    Convert A0 voltage to skin conductance in µS.

    Circuit: 3.3V → R_skin → A0 → R_fixed (100kΩ) → GND
      V_A0 = VCC * R_fixed / (R_skin + R_fixed)
      R_skin = R_fixed * (VCC / V - 1)
      G (S) = 1 / R_skin
      G (µS) = 1e6 / R_skin
    """
    # Guard against divide-by-zero at rail voltages
    voltage = np.clip(voltage, 0.01, VCC - 0.01)
    r_skin = R_FIXED * (VCC / voltage - 1.0)
    return 1e6 / r_skin


def _butter_lowpass(data: np.ndarray, cutoff_hz: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, data)


def _decompose(conductance_us: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Decompose GSR signal into tonic (SCL) and phasic (SCR) components.
      SCL = low-pass at 0.05 Hz  (slow drifting baseline)
      SCR = conductance - SCL    (fast sympathetic responses)
    """
    scl = _butter_lowpass(conductance_us, cutoff_hz=0.05, fs=fs)
    scr = conductance_us - scl
    return scl, scr


# ── BLE scanning ───────────────────────────────────────────────────────────────

async def _scan_for_arduino(timeout: float = 10.0):
    """Scan for the Arduino NeuroPhenotype BLE peripheral."""
    from bleak import BleakScanner

    print(f"[GSR] Scanning for '{ARDUINO_DEVICE_NAME}' BLE device...")
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

    for _, (device, adv_data) in devices.items():
        if device.name and device.name.startswith(ARDUINO_DEVICE_NAME):
            print(f"[GSR] Found: {device.name} ({device.address})")
            return device

    return None


# ── Modality class ─────────────────────────────────────────────────────────────

class GSRModality(BaseModality):
    """
    Streams raw GSR voltage from Arduino over BLE, converts to conductance,
    and extracts tonic/phasic HRV features.
    """
    FEATURE_DIM = 4  # [SCL mean, SCL std, SCR count, SCR amplitude mean]

    def __init__(self, duration: int = COLLECTION_SECONDS):
        self.duration = duration

    def collect(self) -> np.ndarray:
        """Connect to Arduino BLE and collect raw voltage samples."""
        return asyncio.run(self._collect_async())

    def preprocess(self, raw_data: np.ndarray) -> np.ndarray:
        """
        1. Convert voltage → conductance (µS)
        2. Low-pass filter at 5 Hz to remove motion artifacts
        """
        conductance = _voltage_to_conductance_us(raw_data)
        return _butter_lowpass(conductance, cutoff_hz=5.0, fs=SAMPLE_RATE_HZ)

    def extract_features(self, processed_data: np.ndarray) -> np.ndarray:
        """
        Decompose into SCL/SCR and compute 4-element feature vector.
        """
        if len(processed_data) < SAMPLE_RATE_HZ * 10:
            print(f"[GSR] Warning: only {len(processed_data)} samples — need at least 10s for reliable SCL")

        scl, scr = _decompose(processed_data, fs=SAMPLE_RATE_HZ)

        scl_mean = float(np.mean(scl))
        scl_std = float(np.std(scl, ddof=1))

        # SCR peaks: positive deflections ≥ 0.02 µS, min 1s apart
        min_distance = int(SAMPLE_RATE_HZ * 1.0)
        peaks, props = find_peaks(scr, height=0.02, distance=min_distance)
        scr_count = float(len(peaks))
        scr_amp_mean = float(np.mean(props["peak_heights"])) if len(peaks) > 0 else 0.0

        print(f"[GSR] SCL: {scl_mean:.3f} µS (±{scl_std:.3f})  |  SCR peaks: {int(scr_count)}  |  SCR amp: {scr_amp_mean:.4f} µS")
        return np.array([scl_mean, scl_std, scr_count, scr_amp_mean], dtype=np.float32)

    async def _collect_async(self) -> np.ndarray:
        from bleak import BleakClient

        voltage_buffer: list[float] = []

        MAX_RETRIES = 3
        for attempt in range(MAX_RETRIES):
            device = await _scan_for_arduino()
            if not device:
                print(f"[GSR] Arduino not found (attempt {attempt+1}/{MAX_RETRIES})")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(3)
                continue

            try:
                async with BleakClient(device.address) as client:
                    print(f"[GSR] Connected to {device.name}. Collecting for {self.duration}s...")

                    def on_notification(sender, data):
                        # Arduino sends gsrChar.writeValue(gsrVoltage) — 1 float, little-endian
                        voltage = struct.unpack_from("<f", data)[0]
                        voltage_buffer.append(voltage)

                    await client.start_notify(GSR_CHAR_UUID, on_notification)
                    await asyncio.sleep(self.duration)
                    await client.stop_notify(GSR_CHAR_UUID)
                    break  # done

            except Exception as e:
                print(f"[GSR] BLE error: {e}")
                if attempt < MAX_RETRIES - 1:
                    print(f"[GSR] Retrying in 3s... ({attempt+1}/{MAX_RETRIES})")
                    await asyncio.sleep(3)

        samples = np.array(voltage_buffer, dtype=np.float32)
        print(f"[GSR] Collected {len(samples)} samples ({len(samples)/SAMPLE_RATE_HZ:.1f}s at {SAMPLE_RATE_HZ} Hz)")
        return samples


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GSR collection from Arduino via BLE")
    parser.add_argument("--duration", type=int, default=30, help="Collection duration in seconds")
    parser.add_argument("--scan-only", action="store_true", help="List nearby BLE devices and exit")
    args = parser.parse_args()

    if args.scan_only:
        from bleak import BleakScanner

        async def scan():
            print("Scanning for BLE devices (10s)...")
            devices = await BleakScanner.discover(timeout=10.0, return_adv=True)
            for _, (d, adv) in sorted(devices.items(), key=lambda x: x[1][0].name or ""):
                arduino_flag = "  *** ARDUINO ***" if d.name and d.name.startswith(ARDUINO_DEVICE_NAME) else ""
                print(f"  {(d.name or '(unnamed)'):30s}  {d.address}{arduino_flag}")
        asyncio.run(scan())
    else:
        gsr = GSRModality(duration=args.duration)
        features = gsr.run()
        labels = ["SCL mean (µS)", "SCL std (µS)", "SCR count", "SCR amp mean (µS)"]
        print("\n── GSR Feature Vector ──")
        for label, val in zip(labels, features):
            print(f"  {label:20s}: {val:.4f}")
