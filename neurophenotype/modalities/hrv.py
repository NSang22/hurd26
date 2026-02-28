"""
HRV Modality — Autonomic Nervous System
Hardware: Apple Watch via HeartCast app (BLE Heart Rate Profile, UUID 0x180D)

Scans for any BLE device advertising the Heart Rate Service UUID, connects,
and streams RR intervals from the 0x2A37 characteristic to compute HRV features.

Features: SDNN, RMSSD, pNN50, LF/HF ratio (sympathovagal balance)

Requirements:
    pip install bleak scipy numpy
"""
import asyncio
import struct
import numpy as np
from scipy.signal import welch
from scipy.interpolate import interp1d
from bleak import BleakScanner, BleakClient

try:
    from .base import BaseModality  # imported as part of package
except ImportError:
    from base import BaseModality   # run directly: python hrv.py

# Standard BLE GATT Heart Rate Profile
HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# How long to collect RR intervals (seconds). 60s gives good LF/HF resolution.
COLLECTION_SECONDS = 60


# ── Parsing ────────────────────────────────────────────────────────────────────

def _parse_hr_measurement(data: bytearray) -> tuple[int, list[float]]:
    """
    Parse BLE Heart Rate Measurement characteristic (0x2A37).

    Byte 0: flags
      bit 0  — HR format: 0=UINT8, 1=UINT16
      bit 4  — RR intervals present

    RR intervals are UINT16 in units of 1/1024 seconds → convert to ms.
    Returns (heart_rate_bpm, [rr_interval_ms, ...])
    """
    flags = data[0]
    hr_format_16bit = flags & 0x01
    rr_present = (flags >> 4) & 0x01

    offset = 1
    if hr_format_16bit:
        hr = struct.unpack_from("<H", data, offset)[0]
        offset += 2
    else:
        hr = data[offset]
        offset += 1

    rr_intervals_ms = []
    if rr_present:
        while offset + 1 < len(data):
            rr_raw = struct.unpack_from("<H", data, offset)[0]
            rr_ms = (rr_raw / 1024.0) * 1000.0
            rr_intervals_ms.append(rr_ms)
            offset += 2

    return hr, rr_intervals_ms


# ── Feature computation ────────────────────────────────────────────────────────

def _compute_hrv_features(rr_ms: list[float]) -> np.ndarray:
    """
    Compute time-domain and frequency-domain HRV features.
    Returns [SDNN, RMSSD, pNN50, LF_HF_ratio]
    """
    rr = np.array(rr_ms)

    # Time domain
    sdnn = float(np.std(rr, ddof=1))
    successive_diffs = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
    pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)

    # Frequency domain — resample to 4 Hz tachogram, then Welch PSD
    cumtime = np.cumsum(rr) / 1000.0  # seconds
    duration = cumtime[-1]
    fs = 4.0
    t_uniform = np.arange(0, duration, 1.0 / fs)

    if len(rr) >= 4 and duration > 20:
        interp = interp1d(cumtime, rr, kind="cubic", bounds_error=False, fill_value="extrapolate")
        rr_uniform = interp(t_uniform)
        freqs, psd = welch(rr_uniform, fs=fs, nperseg=min(256, len(rr_uniform)))
        lf_mask = (freqs >= 0.04) & (freqs < 0.15)
        hf_mask = (freqs >= 0.15) & (freqs < 0.40)
        lf_power = np.trapz(psd[lf_mask], freqs[lf_mask])
        hf_power = np.trapz(psd[hf_mask], freqs[hf_mask])
        lf_hf = float(lf_power / hf_power) if hf_power > 0 else 0.0
    else:
        lf_hf = 0.0

    return np.array([sdnn, rmssd, pnn50, lf_hf], dtype=np.float32)


# ── BLE scanning ───────────────────────────────────────────────────────────────

async def _scan_for_hr_device(timeout: float = 10.0):
    """
    Scan for a BLE device advertising the Heart Rate Service UUID.
    Falls back to name matching if UUID not in advertisement data.
    """
    print("Scanning for BLE Heart Rate devices...")
    devices = await BleakScanner.discover(timeout=timeout, return_adv=True)

    # Primary: match by advertised service UUID (most reliable)
    for _, (device, adv_data) in devices.items():
        if HR_SERVICE_UUID in [u.lower() for u in adv_data.service_uuids]:
            print(f"Found HR device: {device.name} ({device.address})")
            return device

    # Fallback: match by device name keywords
    for _, (device, adv_data) in devices.items():
        if device.name and any(kw in device.name.lower() for kw in ["heart", "watch", "apple", "hrm"]):
            print(f"Found likely HR device by name: {device.name} ({device.address})")
            return device

    return None


# ── Modality class ─────────────────────────────────────────────────────────────

class HRVModality(BaseModality):
    """
    Connects to HeartCast BLE peripheral, collects RR intervals for
    COLLECTION_SECONDS, then computes HRV feature vector.
    """
    FEATURE_DIM = 4  # [SDNN, RMSSD, pNN50, LF/HF]

    def __init__(self, duration: int = COLLECTION_SECONDS):
        self.duration = duration

    def collect(self) -> list[float]:
        """Scan for HR BLE device and stream RR intervals."""
        return asyncio.run(self._collect_async())

    def preprocess(self, raw_data: list[float]) -> list[float]:
        """
        Filter physiologically implausible RR intervals (300–2000 ms)
        and remove ectopic beats via moving median filter (>20% deviation).
        """
        rr = np.array(raw_data)
        rr = rr[(rr >= 300) & (rr <= 2000)]

        if len(rr) < 4:
            return list(rr)

        window = 5
        filtered = []
        for i, val in enumerate(rr):
            lo = max(0, i - window // 2)
            hi = min(len(rr), i + window // 2 + 1)
            median = np.median(rr[lo:hi])
            if abs(val - median) / median < 0.20:
                filtered.append(val)

        return filtered

    def extract_features(self, processed_data: list[float]) -> np.ndarray:
        if len(processed_data) < 10:
            print(f"[HRV] Warning: only {len(processed_data)} clean RR intervals — features may be unreliable")
        return _compute_hrv_features(processed_data)

    async def _collect_async(self) -> list[float]:
        rr_buffer: list[float] = []

        while True:
            device = await _scan_for_hr_device()
            if not device:
                print("No HR device found. Retrying in 5s...")
                await asyncio.sleep(5)
                continue

            try:
                async with BleakClient(device.address) as client:
                    print(f"[HRV] Connected to {device.name}. Collecting for {self.duration}s...")

                    def on_notification(sender, data):
                        hr, rr_list = _parse_hr_measurement(data)
                        rr_buffer.extend(rr_list)
                        print(f"Apple Watch HR: {hr} BPM  (+{len(rr_list)} RR intervals)")

                    await client.start_notify(HR_MEASUREMENT_UUID, on_notification)
                    await asyncio.sleep(self.duration)
                    await client.stop_notify(HR_MEASUREMENT_UUID)
                    break  # collection complete — exit retry loop

            except Exception as e:
                print(f"[HRV] BLE connection error: {e}")
                print("[HRV] Reconnecting in 3s...")
                await asyncio.sleep(3)

        print(f"[HRV] Collected {len(rr_buffer)} RR intervals")
        return rr_buffer


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test HRV collection from Apple Watch via BLE")
    parser.add_argument("--duration", type=int, default=30, help="Collection duration in seconds")
    parser.add_argument("--scan-only", action="store_true", help="List nearby BLE devices and exit")
    args = parser.parse_args()

    if args.scan_only:
        async def scan():
            print("Scanning for BLE devices (10s)...")
            devices = await BleakScanner.discover(timeout=10.0, return_adv=True)
            for _, (d, adv) in sorted(devices.items(), key=lambda x: x[1][0].name or ""):
                hr_flag = "  *** HR SERVICE ***" if HR_SERVICE_UUID in [u.lower() for u in adv.service_uuids] else ""
                print(f"  {(d.name or '(unnamed)'):30s}  {d.address}{hr_flag}")
        asyncio.run(scan())
    else:
        hrv = HRVModality(duration=args.duration)
        features = hrv.run()
        labels = ["SDNN", "RMSSD", "pNN50", "LF/HF"]
        print("\n── HRV Feature Vector ──")
        for label, val in zip(labels, features):
            print(f"  {label:8s}: {val:.4f}")
