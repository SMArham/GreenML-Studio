"""First-principles dual-layer energy & CO2 telemetry calibrated to CodeCarbon."""

from __future__ import annotations

import json
import os
import time
import tracemalloc
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import psutil
from codecarbon import OfflineEmissionsTracker

from src.config import RESULTS_DIR


@dataclass
class HardwareProfile:
    os_name: str
    cpu_model: str
    cpu_count: int
    cpu_tdp_watts: float
    ram_power_watts: float
    ram_total_gb: float
    country_iso_code: str
    carbon_intensity_kg_per_kwh: float
    tracking_mode: str
    calibrated_at: str


@dataclass
class RunMeasurement:
    wall_time_s: float
    cpu_time_s: float
    cpu_util_ratio: float
    cpu_power_w: float
    cpu_energy_j: float
    total_energy_j: float
    cpu_energy_kwh: float
    total_energy_kwh: float
    co2_grams_cpu: float
    co2_grams_total: float
    peak_mem_kb: float


class EnergyProfiler:
    """High-resolution CPU energy, runtime, memory, and CO2 profiler.

    Uses CodeCarbon to discover hardware TDP, grid carbon intensity (PAK), and RAM
    power model, and wraps long sessions directly with OfflineEmissionsTracker while
    providing nanosecond process_time / perf_counter energy telemetry for sub-second runs.
    """

    def __init__(self, country_iso_code: str = "PAK", profile_cache: Optional[Path] = None) -> None:
        self.country_iso_code = country_iso_code
        self.profile_cache = profile_cache or (RESULTS_DIR / "hardware_profile.json")
        self.profile = self._load_or_calibrate()

    def _load_or_calibrate(self) -> HardwareProfile:
        if self.profile_cache.exists():
            try:
                data = json.loads(self.profile_cache.read_text(encoding="utf-8"))
                return HardwareProfile(**data)
            except Exception:
                pass

        tracker = OfflineEmissionsTracker(
            project_name="greenml_calibration",
            country_iso_code=self.country_iso_code,
            measure_power_secs=0.05,
            save_to_file=False,
            log_level="error",
        )
        tracker.start()
        # Brief deterministic CPU work (~0.15s) to populate CodeCarbon power telemetry
        acc = 0.0
        t_end = time.perf_counter() + 0.15
        while time.perf_counter() < t_end:
            acc += 1.0001 * 0.9999
        _ = acc
        emissions_kg = tracker.stop()
        fd = tracker.final_emissions_data

        energy_kwh = float(getattr(fd, "energy_consumed", 0.0) or 0.0)
        if energy_kwh > 0 and emissions_kg is not None and float(emissions_kg) > 0:
            carbon_intensity = float(emissions_kg) / energy_kwh
        else:
            carbon_intensity = 0.4406  # Fallback PAK grid intensity kg CO2/kWh

        # Infer TDP cleanly from CodeCarbon CPU model / detected hardware
        cpu_model = str(getattr(fd, "cpu_model", "Unknown CPU"))
        cpu_count = int(getattr(fd, "cpu_count", os.cpu_count() or 4))
        ram_power_w = float(getattr(fd, "ram_power", 20.0) or 20.0)
        ram_total_gb = float(getattr(fd, "ram_total_size", 16.0) or 16.0)
        os_name = str(getattr(fd, "os", "Windows"))

        # Check CodeCarbon internal CPU TDP if exposed, else match common Intel TDP tiers
        tdp_w = 15.0
        try:
            for hw in getattr(tracker, "_hardware", []):
                if hasattr(hw, "_tdp") and hw._tdp:
                    tdp_w = float(hw._tdp)
                    break
                if hasattr(hw, "tdp") and hw.tdp:
                    tdp_w = float(hw.tdp)
                    break
        except Exception:
            pass
        if tdp_w <= 0:
            tdp_w = 15.0 if "U CPU" in cpu_model else 65.0

        tracking_mode = (
            "CodeCarbon cpu_load (TDP * effective CPU utilization) + nanosecond process_time"
            if "Windows" in os_name
            else "Intel RAPL / CodeCarbon hardware telemetry"
        )

        profile = HardwareProfile(
            os_name=os_name,
            cpu_model=cpu_model,
            cpu_count=cpu_count,
            cpu_tdp_watts=float(tdp_w),
            ram_power_watts=float(ram_power_w),
            ram_total_gb=round(ram_total_gb, 2),
            country_iso_code=self.country_iso_code,
            carbon_intensity_kg_per_kwh=round(carbon_intensity, 6),
            tracking_mode=tracking_mode,
            calibrated_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        self.profile_cache.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
        return profile

    def measure_callable(self, fn: Callable[[], Any], track_memory: bool = False) -> Tuple[Any, RunMeasurement]:
        """Execute fn() and return (result, RunMeasurement) with nanosecond resolution."""
        if track_memory:
            tracemalloc.start()

        wall_0 = time.perf_counter_ns()
        cpu_0 = time.process_time_ns()
        result = fn()
        cpu_1 = time.process_time_ns()
        wall_1 = time.perf_counter_ns()

        peak_mem_kb = 0.0
        if track_memory:
            _, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mem_kb = peak_bytes / 1024.0

        wall_s = max((wall_1 - wall_0) / 1e9, 1e-9)
        cpu_s = max((cpu_1 - cpu_0) / 1e9, 1e-9)

        # Effective process CPU utilization across logical cores
        util_ratio = min(1.0, max(0.0, cpu_s / (wall_s * max(1, self.profile.cpu_count))))
        # CodeCarbon's cpu_load model allocates 10% static package baseline + 90% dynamic load
        cpu_power_w = self.profile.cpu_tdp_watts * (0.10 + 0.90 * util_ratio)
        cpu_energy_j = cpu_power_w * wall_s
        total_energy_j = (cpu_power_w + self.profile.ram_power_watts) * wall_s

        cpu_kwh = cpu_energy_j / 3_600_000.0
        total_kwh = total_energy_j / 3_600_000.0
        ci = self.profile.carbon_intensity_kg_per_kwh

        meas = RunMeasurement(
            wall_time_s=wall_s,
            cpu_time_s=cpu_s,
            cpu_util_ratio=util_ratio,
            cpu_power_w=cpu_power_w,
            cpu_energy_j=cpu_energy_j,
            total_energy_j=total_energy_j,
            cpu_energy_kwh=cpu_kwh,
            total_energy_kwh=total_kwh,
            co2_grams_cpu=cpu_kwh * ci * 1000.0,
            co2_grams_total=total_kwh * ci * 1000.0,
            peak_mem_kb=peak_mem_kb,
        )
        return result, meas

    @contextmanager
    def track_codecarbon_session(self, session_name: str):
        """Wrap an experiment stage in a genuine CodeCarbon OfflineEmissionsTracker."""
        out_dir = RESULTS_DIR / "codecarbon_logs"
        out_dir.mkdir(parents=True, exist_ok=True)
        tracker = OfflineEmissionsTracker(
            project_name=session_name,
            country_iso_code=self.country_iso_code,
            measure_power_secs=0.5,
            output_dir=str(out_dir),
            output_file=f"emissions_{session_name}.csv",
            save_to_file=True,
            log_level="error",
        )
        session_info: Dict[str, Any] = {"session_name": session_name}
        t0 = time.perf_counter()
        tracker.start()
        try:
            yield session_info
        finally:
            emissions_kg = tracker.stop()
            dt = time.perf_counter() - t0
            fd = tracker.final_emissions_data
            session_info.update({
                "duration_s": dt,
                "codecarbon_energy_kwh": float(getattr(fd, "energy_consumed", 0.0) or 0.0),
                "codecarbon_cpu_energy_kwh": float(getattr(fd, "cpu_energy", 0.0) or 0.0),
                "codecarbon_ram_energy_kwh": float(getattr(fd, "ram_energy", 0.0) or 0.0),
                "codecarbon_co2_grams": float(emissions_kg or 0.0) * 1000.0,
                "cpu_tracking_mode": "CodeCarbon OfflineEmissionsTracker",
            })
