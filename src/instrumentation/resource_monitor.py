"""
Background sampler for peak RAM (process RSS) and peak VRAM (GPU 0 used memory).
Used in Phase 3 to report resource footprint per embedding model.
"""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil


def _gpu_used_mib(gpu_index: int = 0) -> Optional[float]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                "-i",
                str(gpu_index),
            ],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return float(out.decode().strip().splitlines()[0])
    except Exception:
        return None


@dataclass
class ResourceSample:
    peak_rss_mib: float = 0.0
    peak_vram_mib: float = 0.0
    start_vram_mib: float = 0.0
    samples: int = 0
    gpu_available: bool = False


class ResourceMonitor:
    """Polls RSS (psutil) and VRAM (nvidia-smi) every `interval_s` and tracks peak."""

    def __init__(self, interval_s: float = 0.5, gpu_index: int = 0):
        self.interval_s = interval_s
        self.gpu_index = gpu_index
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.stats = ResourceSample()
        self._proc = psutil.Process(os.getpid())

    def _loop(self) -> None:
        vram0 = _gpu_used_mib(self.gpu_index)
        if vram0 is not None:
            self.stats.gpu_available = True
            self.stats.start_vram_mib = vram0
            self.stats.peak_vram_mib = vram0
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss / (1024 * 1024)
                if rss > self.stats.peak_rss_mib:
                    self.stats.peak_rss_mib = rss
                if self.stats.gpu_available:
                    v = _gpu_used_mib(self.gpu_index)
                    if v is not None and v > self.stats.peak_vram_mib:
                        self.stats.peak_vram_mib = v
                self.stats.samples += 1
            except Exception:
                pass
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "ResourceMonitor":
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def summary(self) -> dict:
        return {
            "peak_rss_mib": round(self.stats.peak_rss_mib, 1),
            "peak_vram_mib": round(self.stats.peak_vram_mib, 1),
            "start_vram_mib": round(self.stats.start_vram_mib, 1),
            "delta_vram_mib": round(self.stats.peak_vram_mib - self.stats.start_vram_mib, 1),
            "samples": self.stats.samples,
            "gpu_available": self.stats.gpu_available,
        }
