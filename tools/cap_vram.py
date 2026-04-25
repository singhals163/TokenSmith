"""Import this before loading any torch model to cap VRAM to ~8 GiB.
Controlled by env var VRAM_CAP_GIB; default 8.0. Set VRAM_CAP_GIB=0 to disable.
"""
import os, torch


def apply():
    cap = float(os.environ.get("VRAM_CAP_GIB", "8"))
    if cap <= 0 or not torch.cuda.is_available():
        return
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    frac = min(1.0, cap / total_gib)
    torch.cuda.set_per_process_memory_fraction(frac, device=0)
    print(f"[cap_vram] fraction={frac:.3f} target={cap:.1f}GiB / total={total_gib:.1f}GiB")


if os.environ.get("VRAM_CAP_GIB_AUTO", "0") == "1":
    apply()
