"""Shared helpers for Assignment 3: configuration, build, affinity plan, pinned start and memory measurement.

Reused from Assignment 2 (scripts/common.py), with the affinity mask now depending on the thread count.
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
EXE = ".exe" if os.name == "nt" else ""


def load_config():
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def data_dir(config):
    return Path(config["data_dir"]).expanduser()


def build(config):
    if not shutil.which("gcc"):
        sys.exit("'gcc' not found in PATH")
    BUILD.mkdir(exist_ok=True)
    exe = BUILD / f"parallel{EXE}"
    subprocess.run(["gcc", *config["c_flags"], "-o", str(exe), str(SRC / "parallel.c")], check=True)
    return str(exe)


def affinity_mask(config, method, threads):
    """Logical processors a configuration may use (see config["affinity_plan"])."""
    plan = config["affinity_plan"]
    if method == "seq":
        return plan["sequential_cpu"]
    one, siblings = plan["p_cores_one_thread"], plan["p_cores_siblings"]
    if threads <= len(one):
        return one[:threads]
    if threads <= len(one) + len(siblings):
        return one + siblings[:threads - len(one)]
    return list(range(psutil.cpu_count()))


def start_with_affinity(args, mask, config):
    """Start a process whose affinity mask is `mask` from its first instruction: Windows children inherit the
    parent's mask, so the harness takes the mask while creating the child and then restores its own."""
    me = psutil.Process()
    original = me.cpu_affinity()
    flags = subprocess.HIGH_PRIORITY_CLASS if config.get("priority") == "high" and sys.platform == "win32" else 0
    me.cpu_affinity(mask)
    try:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags,
                                env={**os.environ, "OMP_DYNAMIC": "false"})
    finally:
        me.cpu_affinity(original)


class CorePerformanceCounter:
    """Effective speed of one logical processor (same as Assignment 2): the Windows counter
    '\\Processor Information(0,<cpu>)\\% Processor Performance' (100% = base frequency), read with
    PdhAddEnglishCounterW so that it also works on localised systems."""

    class _Value(ctypes.Structure):
        _fields_ = [("CStatus", ctypes.c_ulong), ("padding", ctypes.c_ulong), ("doubleValue", ctypes.c_double)]

    def __init__(self, cpu):
        self.ok = os.name == "nt"
        if not self.ok:
            return
        self.pdh = ctypes.WinDLL("pdh")
        self.query, self.counters = ctypes.c_void_p(), {}
        self.ok = self.pdh.PdhOpenQueryW(None, None, ctypes.byref(self.query)) == 0
        for name, path in (("perf", f"\\Processor Information(0,{cpu})\\% Processor Performance"),
                           ("base_mhz", f"\\Processor Information(0,{cpu})\\Processor Frequency")):
            handle = ctypes.c_void_p()
            if self.ok and self.pdh.PdhAddEnglishCounterW(self.query, path, None, ctypes.byref(handle)) == 0:
                self.counters[name] = handle
        self.ok = self.ok and len(self.counters) == 2
        if self.ok:
            self.pdh.PdhCollectQueryData(self.query)

    def sample(self):
        """Effective clock in MHz since the previous sample, or None."""
        if not self.ok or self.pdh.PdhCollectQueryData(self.query) != 0:
            return None
        values = []
        for name in ("perf", "base_mhz"):
            v = self._Value()
            if self.pdh.PdhGetFormattedCounterValue(self.counters[name], 0x200, None, ctypes.byref(v)) != 0:
                return None
            values.append(v.doubleValue)
        return values[0] * values[1] / 100


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_mb(proc):
    """Peak working set of a finished subprocess.Popen in MiB (Win32 GetProcessMemoryInfo)."""
    if os.name != "nt":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize / 2**20
