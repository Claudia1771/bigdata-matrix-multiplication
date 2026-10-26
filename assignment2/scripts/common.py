"""Shared helpers: configuration, building, CSR file I/O, pinned process start and memory measurement.

Reused from Assignment 1 (scripts/common.py and run_experiment.py) with the paths and programs of Assignment 2.
"""
import ctypes
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import psutil

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
RESULTS = ROOT / "results"
EXE = ".exe" if os.name == "nt" else ""
STRUCTURES = {"uniform": 0, "banded": 1}


def load_config():
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def data_dir(config):
    """Input matrices are kept outside the repository (and outside cloud-synchronised folders)."""
    path = Path(config["data_dir"]).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def build(config):
    """Compile both C programs with the flags of config.json. Returns {"dense": exe, "sparse": exe}."""
    if not shutil.which("gcc"):
        sys.exit("'gcc' not found in PATH")
    BUILD.mkdir(exist_ok=True)
    programs = {}
    for name in ("dense", "sparse"):
        exe = BUILD / f"{name}{EXE}"
        subprocess.run(["gcc", *config["c_flags"], "-o", str(exe), str(SRC / f"{name}.c")], check=True)
        programs[name] = str(exe)
    return programs


# --- Inputs ---------------------------------------------------------------------------------------

def dense_paths(config, n):
    d = data_dir(config)
    return d / f"A_{n}.bin", d / f"B_{n}.bin"


def sparse_name(n, structure, density_pct, matrix):
    return f"{matrix}_{structure}_n{n}_d{density_pct:g}.csr"


def sparse_paths(config, n, structure, density_pct):
    d = data_dir(config)
    return tuple(d / sparse_name(n, structure, density_pct, m) for m in ("A", "B"))


def write_csr(path, n, row_ptr, col, val):
    with open(path, "wb") as f:
        np.array([n, len(col)], dtype="<i8").tofile(f)
        np.asarray(row_ptr, dtype="<i8").tofile(f)
        np.asarray(col, dtype="<i4").tofile(f)
        np.asarray(val, dtype="<f8").tofile(f)


def read_csr(path):
    """Returns (n, row_ptr, col, val) as NumPy arrays."""
    with open(path, "rb") as f:
        n, nnz = np.fromfile(f, dtype="<i8", count=2)
        row_ptr = np.fromfile(f, dtype="<i8", count=n + 1)
        col = np.fromfile(f, dtype="<i4", count=nnz)
        val = np.fromfile(f, dtype="<f8", count=nnz)
    return int(n), row_ptr, col, val


# --- Pinned process start and peak memory (same method as Assignment 1) ----------------------------

def start_pinned(args, config):
    """Start a process already pinned to config["cpu_affinity"] with the configured priority.
    Windows child processes inherit the affinity of their parent, so the harness pins itself while
    creating the child and then restores its own affinity."""
    me = psutil.Process()
    original = me.cpu_affinity()
    flags = subprocess.HIGH_PRIORITY_CLASS if config.get("priority") == "high" and sys.platform == "win32" else 0
    me.cpu_affinity(config["cpu_affinity"])
    try:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=flags)
    finally:
        me.cpu_affinity(original)


class CorePerformanceCounter:
    """Effective speed of one logical processor, from the Windows performance counter
    '\\Processor Information(0,<cpu>)\\% Processor Performance' (100% = nominal base frequency; values above 100%
    mean turbo). The English counter name is used through PdhAddEnglishCounterW, so it works on localised systems."""

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
        """Returns (performance %, base MHz) since the previous sample, or None."""
        if not self.ok or self.pdh.PdhCollectQueryData(self.query) != 0:
            return None
        values = []
        for name in ("perf", "base_mhz"):
            v = self._Value()
            if self.pdh.PdhGetFormattedCounterValue(self.counters[name], 0x200, None, ctypes.byref(v)) != 0:
                return None
            values.append(v.doubleValue)
        return tuple(values)


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
