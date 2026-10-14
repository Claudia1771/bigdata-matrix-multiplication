"""Shared helpers: configuration, building the three implementations and process memory measurement."""
import ctypes
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
BUILD = ROOT / "build"
DATA = ROOT / "data"
RESULTS = ROOT / "results"
LANGUAGES = ["Python", "Java", "C"]


def load_config():
    with open(ROOT / "config.json", encoding="utf-8") as f:
        return json.load(f)


def java_executable():
    """Path of the real java.exe. On Windows, the `java` found in PATH is often a small launcher
    that starts the JVM as a child process, which would hide the JVM memory from the measurement."""
    out = subprocess.run(["java", "-XshowSettings:properties", "-version"],
                         capture_output=True, text=True).stderr
    java_home = re.search(r"java\.home = (.+)", out).group(1).strip()
    exe = Path(java_home) / "bin" / ("java.exe" if os.name == "nt" else "java")
    return str(exe)


def build(config):
    """Compile the Java and C programs and return {language: base command}."""
    BUILD.mkdir(exist_ok=True)
    for tool in ("gcc", "javac", "java"):
        if not shutil.which(tool):
            sys.exit(f"'{tool}' not found in PATH")
    exe = BUILD / ("matmul.exe" if os.name == "nt" else "matmul")
    subprocess.run(["gcc", *config["c_flags"], "-o", str(exe), str(SRC / "c" / "matmul.c")], check=True)
    subprocess.run(["javac", "-d", str(BUILD), str(SRC / "java" / "MatMul.java")], check=True)
    return {
        "Python": [sys.executable, str(SRC / "python" / "matmul.py")],
        "Java": [java_executable(), *config["java_flags"], "-cp", str(BUILD), "MatMul"],
        "C": [str(exe)],
    }


# --- Peak working set (peak resident memory) of a child process, read through the Win32 API ---

class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def peak_working_set_mb(proc):
    """Peak working set of a subprocess.Popen in MiB. Works after the process has exited, because
    Popen keeps the process handle open. Returns None on non-Windows systems."""
    if os.name != "nt":
        return None
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    psapi = ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(_ProcessMemoryCounters), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(int(proc._handle), ctypes.byref(counters), counters.cb):
        return None
    return counters.PeakWorkingSetSize / 2**20
