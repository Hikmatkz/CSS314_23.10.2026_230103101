import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

from salt import get_student_params

def command_output(args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""

def linux_cpu_info():
    info = {}
    try:
        text = Path("/proc/cpuinfo").read_text(errors="ignore")
        blocks = text.split("\n\n")
        for block in blocks:
            if not block.strip():
                continue
            d = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
            if "model name" in d:
                info.setdefault("model", d["model name"].strip())
            if "cpu MHz" in d:
                info.setdefault("frequency_mhz", float(d["cpu MHz"].strip()))
            if "microcode" in d:
                info.setdefault("microcode", d["microcode"].strip())
        return info
    except Exception:
        return info

def linux_cache():
    result = {"L1_data": None, "L2": None, "L3": None, "cache_line_bytes": None}
    base = Path("/sys/devices/system/cpu/cpu0/cache")
    try:
        for idx in base.glob("index*"):
            level = (idx / "level").read_text().strip()
            typ = (idx / "type").read_text().strip()
            size = (idx / "size").read_text().strip()
            line = int((idx / "coherency_line_size").read_text().strip())
            if result["cache_line_bytes"] is None:
                result["cache_line_bytes"] = line
            m = re.match(r"(\d+)([KMG]?)", size, re.I)
            value = int(m.group(1)) * {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3}[m.group(2).upper()]
            if level == "1" and typ.lower() == "data":
                result["L1_data"] = value
            elif level == "2":
                result["L2"] = value
            elif level == "3":
                result["L3"] = value
    except Exception:
        pass
    return result

def mac_info():
    model = command_output(["sysctl", "-n", "machdep.cpu.brand_string"])
    freq = command_output(["sysctl", "-n", "hw.cpufrequency"])
    return {
        "model": model or platform.processor(),
        "frequency_hz": int(freq) if freq.isdigit() else None,
    }

def get_profile():
    logical = os.cpu_count() or 1
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True) or logical
    except ImportError:
        pass

    cpu = {}
    cache = {"L1_data": None, "L2": None, "L3": None, "cache_line_bytes": 64}
    if sys.platform.startswith("linux"):
        cpu = linux_cpu_info()
        cache = linux_cache()
        cache["cache_line_bytes"] = cache["cache_line_bytes"] or 64
    elif sys.platform == "darwin":
        cpu = mac_info()
        cache["cache_line_bytes"] = 64
    else:
        cpu["model"] = platform.processor() or "Unknown (platform did not expose model)"
        cache["cache_line_bytes"] = 64

    if physical is None:
        physical = logical  # fallback estimate; not a guaranteed physical-core count

    return {
        "cpu": {
            "model": cpu.get("model", "Unknown"),
            "microarchitecture": cpu.get("microarchitecture", "Not reliably exposed by standard Python"),
            "base_frequency_mhz": cpu.get("frequency_mhz"),
            "reported_frequency_hz": cpu.get("frequency_hz"),
        },
        "topology": {
            "physical_cores": physical,
            "logical_threads": logical,
            "smt_or_hyperthreading_detected": logical > physical,
            "physical_core_count_is_estimate": physical == logical and "psutil" not in sys.modules,
        },
        "cache": cache,
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "kernel": platform.uname().release,
        },
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable,
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-id", required=True)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1]
    out = base / "results"
    out.mkdir(exist_ok=True)
    profile = get_profile()
    params = get_student_params(args.student_id)
    (out / "host_profile.json").write_text(json.dumps(profile, indent=2))
    (out / "salted_params.json").write_text(json.dumps(params, indent=2))
    print(json.dumps({"host_profile": str(out / "host_profile.json"),
                      "salted_params": str(out / "salted_params.json"),
                      "N": params["N"]}, indent=2))

if __name__ == "__main__":
    main()
