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
        return subprocess.check_output(
            args,
            text=True,
            stderr=subprocess.DEVNULL
        ).strip()
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

            d = dict(
                line.split(":", 1)
                for line in block.splitlines()
                if ":" in line
            )

            if "model name" in d:
                info.setdefault("model", d["model name"].strip())

            if "cpu MHz" in d:
                info.setdefault(
                    "frequency_mhz",
                    float(d["cpu MHz"].strip())
                )

            if "microcode" in d:
                info.setdefault("microcode", d["microcode"].strip())

        # Try to obtain CPU family/model/stepping metadata.
        for key in ("cpu family", "model", "stepping"):
            pass

        return info

    except Exception:
        return info


def linux_cache():
    result = {
        "L1_data": None,
        "L2": None,
        "L3": None,
        "cache_line_bytes": None
    }

    base = Path("/sys/devices/system/cpu/cpu0/cache")

    try:
        for idx in base.glob("index*"):
            level = (idx / "level").read_text().strip()
            typ = (idx / "type").read_text().strip()
            size = (idx / "size").read_text().strip()

            line = int(
                (idx / "coherency_line_size").read_text().strip()
            )

            if result["cache_line_bytes"] is None:
                result["cache_line_bytes"] = line

            match = re.match(r"(\d+)([KMG]?)", size, re.I)

            if not match:
                continue

            units = {
                "": 1,
                "K": 1024,
                "M": 1024 ** 2,
                "G": 1024 ** 3
            }

            value = (
                int(match.group(1))
                * units[match.group(2).upper()]
            )

            if level == "1" and typ.lower() == "data":
                result["L1_data"] = value

            elif level == "2":
                result["L2"] = value

            elif level == "3":
                result["L3"] = value

    except (OSError, ValueError):
        pass

    return result


def mac_info():
    model = command_output([
        "sysctl", "-n", "machdep.cpu.brand_string"
    ])

    freq = command_output([
        "sysctl", "-n", "hw.cpufrequency"
    ])

    return {
        "model": model or platform.processor() or "Unknown",
        "frequency_hz": int(freq) if freq.isdigit() else None
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
    cache = {
        "L1_data": None,
        "L2": None,
        "L3": None,
        "cache_line_bytes": 64
    }

    if sys.platform.startswith("linux"):
        cpu = linux_cpu_info()
        cache = linux_cache()
        cache["cache_line_bytes"] = (
            cache["cache_line_bytes"] or 64
        )

    elif sys.platform == "darwin":
        cpu = mac_info()
        cache["cache_line_bytes"] = 64

    else:
        cpu["model"] = (
            platform.processor()
            or "Unknown"
        )
        cache["cache_line_bytes"] = 64

    if physical is None:
        physical = logical

    # Required top-level fields for the verifier.
    profile_data = {
        "cpu_model": cpu.get("model", "Unknown"),

        "microarchitecture": cpu.get(
            "microarchitecture",
            "Not reliably exposed by standard Python"
        ),

        "base_frequency_mhz": cpu.get("frequency_mhz"),

        "physical_cores": physical,
        "logical_threads": logical,

        "smt_or_hyperthreading_detected": (
            logical > physical
        ),

        "L1_data_cache_bytes": cache.get("L1_data"),
        "L2_cache_bytes": cache.get("L2"),
        "L3_cache_bytes": cache.get("L3"),

        "cache_line_bytes": cache.get(
            "cache_line_bytes", 64
        ),

        "os_system": platform.system(),
        "os_release": platform.release(),
        "kernel_version": platform.uname().release,

        "python_implementation": (
            platform.python_implementation()
        ),

        "python_version": platform.python_version(),
        "python_executable": sys.executable,

        # Preserve nested fields for compatibility.
        "cpu": {
            "model": cpu.get("model", "Unknown"),
            "microarchitecture": cpu.get(
                "microarchitecture",
                "Not reliably exposed by standard Python"
            ),
            "base_frequency_mhz": cpu.get("frequency_mhz"),
            "reported_frequency_hz": cpu.get("frequency_hz")
        },

        "topology": {
            "physical_cores": physical,
            "logical_threads": logical,
            "smt_or_hyperthreading_detected": (
                logical > physical
            ),
            "physical_core_count_is_estimate": (
                "psutil" not in sys.modules
            )
        },

        "cache": cache,

        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "kernel": platform.uname().release
        },

        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": sys.executable
        }
    }

    return profile_data


def main():
    parser = argparse.ArgumentParser(
        description="CS482 host profiling and student parameter generation"
    )

    parser.add_argument(
        "--student-id",
        required=True,
        help="Your 8-digit Student ID"
    )

    args = parser.parse_args()

    base = Path(__file__).resolve().parents[1]
    output_dir = base / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect host information.
    host_profile = get_profile()

    # Generate salted parameters.
    params = get_student_params(args.student_id)

    # Save results.
    host_file = output_dir / "host_profile.json"
    params_file = output_dir / "salted_params.json"

    host_file.write_text(
        json.dumps(host_profile, indent=2)
    )

    params_file.write_text(
        json.dumps(params, indent=2)
    )

    print("Host profile saved:", host_file)
    print("Salted parameters saved:", params_file)
    print("CPU model:", host_profile["cpu_model"])
    print("Physical cores:", host_profile["physical_cores"])
    print("Logical threads:", host_profile["logical_threads"])
    print("Matrix dimension N:", params["N"])


if __name__ == "__main__":
    main()
