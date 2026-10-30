import json
import math
import multiprocessing as mp
import os
import statistics
import time
from pathlib import Path
import numpy as np

def stencil_chunk(chunk):
    # Five-point stencil: north + south + west + east + center, then scale.
    return (chunk[0:-2, 1:-1] + chunk[2:, 1:-1] +
            chunk[1:-1, 0:-2] + chunk[1:-1, 2:] +
            chunk[1:-1, 1:-1]) * 0.2

def split_with_halo(a, p):
    inner_rows = a.shape[0] - 2
    boundaries = np.linspace(0, inner_rows, p + 1, dtype=int)
    chunks = []
    for i in range(p):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end <= start:
            continue
        # Include one halo row above/below; keep all columns incl. horizontal halos.
        chunks.append(a[start:start + (end - start) + 2, :])
    return chunks

def run_parallel(a, p):
    chunks = split_with_halo(a, p)
    if p == 1:
        return stencil_chunk(chunks[0])
    with mp.Pool(processes=p) as pool:
        outputs = pool.map(stencil_chunk, chunks)
    return np.concatenate(outputs, axis=0)

def median_run(n, p, seed, repeats=3):
    rng = np.random.default_rng(seed)
    a = rng.random((n + 2, n + 2), dtype=np.float64)
    samples = []
    result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = run_parallel(a, p)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result

def main():
    base = Path(__file__).resolve().parents[1]
    params = json.loads((base / "results" / "salted_params.json").read_text())
    profile = json.loads((base / "results" / "host_profile.json").read_text())
    n = params["N"]
    physical = int(profile["topology"]["physical_cores"])
    # Test powers of two up to available physical cores, capped at 8 to keep lab time bounded.
    p_values = [1]
    p = 2
    while p <= min(physical, 8):
        p_values.append(p)
        p *= 2

    strong = {}
    reference = None
    rows = []
    for p in p_values:
        elapsed, result = median_run(n, p, params["seed"])
        if reference is None:
            reference = result
        error = float(np.max(np.abs(result - reference)))
        strong[p] = elapsed
        speedup = strong[1] / elapsed
        efficiency = speedup / p * 100.0
        rows.append({
            "study": "strong", "processes": p, "N": n, "seconds": elapsed,
            "speedup": speedup, "efficiency_percent": efficiency,
            "max_abs_error_vs_P1": error
        })
        print(f"Strong P={p}: {elapsed:.4f}s, speedup={speedup:.2f}, efficiency={efficiency:.1f}%")

    # Weak scaling: area grows proportional to P, so dimension grows by sqrt(P).
    weak_t1 = None
    for p in p_values:
        np_dim = math.ceil(n * math.sqrt(p))
        elapsed, _ = median_run(np_dim, p, params["seed"])
        if p == 1:
            weak_t1 = elapsed
        work_ratio = (np_dim * np_dim) / (n * n)
        scaled_speedup = p * weak_t1 / (work_ratio * elapsed)
        rows.append({
            "study": "weak", "processes": p, "N": np_dim, "seconds": elapsed,
            "speedup": scaled_speedup, "efficiency_percent": scaled_speedup / p * 100.0,
            "max_abs_error_vs_P1": None
        })
        print(f"Weak P={p}: N={np_dim}, {elapsed:.4f}s, scaled speedup={scaled_speedup:.2f}")

    t1 = strong[1]
    estimates = []
    for p in p_values:
        if p > 1:
            s = (1.0 / (t1 / strong[p]) - 1.0 / p) / (1.0 - 1.0 / p)
            estimates.append(s)
    serial_fraction = max(0.0, min(1.0, statistics.median(estimates))) if estimates else 1.0
    smax = (1.0 / serial_fraction) if serial_fraction > 0 else None

    metrics_path = base / "results" / "scaling_metrics.tsv"
    with metrics_path.open("w") as f:
        f.write("study\tprocesses\tN\tseconds\tspeedup_or_scaled_speedup\tefficiency_percent\tmax_abs_error_vs_P1\n")
        for r in rows:
            f.write("\t".join("" if r[k] is None else str(r[k]) for k in
                ("study", "processes", "N", "seconds", "speedup", "efficiency_percent", "max_abs_error_vs_P1")) + "\n")

    # Five-point stencil: 5 FLOPs per output; conservative traffic = 5 reads + 1 write = 48 B.
    ai = 5.0 / 48.0
    cache = profile.get("cache", {})
    roofline = {
        "kernel": "5-point 2D stencil",
        "flops_per_output_cell": 5,
        "conservative_bytes_per_output_cell": 48,
        "arithmetic_intensity": ai,
        "arithmetic_intensity_flops_per_byte": ai,
        "cache_sizes_bytes": cache,
        "peak_memory_bandwidth_bytes_per_second": None,
        "peak_compute_flops_per_second": None,
        "classification": "Likely memory-bandwidth-bound (low arithmetic intensity); exact Roofline ridge-point classification requires documented peak bandwidth and peak FLOP/s for this CPU.",
        "serial_fraction_median_karp_flatt_estimate": serial_fraction,
        "amdahl_asymptotic_speedup_ceiling": smax,
        "note": "Process startup, array serialization, and OS scheduling affect these empirical timings."
    }
    (base / "results" / "roofline_summary.json").write_text(json.dumps(roofline, indent=2))
    print(f"Wrote {metrics_path} and roofline_summary.json")

if __name__ == "__main__":
    mp.freeze_support()
    main()
