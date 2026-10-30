import json
import statistics
import time
from pathlib import Path
import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

if njit:
    @njit
    def ijk(a, b):
        n = a.shape[0]
        c = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += a[i, k] * b[k, j]
                c[i, j] = acc
        return c

    @njit
    def ikj(a, b):
        n = a.shape[0]
        c = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(n):
                aik = a[i, k]
                for j in range(n):
                    c[i, j] += aik * b[k, j]
        return c
else:
    def ijk(a, b):
        n = a.shape[0]
        c = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                acc = 0.0
                for k in range(n):
                    acc += a[i, k] * b[k, j]
                c[i, j] = acc
        return c
    def ikj(a, b):
        n = a.shape[0]
        c = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for k in range(n):
                aik = a[i, k]
                for j in range(n):
                    c[i, j] += aik * b[k, j]
        return c

def timed(fn, a, b):
    samples, result = [], None
    for _ in range(3):
        t0 = time.perf_counter()
        result = fn(a, b)
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples), result

def main():
    base = Path(__file__).resolve().parents[1]
    params = json.loads((base / "results" / "salted_params.json").read_text())
    n = params["N"] // 2
    rng = np.random.default_rng(params["seed"])
    a = rng.random((n, n), dtype=np.float64)
    b = rng.random((n, n), dtype=np.float64)

    # Compile Numba functions before timing, so compilation is excluded.
    if njit:
        ijk(a[:2, :2], b[:2, :2])
        ikj(a[:2, :2], b[:2, :2])

    t_ijk, c_ijk = timed(ijk, a, b)
    t_ikj, c_ikj = timed(ikj, a, b)
    max_error = float(np.max(np.abs(c_ijk - c_ikj)))
    result = {
        "matrix_dimension_M": n,
        "ijk_median_seconds": t_ijk,
        "ikj_median_seconds": t_ikj,
        "speedup_ijk_over_ikj": t_ijk / t_ikj if t_ikj else None,
        "max_absolute_error": max_error,
        "numerically_equal_at_1e-12": bool(np.allclose(c_ijk, c_ikj, rtol=1e-12, atol=1e-12)),
        "numba_available": njit is not None,
        "note": "Numba JIT loops used when installed; otherwise pure Python fallback is very slow."
    }
    (base / "results" / "loop_interchange.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
