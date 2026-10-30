import json
import statistics
import time
from pathlib import Path
import numpy as np
from salt import get_student_params

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]

def main():
    base = Path(__file__).resolve().parents[1]
    params = json.loads((base / "results" / "salted_params.json").read_text())
    count = params["buffer_elements"]
    data = np.ones(count, dtype=np.float64)
    # Touch every page before timing so allocation/page faults do not dominate.
    _ = float(data.sum())
    rows = []
    for stride in STRIDES:
        samples = []
        touched = len(data[::stride])
        for run in range(10):
            start = time.perf_counter_ns()
            total = float(np.sum(data[::stride], dtype=np.float64))
            elapsed = time.perf_counter_ns() - start
            if run >= 5:
                samples.append(elapsed)
        median_ns = int(statistics.median(samples))
        rows.append((stride, touched, median_ns, median_ns / touched, total))
        print(f"stride={stride:3d} touched={touched:,} median={median_ns/1e6:.2f} ms")
    out = base / "results" / "stride_results.tsv"
    with out.open("w") as f:
        f.write("stride\telements_touched\tmedian_time_ns\tlatency_ns_per_element\tchecksum\n")
        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")
    print(f"Wrote {out}")

if __name__ == "__main__":
    main()
