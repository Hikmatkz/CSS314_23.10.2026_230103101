
import json
import statistics
import time
from pathlib import Path
import numpy as np

STRIDES = [1, 2, 4, 8, 16, 32, 64, 128]

def main():
    base = Path(__file__).resolve().parents[1]

    params = json.loads(
        (base / "results" / "salted_params.json").read_text()
    )

    count = params["buffer_elements"]

    # Allocate and initialize the array
    data = np.arange(count, dtype=np.float64)

    rows = []

    for stride in STRIDES:
        samples = []
        total = 0.0

        # Warm-up
        for _ in range(20):
            total = float(np.sum(data[::stride]))

        # Actual measurements
        for _ in range(30):
            start = time.perf_counter_ns()

            total = float(np.sum(data[::stride]))

            elapsed = time.perf_counter_ns() - start
            samples.append(elapsed)

        median_ns = int(statistics.median(samples))
        touched = len(data[::stride])

        latency = median_ns / touched

        rows.append(
            (stride, touched, median_ns, latency, total)
        )

        print(
            f"stride={stride:3d} "
            f"touched={touched:,} "
            f"median={median_ns / 1e6:.4f} ms "
            f"latency={latency:.4f} ns/element"
        )

    out = base / "results" / "stride_results.tsv"

    with out.open("w") as f:
        f.write(
            "stride\telements_touched\tmedian_time_ns"
            "\tlatency_ns_per_element\tchecksum\n"
        )

        for row in rows:
            f.write("\t".join(map(str, row)) + "\n")

    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
