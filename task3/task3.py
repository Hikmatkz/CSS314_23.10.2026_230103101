#!/usr/bin/env python3

import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ARRAY_SIZE = 2_000_000
THREADS = 4


def serial_inclusive_scan(values):
    output = np.empty_like(values, dtype=np.int64)
    running = 0
    for i in range(len(values)):
        running += int(values[i])
        output[i] = running
    return output


def scan_chunk(values, start, end):
    chunk = values[start:end]
    local = np.cumsum(chunk, dtype=np.int64)
    return local, int(local[-1]) if len(local) else 0


def parallel_chunked_scan(values, workers=THREADS):
    n = len(values)
    chunk_size = (n + workers - 1) // workers
    bounds = [
        (start, min(start + chunk_size, n))
        for start in range(0, n, chunk_size)
    ]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        local_results = list(
            pool.map(lambda b: scan_chunk(values, b[0], b[1]), bounds)
        )

    offsets = []
    running = 0
    for _, chunk_total in local_results:
        offsets.append(running)
        running += chunk_total

    output = np.empty(n, dtype=np.int64)

    def apply_offset(index):
        start, end = bounds[index]
        local, _ = local_results[index]
        output[start:end] = local + offsets[index]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(apply_offset, range(len(bounds))))

    return output


def main():
    raw_id = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter your Student ID: ").strip()
    digits = "".join(ch for ch in raw_id if ch.isdigit())
    if not digits:
        raise SystemExit("Student ID must contain at least one digit.")
    seed = int(digits) % (2**32)

    print(f"Student ID: {raw_id}")
    print(f"Array size: {ARRAY_SIZE:,}")
    print(f"Threads: {THREADS}")
    print(f"Seed: {seed}")
    rng = np.random.default_rng(seed)
    values = rng.integers(1, 11, size=ARRAY_SIZE, dtype=np.int64)

    start = time.perf_counter()
    serial_output = serial_inclusive_scan(values)
    serial_seconds = time.perf_counter() - start

    start = time.perf_counter()
    parallel_output = parallel_chunked_scan(values, THREADS)
    parallel_seconds = time.perf_counter() - start

    try:
        assert int(parallel_output[-1]) == int(serial_output[-1]), (
            f"Final prefix mismatch: parallel={parallel_output[-1]}, "
            f"serial={serial_output[-1]}"
        )
        # Stronger check: ensure the complete prefix arrays match.
        assert np.array_equal(parallel_output, serial_output), (
            "Parallel and serial prefix arrays differ."
        )
        verification = "PASS"
    except AssertionError as exc:
        print(f"Verification Status: FAIL — {exc}")
        output = (
            f"Student ID: {raw_id}\n"
            f"Array size: {ARRAY_SIZE}\n"
            f"Serial Time (ms): {serial_seconds * 1000:.2f}\n"
            f"Parallel Time (ms): {parallel_seconds * 1000:.2f}\n"
            f"Speedup: {serial_seconds / parallel_seconds:.3f}x\n"
            f"Verification Status: FAIL — {exc}\n"
        )
        with open("task3_output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        raise SystemExit(1)

    speedup = serial_seconds / parallel_seconds if parallel_seconds else float("inf")
    output = (
        f"Student ID: {raw_id}\n"
        f"Array size: {ARRAY_SIZE:,}\n"
        f"Threads: {THREADS}\n"
        f"Seed: {seed}\n"
        f"Serial Time (ms): {serial_seconds * 1000:.2f}\n"
        f"Parallel Time (ms): {parallel_seconds * 1000:.2f}\n"
        f"Speedup: {speedup:.3f}x\n"
        f"Verification Status: {verification}\n"
        f"Serial final prefix: {int(serial_output[-1])}\n"
        f"Parallel final prefix: {int(parallel_output[-1])}\n"
    )
    print("\n" + output)
    with open("task3_output.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print("Saved results to task3_output.txt")


if __name__ == "__main__":
    main()
