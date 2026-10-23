#!/usr/bin/env python3

import math
import random
import sys
import threading
import time

SAMPLE_POINTS = 40_000_000
THREADS = 4


def split_work(total, workers):
    q, r = divmod(total, workers)
    return [q + (1 if i < r else 0) for i in range(workers)]


def run_shared_prng(total_points):
    rng = random.Random(0xC0FFEE)
    rng_lock = threading.Lock()
    results = [0] * THREADS
    counts = split_work(total_points, THREADS)

    def worker(index, count):
        inside = 0
        # Lock each pair of draws: this deliberately demonstrates contention
        # on one shared mutable generator state.
        for _ in range(count):
            with rng_lock:
                x = rng.random()
                y = rng.random()
            if x * x + y * y <= 1.0:
                inside += 1
        results[index] = inside

    threads = [
        threading.Thread(target=worker, args=(i, counts[i]))
        for i in range(THREADS)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    estimate = 4.0 * sum(results) / total_points
    return estimate, elapsed


def run_independent_prng(student_id, total_points):
    results = [0] * THREADS
    counts = split_work(total_points, THREADS)

    def worker(index, count):
        seed = (student_id * 10) + index
        rng = random.Random(seed)
        inside = 0
        for _ in range(count):
            x = rng.random()
            y = rng.random()
            if x * x + y * y <= 1.0:
                inside += 1
        results[index] = inside

    threads = [
        threading.Thread(target=worker, args=(i, counts[i]))
        for i in range(THREADS)
    ]
    start = time.perf_counter()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    elapsed = time.perf_counter() - start
    estimate = 4.0 * sum(results) / total_points
    return estimate, elapsed


def main():
    raw_id = sys.argv[1].strip() if len(sys.argv) > 1 else input("Enter your Student ID: ").strip()
    digits = "".join(ch for ch in raw_id if ch.isdigit())
    if not digits:
        raise SystemExit("Student ID must contain at least one digit.")
    student_id = int(digits)

    print(f"Student ID: {raw_id}")
    print(f"Threads: {THREADS}")
    print(f"Sample points per run: {SAMPLE_POINTS:,}")
    print("Running shared PRNG with mutex...", flush=True)
    shared_pi, shared_seconds = run_shared_prng(SAMPLE_POINTS)
    print("Running independent per-thread PRNGs...", flush=True)
    independent_pi, independent_seconds = run_independent_prng(student_id, SAMPLE_POINTS)

    shared_error = abs(shared_pi - math.pi)
    independent_error = abs(independent_pi - math.pi)
    speedup = shared_seconds / independent_seconds if independent_seconds else float("inf")

    output_lines = [
        f"Student ID: {raw_id}",
        f"Threads: {THREADS}",
        f"Sample points per run: {SAMPLE_POINTS:,}",
        f"Shared PRNG: Computed pi = {shared_pi:.12f}",
        f"Shared PRNG: Absolute error = {shared_error:.12f}",
        f"Shared PRNG: Wall time = {shared_seconds * 1000:.2f} ms",
        f"Independent PRNG: Computed pi = {independent_pi:.12f}",
        f"Independent PRNG: Absolute error = {independent_error:.12f}",
        f"Independent PRNG: Wall time = {independent_seconds * 1000:.2f} ms",
        f"Speedup factor (shared time / independent time): {speedup:.3f}x",
    ]
    output = "\n".join(output_lines) + "\n"
    print("\n" + output)
    with open("task2_output.txt", "w", encoding="utf-8") as f:
        f.write(output)
    print("Saved results to task2_output.txt")


if __name__ == "__main__":
    main()
