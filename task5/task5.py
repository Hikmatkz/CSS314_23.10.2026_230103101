# task5.py
import os
import time
import hashlib
import statistics
from concurrent.futures import ThreadPoolExecutor

TASK_COUNT = 10_000
HASH_ITERATIONS = 500
DATA = b"A" * 1024
REPEATS = 3


def cpu_task(task_id):
    data = DATA + str(task_id).encode()
    for _ in range(HASH_ITERATIONS):
        data = hashlib.sha256(data).digest()
    return data


def benchmark(pool_size):
    times = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=pool_size) as executor:
            list(executor.map(cpu_task, range(TASK_COUNT)))

        times.append((time.perf_counter() - start) * 1000)

    return statistics.median(times)


def main():
    cores = os.cpu_count() or 1

    pool_sizes = [
        1,
        max(1, cores // 2),
        cores,
        cores * 2,
        cores * 4,
        cores * 16,
        256,
    ]

    pool_sizes = sorted(set(pool_sizes))
    results = []

    print(f"CPU cores detected: {cores}")
    print(f"Tasks: {TASK_COUNT:,} | SHA-256 iterations per task: {HASH_ITERATIONS}")
    print(f"Repeats per pool size: {REPEATS}\n")

    for size in pool_sizes:
        elapsed = benchmark(size)
        results.append((size, elapsed))

    baseline = next(t for size, t in results if size == 1)

    print(
        f"{'Pool Size':>12} | {'Execution Time (ms)':>20} | "
        f"{'Speedup':>10} | {'Status'}"
    )
    print("-" * 75)

    worsening_size = None
    best_time = float("inf")

    for size, elapsed in results:
        speedup = baseline / elapsed

        if size == 1:
            status = "BASELINE"
        elif elapsed > best_time:
            status = "SLOWER"
        elif elapsed < best_time:
            status = "IMPROVED"
        else:
            status = "UNCHANGED"

        if size != 1 and worsening_size is None:
            previous_times = [
                t for s, t in results if s < size
            ]
            if previous_times and elapsed > min(previous_times):
                worsening_size = size

        best_time = min(best_time, elapsed)

        print(
            f"{size:12d} | {elapsed:20.2f} | "
            f"{speedup:9.2f}x | {status}"
        )

    print("\nOversubscription analysis:")

    if worsening_size is not None:
        print(
            f"First tested pool size slower than an earlier result: "
            f"{worsening_size} threads."
        )
    else:
        print(
            "No worsening detected relative to all previously tested "
            "pool sizes."
        )

    best_size, best_elapsed = min(results, key=lambda item: item[1])
    print(
        f"Best tested pool size: {best_size} threads "
        f"({best_elapsed:.2f} ms)."
    )
    print(
        "Note: Python's GIL can limit CPU-bound thread scaling. "
        "Timing alone cannot prove that context switching is the cause."
    )


if __name__ == "__main__":
    main()