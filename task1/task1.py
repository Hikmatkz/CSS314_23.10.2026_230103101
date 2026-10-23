#!/usr/bin/env python3

import os
import sys
import time
import multiprocessing as mp
from multiprocessing import shared_memory

import numpy as np

N = 8_000
DTYPE = np.float64


def physical_core_count():
    try:
        import psutil
        return psutil.cpu_count(logical=False) or (os.cpu_count() or 1)
    except ImportError:
        # Fallback: may be logical rather than physical.
        return os.cpu_count() or 1


def row_sum_worker(args):
    shm_name, shape, start_row, end_row = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        matrix = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
        return float(matrix[start_row:end_row, :].sum(dtype=np.float64))
    finally:
        shm.close()


def column_sum_worker(args):
    shm_name, shape, start_col, end_col = args
    shm = shared_memory.SharedMemory(name=shm_name)
    try:
        matrix = np.ndarray(shape, dtype=DTYPE, buffer=shm.buf)
        subtotal = 0.0
        # Each column is traversed top-to-bottom. In C-order storage,
        # successive elements in a column are separated by N * 8 bytes.
        for col in range(start_col, end_col):
            subtotal += float(matrix[:, col].sum(dtype=np.float64))
        return subtotal
    finally:
        shm.close()


def ranges(total, workers):
    # Balanced, non-overlapping slices; also handles totals not divisible by workers.
    q, r = divmod(total, workers)
    result = []
    start = 0
    for i in range(workers):
        size = q + (1 if i < r else 0)
        end = start + size
        if start < end:
            result.append((start, end))
        start = end
    return result


def main():
    if len(sys.argv) >= 2:
        student_id = sys.argv[1].strip()
    else:
        student_id = input("Enter your Student ID: ").strip()
    if not student_id:
        raise SystemExit("Student ID cannot be empty.")

    # Use the digits in the Student ID as a reproducible NumPy seed.
    digits = ''.join(ch for ch in student_id if ch.isdigit())
    if not digits:
        raise SystemExit("Student ID must contain at least one digit.")
    seed = int(digits) % (2**32)

    logical_cores = os.cpu_count() or 1
    physical_cores = physical_core_count()
    workers = max(1, physical_cores)

    print(f"Student ID: {student_id}")
    print(f"Logical cores: {logical_cores}")
    print(f"Physical cores / worker processes: {physical_cores}")
    print(f"Matrix: {N} x {N}, dtype=float64, seed={seed}")
    print(f"Workers: {workers}")
    print("Allocating and initializing matrix (~512 MB)...", flush=True)

    # Shared memory avoids copying the 512 MB matrix into every process.
    nbytes = N * N * np.dtype(DTYPE).itemsize
    shm = shared_memory.SharedMemory(create=True, size=nbytes)
    try:
        matrix = np.ndarray((N, N), dtype=DTYPE, buffer=shm.buf)
        rng = np.random.default_rng(seed)
        rng.random((N, N), dtype=DTYPE, out=matrix)

        ctx = mp.get_context("spawn")
        row_tasks = [
            (shm.name, (N, N), start, end)
            for start, end in ranges(N, workers)
        ]
        col_tasks = [
            (shm.name, (N, N), start, end)
            for start, end in ranges(N, workers)
        ]

        # Pool creation is outside the timed region; timings focus on traversal/reduction.
        with ctx.Pool(processes=workers) as pool:
            t0 = time.perf_counter()
            row_parts = pool.map(row_sum_worker, row_tasks)
            row_seconds = time.perf_counter() - t0
            row_total = sum(row_parts)

            t0 = time.perf_counter()
            col_parts = pool.map(column_sum_worker, col_tasks)
            col_seconds = time.perf_counter() - t0
            col_total = sum(col_parts)

        # Sanity check: reduction order differs slightly due to floating-point rounding.
        if not np.isclose(row_total, col_total, rtol=1e-10, atol=1e-5):
            print(f"WARNING: sums differ: row={row_total:.12f}, column={col_total:.12f}")

        row_ms = row_seconds * 1000
        col_ms = col_seconds * 1000
        penalty = col_seconds / row_seconds if row_seconds else float("inf")

        lines = [
            f"Student ID: {student_id}",
            f"Logical cores: {logical_cores}",
            f"Physical cores / worker processes: {physical_cores}",
            f"Matrix: {N} x {N}, dtype=float64, seed={seed}",
            f"Mode A (Row-Major) runtime: {row_ms:.2f} ms",
            f"Mode B (Column-Major / Strided) runtime: {col_ms:.2f} ms",
            f"Cache penalty factor (Mode B / Mode A): {penalty:.3f}x",
            f"Mode A sum: {row_total:.12f}",
            f"Mode B sum: {col_total:.12f}",
        ]
        output = "\n".join(lines) + "\n"
        print("\n" + output)
        with open("task1_output.txt", "w", encoding="utf-8") as f:
            f.write(output)
        print("Saved results to task1_output.txt")
    finally:
        shm.close()
        shm.unlink()


if __name__ == "__main__":
    mp.freeze_support()
    main()
