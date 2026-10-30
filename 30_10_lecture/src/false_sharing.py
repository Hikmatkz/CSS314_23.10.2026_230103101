import ctypes
import json
import multiprocessing as mp
import statistics
import time
from pathlib import Path

K = 5_000_000
WORKERS = [1, 2, 4, 8]

def increment(shared, index, iterations, mode, start_event):
    start_event.wait()
    if mode == "privatized":
        local = 0
        for _ in range(iterations):
            local += 1
        shared[index] += local
    else:
        for _ in range(iterations):
            shared[index] += 1

def run_case(p, mode, repeats=2):
    timings = []
    total_count = 0
    for _ in range(repeats):
        slots = p * 8 if mode == "padded" else max(p, 8)
        shared = mp.Array(ctypes.c_int64, slots, lock=False)
        event = mp.Event()
        processes = []
        for worker in range(p):
            index = worker * 8 if mode == "padded" else worker
            proc = mp.Process(target=increment, args=(shared, index, K, mode, event))
            processes.append(proc)
        start = time.perf_counter()
        for proc in processes:
            proc.start()
        event.set()
        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"Worker failed with exit code {proc.exitcode}")
        timings.append(time.perf_counter() - start)
        total_count = sum(shared[i * 8 if mode == "padded" else i] for i in range(p))
    return statistics.median(timings), int(total_count)

def main():
    base = Path(__file__).resolve().parents[1]
    out = base / "results" / "false_sharing_karp_flatt.tsv"
    measurements = {}
    rows = []
    for mode in ("unpadded", "padded", "privatized"):
        for p in WORKERS:
            elapsed, total = run_case(p, mode)
            measurements[(mode, p)] = elapsed
            rows.append((mode, p, K, elapsed, total))
            print(f"{mode:10s} P={p} time={elapsed:.3f}s total={total}")
            if total != p * K:
                raise AssertionError(f"Incorrect count for {mode}, P={p}: {total}")
    with out.open("w") as f:
        f.write("mode\tprocesses\titerations_per_process\tmedian_seconds\ttotal_count\tkarp_flatt_e\n")
        for mode, p, k, elapsed, total in rows:
            if p == 1:
                e = ""
            else:
                t1 = measurements[(mode, 1)]
                e = (p * elapsed / t1 - 1) / (p - 1)
            f.write(f"{mode}\t{p}\t{k}\t{elapsed:.9f}\t{total}\t{e}\n")
    print(f"Wrote {out}")

if __name__ == "__main__":
    mp.freeze_support()
    main()
