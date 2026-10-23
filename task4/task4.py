# task4.py
import multiprocessing as mp
from multiprocessing import shared_memory
import time
import statistics
import numpy as np

SIZES = [100_000, 1_000_000, 10_000_000]
REPEATS = 3
DTYPE = np.float64


def pipe_receiver(conn):
    while True:
        data = conn.recv()
        if data is None:
            break
        checksum = float(data[0]) + float(data[-1])
        conn.send(checksum)
    conn.close()


def shared_memory_receiver(conn):
    while True:
        message = conn.recv()
        if message is None:
            break

        name, shape, dtype_str = message
        shm = shared_memory.SharedMemory(name=name)

        try:
            array = np.ndarray(
                shape, dtype=np.dtype(dtype_str), buffer=shm.buf
            )
            checksum = float(array[0]) + float(array[-1])
            conn.send(checksum)
        finally:
            shm.close()

    conn.close()


def benchmark_pipe(data):
    parent, child = mp.Pipe()
    process = mp.Process(target=pipe_receiver, args=(child,))
    process.start()
    child.close()

    start = time.perf_counter()
    parent.send(data)
    parent.recv()
    elapsed = (time.perf_counter() - start) * 1000

    parent.send(None)
    parent.close()
    process.join()

    return elapsed


def benchmark_shared_memory(data):
    parent, child = mp.Pipe()
    process = mp.Process(
        target=shared_memory_receiver, args=(child,)
    )
    process.start()
    child.close()

    start = time.perf_counter()
    shm = shared_memory.SharedMemory(
        create=True, size=data.nbytes
    )

    try:
        shared_array = np.ndarray(
            data.shape, dtype=data.dtype, buffer=shm.buf
        )
        np.copyto(shared_array, data)

        parent.send((shm.name, data.shape, data.dtype.str))
        parent.recv()

    finally:
        shm.close()
        shm.unlink()

    elapsed = (time.perf_counter() - start) * 1000

    parent.send(None)
    parent.close()
    process.join()

    return elapsed


def main():
    results = [
        "TASK 4 — Host-to-Device Transfer Penalty / CPU IPC Overhead",
        "Path A: Multiprocessing Pipe with Pickle serialization",
        "Path B: Shared memory",
        "Measurements are medians of 3 runs.",
        "",
        f"{'N':>12} {'Size (MiB)':>12} "
        f"{'Path A (ms)':>15} {'Path B (ms)':>15} "
        f"{'Speedup':>12}",
        "-" * 72,
    ]

    for n in SIZES:
        data = np.arange(n, dtype=DTYPE)

        path_a = [
            benchmark_pipe(data) for _ in range(REPEATS)
        ]
        path_b = [
            benchmark_shared_memory(data) for _ in range(REPEATS)
        ]

        a = statistics.median(path_a)
        b = statistics.median(path_b)
        size_mib = data.nbytes / (1024 * 1024)
        speedup = a / b if b > 0 else float("inf")

        results.append(
            f"{n:12,d} {size_mib:12.2f} "
            f"{a:15.3f} {b:15.3f} {speedup:11.2f}x"
        )

    results.extend([
        "",
        "Path A includes serialization, IPC transfer, and deserialization.",
        "Path B includes shared-memory allocation, copying, and synchronization.",
        "Results depend on hardware, operating system, and process scheduling.",
    ])

    output = "\n".join(results)
    print(output)

    with open("task4_output.txt", "w", encoding="utf-8") as file:
        file.write(output + "\n")


if __name__ == "__main__":
    mp.freeze_support()
    main()