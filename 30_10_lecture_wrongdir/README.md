# CS482 Lab Practicum starter scripts

1. Install Python 3.10+, NumPy, and (recommended) Numba:
   `python -m pip install numpy numba`
2. From this folder run:
   `python src/profile_env.py --student-id YOUR_8_DIGIT_ID`
3. Run scripts in order:
   `python src/stride_bench.py`
   `python src/dense_kernel.py`
   `python src/false_sharing.py`
   `python src/scaling_study.py`

Run commands from the project root. `salt.py` is imported by scripts in `src/`.

Notes:
- `dense_kernel.py` uses Numba if installed. Without Numba, its pure-Python fallback can take an extremely long time.
- `stride_bench.py` uses NumPy's strided view reduction; it measures actual strided reads but the exact cliff varies by CPU, cache, OS, and NumPy build.
- `false_sharing.py` intentionally uses an unlocked shared array to expose cache-line contention. Real hardware/OS scheduling may not show the expected collapse consistently.
- The profile script reports unknown values rather than inventing microarchitecture or cache details. `psutil` is optional and improves physical-core detection: `python -m pip install psutil`.
- The supplied assignment refers to `verify_outputs.py`, but its source was not included in the prompt. This package does not fabricate that course-provided verifier or its signature format.
