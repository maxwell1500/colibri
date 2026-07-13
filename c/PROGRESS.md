# colibri MSVC Port — Progress Log

Append-only log. One entry per completed sub-step.

## [1.1] clock_gettime shim + POSIX include guards — dd33d5e — PASS
- Files touched: c/compat.h (added clock_gettime, sched_yield, usleep, strdup, lseek, STDIN_FILENO, dirent shims; added ssize_t, off_t, _CRT_SECURE_NO_WARNINGS), c/iobench.c (guarded unistd.h include, moved OpenMP loop var before pragma)
- What changed: Added QueryPerformanceCounter-based clock_gettime shim. Guarded #include <unistd.h> in iobench.c. Fixed MSVC OpenMP 2.0 requirement (loop var declared before #pragma omp for). Added _CRT_SECURE_NO_WARNINGS + #pragma warning(disable:4996).
- Compile check result: PASS — iobench.c compiles + links, binary runs (prints usage, no crash)
- Deviations from plan: Combined multiple shims into one step (clock_gettime, lseek, strdup, sched_yield, usleep, dirent, STDIN_FILENO, ssize_t, off_t) since they all fit in compat.h and iobench.c was the simplest test file. Also fixed OpenMP loop var declaration which is a MSVC-specific requirement.

## [0.1] Baseline verification — 9c7f6a8 — PASS
- Files touched: none (verification only)
- What changed: Confirmed MSVC toolchain works with exact build flags
- Compile check result: cl.exe 19.44.35227 compiles + links under `/std:c17 /openmp /arch:AVX2 /D_FILE_OFFSET_BITS=64 /experimental:c11atomics`
- Key findings:
  - `/arch:AVX2` defines `__AVX2__` (verified: `#ifndef __AVX2__ #error` compiles)
  - `<stdatomic.h>` requires `/experimental:c11atomics` (without it, `__STDC_NO_ATOMICS__` is defined → fatal error C1189)
  - `static inline`, `_Thread_local`, compound literals all work under `/std:c17`
  - No gcc/make on this machine — MSVC smoke-test is the baseline
- Deviations from plan: Phase 0.1 adapted for no-gcc environment (recorded MSVC smoke-test as baseline instead of GCC control)

## [0.2] Scaffolding created — d3577f3 — PASS
- Files touched: c/ARCHITECTURE.md, c/PROGRESS.md, c/PLAN.md
- What changed: Created architecture document, progress log, and phase plan
- Compile check result: N/A (documentation only)
- Deviations from plan: None

## [0.3] compat_pthread.h stub created — a11d43e — PASS
- Files touched: c/compat_pthread.h (new), c/compat.h (added #include "compat_pthread.h" inside _WIN32 block)
- What changed: Created compat_pthread.h with SRWLOCK/CONDITION_VARIABLE/CreateThread mapping. Verified compiles + links under MSVC /std:c17.
- Compile check result: PASS — compiled and linked test.exe successfully
- Deviations from plan: Added `#ifndef CONDITION_VARIABLE_LOCKMODE_EXCLUSIVE` guard (constant not in this SDK version 10.0.26100.0)

## [0.4] build_msvc.bat created — 24839b6 — PASS
- Files touched: c/build_msvc.bat (new), c/PROGRESS.md
- What changed: Created build script that calls vcvars64.bat then cl.exe with all required flags
- Compile check result: PASS — toolchain verified (iobench.c fails on unistd.h as expected, Phase 1 will fix)
- Deviations from plan: None
