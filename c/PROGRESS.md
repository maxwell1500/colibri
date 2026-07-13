# colibri MSVC Port — Progress Log

Append-only log. One entry per completed sub-step.

## [1.1] clock_gettime shim + POSIX include guards — dd33d5e — PASS
- Files touched: c/compat.h (added clock_gettime, sched_yield, usleep, strdup, lseek, STDIN_FILENO, dirent shims; added ssize_t, off_t, _CRT_SECURE_NO_WARNINGS), c/iobench.c (guarded unistd.h include, moved OpenMP loop var before pragma)
- What changed: Added QueryPerformanceCounter-based clock_gettime shim. Guarded #include <unistd.h> in iobench.c. Fixed MSVC OpenMP 2.0 requirement (loop var declared before #pragma omp for). Added _CRT_SECURE_NO_WARNINGS + #pragma warning(disable:4996).
- Compile check result: PASS — iobench.c compiles + links, binary runs (prints usage, no crash)
- Deviations from plan: Combined multiple shims into one step (clock_gettime, lseek, strdup, sched_yield, usleep, dirent, STDIN_FILENO, ssize_t, off_t) since they all fit in compat.h and iobench.c was the simplest test file. Also fixed OpenMP loop var declaration which is a MSVC-specific requirement.

## [2] st.h POSIX include guards + olmoe.c OpenMP fixes — 08c84a2 — PASS
- Files touched: c/st.h (guarded unistd.h and dirent.h under #ifndef _WIN32), c/olmoe.c (moved loop var declarations before #pragma omp parallel for in 4 sites)
- What changed: st.h now includes POSIX headers only on non-Windows (compat.h provides shims). olmoe.c OpenMP loop vars declared before pragmas (MSVC requirement).
- Compile check result: PASS — olmoe.c compiles + links. Remaining warnings: C4293 (shift count in compat_pread), C4244 (int64→off_t conversions, pre-existing), C4849 (collapse(2) ignored by MSVC OpenMP 2.0), C4244 (uint64→double, pre-existing).
- Deviations from plan: Combined Phase 2.1-2.6 into one commit since all shims were already in compat.h from Phase 1.1. The remaining work was st.h POSIX guards and olmoe.c OpenMP fixes.

## [3.5] POSIX includes guarded + 17 OpenMP loop var fixes + VLA fix — d6f223d — PASS (partial)
- Files touched: c/glm.c (guarded pthread.h/sched.h/unistd.h/sys/select.h under #ifndef _WIN32; moved 17 loop var declarations before #pragma omp parallel for; replaced VLA seen[E] with malloc; wrapped mmap/fstat/madvise in #if __APPLE__||__linux__)
- What changed: All POSIX includes in glm.c now guarded. All OpenMP for-loop variables declared before pragmas (MSVC requirement). VLA replaced with heap allocation. mmap/fstat/madvise gated to non-Windows.
- Compile check result: Partial PASS — compiles past all OpenMP/VLA/mmap errors. Remaining errors: 9 __atomic_* builtins (Phase 4) + fd_set/select (T1-9 serve-mode).
- Deviations from plan: Fixed all 17 OpenMP sites in one commit instead of separate sub-steps (all same pattern, mechanical change).

## [4] Atomics conversion + serve-mode select() rewrite — 89ce71f — PASS
- Files touched: c/glm.c (converted m->eclock to _Atomic uint64_t, pilot_r/pilot_w to _Atomic unsigned; converted 9 __atomic_* builtins to atomic_*_explicit with preserved memory orders; replaced fd_set/select/FD_ZERO/FD_SET/FD_ISSET with WaitForSingleObject on stdin HANDLE under #ifdef _WIN32)
- What changed: All GCC __atomic_* builtins replaced with C11 atomic_*_explicit. Memory orders preserved exactly (RELAXED→relaxed, ACQUIRE→acquire, RELEASE→release). Serve-mode select() replaced with Windows-native WaitForSingleObject.
- Compile check result: PASS — glm.c compiles to 265 KB .obj with only warnings (C4244 int64→off_t conversions, C4849 collapse(2) ignored by MSVC OpenMP 2.0, C4293 shift count). No errors.
- Deviations from plan: None — followed the handoff document exactly.

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
