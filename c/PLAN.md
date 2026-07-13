# colibri MSVC Port — Phase Plan

## PHASE 0 — Baseline & Scaffolding
- [x] 0.1 Confirm current GCC build compiles clean (control baseline). Record compiler version, flags, output binary size/hash in ARCHITECTURE.md.
      DEVIATION: no gcc on this machine. Recorded MSVC smoke-test as baseline. cl.exe 19.44.35227 verified with `/std:c17 /openmp /arch:AVX2 /D_FILE_OFFSET_BITS=64 /experimental:c11atomics`.
- [x] 0.2 Create ARCHITECTURE.md and PROGRESS.md with the sections above, empty bodies.
- [x] 0.3 Create compat_pthread.h (empty stub, just include guard) and add it to compat.h's include chain behind #ifdef _WIN32. Confirm it changes nothing on non-Windows build.
- [x] 0.4 Set up MSVC toolchain invocation. Create build_msvc.bat with the vcvars64 call + cl invocation. Smoke-test by compiling an empty .c file. Record exact command in ARCHITECTURE.md.

## PHASE 1 — iobench.c (simplest file, validates toolchain end-to-end)
- [x] 1.1 Add clock_gettime/CLOCK_MONOTONIC shim to compat.h (QueryPerformanceCounter-based). Wire in iobench.c:19 only.
- [x] 1.2 Add lseek → _lseeki64 shim to compat.h. Wire into iobench.c:39. (combined with 1.1)
- [x] 1.3 Confirm iobench.c's reduction(+:tot) OpenMP clause compiles under /openmp (verification step). (verified in 1.1 compile)
- [x] 1.4 Full compile + link of iobench.c under MSVC. Toolchain validation gate. (PASS — iobench.exe produced, runs without crash)

## PHASE 2 — compat.h mechanical shims (Tier 2, remaining files)
- [x] 2.1 strdup → _strdup shim in compat.h. Confirm st.h:81,140 resolve. (combined with 1.1)
- [x] 2.2 lseek shim reused for st.h — verify st.h does NOT call lseek directly. (verified)
- [x] 2.3 clock_gettime shim reused for glm.c:227,3592 and olmoe.c:53. (olmoe.c compiles)
- [x] 2.4 O_DIRECT twin-fd at st.h:82-85: verify existing #else branch handles Windows. (already handled by existing #else)
- [x] 2.5 dirent.h shim: FindFirstFileA/FindNextFileA wrapper in compat.h. Guard st.h:16. (combined with 1.1)
- [x] 2.6 Compile st.h-dependent translation units (olmoe.c) to confirm Tier 2 shims resolve. (PASS)

## PHASE 3 — pthread compat layer (Tier 1, the real work)
- [ ] 3.1 In compat_pthread.h: define pthread_t → HANDLE, pthread_mutex_t → SRWLOCK, pthread_cond_t → CONDITION_VARIABLE. Write mapping table in ARCHITECTURE.md.
- [ ] 3.2 Implement pthread_create via CreateThread with adapter shim.
- [ ] 3.3 Implement pthread_mutex_init/lock/unlock via SRWLOCK.
- [ ] 3.4 Implement pthread_cond_init/wait/broadcast via CONDITION_VARIABLE.
- [ ] 3.5 Guard the 4 POSIX includes at glm.c:26-30 with #ifndef _WIN32 / #else include compat_pthread.h #endif.
- [ ] 3.6 Add sched_yield→YieldProcessor() and usleep→compat_usleep shims to compat.h.
- [ ] 3.7 Compile glm.c — expect Tier 1 (pthread) errors to be GONE. List remaining errors.
- [ ] 3.8 Walk all 27 pthread call sites one by one, confirm each resolves to a shim symbol.
- [ ] 3.9 T1-9 serve-mode select() rewrite: STDIN_FILENO + WaitForSingleObject on stdin HANDLE.

## PHASE 4 — Atomics conversion
- [ ] 4.1 Change glm.c:169 `uint64_t eclock` → `_Atomic uint64_t eclock`. Change glm.c:2022 `volatile unsigned` → `_Atomic unsigned`.
- [ ] 4.2 Convert glm.c:1741, 1960, 2049 (`__atomic_add_fetch` → `atomic_fetch_add_explicit`).
- [ ] 4.3 Convert glm.c:2063, 2064 (`__atomic_load_n` → `atomic_load_explicit`).
- [ ] 4.4 Convert glm.c:2068, 2099, 2100 (store/load mix).
- [ ] 4.5 Convert glm.c:2102 (`__atomic_store_n` → `atomic_store_explicit`).
- [ ] 4.6 Confirm `#include <stdatomic.h>` at glm.c:27 is unguarded.
- [ ] 4.7 Full glm.c compile attempt. Resolve remaining errors one at a time.

## PHASE 5 — Full link & smoke test
- [ ] 5.1 Full solution build: all 4 .c files + compat.h + compat_pthread.h under build_msvc.bat.
- [ ] 5.2 Run iobench.exe on a small test file. Confirm no crash, sane timing output.
- [ ] 5.3 Run minimal glm.exe workload exercising PipePool/pilot_worker threading path. Timeout test for deadlock/hang.
- [ ] 5.4 Diff behavioral output against expected baseline (oracle self-test "32/32 positions" if glm_tiny available).
- [ ] 5.5 Final ARCHITECTURE.md pass: confirm every "Known behavioral delta" is complete and accurate.
