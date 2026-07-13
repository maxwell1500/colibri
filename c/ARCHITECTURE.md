# colibri MSVC Port — Architecture Document

## Scope

This port covers building the colibri C engine on native Windows 11 x86-64 using ONLY the
Microsoft Visual C++ compiler (cl.exe, MSVC 19.44). It explicitly EXCLUDES:
- WSL2 / Linux build paths
- MinGW-w64 / MSYS2 build paths
- CUDA backend (`backend_cuda.cu`) — separate workstream
- Metal backend (`backend_metal.mm`) — macOS only
- Python converter tools (`c/tools/`) — already run on Windows
- The Makefile (`c/Makefile`) — replaced by `build_msvc.bat`
- The setup script (`c/setup.sh`) — shell script, not used on native Windows

## Build System

### Toolchain
- **Compiler**: cl.exe (MSVC 19.44.35227)
- **Path**: `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`
- **Environment setup**: `vcvars64.bat` at `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`

### Build Flags
```
cl /std:c17 /openmp /arch:AVX2 /D_FILE_OFFSET_BITS=64 /experimental:c11atomics /W3 /Fe:<out.exe> <src.c>
```

| Flag | Purpose |
|------|---------|
| `/std:c17` | C17 standard (C99 features like compound literals, mid-block declarations, `snprintf`) |
| `/openmp` | OpenMP 2.0 support (all 24 `#pragma omp` sites) |
| `/arch:AVX2` | Enables AVX2 intrinsics (`__AVX2__` defined, `<immintrin.h>` works) |
| `/D_FILE_OFFSET_BITS=64` | 64-bit file offsets (required for >4 GB pread) |
| `/experimental:c11atomics` | **Required** — enables C11 `<stdatomic.h>`. Without it, `__STDC_NO_ATOMICS__` is defined and atomics fail. |
| `/W3` | Warning level 3 (reduces noise from intentional `-Wno-unused-parameter` patterns) |

### Build Script
`build_msvc.bat` — invokes vcvars64.bat then cl.exe for each target (glm, olmoe, iobench).

## Compat Layer Design

### compat.h (existing, extended)
The existing `compat.h` already provides these Win32 shims under `#ifdef _WIN32`:
- `pread` → `ReadFile` + `OVERLAPPED` (thread-safe, 64-bit offset, binary mode)
- `posix_fadvise` → no-op macro
- `posix_memalign` → `_aligned_malloc`
- `rename` → `MoveFileExA` with `MOVEFILE_REPLACE_EXISTING`
- `getpid` → `_getpid`
- `getrusage` → `GetProcessMemoryInfo`
- `getline` → `compat_getline` (fgets + realloc)
- `setenv` → `SetEnvironmentVariableA`
- `COMPAT_O_RDONLY` → `O_RDONLY | O_BINARY`
- `#pragma comment(lib, "psapi.lib")` — auto-links psapi

**New shims added in this port:**
- `clock_gettime` / `CLOCK_MONOTONIC` → `QueryPerformanceCounter`
- `sched_yield` → `YieldProcessor()`
- `usleep` → `Sleep(ms)` (microsecond→millisecond rounding)
- `strdup` → `_strdup`
- `lseek` → `_lseeki64`
- `STDIN_FILENO` → `_fileno(stdin)`
- `struct timespec` → compat definition (guarded by `_TIMESPEC_DEFINED`)
- `opendir`/`readdir`/`closedir`/`DIR`/`struct dirent` → `FindFirstFileA`/`FindNextFileA` wrapper
- `select()` stdin polling → `WaitForSingleObject` on stdin HANDLE

### compat_pthread.h (new file)
Maps pthread primitives to Win32 primitives:

| pthread | Win32 | Notes |
|---------|-------|-------|
| `pthread_t` | `HANDLE` | Thread handle from `CreateThread` |
| `pthread_mutex_t` | `SRWLOCK` | Static init via `SRWLOCK_INIT`; non-recursive (verified no recursive locking in codebase) |
| `pthread_cond_t` | `CONDITION_VARIABLE` | Paired with SRWLOCK; static init via `{0}` |
| `PTHREAD_MUTEX_INITIALIZER` | `SRWLOCK_INIT` | Compile-time constant |

**Functions mapped:**
- `pthread_create` → `CreateThread` + adapter shim (`compat_thread_arg` struct + `compat_thread_proc`)
- `pthread_mutex_init` → `InitializeSRWLock`
- `pthread_mutex_lock` → `AcquireSRWLockExclusive`
- `pthread_mutex_unlock` → `ReleaseSRWLockExclusive`
- `pthread_cond_init` → `InitializeConditionVariable`
- `pthread_cond_wait` → `SleepConditionVariableSRW` (takes SRWLOCK + CONDITION_VARIABLE)
- `pthread_cond_broadcast` → `WakeAllConditionVariable`

**NOT mapped (not called in codebase):**
- `pthread_join`, `pthread_cancel`, `pthread_detach`, `pthread_mutex_destroy`,
  `pthread_cond_destroy`, `pthread_cond_signal`, `pthread_self`, `pthread_setspecific`

**Thread function adapter:** pthread's `void* (*)(void*)` → Win32's `DWORD WINAPI (*)(LPVOID)`.
Both worker threads (`pipe_worker`, `pilot_worker`) run forever and are never joined, so the
adapter discards the return value safely. The adapter wraps `(fn, arg)` in a heap-allocated
`compat_thread_arg` struct, passes it to `CreateThread`, and frees it when the thread exits.

## Atomics Model

### Variables converted to `_Atomic`

| Variable | Type | Declaration site |
|----------|------|-----------------|
| `m->eclock` (Model field) | `_Atomic uint64_t` | glm.c:169 (was plain `uint64_t`) |
| `pilot_r` | `_Atomic unsigned` | glm.c:2022 (was `volatile unsigned`) |
| `pilot_w` | `_Atomic unsigned` | glm.c:2022 (was `volatile unsigned`) |

### Memory order mapping (preserving original `__ATOMIC_*` order)

| Original call | C11 equivalent | Order | Rationale |
|--------------|----------------|-------|-----------|
| `__atomic_add_fetch(&m->eclock, 1, __ATOMIC_RELAXED)` ×3 | `atomic_fetch_add_explicit(&m->eclock, 1, memory_order_relaxed)` | RELAXED | LRU clock — not a synchronization point; OMP regions provide their own barriers |
| `__atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE)` | `atomic_load_explicit(&pilot_r, memory_order_acquire)` | ACQUIRE | Ring buffer read index — must see the writer's store before reading queue data |
| `__atomic_load_n(&pilot_w, __ATOMIC_ACQUIRE)` | `atomic_load_explicit(&pilot_w, memory_order_acquire)` | ACQUIRE | Ring buffer write index — same pattern as pilot_r |
| `__atomic_store_n(&pilot_r, r+1, __ATOMIC_RELEASE)` | `atomic_store_explicit(&pilot_r, r+1, memory_order_release)` | RELEASE | Ring buffer read index — publishes the consumed slot |
| `__atomic_load_n(&pilot_w, __ATOMIC_RELAXED)` | `atomic_load_explicit(&pilot_w, memory_order_relaxed)` | RELAXED | Polling the write index — no ordering dependency on this load |
| `__atomic_load_n(&pilot_r, __ATOMIC_ACQUIRE)` (in comparison) | `atomic_load_explicit(&pilot_r, memory_order_acquire)` | ACQUIRE | Ring buffer fullness check — must see the latest read position |
| `__atomic_store_n(&pilot_w, w+1, __ATOMIC_RELEASE)` | `atomic_store_explicit(&pilot_w, w+1, memory_order_release)` | RELEASE | Ring buffer write index — publishes the enqueued slot |

### Semantic change: `volatile` → `_Atomic` on pilot_r/pilot_w
The original code used `static volatile unsigned pilot_w=0, pilot_r=0;` with GCC's
`__atomic_*` builtins. The `volatile` keyword was the GCC idiom for "don't optimize this
away." In C11, `_Atomic` is the correct mechanism — it provides both the "don't optimize
away" guarantee AND the memory ordering semantics that the `__atomic_*` calls were already
requesting. This is a deliberate semantic change: `volatile` is neither necessary nor
correct for atomics in C11.

### Variables NOT converted (preserve existing behavior)
- `m->hits`, `m->miss`, `m->ereq` — incremented with non-atomic `++` inside OMP parallel
  regions. This is a pre-existing benign race in the GCC build; do NOT "fix" it.

## Known Behavioral Deltas from POSIX Build

1. **O_DIRECT → buffered I/O**: Windows uses the page cache as a free L2. The `#else` branch
   at st.h:87 sets `dfds=-1` (no direct fd). Buffered reads route through `compat_pread`
   (ReadFile+OVERLAPPED). This matches the README's stated Windows behavior.

2. **select() → WaitForSingleObject**: Serve-mode stdin polling uses `WaitForSingleObject`
   on the stdin HANDLE instead of POSIX `select()`. This is serve-mode only; chat mode is
   unaffected. The timeout semantics are equivalent (0 = poll, INFINITE = block).

3. **sched_yield() → YieldProcessor()**: CPU spin-yield (gives up timeslice but stays
   runnable). Matches the original `sched_yield` intent in spin loops. Slightly more
   aggressive than `Sleep(0)` which can yield to lower-priority threads.

4. **usleep() → Sleep()**: Microsecond→millisecond rounding. `usleep(200)` becomes
   `Sleep(1)`. Acceptable for the pilot worker's poll loop (readahead thread).

5. **clock_gettime(CLOCK_MONOTONIC) → QueryPerformanceCounter**: Sub-microsecond resolution
   on Windows. Equivalent or better than POSIX monotonic clock.

6. **pthread → Win32 threading**: No `pthread_join` in the codebase, so threads are "fire
   and forget" — no join/destroy needed. SRWLOCK is non-recursive (verified no recursive
   locking). CONDITION_VARIABLE + SleepConditionVariableSRW has equivalent semantics to
   pthread_cond_wait.

7. **OpenMP runtime**: MSVC's `/openmp` uses the Microsoft OpenMP runtime (not libgomp).
   All used constructs are OpenMP 2.0-compatible, so no behavioral differences expected.

8. **Compound literals**: 5 sites in glm.c (3047, 3318, 3328, 3732). C99 feature, supported
   in MSVC C17 mode. No behavioral change — same semantics as POSIX.
