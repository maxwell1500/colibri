/* compat_pthread.h — pthread → Win32 primitive mapping
 *
 * On Windows this file is included from compat.h (under #ifdef _WIN32).
 * On Linux/macOS this header is never included (pthread.h is used directly).
 *
 * Mapping:
 *   pthread_t          → HANDLE
 *   pthread_mutex_t    → SRWLOCK
 *   pthread_cond_t     → CONDITION_VARIABLE
 *   PTHREAD_MUTEX_INITIALIZER → SRWLOCK_INIT
 *
 * Functions:
 *   pthread_create     → CreateThread + adapter shim
 *   pthread_mutex_init → InitializeSRWLock
 *   pthread_mutex_lock → AcquireSRWLockExclusive
 *   pthread_mutex_unlock → ReleaseSRWLockExclusive
 *   pthread_cond_init  → InitializeConditionVariable
 *   pthread_cond_wait  → SleepConditionVariableSRW
 *   pthread_cond_broadcast → WakeAllConditionVariable
 *
 * NOT mapped (not called in codebase):
 *   pthread_join, pthread_cancel, pthread_detach, pthread_mutex_destroy,
 *   pthread_cond_destroy, pthread_cond_signal, pthread_self, pthread_setspecific
 */
#ifndef COMPAT_PTHREAD_H
#define COMPAT_PTHREAD_H

#include <windows.h>

/* --- Types --- */
typedef HANDLE pthread_t;
typedef SRWLOCK pthread_mutex_t;
typedef CONDITION_VARIABLE pthread_cond_t;

/* --- Constants --- */
#define PTHREAD_MUTEX_INITIALIZER SRWLOCK_INIT
#ifndef CONDITION_VARIABLE_LOCKMODE_EXCLUSIVE
#define CONDITION_VARIABLE_LOCKMODE_EXCLUSIVE 1
#endif

/* --- Mutex --- */
static inline void pthread_mutex_init(pthread_mutex_t *m, void *attr) {
    (void)attr;
    InitializeSRWLock(m);
}

static inline void pthread_mutex_lock(pthread_mutex_t *m) {
    AcquireSRWLockExclusive(m);
}

static inline void pthread_mutex_unlock(pthread_mutex_t *m) {
    ReleaseSRWLockExclusive(m);
}

/* --- Condition variable --- */
static inline void pthread_cond_init(pthread_cond_t *c, void *attr) {
    (void)attr;
    InitializeConditionVariable(c);
}

static inline void pthread_cond_wait(pthread_cond_t *c, pthread_mutex_t *m) {
    SleepConditionVariableSRW(c, m, INFINITE, CONDITION_VARIABLE_LOCKMODE_EXCLUSIVE);
}

static inline void pthread_cond_broadcast(pthread_cond_t *c) {
    WakeAllConditionVariable(c);
}

/* --- Thread creation adapter --- */
/* pthread's thread proc is void* (*)(void*); Win32's is DWORD WINAPI (*)(LPVOID).
 * The adapter wraps (fn, arg) in a heap-allocated struct, passes it to CreateThread,
 * and frees it when the thread exits. Both worker threads run forever and are never
 * joined, so discarding the return value is safe. */
typedef struct {
    void* (*fn)(void*);
    void *arg;
} compat_thread_arg;

static DWORD WINAPI compat_thread_proc(LPVOID p) {
    compat_thread_arg *a = (compat_thread_arg*)p;
    a->fn(a->arg);   /* discard return value — both workers return NULL/never join */
    free(a);
    return 0;
}

static inline int pthread_create(pthread_t *t, void *attr,
                                  void* (*fn)(void*), void *arg) {
    (void)attr;
    compat_thread_arg *a = (compat_thread_arg*)malloc(sizeof(*a));
    if (!a) return -1;
    a->fn = fn;
    a->arg = arg;
    *t = CreateThread(NULL, 0, compat_thread_proc, a, 0, NULL);
    return *t ? 0 : -1;
}

#endif /* COMPAT_PTHREAD_H */
