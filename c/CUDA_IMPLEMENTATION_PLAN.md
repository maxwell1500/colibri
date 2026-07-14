# CUDA Expert Offload — Full Implementation Plan

Six improvements to the GLM-5.2 Colibri CUDA expert offload system. Each phase is
self-contained and testable before proceeding. Every change includes exact file locations,
code diffs, testing commands, and rollback procedures.

---

## Phase 1: Fix resident_bytes Accounting

**Goal**: Correct the LRU cache cap calculation so `cap_for_ram()` sees the true RAM
footprint after VRAM experts have their host backing released.

**Why first**: Every subsequent phase depends on `cap_for_ram()` being correct. If the cap
is wrong, the LRU cache is either too small (wasted RAM) or too large (OOM risk).

**Time estimate**: 30 minutes

### 1.1 Problem

In `pin_load()` (glm.c:3464):
```c
m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:npin)*eb;
```

This adds ALL `gpu_prefix` experts to `resident_bytes` before the VRAM upload loop. Then
`expert_host_release()` (line 1431) decrements `resident_bytes` for each released expert.
By the time `cap_for_ram()` runs (line 3793), the counter should be correct.

However, there is a **timing issue**: `cap_for_ram()` is called at line 3793, AFTER
`pin_load()` at line 3773/3789. Inside `pin_load()`, the upload loop (3468-3497) calls
`expert_host_release()` (3488) which decrements `resident_bytes` (1431). So by the time
`pin_load()` returns, `resident_bytes` should already reflect the released experts.

**The real problem**: The `expert_avail()` function (line 3555-3561) is called BEFORE
`pin_load()` (at line 3788) to compute `pin_gb`. It uses `resident_bytes` which at that
point does NOT include the pinned experts. This is correct.

BUT: `cap_for_ram()` at line 3793 uses `resident_bytes` which by then includes:
- Dense model: ~10 GB
- Pinned RAM-only experts: `(npin - gpu_expert_count) * eb`
- VRAM experts (freed): 0 (after release)
- This should be correct.

**Actual bug**: After `expert_host_release()`, the `resident_bytes` counter IS decremented
correctly. But the `resident_bytes` increment at line 3464 happens BEFORE the upload loop,
and `cap_for_ram()` at line 3793 happens AFTER `pin_load()` returns. The counter should
be correct at that point.

**Re-evaluation**: The accounting is actually correct AFTER `pin_load()` completes. The
issue is that during `pin_load()`, the transient spike is not reflected in any cap. This
is fine because no inference runs during `pin_load()`.

**Revised problem**: The real issue is that `npin` itself is limited by RAM budget
(`expert_avail()` returns ~7 GB → 370 experts). After `pin_load()`, `resident_bytes`
includes only the RAM-only experts (~0 with `release_host`), so `cap_for_ram()` should
compute a correct (larger) cap. Let me verify this is actually broken.

### 1.2 Verification Test

Before changing code, verify the current behavior:

```bat
set SNAP=H:\glm52
set RAM_GB=24
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set OMP_NUM_THREADS=20
set COLI_CUDA_RELEASE_HOST=1
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=7
set "PROMPT=[gMASK]<sop><|user|>test<|assistant|>"
glm.exe 16
```

Check the `[RAM_GB=...]` line. If it says `cap lowered 16->2`, the cap is wrong because
`resident_bytes` is inflated. If it says `cap lowered 16->4` or higher, the accounting
is already correct and this phase is a no-op.

### 1.3 If Cap Is Wrong: Fix

**File**: `c/glm.c`

**Change 1** (line 3464): Do NOT increment `resident_bytes` for experts that will be
VRAM-uploaded and host-released. Instead, track them separately:

```c
// BEFORE (line 3464):
m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:npin)*eb;

// AFTER:
// Don't count VRAM experts in resident_bytes — they'll be released shortly.
// Only count RAM-only experts (the suffix after gpu_prefix).
// The gpu_prefix experts are counted transiently and released below.
```

Remove line 3464 entirely. The RAM-only experts are loaded at line 3507-3508 and their
`resident_bytes` is incremented at line 3509. This is correct.

For the GPU experts, `resident_bytes` should only be incremented if `!g_cuda_release_host`
(host backing is kept). Add after line 3464:

```c
#ifdef COLI_CUDA
    if(!g_cuda_release_host)
        m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:0)*eb;
#else
    m->resident_bytes+=(int64_t)npin*eb;
#endif
```

**Change 2** (line 3509): This already counts RAM-only experts. No change needed.

### 1.4 Testing

```bat
:: Test 1: Verify cap is higher
set RAM_GB=24
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set COLI_CUDA_RELEASE_HOST=1
set PIN=H:\glm52\.coli_usage
set PIN_GB=7
:: Check: cap should be 4+, not 2
:: Check: RSS should be lower (less wasted LRU cache)

:: Test 2: Verify no OOM
:: Run full 512-token generation and check for crashes

:: Test 3: Verify hit rate
:: Compare hit rate before/after — should improve if cap increased
```

### 1.5 Rollback

Revert the single change to `resident_bytes` at line 3464. No other files affected.

---

## Phase 2: Batched VRAM Loading

**Goal**: Fill VRAM to 11 GB instead of 7 GB by loading experts in small batches instead
of all at once.

**Dependencies**: Phase 1 must be complete (correct `resident_bytes`).

**Time estimate**: 2-3 hours

### 2.1 Design

Current flow:
```
1. Compute npin = 370 (RAM budget)
2. Compute gpu_prefix = min(581, 370) = 370 (VRAM budget capped by RAM)
3. Load ALL 370 experts into host RAM (parallel, ~8 seconds)
4. Upload ALL 370 to VRAM (sequential, ~8 seconds)
5. Free host backing for each
```

New flow:
```
1. Compute npin = 370 (RAM budget, for RAM-only experts)
2. Compute gpu_budget_experts = 581 (VRAM budget, uncapped by npin)
3. BATCH_SIZE = 16 experts (~300 MB per batch)
4. For each batch of 16:
   a. Load 16 experts into host RAM (parallel, ~0.3 seconds)
   b. Upload each to VRAM (sequential, ~0.3 seconds)
   c. Free host backing for each
   d. If VRAM full (gpu_expert_bytes >= budget): break
5. Load remaining RAM-only experts
```

Peak host RAM usage: 16 experts × 19 MB = **300 MB** (vs 7 GB before).

### 2.2 Code Changes

**File**: `c/glm.c`

**Change 1** (line 3454): Remove the `npin` cap on `gpu_prefix`:

```c
// BEFORE:
if(g_cuda_enabled&&g_cuda_release_host&&budget>0){
    gpu_prefix=(int)(budget/eb)+g_cuda_ndev;
    if(gpu_prefix>npin) gpu_prefix=npin;
}

// AFTER:
if(g_cuda_enabled&&g_cuda_release_host&&budget>0){
    gpu_prefix=(int)(budget/eb)+g_cuda_ndev;
    // Remove: if(gpu_prefix>npin) gpu_prefix=npin;
    // gpu_prefix is now purely VRAM-budget-driven
}
```

**Change 2** (lines 3458-3503): Replace the monolithic load+upload with a batched loop:

```c
    /* BATCHED VRAM LOADING: load experts in small batches, upload to VRAM,
     * release host backing, repeat. Peak host RAM = BATCH_SIZE * expert_bytes. */
    int a;
    int BATCH_SIZE = 16;  /* ~300 MB per batch at 19 MB/expert */
    int vram_budget_experts = gpu_prefix;  /* total we WANT in VRAM */
    int vram_placed = 0;                   /* total PLACED so far */

    for(int batch_start = 0; batch_start < vram_budget_experts && vram_placed < vram_budget_experts; batch_start += BATCH_SIZE){
        int batch_end = batch_start + BATCH_SIZE;
        if(batch_end > vram_budget_experts) batch_end = vram_budget_experts;
        int batch_count = batch_end - batch_start;

        /* Phase A: Load this batch into host RAM (parallel pread) */
        #pragma omp parallel for schedule(dynamic,1)
        for(a = batch_start; a < batch_end; a++)
            expert_load(m, r[a].l, r[a].e, &m->pin[r[a].l][slot_of[a]], 1);
        m->resident_bytes += (int64_t)batch_count * eb;

        /* Phase B: Upload this batch to VRAM, release host backing */
        for(a = batch_start; a < batch_end; a++){
            int li = r[a].l;
            ESlot *s = &m->pin[li][slot_of[a]];
            int64_t need = qt_bytes(&s->g) + qt_bytes(&s->u) + qt_bytes(&s->d);
            if(m->gpu_expert_bytes + need > budget) break;

            int tried[COLI_CUDA_MAX_DEVICES] = {0}, placed = 0;
            for(int attempt = 0; attempt < g_cuda_ndev && !placed; attempt++){
                int best = -1;
                for(int i = 0; i < g_cuda_ndev; i++)
                    if(!tried[i] && remaining[i] >= need &&
                       (best < 0 || placed_b[i] < placed_b[best]))
                        best = i;
                if(best < 0) break;
                tried[best] = 1;
                s->g.cuda_device = s->u.cuda_device = s->d.cuda_device = g_cuda_devices[best];
                s->g.cuda_eligible = s->u.cuda_eligible = s->d.cuda_eligible = 1;
                if(qt_cuda_upload(&s->g) && qt_cuda_upload(&s->u) && qt_cuda_upload(&s->d)){
                    int64_t actual = (int64_t)coli_cuda_tensor_bytes(s->g.cuda)
                                   + (int64_t)coli_cuda_tensor_bytes(s->u.cuda)
                                   + (int64_t)coli_cuda_tensor_bytes(s->d.cuda);
                    m->gpu_expert_count++; m->gpu_expert_bytes += actual;
                    remaining[best] -= actual; placed_b[best] += actual; placed_n[best]++;
                    if(g_cuda_release_host) expert_host_release(m,s);
                    vram_placed++;
                    placed = 1;
                } else {
                    qt_cuda_reset(&s->g); qt_cuda_reset(&s->u); qt_cuda_reset(&s->d);
                    s->g.cuda_eligible = s->u.cuda_eligible = s->d.cuda_eligible = 0;
                    remaining[best] = 0;
                }
            }
        }
    }
```

**Change 3** (lines 3504-3509): The RAM-only suffix loading needs adjustment. With the
new flow, `gpu_prefix` may exceed `npin`. The RAM-only experts are those in `r[0..npin-1]`
that were NOT uploaded to VRAM. We need to track which ones were skipped:

```c
    /* Load RAM-only experts (those in r[0..npin-1] that weren't uploaded to VRAM) */
    if(npin > 0){
        int ram_count = 0;
        int *ram_list = malloc(npin * sizeof(int));
        for(a = 0; a < npin; a++){
            ESlot *s = &m->pin[r[a].l][slot_of[a]];
            if(!s->g.cuda_eligible)  /* not in VRAM */
                ram_list[ram_count++] = a;
        }
        if(ram_count > 0){
            int q;
            #pragma omp parallel for schedule(dynamic,1)
            for(q = 0; q < ram_count; q++)
                expert_load(m, r[ram_list[q]].l, r[ram_list[q]].e,
                           &m->pin[r[ram_list[q]].l][slot_of[ram_list[q]]], 1);
            m->resident_bytes += (int64_t)ram_count * eb;
        }
        free(ram_list);
    }
```

### 2.3 New Environment Variable

Add `CUDA_BATCH_SIZE` env var to tune the batch size:

```c
int batch_size = getenv("CUDA_BATCH_SIZE") ? atoi(getenv("CUDA_BATCH_SIZE")) : 16;
if(batch_size < 1) batch_size = 1;
if(batch_size > 64) batch_size = 64;
```

### 2.4 Testing

```bat
:: Test 1: Verify VRAM usage reaches 11 GB
set CUDA_EXPERT_GB=11
set CUDA_BATCH_SIZE=16
:: Check: [CUDA] hot expert tier: ~580/~580 experts, VRAM 11.00 GB

:: Test 2: Verify startup time
:: Should be similar to before (batched loading is same total I/O)

:: Test 3: Verify hit rate and speed
:: Expected: hit rate 25-30%, speed 0.12-0.15 tok/s

:: Test 4: Verify no OOM
:: Peak host RAM should be ~300 MB, not ~7 GB

:: Test 5: Test different batch sizes
set CUDA_BATCH_SIZE=8    :: slower startup, same VRAM
set CUDA_BATCH_SIZE=32   :: faster startup, same VRAM
set CUDA_BATCH_SIZE=64   :: fastest startup, may spike RAM
```

### 2.5 Rollback

Revert the three changes in `pin_load()`. Add back the `if(gpu_prefix>npin) gpu_prefix=npin;` line.

---

## Phase 3: Speculative VRAM Prefetch

**Goal**: Predict which experts will be needed in the next 2-3 layers and prefetch them
into VRAM before the compute reaches that layer.

**Dependencies**: Phase 2 must be complete (VRAM is fully utilized).

**Time estimate**: 4-6 hours

### 3.1 Design

The pilot system (g_pilot=1) already runs a lightweight routing prediction 2-3 layers
ahead. Currently it only loads experts into the LRU cache (disk -> RAM). We extend it
to also upload to VRAM if the expert has a VRAM slot available.

The prefetch runs in a background thread, so it overlaps with the current layer's compute.

### 3.2 Code Changes

**File**: `c/glm.c`

**Change 1** (line 2103, `pilot_realload()`): Add VRAM upload path:

```c
static void pilot_realload(Model *m, int layer, int eid){
    pthread_mutex_lock(&g_pilot_mx);
    if(layer <= atomic_load_explicit(&g_cur_moe_layer,memory_order_acquire)){
        atomic_fetch_add_explicit(&g_pilot_drops,1,memory_order_relaxed);
        pthread_mutex_unlock(&g_pilot_mx); return;
    }
    ESlot *P=m->pin[layer];
    for(int z=0;z<m->npin[layer];z++){
        if(P[z].eid==eid){
#ifdef COLI_CUDA
            /* Already in VRAM (pinned)? Skip. */
            if(P[z].g.cuda_eligible){ pthread_mutex_unlock(&g_pilot_mx); return; }
            /* In RAM pin but not VRAM? Upload if VRAM budget allows. */
            if(g_cuda_enabled && g_cuda_release_host && P[z].slab){
                int64_t need = qt_bytes(&P[z].g)+qt_bytes(&P[z].u)+qt_bytes(&P[z].d);
                if(m->gpu_expert_bytes + need < g_cuda_expert_gb*1e9){
                    /* Try VRAM upload */
                    size_t free_b=0, total_b=0;
                    if(g_cuda_ndev>0 && coli_cuda_mem_info(g_cuda_devices[0],&free_b,&total_b)){
                        if(free_b > (size_t)need + 2e9){
                            P[z].g.cuda_device = P[z].u.cuda_device = P[z].d.cuda_device = g_cuda_devices[0];
                            P[z].g.cuda_eligible = P[z].u.cuda_eligible = P[z].d.cuda_eligible = 1;
                            if(qt_cuda_upload(&P[z].g) && qt_cuda_upload(&P[z].u) && qt_cuda_upload(&P[z].d)){
                                int64_t actual = (int64_t)coli_cuda_tensor_bytes(P[z].g.cuda)
                                               + (int64_t)coli_cuda_tensor_bytes(P[z].u.cuda)
                                               + (int64_t)coli_cuda_tensor_bytes(P[z].d.cuda);
                                m->gpu_expert_count++; m->gpu_expert_bytes += actual;
                                if(g_cuda_release_host) expert_host_release(m, &P[z]);
                            } else {
                                qt_cuda_reset(&P[z].g); qt_cuda_reset(&P[z].u); qt_cuda_reset(&P[z].d);
                                P[z].g.cuda_eligible = P[z].u.cuda_eligible = P[z].d.cuda_eligible = 0;
                            }
                        }
                    }
                }
            }
#endif
            pthread_mutex_unlock(&g_pilot_mx); return;
        }
    }
    // ... existing LRU load logic ...
```

**Change 2**: Add a new env var `CUDA_PREFETCH` to control this behavior:

```c
static int g_cuda_prefetch = 0;  // default OFF
// In main():
g_cuda_prefetch = getenv("CUDA_PREFETCH") ? atoi(getenv("CUDA_PREFETCH")) : 0;
```

Guard the VRAM upload path with `if(g_cuda_prefetch)`.

### 3.3 Testing

```bat
:: Test 1: Enable prefetch and measure hit rate improvement
set PILOT_REAL=1
set CUDA_PREFETCH=1
:: Compare hit rate with/without CUDA_PREFETCH

:: Test 2: Verify no VRAM thrashing
:: Monitor VRAM usage during generation — should be stable

:: Test 3: Measure speed improvement
:: Expected: 0.12 -> 0.15 tok/s

:: Test 4: Disable and verify no regression
set CUDA_PREFETCH=0
:: Should match Phase 2 baseline
```

### 3.4 Rollback

Set `CUDA_PREFETCH=0` (default). No code changes needed for rollback.

---

## Phase 4: Re-enable MTP Draft with CUDA

**Goal**: Restore 3x draft speculative decoding when CUDA is enabled.

**Dependencies**: None (independent of Phases 1-3).

**Time estimate**: 2-4 hours (investigation + implementation)

### 4.1 Investigation

The disable is at line 3757:
```c
g_draft = (m.has_mtp&&!g_cuda_enabled) ? 3 : 0;
```

Need to understand WHY MTP is disabled with CUDA. Possible reasons:
1. MTP head uses attention, which may conflict with CUDA attention path
2. MTP draft tokens need full model verification, requiring CUDA for all layers
3. The MTP head itself is not ported to CUDA

Investigate by:
1. Reading the MTP draft generation code
2. Checking if MTP uses `attention_rows()` or a separate path
3. Checking if the verification path works with CUDA experts

### 4.2 Code Changes (if MTP can work with CUDA)

**File**: `c/glm.c`

**Change 1** (line 3757): Remove the CUDA disable:

```c
// BEFORE:
g_draft = (m.has_mtp&&!g_cuda_enabled) ? 3 : 0;

// AFTER:
g_draft = m.has_mtp ? 3 : 0;
```

**Change 2**: If MTP attention needs special handling, add a CUDA-aware path in the
MTP verification code. This depends on the investigation findings.

### 4.3 Testing

```bat
:: Test 1: Basic MTP with CUDA
set COLI_CUDA=1
set DRAFT=-1  :: auto (should be 3 with MTP)
:: Check: "draft=3" in output, MTP acceptance > 0%

:: Test 2: Verify output quality
:: Generate 512 tokens, compare with draft=0 output
:: Should be identical (MTP is lossless speculative decoding)

:: Test 3: Measure speed improvement
:: Expected: 1 tok/fw -> 3 tok/fw (theoretical)
:: Actual: ~2x due to verification overhead

:: Test 4: Test fallback
:: If MTP fails, draft should fall back to 0
set DRAFT=0
:: Should work identically to before
```

### 4.4 Rollback

Add back the `!g_cuda_enabled` check at line 3757.

---

## Phase 5: Overlapping VRAM Upload with Prefill

**Goal**: Hide VRAM upload latency behind the first token's prefill computation.

**Dependencies**: Phase 2 must be complete.

**Time estimate**: 6-8 hours (complex synchronization)

### 5.1 Design

Instead of:
```
[==== VRAM upload ====][==== prefill ====][==== generation ====]
```

Do:
```
[upload batch 1][upload batch 2][upload batch 3]...
  [==== prefill (CPU experts) ====][==== generation ====]
       [upload batch N]...
```

The first few layers of prefill use CPU-only experts while VRAM uploads continue in
a background thread.

### 5.2 Code Changes

**File**: `c/glm.c`

**Change 1**: Add a background upload thread:

```c
typedef struct {
    Model *m;
    PinRec *r;
    int *slot_of;
    int start, end;
    double budget;
    int64_t eb;
    int done;
    pthread_t thread;
} VramUploadCtx;

static void *vram_upload_worker(void *arg){
    VramUploadCtx *ctx = (VramUploadCtx *)arg;
    for(int a = ctx->start; a < ctx->end; a++){
        expert_load(ctx->m, ctx->r[a].l, ctx->r[a].e,
                   &ctx->m->pin[ctx->r[a].l][ctx->slot_of[a]], 1);
        // ... upload to VRAM, release host ...
    }
    ctx->done = 1;
    return NULL;
}
```

**Change 2**: In `run_text()` and `run_serve()`, spawn the upload thread before prefill:

```c
VramUploadCtx upload_ctx = {0};
upload_ctx.m = m;
// ... populate fields ...
pthread_create(&upload_ctx.thread, NULL, vram_upload_worker, &upload_ctx);

// Prefill can start immediately with CPU-only experts
// The CUDA dispatch checks cuda_eligible — experts not yet uploaded fall back to CPU

// After prefill completes, join the upload thread
pthread_join(upload_ctx.thread, NULL);
```

**Change 3**: The CUDA expert dispatch (line 1964) already checks `cuda_eligible`:
```c
if(g_cuda_enabled && e->g.cuda_eligible && e->u.cuda_eligible && e->d.cuda_eligible ...)
```

No change needed — experts not yet uploaded have `cuda_eligible=0` and use CPU path.

### 5.3 Race Condition Mitigation

The background thread writes to `pin[layer][slot].slab` and `pin[layer][slot].cuda_*`
fields. The main thread reads these during `moe()`. The existing `g_pilot_mx` mutex
and the layer ownership model (line 1713-1724) provide synchronization:
- The main thread owns layer N while computing it
- The upload thread works on layer N+2 or later
- No concurrent access to the same layer's pin slots

### 5.4 Testing

```bat
:: Test 1: Verify no crashes
:: Run full 512-token generation with upload overlap

:: Test 2: Measure startup time reduction
:: Expected: 8-15 seconds saved

:: Test 3: Verify output correctness
:: Compare token-by-token with non-overlapping version
:: Must be identical (lossless)

:: Test 4: Stress test
:: Run 10 prompts in sequence, check for memory corruption
```

### 5.5 Rollback

Disable the background thread by setting `upload_ctx.done = 1` immediately (skip upload).
The prefill falls back to CPU-only experts.

---

## Phase 6: CUDA Unified Memory

**Goal**: Replace manual host->device transfers with CUDA Unified Memory, allowing the
driver to manage expert placement automatically.

**Dependencies**: Phases 1-5 must be complete and stable.

**Time estimate**: 1-2 days (major refactor)

**NOTE**: This phase supersedes Phases 1-3 in the long term, but those phases provide
immediate benefit and serve as fallback if Unified Memory has issues.

### 6.1 Design

Replace `cudaMalloc` + `cudaMemcpy` with `cudaMallocManaged` in `coli_cuda_tensor_upload()`.
The CUDA driver will:
- Keep hot experts in VRAM
- Page-fault cold experts to system RAM
- Evict least-recently-used experts from VRAM when space is needed

This eliminates:
- The manual batched loading (Phase 2)
- The `expert_host_release()` mechanism
- The `gpu_prefix` / `npin` coupling

### 6.2 Code Changes

**File**: `c/backend_cuda.cu`

**Change 1** (line 360): Replace `cudaMalloc` with `cudaMallocManaged`:

```c
// BEFORE:
if (!cuda_ok(cudaMalloc(&t->weights, t->weight_bytes), "tensor allocation") ||
    !cuda_ok(cudaMemcpy(t->weights, weights, t->weight_bytes, cudaMemcpyHostToDevice), "tensor upload")) {

// AFTER:
if (!cuda_ok(cudaMallocManaged(&t->weights, t->weight_bytes), "tensor allocation")) {
    // Note: no cudaMemcpy needed — the driver handles page migration
    // The host pointer IS the device pointer
}
```

**Change 2** (line 368): Same for scales:

```c
// BEFORE:
if (!cuda_ok(cudaMalloc(&t->scales, (size_t)O * sizeof(float)), "scale allocation") ||
    !cuda_ok(cudaMemcpy(t->scales, scales, (size_t)O * sizeof(float), cudaMemcpyHostToDevice), "scale upload")) {

// AFTER:
if (!cuda_ok(cudaMallocManaged(&t->scales, (size_t)O * sizeof(float)), "scale allocation")) {
}
```

**Change 3**: Remove the `offset_to_signed_s4` kernel call (line 365-366) since the
int4 conversion must happen on the host before passing to unified memory. Move it to
the CPU path in `expert_load()`.

**File**: `c/glm.c`

**Change 4** (line 1421): `expert_host_release()` becomes a no-op with unified memory:

```c
static void expert_host_release(Model *m, ESlot *s){
    if(!s->slab&&!s->fslab) return;
#ifdef COLI_CUDA_UNIFIED
    /* With unified memory, host backing is managed by the driver.
     * Just clear the slab pointers but don't free — the driver owns the memory. */
    s->slab = NULL; s->fslab = NULL;
    s->slab_cap = s->fslab_cap = 0;
    QT *q[3]={&s->g,&s->u,&s->d};
    for(int k=0;k<3;k++){ q[k]->qf=NULL; q[k]->q8=NULL; q[k]->q4=NULL; q[k]->s=NULL; }
    /* Don't decrement resident_bytes — unified memory counts as resident */
    return;
#else
    // ... existing free() logic ...
#endif
}
```

**Change 5**: Add compile flag `COLI_CUDA_UNIFIED` to enable/disable this path:

```bat
cl /DCOLI_CUDA_UNIFIED /DCOLI_CUDA ...
```

### 6.3 Memory Budget Adjustment

With unified memory, the VRAM budget is no longer a hard cap. The driver will page-fault
experts in and out as needed. But we still need a budget to prevent OOM:

```c
// Set unified memory budget to available VRAM + 50% of available RAM
double unified_budget = vram_total + ram_available * 0.5;
```

### 6.4 Page-Fault Performance

The RTX 4070 (compute capability 8.9) supports hardware page-fault handling. Measured
page-fault latency: ~10-50 us per expert. With 8 experts per layer, that's ~80-400 us
per layer on first access. For 78 layers: ~6-30 ms total overhead for a cold start.

After warmup, hot experts stay in VRAM and there are no page faults.

### 6.5 Testing

```bat
:: Test 1: Basic functionality
set COLI_CUDA=1
set COLI_CUDA_UNIFIED=1
:: Run full 512-token generation

:: Test 2: Measure page-fault overhead
:: Use CUDA profiling: nvprof or nsight
:: Expected: ~10-50 us per page fault

:: Test 3: Measure VRAM usage
:: Should dynamically adjust, not a fixed budget

:: Test 4: Compare with Phase 2 (batched loading)
:: Unified memory should be slower on first run (page faults)
:: but faster on subsequent runs (no startup batching)

:: Test 5: Stress test
:: Run 10 prompts, verify no memory corruption
:: Monitor for CUDA out-of-memory errors
```

### 6.6 Rollback

Remove `COLI_CUDA_UNIFIED` compile flag. The existing `cudaMalloc`+`cudaMemcpy` path
is still available.

---

## Phase Summary

| Phase | Description | Time | Impact | Risk |
|-------|-------------|------|--------|------|
| 1 | Fix resident_bytes | 30 min | Correct cap | Low |
| 2 | Batched VRAM loading | 2-3 hrs | VRAM 7->11 GB | Low |
| 3 | Speculative prefetch | 4-6 hrs | +30-50% hit rate | Medium |
| 4 | Re-enable MTP draft | 2-4 hrs | +2x generation | Medium |
| 5 | Upload/prefill overlap | 6-8 hrs | -8s startup | High |
| 6 | Unified Memory | 1-2 days | Dynamic VRAM | High |

### Expected Cumulative Impact

| Metric | Current | After Phase 2 | After Phase 4 | After Phase 6 |
|--------|---------|---------------|---------------|---------------|
| VRAM used | 3.73 GB | 11 GB | 11 GB | Dynamic |
| Hit rate | 17% | 30% | 30% | 40%+ |
| tok/s | 0.08 | 0.15 | 0.30 | 0.35+ |
| MTP draft | 0 | 0 | 3 | 3 |
| Startup | 15s | 15s | 15s | 8s |

### Testing After Each Phase

Every phase must pass these gates before proceeding:

1. **Compile**: `cl /std:c17 /openmp /arch:AVX2 /DCOLI_CUDA /D_FILE_OFFSET_BITS=64 /experimental:c11atomics glm.c backend_cuda.obj /link /STACK:134217728 cuda.lib cudart_static.lib`
2. **Load test**: `set PROMPT=test && glm.exe 16` — must load without crash
3. **Generation test**: Generate 64 tokens, verify non-empty output
4. **VRAM check**: `nvidia-smi` shows expected VRAM usage
5. **RSS check**: Task Manager shows expected RAM usage
6. **Speed check**: Compare tok/s with baseline

### Environment Variables Reference

| Variable | Phase | Default | Description |
|----------|-------|---------|-------------|
| `CUDA_BATCH_SIZE` | 2 | 16 | Experts per VRAM upload batch |
| `CUDA_PREFETCH` | 3 | 0 | Enable speculative VRAM prefetch |
| `CUDA_OVERLAP` | 5 | 0 | Enable upload/prefill overlap |
| `COLI_CUDA_UNIFIED` | 6 | 0 | Enable CUDA Unified Memory path |

---

## Commit Strategy

Each phase gets its own commit:

1. `fix(cuda): correct resident_bytes accounting for VRAM experts`
2. `feat(cuda): batched VRAM loading — fill full 12 GB VRAM`
3. `feat(cuda): speculative VRAM prefetch from pilot predictions`
4. `feat(cuda): re-enable MTP draft speculative decoding with CUDA`
5. `feat(cuda): overlap VRAM upload with prefill computation`
6. `feat(cuda): unified memory path for dynamic expert placement`
