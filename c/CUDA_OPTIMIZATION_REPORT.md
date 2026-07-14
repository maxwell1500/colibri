# CUDA Expert Offload Optimization Report

## Executive Summary

The GLM-5.2 Colibri engine runs a 384 GB MoE model (78 layers x 256 experts, int4) on a
single RTX 4070 (12 GB VRAM) with 32 GB system RAM. Current throughput is ~0.08 tok/s,
with 87% of time spent in disk I/O for expert loading. The CUDA expert tier pins experts
to VRAM at startup, but a fundamental design limitation constrains VRAM usage to only
~7 GB of the available 12 GB because all experts must be loaded into host RAM before
upload to GPU.

This report documents the full analysis and proposes a series of changes, ordered by
impact and implementation complexity.

---

## 1. Current Architecture

### 1.1 Expert Memory Hierarchy

```
SSD (384 GB)
  |-- pread() -->
Host RAM (LRU cache, cap=2-4 experts/layer)
  |-- cudaMemcpy --> 
GPU VRAM (pinned experts, ~7 GB / 12 GB available)
  |-- kernel -->
Compute
```

### 1.2 Startup Flow (pin_load, glm.c:3408-3514)

1. Read `.coli_usage` file (expert selection counts from prior runs)
2. Sort experts by usage count (descending)
3. Compute `npin = min(ram_budget / expert_bytes, total_experts)`
   - `ram_budget` = `expert_avail()` which uses `resident_bytes` to compute available RAM
4. Compute `gpu_prefix = min(vram_budget / expert_bytes, npin)`
   - **KEY LIMITATION**: `gpu_prefix` is capped by `npin` (line 3454)
5. Load ALL `gpu_prefix` experts into host RAM simultaneously (parallel OMP loop, line 3461-3463)
6. Increment `resident_bytes` by `gpu_prefix * expert_bytes` (line 3464)
7. Upload each expert to VRAM (sequential loop, line 3468-3497)
8. Free host backing after each upload if `COLI_CUDA_RELEASE_HOST=1` (line 3488)
9. Load remaining RAM-only experts (line 3504-3509)

### 1.3 Inference Flow (moe(), glm.c:1712-2038)

Per token, per layer:
1. Router selects top-8 experts (lines 1747-1773)
2. Batch-union: collect unique experts across all positions (lines 1790-1795)
3. For each batch of 64 unique experts (line 1808):
   a. **Resolve**: check pin[] (pinned) -> ecache[] (LRU) -> disk (miss) (lines 1811-1816)
   b. **Load misses**: OMP parallel pread from SSD (lines 1886-1890)
   c. **Compute**: CUDA grouped matmul for VRAM-resident experts (lines 1980-2025)
   d. **Promote to LRU**: swap missed experts into ecache (lines 2031-2037)

### 1.4 Runtime Expert Migration (repin_pass, glm.c:2867-2905)

Periodic pass that replaces cold pinned experts with hot experts:
- Picks top-4 candidates by heat delta
- Loads new expert into same pin slot (disk -> RAM -> VRAM)
- Releases host backing
- This is the ONLY runtime mechanism for VRAM contents to change

---

## 2. Identified Issues

### 2.1 Issue: VRAM Capped by RAM Budget (CRITICAL)

**Location**: `glm.c:3454`
```c
if(g_cuda_enabled&&g_cuda_release_host&&budget>0){
    gpu_prefix=(int)(budget/eb)+g_cuda_ndev;
    if(gpu_prefix>npin) gpu_prefix=npin;  // <-- THE BOTTLENECK
}
```

**Problem**: `gpu_prefix` (experts to upload to VRAM) is capped by `npin` (experts loaded
into host RAM). Since host RAM is shared with the dense model, KV cache, working set, and
OS page cache, only ~370 experts can be loaded simultaneously, using 7 GB of the 12 GB
available VRAM.

**Impact**: 5 GB VRAM wasted. With 370 experts (7 GB), each layer gets ~4.7 VRAM experts
but needs 8. The remaining 3.3 must come from disk every token.

**Root Cause**: The "load-all-then-upload" pattern requires ALL experts to be resident in
host RAM before any are uploaded. `COLI_CUDA_RELEASE_HOST` frees host RAM *after* upload,
but the initial peak is unavoidable.

### 2.2 Issue: resident_bytes Bookkeeping Error (MODERATE)

**Location**: `glm.c:3464`
```c
m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:npin)*eb;
```

**Problem**: This increments `resident_bytes` for ALL loaded experts, including those that
will be uploaded to VRAM and have their host backing released. After `expert_host_release()`
(line 1431) correctly decrements `resident_bytes`, the counter is accurate *eventually*. But
`cap_for_ram()` (line 3793) is called AFTER `pin_load()` (line 3793), and uses the
`resident_bytes` value that was already decremented by `expert_host_release()`. So this
particular path may actually be correct after release.

However, the **real problem** is that `cap_for_ram()` computes available RAM for the LRU
cache based on `resident_bytes`, which now includes the VRAM-pinned experts' host backing
*during* the pin_load phase (before release). The cap is computed from a snapshot that
includes transient allocations.

**Impact**: Cap is 1-2 slots lower than optimal. With cap=2 instead of cap=4, each layer
only caches 2 LRU experts, forcing ~6 disk reads per layer per token instead of ~4.

### 2.3 Issue: No Overlap Between Expert Upload and Compute (MODERATE)

**Location**: `glm.c:3461-3497`

**Problem**: The startup flow is strictly sequential:
1. Load ALL experts to host RAM (OMP parallel, fast)
2. Upload ALL experts to VRAM (sequential, slow - involves cudaMalloc + cudaMemcpy per expert)
3. Only THEN does inference begin

There is no overlap between VRAM uploads and any useful work. On a 384 GB model with ~370
experts to upload, this adds ~8-15 seconds to startup with no benefit.

### 2.4 Issue: No Runtime VRAM Eviction (MINOR)

**Location**: `repin_pass()` (line 2867)

**Problem**: The only mechanism for changing VRAM contents at runtime is `repin_pass()`,
which replaces cold pinned experts with hot ones. But it has limitations:
- Only runs every N tokens (controlled by `REPINE` env var)
- Only replaces 4 experts per pass (hardcoded)
- Cannot grow the total number of VRAM experts beyond the initial `gpu_prefix`
- Cannot reclaim VRAM from experts that became cold after initial pin

### 2.5 Issue: MTP Draft Disabled with CUDA (MINOR)

**Location**: `glm.c:3757`
```c
g_draft = (m.has_mtp&&!g_cuda_enabled) ? 3 : 0;
```

**Problem**: MTP (Multi-Token Prediction) draft speculative decoding is disabled when CUDA
is enabled. This halves generation throughput (1 token/forward instead of 3).

**Root Cause**: The MTP head requires attention computation that may not be compatible with
the CUDA attention path. Not investigated in detail.

---

## 3. Proposed Improvements

### 3.1 Batched VRAM Loading (HIGH IMPACT, MEDIUM COMPLEXITY)

**Goal**: Fill VRAM to 11 GB without requiring all experts in host RAM simultaneously.

**Design**:
Instead of the current pattern:
```
load [N experts] -> host RAM -> upload all to VRAM -> free host
```

Use a sliding-window pattern:
```
for each expert in sorted list:
    load expert -> host RAM (small batch: 10-20 experts)
    upload to VRAM
    free host backing
    if VRAM full: break
```

**Changes Required**:

1. **Split the parallel load into batches** (`glm.c:3461-3463`):
```c
// CURRENT: load all gpu_prefix experts at once
#pragma omp parallel for schedule(dynamic,1)
for(a=0;a<gpu_prefix;a++)
    expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1);

// PROPOSED: load in batches of BATCH_SIZE
int BATCH_SIZE = 16;  // tune based on expert size (~19 MB each -> ~300 MB batch)
for(int batch_start=0; batch_start<gpu_prefix; batch_start+=BATCH_SIZE){
    int batch_end = min(batch_start+BATCH_SIZE, gpu_prefix);
    #pragma omp parallel for schedule(dynamic,1)
    for(a=batch_start; a<batch_end; a++)
        expert_load(m,r[a].l,r[a].e,&m->pin[r[a].l][slot_of[a]],1);
    m->resident_bytes += (int64_t)(batch_end-batch_start)*eb;
    
    // Upload this batch to VRAM
    for(a=batch_start; a<gpu_limit && m->gpu_expert_bytes<budget; a++){
        // ... existing upload logic ...
        if(g_cuda_release_host) expert_host_release(m,s);
    }
    // After upload, resident_bytes is decremented by expert_host_release
}
```

2. **Adjust `gpu_prefix` calculation** to not depend on `npin`:
```c
// CURRENT
if(gpu_prefix>npin) gpu_prefix=npin;

// PROPOSED: gpu_prefix is purely VRAM-budget-driven
gpu_prefix = (int)(budget/eb) + g_cuda_ndev;
// npin remains RAM-driven for the non-VRAM experts
```

3. **Update `resident_bytes` accounting**: The counter should be decremented after each
   batch upload + release, not accumulated at the start.

**Expected Impact**:
- VRAM usage: 7 GB -> 11 GB (57% more experts)
- VRAM experts per layer: 4.7 -> 7.0
- Disk reads per token: ~3.3 per layer -> ~1.0 per layer
- Overall speed: ~0.08 tok/s -> ~0.12-0.15 tok/s (estimated 50-80% improvement)

**Risk**: Slightly slower startup (more cudaMalloc calls), but well under 1 second total.

### 3.2 Fix resident_bytes Accounting (HIGH IMPACT, LOW COMPLEXITY)

**Goal**: Allow `cap_for_ram()` to compute the correct LRU cache cap.

**Changes Required**:

1. **Decouple VRAM expert counting from `resident_bytes`** (`glm.c:3464`):
```c
// CURRENT: counts VRAM experts in resident_bytes
m->resident_bytes+=(int64_t)(gpu_prefix?gpu_prefix:npin)*eb;

// PROPOSED: only count RAM-only experts in resident_bytes
int ram_only = npin - m->gpu_expert_count;  // after VRAM upload
m->resident_bytes += (int64_t)ram_only * eb;
```

2. **Move `cap_for_ram()` call after `pin_load()` completes** and VRAM experts are
   released from host. This is already the case (line 3793), but ensure
   `resident_bytes` reflects the post-release state.

**Expected Impact**:
- LRU cache cap: 2 -> 4-5 (doubles the per-layer expert cache)
- Hit rate: 17% -> 25-30%
- Speed: ~0.08 tok/s -> ~0.10 tok/s

### 3.3 Overlapping VRAM Upload with Prefill (LOW IMPACT, HIGH COMPLEXITY)

**Goal**: Hide VRAM upload latency behind the first token's prefill computation.

**Design**:
- Start uploading VRAM experts in a background thread
- Begin prefill computation with CPU-only experts for the first few layers
- As VRAM uploads complete, switch those layers to CUDA

**Changes Required**:
1. Spawn upload thread before `run_text()` or `run_serve()`
2. During prefill, check `cuda_eligible` flag per expert per layer
3. Layer N uses CUDA only if its experts are uploaded

**Expected Impact**:
- Startup latency reduced by ~8 seconds (upload time hidden)
- No throughput improvement during steady-state generation

**Risk**: Complex synchronization, may introduce race conditions in `expert_load()`.

### 3.4 Runtime VRAM Expansion via CUDA Unified Memory (MEDIUM IMPACT, HIGH COMPLEXITY)

**Goal**: Allow the VRAM expert set to grow beyond the initial `gpu_prefix` during inference.

**Design**:
- Use CUDA Unified Memory (`cudaMallocManaged`) for expert tensors
- Let the CUDA driver page-fault experts into VRAM on demand
- The OS/driver handles eviction automatically

**Changes Required**:
1. Replace `cudaMalloc` + `cudaMemcpy` with `cudaMallocManaged` in `coli_cuda_tensor_upload()`
2. Remove `expert_host_release()` — unified memory handles the host/GPU mapping
3. Adjust memory budget calculations to account for unified memory overhead

**Expected Impact**:
- VRAM usage: dynamically adjusts to actual demand
- Hot experts naturally migrate to VRAM, cold experts page out
- No startup batching needed

**Risk**: 
- Page-fault overhead on first access (~10-50 us per fault)
- May cause VRAM thrashing if too many experts compete for space
- Requires CUDA compute capability 6.0+ (RTX 4070 is 8.9, OK)

### 3.5 Speculative VRAM Prefetch for Next-Layer Experts (MEDIUM IMPACT, MEDIUM COMPLEXITY)

**Goal**: Predict which experts will be needed in the next layer and prefetch them to VRAM.

**Design**:
- The pilot (layer N-1) already predicts routing for layer N
- Use this prediction to prefetch experts from layer N into VRAM before the compute reaches that layer
- If the expert is already in VRAM (pinned), skip. If in RAM, upload. If on disk, load+upload.

**Changes Required**:
1. Extend the pilot system to output predicted expert IDs for the next 2-3 layers
2. In `repin_pass()`, add a prefetch path that uploads predicted experts to VRAM
3. Limit prefetch to maintain a VRAM headroom budget (e.g., keep 2 GB free)

**Expected Impact**:
- Disk reads during inference reduced by ~30-50%
- Speed: ~0.08 tok/s -> ~0.12 tok/s

### 3.6 Re-enable MTP Draft with CUDA (MEDIUM IMPACT, UNKNOWN COMPLEXITY)

**Goal**: Restore 3x draft speculative decoding when CUDA is enabled.

**Current Status**: Disabled at line 3757. The MTP head likely requires the attention
computation to be available on CPU, which conflicts with the CUDA attention path.

**Investigation Needed**:
1. Determine if MTP head can use CPU attention while routed experts use CUDA
2. Check if the MTP draft tokens need to be verified through the full model (requiring
   CUDA for all layers)
3. If MTP can be CPU-only, it may already work — the disable flag may be overly conservative

**Expected Impact**:
- Generation speed: 1 token/forward -> 3 tokens/forward (theoretical 3x)
- Actual: ~2x due to verification overhead and acceptance rate

---

## 4. Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. **Fix resident_bytes accounting** (3.2) — simple counter fix
2. **Batched VRAM loading** (3.1) — moderate refactor of `pin_load()`
3. Test with `CUDA_EXPERT_GB=11` and verify VRAM usage reaches 11 GB

### Phase 2: Runtime Optimization (3-5 days)
4. **Speculative VRAM prefetch** (3.5) — extend pilot system
5. **Profile and tune batch size** for optimal startup vs throughput tradeoff
6. **Investigate MTP re-enablement** (3.6)

### Phase 3: Architectural Improvements (1-2 weeks)
7. **CUDA Unified Memory** (3.4) — major refactor but best long-term solution
8. **Overlapping upload with prefill** (3.3) — if startup latency matters

---

## 5. Benchmarking Plan

### Current Baseline
```
Model: GLM-5.2 int4 (384 GB, 78 layers, 256 experts)
Hardware: RTX 4070 (12 GB VRAM) + 32 GB RAM + NVMe SSD
Settings: CUDA_EXPERT_GB=8, OMP_NUM_THREADS=20, ABSORB=1

Results:
- VRAM: 3.73 GB (197 experts)
- Hit rate: 14-17%
- Speed: 0.08 tok/s
- Disk I/O: 87% of total time
- Cap: 2 experts/layer LRU
```

### Target After Phase 1
```
Expected:
- VRAM: 11 GB (~580 experts)
- Hit rate: 25-30%
- Speed: 0.12-0.15 tok/s
- Cap: 4-5 experts/layer LRU
```

### Target After Phase 2
```
Expected:
- VRAM: 11 GB (dynamic)
- Hit rate: 35-40%
- Speed: 0.15-0.20 tok/s
- MTP: 3x draft enabled
```

---

## 6. Risk Assessment

| Change | Risk | Mitigation |
|--------|------|------------|
| Batched VRAM load | Low | Well-isolated in `pin_load()`, no runtime impact |
| resident_bytes fix | Low | Pure bookkeeping, verify with RSS monitoring |
| Speculative prefetch | Medium | Add env var to disable, verify no regression |
| MTP re-enable | Medium | Start with `MTP_DRAFT=1` env var, fallback to 0 |
| Unified Memory | High | Requires extensive testing, may hit page-fault overhead |

---

## 7. Files to Modify

| File | Changes | Lines Affected |
|------|---------|----------------|
| `c/glm.c` | Batched VRAM load, resident_bytes fix, prefetch | ~3408-3514, ~2867-2905, ~1892-1902 |
| `c/backend_cuda.cu` | Unified Memory path (Phase 3) | ~400-470 |
| `c/compat.h` | Nothing needed | - |
| `c/PROGRESS.md` | Document changes | Append |

---

## 8. Conclusion

The most impactful change is **batched VRAM loading** (3.1), which directly addresses the
5 GB wasted VRAM. Combined with the **resident_bytes fix** (3.2), this should improve
throughput by 50-80% with moderate code changes.

The long-term solution is **CUDA Unified Memory** (3.4), which eliminates the manual
host->device transfer entirely and allows the CUDA driver to manage expert placement
automatically.

Both changes are safe, well-scoped, and can be tested incrementally without affecting
CPU-only or Metal backends.
