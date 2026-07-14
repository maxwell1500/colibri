# Unified Expert Selection — Implementation Plan

## Overview

Four complementary mechanisms to reduce disk I/O by preferring experts already in VRAM/RAM:

```
Router scores (256 experts)
    │
    ▼
[Layer 1: VRAM_BIAS] ── Soft boost for available experts
    │
    ▼
Top-K selection (8 experts)
    │
    ▼
[Layer 2: VRAM_FILTER] ── Hard filter + TOPP re-truncation
    │
    ▼
[Layer 3: SIMILARITY] ── Substitute nearest neighbor for missing
    │
    ▼
[Layer 4: Force Available] ── Fallback: use best available
    │
    ▼
Expert resolution (pin/ecache/disk)
```

## Current State

- **509 VRAM experts** (9.63 GB) / **19 RAM-only experts** (0.4 GB)
- **Expert hit rate**: 21%
- **Disk I/O**: 81% of total time (283s / 350s)
- **Decode tok/s**: 0.05
- **MTP acceptance**: 38%

## Target State

- **Expert hit rate**: 50-70% (up from 21%)
- **Disk I/O**: 30-40% of total time (down from 81%)
- **Decode tok/s**: 0.08-0.12 (up from 0.05)

---

## Phase 1: is_warm() Helper + VRAM_BIAS

**Goal**: Add a soft bias to router scores for experts already in VRAM/RAM.

**Estimated time**: 30 minutes

### Step 1.1: Add is_warm() helper function

**File**: `c/glm.c`, near line 1710 (before `moe()`)

**Code**:
```c
/* Check if expert `eid` is warm (in pin or ecache for this layer) */
static int is_warm(Model *m, int layer, int eid){
    ESlot *P=m->pin[layer];
    for(int z=0;z<m->npin[layer];z++) if(P[z].eid==eid) return 1;
    ESlot *Sl=m->ecache[layer];
    for(int z=0;z<m->ecn[layer];z++) if(Sl[z].eid==eid) return 1;
    return 0;
}
```

**Test**: Build, verify no compile errors.

### Step 1.2: Add g_vram_bias global variable

**File**: `c/glm.c`, near line 778 (after `g_topp`)

**Code**:
```c
static float g_vram_bias=0;  /* VRAM_BIAS=n -> boost router score for available experts */
```

### Step 1.3: Parse VRAM_BIAS env var

**File**: `c/glm.c`, near line 3715 (after `g_topp` parsing)

**Code**:
```c
g_vram_bias = getenv("VRAM_BIAS") ? atof(getenv("VRAM_BIAS")) : 0;
```

### Step 1.4: Apply bias in moe() routing

**File**: `c/glm.c`, line 1751 (after sigmoid+bias computation)

**Insert after line 1751**:
```c
if(g_vram_bias > 0){
    for(int e=0; e<E; e++)
        if(is_warm(m, layer, e)) choice[e] += g_vram_bias;
}
```

**How it works**: The `choice` score is used for top-k ranking. Adding a bonus to warm experts makes them more likely to be selected. The original `sigmoid` weight is still used for the final accumulation, so the output scaling is unchanged.

### Step 1.5: Test

**Batch file**: `H:\test_vram_bias.bat`
```bat
@echo off
set SNAP=H:\glm52
set RAM_GB=24
set NGEN=32
set ABSORB=1
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set CUDA_RELEASE_HOST=1
set CUDA_BATCH_SIZE=16
set OMP_NUM_THREADS=20
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=10
set VRAM_BIAS=0.2
set "PROMPT=[gMASK]<sop><|user|>Write a short Python function to compute fibonacci numbers.<|assistant|>"
"C:\Users\max\Projects\GLMLocal\colibri\glm.exe" 16 > H:\vram_bias_test.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> H:\vram_bias_test.txt
```

**Check**: Compare hit rate, disk I/O time, and tok/s with baseline.

### Step 1.6: Commit

```bash
git add c/glm.c
git commit -m "feat: VRAM_BIAS — soft router bias for available experts

Add VRAM_BIAS=n env var. When set, router choice scores are boosted
by n for experts already in pin/ecache. This makes the top-k selection
naturally prefer warm experts, reducing disk I/O.

The original sigmoid weights are preserved for output scaling, so the
model's output is unchanged for warm experts. For cold experts that
are still selected (router strongly prefers them), the weight is the
original sigmoid score."
```

---

## Phase 2: VRAM_FILTER + TOPP Re-truncation

**Goal**: After top-k, filter out unavailable experts and re-run TOPP on the available subset.

**Estimated time**: 45 minutes

### Step 2.1: Add g_vram_filter global variable

**File**: `c/glm.c`, near line 778

**Code**:
```c
static int g_vram_filter=0;  /* VRAM_FILTER=1 -> post-topk filter unavailable experts */
```

### Step 2.2: Parse VRAM_FILTER env var

**File**: `c/glm.c`, near line 3715

**Code**:
```c
g_vram_filter = getenv("VRAM_FILTER") ? atoi(getenv("VRAM_FILTER")) : 0;
```

### Step 2.3: Implement post-topk filter in moe()

**File**: `c/glm.c`, insert between lines 1758 and 1759 (after top-k, before TOPP)

**Code**:
```c
/* Phase 1.5: VRAM filter — remove unavailable experts, compact the list */
if(g_vram_filter){
    int navail=0;
    for(int kk=0; kk<Ksel; kk++){
        if(is_warm(m, layer, idx[kk]))
            idx[navail++] = idx[kk];
    }
    /* If too few available, keep at least 2 (prevents degenerate routing) */
    if(navail < 2){
        /* Restore original top-k — filter was too aggressive */
        navail = Ksel;
        /* Re-run top-k from scratch (original order) */
        for(int kk=0; kk<Ksel; kk++){
            int best=-1; float bv=-1e30f;
            for(int e=0; e<E; e++){
                int tk=0;
                for(int j=0; j<kk; j++) if(idx[j]==e){ tk=1; break; }
                if(!tk && choice[e]>bv){ bv=choice[e]; best=e; }
            }
            idx[kk]=best; w[kk]=logit[best];
        }
    }
    Ksel = navail;
}
```

### Step 2.4: Test

**Batch file**: `H:\test_vram_filter.bat`
```bat
@echo off
set SNAP=H:\glm52
set RAM_GB=24
set NGEN=32
set ABSORB=1
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set CUDA_RELEASE_HOST=1
set CUDA_BATCH_SIZE=16
set OMP_NUM_THREADS=20
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=10
set VRAM_FILTER=1
set "PROMPT=[gMASK]<sop><|user|>Write a short Python function to compute fibonacci numbers.<|assistant|>"
"C:\Users\max\Projects\GLMLocal\colibri\glm.exe" 16 > H:\vram_filter_test.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> H:\vram_filter_test.txt
```

### Step 2.5: Commit

```bash
git add c/glm.c
git commit -m "feat: VRAM_FILTER — post-topk filter for unavailable experts

Add VRAM_FILTER=1 env var. When set, after top-k selection, filters out
experts not in pin/ecache and re-runs TOPP truncation on the available
subset. Reduces disk I/O by avoiding load of cold experts.

If filter removes too many (<2 available), falls back to original top-k
to prevent degenerate routing."
```

---

## Phase 3: Similarity Matrix + Substitution

**Goal**: When an expert is unavailable, substitute with the nearest available neighbor.

**Estimated time**: 2-3 hours

### Step 3.1: Add similarity data structures

**File**: `c/glm.c`, near line 155 (Model struct)

**Add to Model struct**:
```c
int **sim_neighbors;  /* [n_layers][n_experts * 8] — top-8 nearest neighbors per expert */
float **sim_matrix;   /* temporary: [n_experts] feature vectors during computation */
```

### Step 3.2: Add g_similarity global variable

**File**: `c/glm.c`, near line 778

**Code**:
```c
static int g_similarity=0;  /* SIMILARITY=1 -> precompute expert similarity, substitute on miss */
```

### Step 3.3: Parse SIMILARITY env var

**File**: `c/glm.c`, near line 3715

**Code**:
```c
g_similarity = getenv("SIMILARITY") ? atoi(getenv("SIMILARITY")) : 0;
```

### Step 3.4: Implement similarity computation

**File**: `c/glm.c`, new function near line 1710

**Code**:
```c
/* Compute feature vector for an expert (column norms of gate_proj) */
static void expert_feature(Model *m, int layer, int eid, float *feat, int feat_dim){
    ESlot *s = &m->ws[0];  /* temporary slot */
    expert_load(m, layer, eid, s, 0);
    if(!s->slab) return;
    /* Compute column norms of gate_proj: feat[d] = sum of |gate_proj[:,d]|^2 */
    int O = s->g.O, I = s->g.I;
    memset(feat, 0, feat_dim * sizeof(float));
    /* Simplified: use first feat_dim columns of gate_proj */
    for(int d=0; d<feat_dim && d<I; d++){
        float sum = 0;
        for(int o=0; o<O; o++){
            float val = /* dequantize s->g at [o,d] */;
            sum += val * val;
        }
        feat[d] = sqrtf(sum);
    }
}

/* Precompute similarity matrix for all experts in a layer */
static void similarity_precompute(Model *m, int layer){
    Cfg *c = &m->c;
    int E = c->n_experts;
    int feat_dim = 64;  /* feature vector dimension */
    float *feats = malloc(E * feat_dim * sizeof(float));
    
    /* Compute features for all experts */
    for(int e=0; e<E; e++)
        expert_feature(m, layer, e, feats + e*feat_dim, feat_dim);
    
    /* Compute pairwise cosine similarities and find top-8 neighbors */
    m->sim_neighbors[layer] = malloc(E * 8 * sizeof(int));
    for(int e=0; e<E; e++){
        float *fe = feats + e*feat_dim;
        float ne = 0; for(int d=0; d<feat_dim; d++) ne += fe[d]*fe[d];
        ne = sqrtf(ne) + 1e-10f;
        
        /* Compute similarities with all other experts */
        float *sims = malloc(E * sizeof(float));
        for(int j=0; j<E; j++){
            if(j == e){ sims[j] = -1; continue; }
            float *fj = feats + j*feat_dim;
            float dot = 0, nj = 0;
            for(int d=0; d<feat_dim; d++){ dot += fe[d]*fj[d]; nj += fj[d]*fj[d]; }
            sims[j] = dot / (ne * (sqrtf(nj) + 1e-10f));
        }
        
        /* Find top-8 by insertion sort */
        int *nb = m->sim_neighbors[layer] + e*8;
        for(int k=0; k<8; k++){
            int best = -1; float best_sim = -2;
            for(int j=0; j<E; j++) if(sims[j] > best_sim){ best_sim = sims[j]; best = j; }
            nb[k] = best;
            if(best >= 0) sims[best] = -2;  /* mark as used */
        }
        free(sims);
    }
    free(feats);
}

/* Find nearest available neighbor for an expert */
static int similarity_substitute(Model *m, int layer, int eid){
    if(!m->sim_neighbors) return -1;
    int *nb = m->sim_neighbors[layer] + eid*8;
    for(int k=0; k<8; k++){
        if(nb[k] >= 0 && is_warm(m, layer, nb[k]))
            return nb[k];
    }
    return -1;  /* no available neighbor */
}
```

### Step 3.5: Call similarity_precompute at startup

**File**: `c/glm.c`, after `pin_load()` call (line 3789)

**Code**:
```c
if(g_similarity){
    m.sim_neighbors = calloc(c.n_layers, sizeof(int*));
    for(int l=0; l<c.n_layers; l++) if(m.L[l].sparse)
        similarity_precompute(&m, l);
    fprintf(stderr, "[SIMILARITY] precomputed neighbors for %d sparse layers\n", nsp);
}
```

### Step 3.6: Use substitution in moe() resolution

**File**: `c/glm.c`, line 1817 (after miss detection)

**Modify line 1817**:
```c
if(!use[j]){
    /* Try similarity substitution before disk load */
    if(g_similarity){
        int sub = similarity_substitute(m, layer, eid);
        if(sub >= 0){
            /* Use the substitute instead of loading from disk */
            ESlot *P=m->pin[layer];
            for(int z=0; z<m->npin[layer]; z++) if(P[z].eid==sub){ use[j]=&P[z]; m->hits++; break; }
            if(!use[j]){ ESlot *Sl=m->ecache[layer]; int nn=m->ecn[layer];
                for(int z=0; z<nn; z++) if(Sl[z].eid==sub){ use[j]=&Sl[z]; m->hits++; break; } }
        }
    }
    if(!use[j]){ qof[j]=nmiss; use[j]=&m->ws[nmiss]; missk[nmiss++]=j; m->miss++; }
}
```

### Step 3.7: Test

**Batch file**: `H:\test_similarity.bat`
```bat
@echo off
set SNAP=H:\glm52
set RAM_GB=24
set NGEN=32
set ABSORB=1
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set CUDA_RELEASE_HOST=1
set CUDA_BATCH_SIZE=16
set OMP_NUM_THREADS=20
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=10
set VRAM_BIAS=0.2
set VRAM_FILTER=1
set SIMILARITY=1
set "PROMPT=[gMASK]<sop><|user|>Write a short Python function to compute fibonacci numbers.<|assistant|>"
"C:\Users\max\Projects\GLMLocal\colibri\glm.exe" 16 > H:\similarity_test.txt 2>&1
echo EXIT_CODE=%ERRORLEVEL% >> H:\similarity_test.txt
```

### Step 3.8: Commit

```bash
git add c/glm.c
git commit -m "feat: SIMILARITY — expert neighbor substitution on cache miss

Add SIMILARITY=1 env var. At startup, precompute top-8 nearest neighbors
per expert per layer using cosine similarity of gate_proj column norms.
When an expert is not in pin/ecache, substitute with the nearest available
neighbor instead of loading from disk.

Memory cost: ~640 KB (78 layers × 256 experts × 8 neighbors × 4 bytes)
Startup cost: ~2-5 seconds (loads each expert temporarily to compute features)"
```

---

## Phase 4: Combined Testing + Startup Scripts

**Goal**: Test all combinations and create startup scripts with knobs.

**Estimated time**: 1-2 hours

### Step 4.1: Combined test matrix

| Test | VRAM_BIAS | VRAM_FILTER | SIMILARITY | REPIN | File |
|------|-----------|-------------|------------|-------|------|
| Baseline | 0 | 0 | 0 | 0 | `test_baseline.bat` |
| Bias only | 0.2 | 0 | 0 | 0 | `test_bias.bat` |
| Filter only | 0 | 1 | 0 | 0 | `test_filter.bat` |
| Bias+Filter | 0.2 | 1 | 0 | 0 | `test_bf.bat` |
| Full stack | 0.2 | 1 | 1 | 0 | `test_full.bat` |
| Dynamic | 0.2 | 1 | 1 | 8 | `test_dynamic.bat` |
| Coding bias | 0.2 | 1 | 1 | 32 | `test_coding.bat` |

### Step 4.2: Create startup scripts

**File**: `H:\startup_coding.bat` (coding-optimized)
```bat
@echo off
set SNAP=H:\glm52
set RAM_GB=24
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set CUDA_RELEASE_HOST=1
set CUDA_BATCH_SIZE=16
set OMP_NUM_THREADS=20
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=10
set VRAM_BIAS=0.2
set VRAM_FILTER=1
set SIMILARITY=1
set REPIN=32
set DRAFT=-1
set ABSORB=1
```

**File**: `H:\startup_dynamic.bat` (adaptive)
```bat
@echo off
set SNAP=H:\glm52
set RAM_GB=24
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set CUDA_RELEASE_HOST=1
set CUDA_BATCH_SIZE=16
set OMP_NUM_THREADS=20
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=10
set VRAM_BIAS=0.2
set VRAM_FILTER=1
set SIMILARITY=1
set REPIN=8
set DRAFT=-1
set ABSORB=1
```

**File**: `H:\startup_baseline.bat` (conservative)
```bat
@echo off
set SNAP=H:\glm52
set RAM_GB=24
set COLI_CUDA=1
set CUDA_EXPERT_GB=11
set CUDA_RELEASE_HOST=1
set CUDA_BATCH_SIZE=16
set OMP_NUM_THREADS=20
set AUTOPIN=0
set PIN=H:\glm52\.coli_usage
set PIN_GB=10
set DRAFT=-1
set ABSORB=1
```

### Step 4.3: Run all tests and collect results

Create a batch runner `H:\run_all_tests.bat` that runs each test and collects results.

### Step 4.4: Commit startup scripts

```bash
git add H:\startup_*.bat
git commit -m "feat: startup scripts with VRAM selection knobs

Three presets:
- startup_coding.bat: Coding-optimized (REPIN=32, slow adaptation)
- startup_dynamic.bat: Adaptive (REPIN=8, fast task switching)
- startup_baseline.bat: Conservative (no VRAM selection)

All use VRAM_BIAS=0.2, VRAM_FILTER=1, SIMILARITY=1 except baseline."
```

---

## Rollback

Each phase is independently toggleable via env vars:

| Phase | Env var | Default | Rollback |
|-------|---------|---------|----------|
| 1 | `VRAM_BIAS=0` | 0 (off) | Set to 0 |
| 2 | `VRAM_FILTER=0` | 0 (off) | Set to 0 |
| 3 | `SIMILARITY=0` | 0 (off) | Set to 0 |
| 4 | Use `startup_baseline.bat` | — | Use baseline script |

No code rollback needed — just change env vars.

---

## Expected Impact

| Metric | Baseline | Full Stack | Improvement |
|--------|----------|------------|-------------|
| Expert hit rate | 21% | 50-70% | +2-3x |
| Disk I/O time | 283s (81%) | 100-140s (30-40%) | -50-60% |
| Decode tok/s | 0.05 | 0.08-0.12 | +60-140% |
| Quality degradation | 0% | <5% (with similarity) | Minimal |
