# GLM-5.2 Architecture Deep Dive — Performance Optimization Analysis

## Model Dimensions

| Parameter | Value |
|-----------|-------|
| hidden_size (D) | 6144 |
| num_attention_heads (H) | 64 |
| q_lora_rank | 2048 |
| kv_lora_rank (K) | 512 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim (R) | 64 |
| v_head_dim (vh) | 256 |
| n_routed_experts (E) | 256 per MoE layer |
| num_experts_per_tok (K) | 8 (top-k routing) |
| n_shared_experts | 1 per MoE layer |
| moe_intermediate_size (I) | 2048 |
| num_hidden_layers | 78 (3 dense + 75 MoE + 1 MTP) |
| expert bytes (int4) | 18.04 MB per expert |

## Full Decode Forward Pass (Single Token)

```
embed_row(x, token)                          [CPU]
├── for each layer (0..77):
│   ├── rmsnorm(x, in_ln)                    [CPU]
│   ├── attention_rows(x)                     [CPU or CUDA]
│   │   ├── matmul_qt(qresid, x, q_a)        [CPU/CUDA]
│   │   ├── rmsnorm(qresid, q_a_ln)          [CPU]
│   │   ├── matmul_qt(qfull, qresid, q_b)    [CPU/CUDA]
│   │   ├── rope_interleave(q_rope)           [CPU]
│   │   ├── matmul_qt(comp, x, kv_a)          [CPU/CUDA]
│   │   ├── store KV cache (latent + k_rot)   [CPU]
│   │   ├── weight absorption attention       [CPU]
│   │   │   ├── q_abs = q_nope · kv_b rows    [CPU]
│   │   │   ├── score = q_abs·Lc + q_rope·Rc  [CPU]
│   │   │   ├── softmax(scores)               [CPU]
│   │   │   ├── context = softmax · Lc         [CPU]
│   │   │   └── V_out = kv_b · context         [CPU]
│   │   └── matmul_qt(out, ctx, o_proj)       [CPU/CUDA]
│   ├── residual_add(x, attn_output)          [CPU]
│   ├── rmsnorm(x, post_ln)                   [CPU]
│   ├── if sparse: moe(x)                     [CPU or CUDA]
│   │   ├── router: matmul(logit, x, router)  [CPU]
│   │   ├── sigmoid + top-k selection          [CPU]
│   │   ├── for each expert (8 per token):
│   │   │   ├── resolve (pin/ecache/disk)      [CPU]
│   │   │   ├── load from disk if needed       [Disk I/O]
│   │   │   ├── expert_gate_up() + silu        [CPU/CUDA]
│   │   │   └── expert_down()                  [CPU/CUDA]
│   │   └── shared expert: gate+up+silu+down   [CPU/CUDA]
│   └── residual_add(x, mlp_output)           [CPU]
├── rmsnorm(x, final_norm)                    [CPU]
└── matmul_qt(logit, x, lm_head)              [CPU/CUDA]
```

## Dense Tensor Sizes

### Per-Layer Attention Tensors (int8)

| Tensor | Shape [O,I] | int8 bytes | int4 bytes |
|--------|-------------|------------|------------|
| q_a_proj | [2048, 6144] | 12.01 MB | 6.01 MB |
| q_b_proj | [16384, 2048] | 32.05 MB | 16.07 MB |
| kv_a_proj | [576, 6144] | 3.38 MB | 1.69 MB |
| kv_b_proj | [28672, 512] | 14.11 MB | 7.11 MB |
| o_proj | [6144, 16384] | 96.03 MB | 48.03 MB |
| **Attention subtotal** | | **157.58 MB** | **78.91 MB** |

### Per-Dense-Layer MLP (int8)

| Tensor | Shape [O,I] | int8 bytes | int4 bytes |
|--------|-------------|------------|------------|
| gate_proj | [18432, 6144] | 108.13 MB | 54.08 MB |
| up_proj | [18432, 6144] | 108.13 MB | 54.08 MB |
| down_proj | [6144, 18432] | 108.07 MB | 54.03 MB |
| **Dense MLP subtotal** | | **324.33 MB** | **162.19 MB** |

### Per-Sparse-Layer Shared Expert (int8)

| Tensor | Shape [O,I] | int8 bytes | int4 bytes |
|--------|-------------|------------|------------|
| sh_gate_proj | [2048, 6144] | 12.01 MB | 6.01 MB |
| sh_up_proj | [2048, 6144] | 12.01 MB | 6.01 MB |
| sh_down_proj | [6144, 2048] | 11.98 MB | 6.02 MB |
| **Shared expert subtotal** | | **36.00 MB** | **18.04 MB** |

### Embedding + LM Head

| Tensor | Shape | int4 bytes | f32 bytes |
|--------|-------|------------|-----------|
| embed_tokens | [151552, 6144] | 444.5 MB | 3,552.5 MB |
| lm_head | [151552, 6144] | 444.5 MB | 3,552.5 MB |
| **Subtotal** | | **889.0 MB** | **7,105.0 MB** |

### Total Dense Tensor VRAM

| Precision | All Dense Tensors |
|-----------|-------------------|
| int4 | **8.55 GB** |
| int8 | **22.20 GB** |

### Attention-Only (All 78 Layers)

| Precision | Total |
|-----------|-------|
| int4 | **5.87 GB** |
| int8 | 11.72 GB |

## Expert Sizes

### One Expert (int4)

| Matrix | Shape | Weights | Scales | Total |
|--------|-------|---------|--------|-------|
| gate_proj | [2048, 6144] | 6.0 MB | 8 KB | 6.01 MB |
| up_proj | [2048, 6144] | 6.0 MB | 8 KB | 6.01 MB |
| down_proj | [6144, 2048] | 6.0 MB | 24 KB | 6.02 MB |
| **TOTAL** | | | | **18.04 MB** |

### All Experts on Disk

| Component | Count | Total |
|-----------|-------|-------|
| 75 MoE layers × 256 experts | 19,200 | 338.2 GB |
| MTP layer × 256 experts | 256 | 4.51 GB |
| **Grand total** | **19,456** | **342.7 GB** |

### VRAM Expert Budget (12 GB RTX 4070)

| Configuration | VRAM Used | Free |
|---------------|-----------|------|
| 509 VRAM experts | 8.80 GB | 3.20 GB |
| 170 VRAM experts | 3.07 GB | 8.93 GB |
| Attention tensors (int4) | 5.87 GB | 6.13 GB |
| 2 full MoE layers | 9.20 GB | 2.80 GB |

## MLA (Multi-head Latent Attention)

### KV Cache Per Token Per Layer

| Component | Size |
|-----------|------|
| Lc (compressed latent) | [512] floats = 2,048 bytes |
| Rc (rope keys) | [64] floats = 256 bytes |
| **Total** | **2,304 bytes/token/layer** |

vs full KV cache: 64 × 448 × 4 = 114,688 bytes/token/layer (50x compression)

### Weight Absorption Decode Path

```
For each head h:
  q_abs[kv_lora] = q_nope · kv_b[h*(qk_nope+vh)+d, :kv_lora]  # project to latent space
  score[t] = (q_abs · Lc[t] + q_rope · Rc[t]) × attn_scale     # compute scores
  context_latent = softmax(scores) · Lc                           # weighted sum
  output[h] = kv_b[h*(qk_nope+vh)+qk_nope:, :] · context_latent  # reconstruct V
```

Cost: O(T × kv_lora) per head vs O(T × (qk_nope + v_head)) = 50x savings

## MoE Routing

### Top-K Selection
- Default: 8 experts per token (config `topk=8`)
- Override: `TOPK=n` env var
- Effect: Each expert call loads 3 matrices from disk (~18 MB each)

### Top-P Truncation
- `TOPP=n` env var (e.g., 0.7)
- Keeps only experts whose cumulative weight reaches threshold
- Reduces disk I/O: fewer experts loaded per token
- Example: TOPP=0.7 reduces from 8 to ~5-6 experts per token

## CUDA Expert Kernels

### Kernel Inventory

| Kernel | Purpose | Notes |
|--------|---------|-------|
| quant_matmul | Generic quantized matmul | y = x @ W^T |
| silu_mul | SiLU activation + element-wise multiply | Fused |
| grouped_hidden_w4 | Fused gate+up projection for int4 | Primary fusion |
| grouped_down_w4 | Packed int4 grouped down projection | |
| grouped_s4_wmma | WMMA int4 expert matmul (Tensor Cores) | SM 7.5+ |
| attention_absorb_kernel | Full MLA decode attention on GPU | Requires kv_b in VRAM |

### Kernel Launch Overhead

Each `coli_cuda_expert_group()` call:
1. 1x cudaMemcpyAsync (group descriptors)
2. 1x cudaMemcpyAsync (input activations H2D)
3. 1-2x gate+up kernels
4. 1x silu_mul kernel
5. 1x grouped_down kernel
6. 1x cudaMemcpyAsync (output D2H)
7. 1x cudaStreamSynchronize

**Minimum: 5-6 kernel launches per expert group call**

### Fused Kernels Available

1. `grouped_hidden_w4_dual`: Fuses gate_proj + up_proj into single kernel
2. `silu_mul`: Fuses SiLU activation with multiply
3. `grouped_s4_wmma`: Uses Tensor Core WMMA for int4 matmul
4. `attention_absorb_kernel`: Fuses full MLA decode attention

## CPU-GPU Memory Transfer

### Per Expert Group Call

| Transfer | Direction | Size |
|----------|-----------|------|
| Group descriptors | H2D | ~few KB |
| Input activations | H2D | 8 × 6144 × 4 = 192 KB |
| Output activations | D2H | 8 × 6144 × 4 = 192 KB |
| **Total PCIe** | | **~384 KB** |

### Sync Points

1. `cudaMemcpy` (synchronous) in tensor upload
2. `cudaStreamSynchronize` in expert group call
3. These block the CPU until GPU finishes

## Current Bottleneck Analysis

### Profile Output

```
PROFILE: expert-disk 283.4s (81%) | expert-matmul 39s (11%) | attention 28s (8%)
```

### If Disk I/O Were Eliminated

| Component | Current | Without Disk |
|-----------|---------|--------------|
| expert-disk | 283s | 0s |
| expert-matmul | 39s | 39s |
| attention | 28s | 28s |
| **Total** | **350s** | **67s** |
| **tok/s** | **0.09** | **0.48** |
| **Speedup** | | **5.3x** |

## Optimization Paths

### Path A: Attention on GPU (COLI_CUDA_ATTN)

**What**: Upload attention tensors (q/k/v/o projections) to VRAM. Enable the existing `attention_absorb_kernel`.

**VRAM cost**: 5.87 GB at int4 (all 78 layers)

**Tradeoff**: Must reduce VRAM experts from 509 (8.80 GB) to ~170 (3.07 GB) to fit.

**Impact**:
- Saves 28s of attention compute (runs on GPU instead of CPU)
- But 170 VRAM experts means more disk I/O for experts
- Net effect: uncertain — depends on whether attention savings > expert I/O increase

**Implementation**:
1. Mark q_a, q_b, kv_a, kv_b, o_proj as `cuda_eligible` during model load
2. Upload them to VRAM at startup
3. Set `COLI_CUDA_ATTN=1` to enable the CUDA attention path
4. Reduce `CUDA_EXPERT_GB` to ~3.5 to keep VRAM usage under 12 GB

### Path B: Layer-Level Offloading

**What**: Keep 2-3 full MoE layers entirely on GPU (dense + experts). Like llama.cpp's GPU offloading.

**VRAM cost**: 2 layers × 4.60 GB = 9.20 GB

**Tradeoff**: Only 2 layers are fast. Remaining 76 layers still have expert disk I/O.

**Impact**:
- 2 layers run entirely on GPU (fast)
- 76 layers on CPU (same speed as before)
- Net effect: ~5% improvement (2/78 layers)

**Implementation**:
1. Identify which 2 layers to offload (most frequently used?)
2. Upload all dense + expert tensors for those layers to VRAM
3. Modify the forward pass to skip CPU computation for offloaded layers
4. This is a major refactor — different code path for GPU-resident layers

### Path C: Hybrid — Attention + Fewer VRAM Experts

**What**: Upload attention tensors at int4 (5.87 GB) + keep ~170 VRAM experts (3.20 GB). Use TOPP=0.7 to reduce disk I/O.

**VRAM cost**: 5.87 + 3.20 = 9.07 GB (fits in 12 GB)

**Tradeoff**: Fewer VRAM experts (170 vs 509) means more disk I/O, but TOPP=0.7 reduces the number of experts needed per token.

**Impact**:
- Saves 28s of attention compute
- TOPP=0.7 reduces experts per token from 8 to ~5-6
- Net effect: likely positive (attention savings > I/O increase)

**Implementation**:
1. Upload attention tensors to VRAM
2. Reduce CUDA_EXPERT_GB to ~3.5
3. Set TOPP=0.7 to reduce disk I/O
4. Monitor hit rate — should stay similar with TOPP

### Path D: Fused CUDA Kernels + Async Streams

**What**: Optimize existing CUDA kernels, reduce launch overhead, overlap computation with data transfer.

**Impact**:
- Reduces expert-matmul time from 39s to ~30s (23% improvement)
- Net effect: ~3% overall improvement

**Implementation**:
1. Use CUDA graphs to batch kernel launches
2. Overlap H2D/D2H transfers with computation
3. Reduce sync points
4. Optimize kernel grid/block dimensions

## Recommended Approach

**Path C (Hybrid)** is the best balance:

1. **Upload attention tensors to VRAM** (5.87 GB at int4)
   - q_a, q_b, kv_a, kv_b, o_proj for all 78 layers
   - Enable `COLI_CUDA_ATTN=1`
   - Saves 28s of attention compute

2. **Reduce VRAM experts to ~170** (3.07 GB)
   - Keep the most frequently used experts
   - Total VRAM: 9.07 GB (fits in 12 GB)

3. **Use TOPP=0.7** to reduce disk I/O
   - Reduces experts per token from 8 to ~5-6
   - Compensates for fewer VRAM experts

4. **Expected result**:
   - Attention: 28s → ~5s (GPU-accelerated)
   - Expert disk I/O: 283s → ~200s (TOPP reduces I/O)
   - Expert matmul: 39s → ~30s (fused kernels)
   - **Total: ~235s → ~0.14 tok/s (2x faster)**
