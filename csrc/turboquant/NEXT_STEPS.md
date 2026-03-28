# TurboQuant Next Steps: Detailed Feasibility Analysis

## Priority Matrix

| # | Item | Impact | Effort | Priority |
|---|------|--------|--------|----------|
| 1 | GPU KV quantize kernel | HIGH (30x compress speed) | Medium (1-2 weeks) | **P0** |
| 2 | Rotation fusion into kv_a_proj | MEDIUM (eliminate rotation cost) | Complex (blocked by layernorm) | **P2** |
| 3 | MoE expert weight quantization | HIGH (58% of decode traffic) | High (existing INT4 MoE paths) | **P1** |

---

## 1. GPU KV Quantize Kernel (P0)

### Current State

`MLATokenToKVPoolTQ._tq_compress()` runs in Python on every `set_kv_buffer` call:

```
Per token, per layer:
  k_nope (512 dims, bf16)
  → for each group g (4 groups, gs=128):
      1. L2 norm:        ||k_g||                           # 128 FMAs
      2. Normalize:      k_g / ||k_g||                     # 128 divs
      3. Load Pi_g:      (128, 128) float32 matrix          # 64 KB
      4. Rotate:         k_g_norm @ Pi_g.T                  # 128×128 = 16384 FMAs
      5. Scale:          × √128                             # 128 muls
      6. Quantize:       searchsorted(15 boundaries)        # 128 × ~4 comparisons
      7. Pack:           2 nibbles → 1 byte                 # 64 byte ops
  → store: 256 bytes (packed) + 4×4 bytes (norms) + 128 bytes (rope)
  Total: 400 bytes per token per layer (vs 1152 bytes FP16)
```

Measured: **~0.5 ms per batch** in Python. Bottleneck is the 4× rotation matmuls (128×128).

### Proposed GPU Kernel

Single HIP kernel: `turboquant_kv_compress_kernel`

```
Inputs:  k_nope (T, 512, bf16), rope (T, 64, bf16)
Preload: Pi_0..Pi_3 (4 × 128×128 float32 = 256 KB) → shared memory or constant
Output:  compressed (T, 400, uint8)

Per-token work (one warp per token):
  For each group (4 groups, 128 dims):
    1. Load 128 bf16 values from k_nope             # 256 bytes, coalesced
    2. Warp-reduce L2 norm                           # ~7 butterfly steps
    3. Normalize (broadcast scalar)                  # 128 FMAs
    4. Matrix-vector multiply via MFMA:              # 128×128 = 16K FMAs
       - Pi in LDS (128×128×2 = 32 KB bf16)
       - 8 MFMA 16×16×16 ops (K=128 / 16 = 8 iters)
    5. Scale by √128 and searchsorted               # 128 comparisons
    6. Pack pairs into bytes                         # 64 byte ops
  Copy rope (64 bf16 → 128 bytes) unchanged
  Store norms (4 × float32 = 16 bytes)
```

**Expected performance**:
- Rotation MFMA: 4 groups × 8 MFMA ops = 32 MFMA instructions per token
- At ~1024 MFMA TFLOPS: 32 × 16×16×16 × 2 / 1024e12 ≈ **0.003 µs** (pure MFMA)
- Memory: load 1024 bytes (k_nope) + store 400 bytes = 1424 bytes/token
- At 8 TB/s: 1424 / 8e12 ≈ **0.0002 µs** per token
- **Bottleneck: LDS bandwidth for Pi matrices** (~256 KB total across groups)
- Realistic estimate: **~15-50 µs per batch** (vs 500 µs Python = **10-30x speedup**)

### LDS Layout for Pi Matrices

4 rotation matrices × 128×128 × 2 bytes (bf16) = 128 KB → fits in MI355X LDS (64 KB per CU, but can split across groups or use constant memory for smaller groups).

Alternative: keep Pi in global memory (256 KB total, cached in L2). Each token loads the same Pi — high cache hit rate for batched tokens.

### Implementation Plan

1. Write `turboquant_kv_compress.hip` (~200 lines):
   - Kernel: one threadblock per token (or per batch of tokens)
   - Pi matrices as kernel arguments (preloaded to device at model init)
   - Output: packed indices (256 bytes) + norms (16 bytes) + rope (128 bytes) = 400 bytes
2. Python wrapper in `turboquant_engine.py`: `turboquant_kv_compress_gpu(k_nope, rope, Pi_list)`
3. Update `MLATokenToKVPoolTQ._tq_compress()` to call GPU kernel
4. Verify correctness: CosSim vs Python reference

### Decompression Kernel (Already Partially Exists)

Dequant path is simpler (codebook lookup + rotate back + denormalize). Could reuse the existing flatmm dequant pattern, but for KV cache the operation is different: it's a codebook-lookup + matmul (Pi^T) per token, not a batched GEMM.

---

## 2. Rotation Fusion (P2)

### Why It's Blocked

The DeepSeek-V3 MLA forward path:

```
h (7168) → fused_qkv_a_proj → split → kv_a_layernorm(gamma) → k_nope (512)
                                                                    ↓
                                                        TurboQuant: norm → Pi @ x → quantize
```

**`kv_a_layernorm` has a learned per-dimension gamma (512 dims)**. This RMSNorm applies:
```
k_nope = (raw_kv / RMS(raw_kv)) * gamma
```

To fold Pi into W_kv, we'd need `Pi @ diag(gamma)` to equal `diag(gamma') @ Pi` for some gamma'. This only works if Pi commutes with diag(gamma), which it doesn't for a random orthogonal Pi.

### Three Possible Approaches

| Approach | Exactness | Feasibility | Notes |
|----------|-----------|-------------|-------|
| **A: Fuse Pi into GPU kernel** | Exact | Easy | Pi @ x is a 128×128 matmul, trivially fused into the compress kernel. This is item 1. |
| **B: Absorb Pi into W_kv** | Approximate | Medium | Define W_kv' = Pi @ diag(gamma) @ W_kv for each group. But then layernorm gamma must be set to 1 (or a new gamma' found). Quality depends on gamma variance. |
| **C: Pre-rotate gamma** | Approximate | Easy | gamma' = Pi @ gamma per group. The layernorm output is then approximately rotated. Error ∝ variance of gamma within each group. |

**Approach A is the right answer**: fuse the rotation into the GPU compress kernel (item 1). The rotation matmul is 128×128 = 16K FMAs, which is negligible compared to kernel launch overhead. There's no need to eliminate it algebraically.

**Approach B/C might be worth revisiting** if we move to a custom-trained model where we can control the layernorm. For existing checkpoints, the layernorm gamma blocks algebraic fusion.

### One Exception: Per-Group Norm Commutativity

Since TurboQuant applies per-group L2 normalization AFTER extracting the group from k_nope, and Pi preserves L2 norms (orthogonal), the rotation CAN be moved before normalization:

```
Original: norm(k_g) → Pi @ (k_g / norm(k_g))
Equivalent: Pi @ k_g → norm(Pi @ k_g) = norm(k_g) → (Pi @ k_g) / norm(k_g)
```

This means the rotation could theoretically be applied to the raw kv_a_proj output (before layernorm), but only if we also move it before the layernorm gamma — which breaks the model.

**Conclusion**: Rotation fusion is not worth pursuing independently. The GPU kernel (item 1) naturally fuses it.

---

## 3. MoE Expert Weight Quantization (P1)

### Why It Matters

MoE FFN accounts for **58.4% of per-layer weight traffic** at B=1 decode:

| Component | BF16 MB/layer/GPU | Active params |
|-----------|------------------|---------------|
| 8 routed experts (gate_up) | 56.0 | 29.4M |
| 8 routed experts (down) | 28.0 | 14.7M |
| Shared expert | 10.5 | 5.5M |
| Router | 3.5 | 1.8M |
| **Total MoE** | **98.0** | **51.4M** |

Each expert is small: gate_up = (512, 7168), down = (7168, 256) at TP=8.

### Existing INT4 MoE Paths in SGLang

| Path | Format | Backend | Status |
|------|--------|---------|--------|
| Triton W4A16 | GPTQ/AWQ group quant | `fused_moe_kernel_gptq_awq` | Working |
| FlashInfer MxInt4 | packed int32 | FlashInfer TRT-LLM | NVIDIA only |
| Aiter INT4 packed | uint32, `SGLANG_INT4_WEIGHT` | Aiter `fused_moe` | Working on ROCm |

**Critical finding: INT4 MoE already exists on AMD** via `SGLANG_INT4_WEIGHT` + Aiter. The question is whether TurboQuant's codebook-based INT4 offers better quality than the existing uniform/GPTQ quantization.

### TurboQuant INT4 vs GPTQ INT4 for MoE

| Aspect | TurboQuant INT4 | GPTQ/AWQ INT4 |
|--------|----------------|---------------|
| Quantization | Data-oblivious (rotation + Lloyd-Max) | Data-dependent (calibration set) |
| Codebook | Fixed 16 centroids for N(0,1) | Per-group min/max or learned |
| Quality | ~+10% PPL on 8B model | ~+3-5% PPL (calibrated) |
| Offline cost | None (no calibration) | Requires calibration data |
| Kernel support | Custom CK flatmm (kv_b_proj only) | Triton fused_moe (all experts) |

**Honest assessment**: GPTQ/AWQ INT4 will have **better quality** (lower PPL degradation) because it's calibrated on real data, while TurboQuant is data-oblivious. The advantage of TurboQuant is zero calibration cost.

### Three Integration Strategies

#### Strategy A: Use Existing GPTQ INT4 MoE (Lowest effort)

- Apply GPTQ/AWQ quantization to MoE expert weights offline
- Use existing Triton `fused_moe_kernel_gptq_awq` or Aiter INT4 path
- **Effort**: 1-2 days (weight conversion script + config)
- **Quality**: Best (calibrated)
- **No new kernel work needed**

#### Strategy B: TurboQuant INT4 via Existing Aiter Path

- Quantize expert weights using `turboquant_quantize_packed()`
- Convert to the packed format expected by `SGLANG_INT4_WEIGHT` path
- **Effort**: 3-5 days (format conversion + validation)
- **Quality**: Moderate (data-oblivious)
- **Advantage**: Zero calibration, consistent with kv_b_proj quantization

#### Strategy C: Custom CK FlatMM MoE Kernel (Highest effort)

- Extend `turboquant_int4_flatmm.hip` to handle batched expert GEMMs
- Integrate with FusedMoE dispatch
- **Effort**: 3-4 weeks (kernel + integration + tuning)
- **Quality**: Same as B but potentially faster
- **Advantage**: Full control over preshuffle/codebook/scheduling

### Expert Shape Analysis

At TP=8, per expert:

| Layer | Shape (N, K) | BF16 MB | INT4 MB | Bandwidth time at 8 TB/s |
|-------|-------------|---------|---------|--------------------------|
| gate_up | (512, 7168) | 7.0 | 1.8 | 0.9 µs → 0.2 µs |
| down | (7168, 256) | 3.5 | 0.9 | 0.4 µs → 0.1 µs |

These are all **well below the dispatch floor** (~11 µs). INT4 won't help individual expert latency at all — they're already at the floor.

**BUT**: FusedMoE batches 8 experts into 1-2 kernel launches:
- gate_up batch: 8 × 7.0 MB = 56 MB → 7.0 µs (above floor, bandwidth-bound)
- down batch: 8 × 3.5 MB = 28 MB → 3.5 µs (near floor)

With INT4:
- gate_up batch: 8 × 1.8 MB = 14 MB → 1.8 µs (but FusedMoE dispatch overhead ~15 µs)
- **INT4 savings are eaten by FusedMoE kernel dispatch overhead**

### Revised Impact Estimate

At B=1 decode with FusedMoE (2 launches per MoE layer):

| Component | BF16 est. µs | INT4 est. µs | Saving |
|-----------|-------------|-------------|--------|
| FusedMoE gate_up (8 exp) | ~15 | ~15 | 0 (dispatch floor) |
| FusedMoE down (8 exp) | ~15 | ~15 | 0 (dispatch floor) |
| Shared expert (2 GEMMs) | ~22 | ~22 | 0 (dispatch floor) |

**At B=1 decode, INT4 MoE provides near-zero speedup** because FusedMoE dispatch overhead dominates.

At **larger batch sizes** (B=64-128, prefill-like):
- FusedMoE becomes compute-bound (each token routes to 8 experts)
- INT4 weight dequant overhead could HURT performance
- FP8 w8a8 is the right choice for compute-bound MoE

### MoE INT4 Conclusion

**MoE INT4 weight quantization is NOT a high-impact optimization for latency.** The primary benefit would be **memory savings** (~2.8x reduction in expert weights = ~60 GB saved across 256×58 experts on TP=8). This matters for:
- Fitting more experts per GPU (EP < 8 scenarios)
- Reducing model load time
- NOT for decode latency (dispatch-floor-bound at B=1, compute-bound at B>1)

---

## Revised Priority Matrix

| # | Item | Latency Impact | Memory Impact | Effort | Priority |
|---|------|---------------|---------------|--------|----------|
| 1 | GPU KV compress kernel | Medium (faster set_kv) | None (already 2.94x) | 1-2 weeks | **P0** |
| 2 | Rotation fusion | None (absorbed into #1) | None | N/A | **Dropped** |
| 3a | GPTQ INT4 MoE (existing path) | ~0% at B=1 | 2.8x expert weights | 2 days | **P1 (memory)** |
| 3b | TurboQuant INT4 MoE (custom) | ~0% at B=1 | 2.8x expert weights | 3-4 weeks | **P3** |

### What Actually Moves the Needle

| Optimization | Decode Latency | Throughput | Effort |
|-------------|---------------|------------|--------|
| **KV cache 2.94x compression** (done) | 0% | +2-3x (more concurrent users) | Done |
| **kv_b_proj INT4 GEMM** (done) | -5% | marginal | Done |
| **GPU KV compress kernel** (next) | -1-2% (faster set_kv) | slight | 1-2 weeks |
| **FP8 MoE** (existing in Aiter) | -5-10% at B>1 | significant | Config-only |
| **Speculative decoding** | -30-50% | major | Large project |
| **Batch scheduling optimization** | 0% | +20-50% | Medium |

The biggest remaining wins are in **serving-level optimizations** (batch scheduling, speculative decoding, continuous batching efficiency), not kernel-level weight quantization.

---

## Recommended Immediate Actions

1. **Ship what we have**: The KV cache compression (2.94x) + kv_b_proj INT4 (2.16x) are ready for PR
2. **GPU KV compress kernel**: Reduces the Python overhead on the hot path
3. **Enable FP8 MoE via Aiter**: This is a config flag, not new code, and helps at B>1
4. **End-to-end benchmark**: Run DeepSeek-R1 with `SGLANG_KV_CACHE_TURBOQUANT=1` on 8x MI355X
