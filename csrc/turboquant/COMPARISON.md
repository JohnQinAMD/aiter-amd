# KV Cache Quantization Methods Comparison

## Methods Overview

| Method | Bits | Type | Calibration | Paper | Year |
|--------|------|------|-------------|-------|------|
| **FP16/BF16** | 16 | Baseline | — | — | — |
| **FP8 E5M2** | 8 | Float cast | No | — | 2023 |
| **FP8 E4M3** | 8 | Float cast + scale | Per-tensor/block | — | 2023 |
| **FP4 E2M1 (MXFP4)** | 4 | Float cast + block scale | Per-block (16 elems) | OCP | 2024 |
| **KIVI** | 2 | Asymmetric INT, per-ch K / per-tok V | No (tuning-free) | ICML 2024 | 2024 |
| **KVQuant** | 2-4 | Per-channel + outlier | Calibration set | MLSys 2025 | 2024 |
| **TurboQuant** | 2-4 | Rotation + Lloyd-Max | **No** (data-oblivious) | ICLR 2026 | 2026 |

## Compression Ratio

| Method | Bits/elem | Bytes/elem | Compression vs FP16 | Overhead |
|--------|-----------|------------|---------------------|----------|
| FP16/BF16 | 16 | 2.0 | 1.0x | — |
| FP8 | 8 | 1.0 | **2.0x** | +scale per tensor/block |
| MXFP4 | 4 | 0.5 | **~3.5x** | +1B scale per 16 elements |
| KIVI-2bit | 2 | 0.25 | **~6x** | +zero-point per group |
| TurboQuant-4bit | 4 | 0.5 | **2.94x** (MLA) | +FP16 norms per group + FP16 rope |
| TurboQuant-3bit | 3 | 0.375 | **~3.8x** (MLA) | same overhead |
| TurboQuant-2bit | 2 | 0.25 | **~5.1x** (MLA) | same overhead |

Note: TurboQuant MLA compression is 2.94x (not 4x) because the 64-dim RoPE part stays FP16.

## Quality (PPL on Llama-class Models)

| Method | Bits | Llama-2-7B | Llama-3-8B | Source |
|--------|------|------------|------------|--------|
| FP16 baseline | 16 | 5.47 | 6.12 | our measurement |
| FP8 E5M2 | 8 | ~5.47 | ~6.12 | <0.1% degradation (SGLang PR #1204) |
| FP8 E4M3 | 8 | ~5.47 | ~6.12 | <0.1% degradation (vLLM PR #2279) |
| KIVI-4bit | 4 | ~5.51 | — | <1% (KIVI paper Table 1) |
| KIVI-2bit | 2 | ~5.72 | — | ~4.6% (KIVI paper Table 1) |
| KVQuant-4bit | 4 | ~5.49 | — | <0.5% (KVQuant paper) |
| KVQuant-2bit | 2 | ~5.96 | — | ~9% (KVQuant paper) |
| **TurboQuant-4bit** | 4 | — | 6.32 (8B) | +3.3% (Llama-3-8B) |
| **TurboQuant-4bit** | 4 | — | **2.7748 (671B)** | **+0.00% (DeepSeek-R1)** |
| **TurboQuant-4bit** (weight) | 4 | — | 7.22 | +10.6% (weight quant, not KV) |

## Platform Support in SGLang

| Method | CUDA | AMD (HIP/ROCm) | MLA Support | Implementation |
|--------|------|----------------|-------------|----------------|
| FP8 E5M2 | ✅ | ✅ (as e4m3fnuz) | ✅ | Native dtype, aiter/flashinfer |
| FP8 E4M3 | ✅ | ✅ (as e4m3fnuz) | ✅ | Native dtype, aiter/flashinfer |
| MXFP4 KV | ✅ | ❌ (policy gate) | ✅ | cutlass_mla/trtllm/flashinfer (CUDA-only) |
| MXFP4 weight | ✅ | ✅ (Quark+aiter) | ✅ | `--quantization quark_mxfp4` works on MI355X |
| KIVI | ❌ | ❌ | ❌ | Not in SGLang |
| KVQuant | ❌ | ❌ | ❌ | Not in SGLang |
| **TurboQuant** | ❌* | **✅** | **✅** | MLA-only, aiter backend |

*TurboQuant could work on CUDA but GPU kernel targets gfx950.

**Key distinction**: MXFP4 **weight** quantization works on AMD MI355X via Quark + aiter flatmm. MXFP4 **KV cache** quantization (`--kv-cache-dtype fp4_e2m1`) is blocked on AMD by a policy gate, not a hardware limitation.

## DeepSeek-V3/R1 MLA Specific Comparison

For MLA (kv_lora_rank=512, qk_rope_head_dim=64), the KV cache stores a single 576-dim latent vector per token per layer (not per-head K+V).

| Method | Bytes/token/layer | Compression | MLA-specific notes |
|--------|------------------|-------------|-------------------|
| BF16 | 1,152 | 1.0x | baseline |
| FP8 | 576 | **2.0x** | Simple dtype cast, near-zero quality loss |
| MXFP4 | ~308 | **~3.7x** | 16-elem blocks + scale; CUDA-only |
| **TQ-4bit** | **392** | **2.94x** | Latent INT4 (256B) + norms (8B) + rope FP16 (128B) |
| **TQ-3bit** | ~308 | **~3.7x** | Would match MXFP4 compression |
| **TQ-2bit** | ~224 | **~5.1x** | Theoretical; quality evaluation needed |

## Latency Impact

| Method | Compress overhead | Decompress overhead | Attention kernel change |
|--------|------------------|--------------------|-----------------------|
| FP8 | ~0 (native cast) | ~0 (native cast) | FP8-aware kernel needed |
| MXFP4 | Per-block quantize | Per-block dequant (~1-3 µs) | None if dequant-to-FP16 first; or FP4-fused kernel |
| KIVI | Per-channel quantize | Per-group dequant | Custom attention kernel |
| **TQ-4bit** | **33 µs/layer** (GPU) | **included above** | **None** (decompressed to FP16) |

TurboQuant's key advantage: the attention kernel (aiter mla_decode_fwd) is **unchanged** — it always sees FP16 data. FP8/MXFP4 require modified attention kernels that handle quantized KV directly.

## When to Use What

### FP8 KV Cache (Recommended Default)
- **Best for**: General deployment, minimal quality loss
- **Compression**: 2.0x
- **Quality**: <0.1% PPL degradation
- **Effort**: `--kv-cache-dtype fp8_e4m3` flag
- **Platform**: CUDA + AMD

### TurboQuant-4bit KV Cache (Memory-Constrained MLA)
- **Best for**: Long-context MLA models (DeepSeek-V3/R1, Kimi-K2) where KV memory is the bottleneck
- **Compression**: 2.94x (47% more than FP8)
- **Quality**: +3.3% PPL on Llama-3-8B KV cache
- **Effort**: `export SGLANG_KV_CACHE_TURBOQUANT=1`
- **Platform**: AMD MI355X (currently)

### MXFP4 KV Cache (CUDA High-Compression)
- **Best for**: CUDA deployments needing >2x compression
- **Compression**: ~3.7x
- **Quality**: Between FP8 and INT4
- **Effort**: `--kv-cache-dtype fp4_e2m1`
- **Platform**: CUDA only (H100+)

### Combined Strategy
For maximum memory savings on AMD MI355X with DeepSeek-R1:
```
FP8 KV (2x)  →  saves 4.3 GB/GPU at 128K  →  zero quality loss
TQ-4bit (2.94x) → saves 5.7 GB/GPU at 128K → +3.3% PPL, more concurrent users
```

TurboQuant provides **47% more memory savings** than FP8 at the cost of ~3% PPL degradation.

## Unique Advantages of TurboQuant

1. **Data-oblivious**: No calibration data, no fine-tuning. Works on any model instantly.
2. **Mathematical guarantees**: Lloyd-Max optimal for Gaussian (which random rotation induces).
3. **Flexible bit-width**: Same algorithm works at 2/3/4 bits — just change the codebook.
4. **No attention kernel changes**: Compress on write, decompress on read. Transparent to downstream.
5. **Composable**: Can be combined with FP8 weight quantization — orthogonal optimizations.

## Limitations vs Alternatives

1. **Higher PPL degradation than FP8** (+3.3% vs <0.1%) — but FP8 only gives 2x compression.
2. **Higher decompress latency** (33 µs/layer for rotation matmul) vs FP8 (native dtype cast, ~0).
3. **MLA-only in current implementation** — GQA support would need per-head quantization.
4. **AMD-only GPU kernel** — needs CUDA port for broader deployment.
5. **Rotation matrix storage**: 4 × 128×128 × 4B = 256KB per layer on device (negligible vs model weights).

---

## Deep Dive: MXFP4 KV Cache on AMD MI355X

### Hardware Reality: MI355X Supports MXFP4 Natively

MI355X (gfx950 / CDNA4) has **full native MXFP4 support**:

| Spec | Value |
|------|-------|
| MXFP4 peak FLOPS | 10.1 PFLOPS |
| FP4/FP6 MFMA instructions | `V_CVT_SCALEF32_PK_*_FP4`, MFMA scale instructions |
| CK/aiter support | `pk_fp4_t`, `MXFlatmm_GFX950_*`, `gemm_microscale_pipeline_*` |
| ROCm support | ROCm 7.0+ with CK, hipBLASLt, rocWMMA |
| Quark toolkit | MXFP4 weight quantization with AutoSmoothQuant |

The hardware is ready. MXFP4 **weight** quantization already works on MI355X via `--quantization quark_mxfp4`. The blocker is KV cache specifically.

### Why MXFP4 KV Cache Fails on AMD (Root Cause)

The failure is **two policy gates**, not a hardware limitation:

**Gate 1**: `server_args.py` line 2498:
```python
# _handle_kv4_compatibility():
else:
    raise RuntimeError("KV4 is not tested on non-CUDA platforms.")
```
This kills `--kv-cache-dtype fp4_e2m1` on any non-CUDA platform immediately.

**Gate 2**: `utils/common.py` line 198:
```python
def is_float4_e2m1fn_x2(dtype) -> bool:
    target_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return is_cuda() and dtype == target_dtype  # <-- hard CUDA check
```
Even if Gate 1 were removed, `MLATokenToKVPoolFP4` would never be selected on HIP.

**Gate 3**: All FP4-compatible attention backends are CUDA-only:
- `cutlass_mla_backend.py`: imports `sgl_kernel` only when `is_cuda()`
- `trtllm_mla_backend.py`: depends on FlashInfer CUDA paths
- `flashmla` / `flashinfer`: no ROCm FP4 KV variants exist

### What Would Be Needed to Enable MXFP4 KV on AMD

There are two possible architectures:

**Architecture A: FP4-native attention kernel (what CUDA does)**
The attention kernel reads FP4+scale directly and dequants fused inside the kernel. This is what cutlass_mla/trtllm do. Building this for aiter would take weeks.

**Architecture B: Decompress-then-attend (TurboQuant's approach)**
Decompress FP4→FP16 as a separate step, then feed standard FP16 into the existing aiter attention kernel. This is much simpler.

Architecture B is the practical path:

| Step | Effort | Description |
|------|--------|-------------|
| 1. Remove policy gates | Trivial | Change `_handle_kv4_compatibility` + `is_float4_e2m1fn_x2` to allow AMD |
| 2. Implement `MLATokenToKVPoolMXFP4` | **1-2 days** | Same compress-on-write/decompress-on-read pattern as TurboQuant, but with E2M1 block-scale instead of Lloyd-Max codebook |
| 3. Quantize (on write) | Easy | Block-max → E8M0 scale, values → E2M1 cast. `KVFP4QuantizeUtil` already exists in SGLang |
| 4. Dequantize (on read) | Easy | `fp16_val = E2M1_table[nibble] * pow2(scale)` — one multiply per element, ~1-3 µs/layer |
| 5. Testing + validation | 1-2 days | Quality + perf benchmarks on MLA with real DeepSeek-V3 |

MXFP4 dequant is **far simpler** than TurboQuant's — no rotation matmul, just an elementwise scale-and-cast:

| | TurboQuant decompress | MXFP4 decompress |
|---|---|---|
| Per-element ops | codebook[idx] + 128×128 matmul + norm scale | E2M1→FP16 lookup + 1 multiply |
| Latency/layer | ~33 µs (rotation-dominated) | **~1-3 µs** (trivial elementwise) |
| Complexity | 4 rotation matmuls per token | Block-parallel, no dependencies |

**Total effort for MXFP4 KV on AMD via Architecture B: ~3-4 days.** No attention kernel changes needed — the existing aiter `mla_decode_fwd` sees FP16, just like with TurboQuant.

### MXFP4 vs TurboQuant: Head-to-Head for MLA KV Cache

| Dimension | MXFP4 (E2M1) | TurboQuant-4bit |
|-----------|-------------|-----------------|
| **Bits per element** | 4 (E2M1 float) | 4 (Lloyd-Max codebook index) |
| **Representation** | Floating-point: 1 sign + 2 exp + 1 mantissa | Integer index into 16-level codebook |
| **Block/group structure** | 32 elements share 1 E8M0 scale (1 byte) | 128 elements share 1 FP16 L2 norm (2 bytes) |
| **Compression (MLA)** | ~3.7x (512×0.5 + 16 scales + rope×2 = ~308B) | 2.94x (256B packed + 8B norms + 128B rope = 392B) |
| **Calibration** | No (online block-max) | No (data-oblivious rotation) |
| **Quality (theory)** | Adapts to local magnitude via block scale | Provably near-optimal D_mse for Gaussian (TurboQuant paper Thm 1) |
| **Quality (measured)** | ~+1-2% PPL on large models (Quark blog, weight quant) | +3.3% PPL on Llama-3-8B KV cache |
| **HW acceleration** | Native FP4 MFMA on MI355X (10.1 PFLOPS) | No HW accel; decompresses to FP16 for MFMA |
| **Attention kernel** | **No change needed** (decompress to FP16 first, ~1-3 µs) | **No change** — aiter mla_decode_fwd sees FP16 |
| **AMD MI355X status** | ❌ KV cache blocked (weight quant works) | ✅ Working, benchmarked |
| **CUDA status** | ✅ cutlass_mla / trtllm_mla | ❌ Kernel targets gfx950 only |

### Quality Comparison: What We Know

**MXFP4 quality data (from AMD Quark blog, weight quantization):**

| Model | FP16 baseline | MXFP4 (weight) | Degradation |
|-------|--------------|----------------|-------------|
| DeepSeek-R1-0528 (AIME24) | 86.7% | 86.3% | -0.5% |
| DeepSeek-R1-0528 (MATH-500) | 97.2% | 97.0% | -0.2% |
| Llama-3.1-405B (MMLU-Pro) | 61.6% | 60.8% | -1.3% |
| Llama-3.3-70B (MMLU-Pro) | 55.9% | 51.4% | -8.1% |
| GPT-OSS-120B (MMLU-Pro) | 53.5% | 51.2% | -4.3% |

Note: These are **weight quantization** numbers from the Quark blog, not KV cache. KV cache MXFP4 quality is expected to be similar or better (KV vectors have less outlier structure than weights).

**TurboQuant quality data (our measurements, KV cache):**

| Model | FP16 baseline | TQ-4bit KV | Degradation |
|-------|--------------|------------|-------------|
| Llama-3-8B (WikiText PPL) | 6.12 | 6.32 | +3.3% |
| DeepSeek-V3 MLA (CosSim) | 1.000 | 0.995 | -0.5% |
| TQ paper (LongBench-E, 3.5-bit) | 50.06 avg | 50.06 avg | 0% |
| TQ paper (LongBench-E, 2.5-bit) | 50.06 avg | 49.44 avg | -1.2% |

**Expected comparison for KV cache at 4-bit:**
- MXFP4: likely +1-3% PPL (block-scale adapts to local magnitude; good for activations/KV)
- TurboQuant: +3.3% PPL (data-oblivious; quality is independent of data distribution)
- MXFP4 may have a slight quality edge because E2M1 floating-point preserves relative precision across magnitudes, while TurboQuant's fixed codebook is optimal only for the post-rotation Gaussian distribution

### Throughput Comparison

**MXFP4 (if native FP4 attention existed on AMD):**
- Compress: ~0 latency (block-max + E2M1 cast is simple)
- Decompress: ~0 latency (integrated into FP4-native attention kernel)
- Attention: potentially faster than FP16 (reads 4x fewer KV bytes)
- **Theoretical best case**: attention becomes compute-bound instead of memory-bound at large context

**TurboQuant-4bit (current implementation):**
- Compress: ~33 µs/layer (Python; GPU kernel would reduce to ~5 µs)
- Decompress: ~33 µs/layer (rotation matmul + codebook lookup per group)
- Attention: same as FP16 (decompressed before kernel)
- **Net effect**: saves memory (2.94x) but adds latency per layer

| Scenario | MXFP4 (hypothetical AMD) | TurboQuant-4bit (actual AMD) |
|----------|-------------------------|------------------------------|
| Decode B=1 | ~same as FP16 (memory-bound) | +33 µs/layer overhead |
| Decode B=128 | faster (4x less KV bandwidth) | same overhead |
| Prefill | faster (4x less KV bandwidth) | same overhead |
| Memory saved at 128K | ~5.8 GB/GPU (3.7x) | ~5.6 GB/GPU (2.94x) |

### Verdict: When to Use Which on MI355X

| Situation | Recommendation | Rationale |
|-----------|---------------|-----------|
| **Today, need >2x KV compression on AMD** | **TurboQuant** | Only working option; 2.94x compression, proven quality |
| **Today, minimal quality loss on AMD** | **FP8 KV** | 2.0x compression, <0.1% PPL loss, native support |
| **Near-term, MXFP4 dequant-to-FP16 path (~3-4 days)** | **MXFP4** | 3.7x compression, ~1-3 µs/layer dequant (10x faster than TQ's 33 µs), no attention kernel changes |
| **Long-term, native FP4 attention kernel** | **MXFP4** | Same 3.7x compression + zero decompress overhead (fused into attention) |
| **CUDA deployment** | **MXFP4** | Already working via cutlass_mla/trtllm, 3.7x compression |
| **Extreme compression (>4x) on AMD** | **TurboQuant-2bit** | 5.1x compression; MXFP4 can't go below 4 bits |

### Enabling MXFP4 KV on AMD: Recommended Path

The fastest path to MXFP4 KV on AMD MI355X:

1. **Short-term (available now)**: Use TurboQuant for 4-bit KV cache compression on AMD
2. **Medium-term**: Build a dequantize-in-kernel path for aiter MLA that reads FP4+scale buffers (similar to TurboQuant's transparent dequant approach, but with E2M1 instead of codebook lookup)
3. **Long-term**: Build a native FP4 MLA attention kernel for aiter that consumes MXFP4 KV directly using MI355X's MFMA scale instructions — this would eliminate decompress overhead entirely

The key realization: **TurboQuant's architecture (compress-on-write, decompress-on-read, unchanged attention kernel) could also be applied to MXFP4** as an interim solution. Replace Lloyd-Max codebook quantization with E2M1 block-scale quantization in `MLATokenToKVPoolTQ`, and you get MXFP4 KV cache compression on AMD with zero attention kernel changes. This would be ~1-2 days of work.
