# QuickReduce Fused AllReduce + RMSNorm for ROCm

## Summary

Single-kernel fused AllReduce + Residual Add + RMSNorm for AMD MI300/MI355X GPUs, built into the QuickReduce IPC communication library. Eliminates the intermediate HBM write between AllReduce and RMSNorm by applying normalization directly on register data after the twoshot reduce completes.

**Status**: Kernel implemented, compiled, correctness verified (280.9 dB SNR), 1.37× micro speedup. E2E integration with vLLM v0.19.0 tested — neutral at CONC=4/16 due to CUDA graph splitting-op overhead and 32KB tile limit. Ready for upstream as infrastructure.

## Motivation

In TP>1 LLM inference, every transformer layer has two AllReduce + RMSNorm pairs:
1. Post-attention: `o_proj` AllReduce → `post_attention_layernorm`
2. Post-MoE: MoE AllReduce → `input_layernorm` (next layer)

Profiling MiniMax-M2.5 on MI355X with TP=4 EP=4 shows AllReduce is **27.1% of decode time** — the single largest bottleneck. The AllReduce output goes directly into RMSNorm with no other consumers in between, making this an ideal fusion target.

NVIDIA's vLLM implementation fuses this via FlashInfer's `allreduce_fusion()` kernel (Hopper/Blackwell only). No ROCm equivalent existed.

## Architecture

### QuickReduce Twoshot AllReduce (Existing)

```
Phase 0: Read input into registers tA[8] (128 bytes/thread, 64 bf16 elements)
Phase 1A: Broadcast segments to all ranks via P2P (with INT4/FP8 codec)
Phase 1B: Reduce segments from all ranks → tR[] in registers
Phase 2: Broadcast reduced result → tA[8] holds final allreduced data
Phase 3: Write tA[] to output HBM  ← THIS IS WHERE WE FUSE
```

### Fused Variant (New)

```
Phases 0-2: Identical to standard AllReduceTwoshot (INT4 compression preserved)
Phase 3 (NEW — replaces simple write):
  3a: Load residual_in from HBM, compute residual_out = tA + residual_in
  3b: Per-row sum-of-squares on tA (register-local)
  3c: Block-reduce sum_sq via warp shuffle + LDS cross-warp reduction
  3d: rsqrt_val = rsqrt(mean_sq + epsilon)
  3e: output = tA * rsqrt_val * weight (load weight from HBM)
  3f: Write output + residual_out to HBM
```

**Key insight**: After Phase 2, the allreduced data is in `tA[kAtoms]` registers — 64 bf16 elements per thread, 16384 per block. RMSNorm's sum-of-squares reduction happens in registers + LDS, not HBM. The only additional HBM accesses are: 1 read (residual_in), 1 read (weight), 2 writes (output, residual_out) — same as a standalone RMSNorm kernel. The AllReduce output never touches HBM.

### Small-M Constraint

The fused kernel requires the entire tensor to fit in one `kTileSize` (32KB) for block-local RMSNorm reduction:

```
Max elements per tile: 32KB / 2 bytes = 16384 bf16 elements
For hidden=3072: max M = 16384 / 3072 = 5 rows
For hidden=7168: max M = 16384 / 7168 = 2 rows
```

This covers decode batches (M=1-4) but not prefill. Larger M falls back to separate AllReduce + RMSNorm.

## Files

### New Files

| File | Description |
|------|-------------|
| `csrc/include/quick_all_reduce_rmsnorm.cuh` | `AllReduceTwoshotFusedRMSNorm` GPU kernel template (~270 lines) |
| `csrc/kernels/quick_all_reduce_rmsnorm.cu` | PyTorch C++ binding + dispatch + template instantiations (~200 lines) |

### Modified Files

| File | Change |
|------|--------|
| `csrc/include/quick_all_reduce.h` | Added `qr_fused_allreduce_rmsnorm` declaration |
| `csrc/include/rocm_ops.hpp` | Added pybind11 binding in `QUICK_ALL_REDUCE_PYBIND` macro |
| `aiter/ops/quick_all_reduce.py` | Added `qr_fused_allreduce_rmsnorm` Python JIT stub |
| `aiter/jit/optCompilerConfig.json` | Added new `.cu` to JIT source list |

## API

### C++

```cpp
void qr_fused_allreduce_rmsnorm(
    fptr_t fa,                  // QuickReduce DeviceComms handle
    torch::Tensor& inp,         // [M, hidden] input (will be allreduced)
    torch::Tensor& out,         // [M, hidden] RMSNorm output
    torch::Tensor& residual_in, // [M, hidden] residual input
    torch::Tensor& residual_out,// [M, hidden] allreduced + residual
    torch::Tensor& rms_weight,  // [hidden] RMSNorm gamma
    double rms_epsilon,         // RMSNorm epsilon (e.g., 1e-5)
    int64_t quant_level,        // 0=FP16, 1=FP8, 2=INT6, 3=INT4
    bool cast_bf2half            // Convert bf16→fp16 for faster math
);
```

### Python

```python
from aiter.ops.quick_all_reduce import qr_fused_allreduce_rmsnorm

qr_fused_allreduce_rmsnorm(
    fa, inp, out, residual_in, residual_out,
    weight, epsilon, quant_level, cast_bf2half
)
```

### vLLM Integration (QuickAllReduce class)

```python
class QuickAllReduce:
    def fused_allreduce_rmsnorm(self, inp, residual_in, weight, epsilon):
        """Returns (rmsnorm_out, residual_out) or None if not applicable."""
        if self.disabled or not self.should_quick_allreduce(inp):
            return None
        if inp.dim() != 2 or inp.numel() * inp.element_size() > 32 * 1024:
            return None
        out = torch.empty_like(inp)
        residual_out = torch.empty_like(inp)
        qr_fused_allreduce_rmsnorm(
            self._ptr, inp, out, residual_in, residual_out,
            weight, epsilon, self.qr_quant_level.value, self.use_fp16_kernels
        )
        return out, residual_out
```

## Benchmarks

### Microbenchmark (4× MI355X, TP=4, M=4, hidden=3072)

```
Separate (QR AllReduce + Python RMSNorm): 61.5 us
Fused (single kernel):                    44.9 us
Speedup: 1.37×
Savings: 16.7 us per call
```

### Correctness

```
residual_out SNR: 287.8 dB  PASS
rmsnorm_out SNR:  280.9 dB  PASS
Output match: exact (ref[0,:4] == fused[0,:4])
```

### E2E (MiniMax-M2.5, vLLM v0.19.0, TP=4 EP=4, ISL/OSL=1024)

| CONC | Baseline tok/s | Fused tok/s | Delta |
|-----:|:--------------:|:-----------:|:-----:|
| 4 | 365.5 | 364.2 | -0.3% |
| 16 | 1154.8 | 1158.0 | +0.3% |

E2E neutral because:
1. The fused op runs as a CUDA graph **splitting op** (between compiled segments), adding Python dispatch overhead that offsets the kernel-level gains
2. At CONC≥6 (M≥6), the 32KB tile limit triggers fallback to separate path
3. The baseline QuickReduce INT4 + CK RMSNorm is already heavily optimized

### When This Kernel Wins

The fused kernel will show E2E gains when:
- **Models with larger hidden dims** (e.g., hidden=7168 at M=1-2): more RMSNorm compute to overlap
- **The 32KB tile limit is lifted**: extend to multi-tile RMSNorm with cross-block reduction
- **The AllReduce op is moved inside torch.compile scope**: eliminates splitting-op overhead
- **Higher TP counts** (TP=8): AllReduce latency increases, making fusion savings proportionally larger

## vLLM Integration Approaches Tested

| Approach | Result | Issue |
|----------|--------|-------|
| **v1**: torch.compile FX pattern matcher | -20% regression | Custom op wrapper calls 2 separate ops → Python overhead |
| **v2**: aiter CustomAllreduce `_ptr` | Crash | vLLM and aiter have incompatible C++ handle types |
| **v3**: Replace vLLM CA with aiter CA | -3.2% regression | Lost INT4 QuickReduce acceleration |
| **v4**: QuickReduce fused kernel + custom op | Neutral | Splitting-op overhead + 32KB limit |
| **v5**: Model-level direct call | Neutral | Same splitting-op overhead |

The recommended integration path is **v4/v5** once torch.compile supports AllReduce as a non-splitting op, or via model-specific patches for critical workloads.

## Build

```bash
# aiter builds this via JIT on first import
# The new .cu file is added to optCompilerConfig.json
python -c "from aiter.ops.quick_all_reduce import qr_fused_allreduce_rmsnorm"
# First run triggers ~25s JIT compilation
```

## Future Work

1. **Multi-tile RMSNorm**: Extend to M>5 via cross-block sum-of-squares reduction (atomic add or two-pass)
2. **Fused AR+RMS+FP8Quant**: Add FP8 quantization to the output (common pattern: RMSNorm → FP8 quant → GEMM)
3. **Non-splitting-op AllReduce in vLLM**: Would allow torch.compile to see AR+RMS as one graph segment
4. **Upstream to aiter**: PR to `amd/aiter` repository with the 6 files listed above
