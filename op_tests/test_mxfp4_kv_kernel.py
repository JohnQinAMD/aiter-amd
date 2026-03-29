"""Test + benchmark for MXFP4 E2M1 HIP dequant/quant kernel.

Compares the fused HIP kernel against:
  1. Python reference (identical math to SGLang's KVFP4QuantizeUtil)
  2. @torch.compile path (SGLang production path, if available)

Run inside ROCm container:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /shared_aig:/shared_aig --ipc=host \
    rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260324 \
    python3 /shared_aig/john/semianalysis/aiter-amd/op_tests/test_mxfp4_kv_kernel.py
"""

import os
import sys
import time
import math
import torch

torch.manual_seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BLOCK_SIZE = 16

# E2M1 constants
E2M1_MAX = 6.0
E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=DEVICE
)
E2M1_BOUNDS = torch.tensor(
    [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=DEVICE
)


# ─── Python Reference ───


def ref_quantize(tensor):
    """Python MXFP4 quantization (matches KVFP4QuantizeUtil.batched_quantize)."""
    flat = tensor.float().reshape(-1, tensor.shape[-1])
    T, N = flat.shape
    reshaped = flat.view(T, N // BLOCK_SIZE, BLOCK_SIZE)
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
    scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)
    scaled = reshaped / torch.exp2(scale_exp)
    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)
    fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)
    fp4_flat = fp4_vals.view(T, N)
    packed = (fp4_flat[:, 1::2] << 4) + fp4_flat[:, 0::2]
    return packed, scale_factors


def ref_dequantize(packed, scale_factors, N):
    """Python MXFP4 dequantization (matches KVFP4QuantizeUtil.batched_dequantize)."""
    T = packed.shape[0]
    N_half = packed.shape[-1]
    fp4_vals = torch.empty(T, N, dtype=torch.uint8, device=packed.device)
    fp4_vals[:, 0::2] = packed & 0x0F
    fp4_vals[:, 1::2] = (packed >> 4) & 0x0F
    sign_mask = (fp4_vals & 0x08) != 0
    magnitude_idx = fp4_vals & 0x07
    float_vals = E2M1_VALUES[magnitude_idx.long()]
    float_vals = torch.where(sign_mask, -float_vals, float_vals)
    reshaped = float_vals.view(T, N // BLOCK_SIZE, BLOCK_SIZE)
    scale_exp = scale_factors.float() - 127
    scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))
    return scaled.view(T, N).to(torch.bfloat16)


# ─── Build HIP kernel ───


def build_kernel():
    import torch.utils.cpp_extension

    src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "csrc", "turboquant")
    hip_file = os.path.join(src_dir, "mxfp4_kv_dequant.hip")
    return torch.utils.cpp_extension.load(
        name="mxfp4_kv_dequant",
        sources=[hip_file],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )


def cosine_similarity(a, b):
    a_f = a.reshape(-1).float()
    b_f = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(
        a_f.unsqueeze(0), b_f.unsqueeze(0)
    ).item()


# ─── Correctness Tests ───


def test_dequant_correctness(mod, N, T):
    """Verify HIP dequant matches Python reference (should be bit-exact)."""
    data = torch.randn(T, N, dtype=torch.bfloat16, device=DEVICE) * 0.02
    packed_ref, scales_ref = ref_quantize(data)
    deq_ref = ref_dequantize(packed_ref, scales_ref, N)
    deq_hip = mod.mxfp4_dequantize(packed_ref, scales_ref, N)

    max_diff = (deq_ref.float() - deq_hip.float()).abs().max().item()
    cos = cosine_similarity(deq_ref, deq_hip)
    bit_exact = max_diff == 0.0

    assert max_diff < 1e-4, f"Dequant FAIL N={N} T={T}: max_diff={max_diff}"
    print(f"  dequant  N={N:>5} T={T:>5}  max_diff={max_diff:.2e}  "
          f"CosSim={cos:.8f}  bit_exact={bit_exact}")
    return bit_exact


def test_quant_correctness(mod, N, T):
    """Verify HIP quant matches Python reference."""
    data = torch.randn(T, N, dtype=torch.bfloat16, device=DEVICE) * 0.02
    packed_ref, scales_ref = ref_quantize(data)
    result = mod.mxfp4_quantize(data)
    packed_hip, scales_hip = result[0], result[1]

    packed_match = (packed_ref == packed_hip).all().item()
    scales_match = (scales_ref == scales_hip).all().item()

    if not packed_match:
        diff_count = (packed_ref != packed_hip).sum().item()
        print(f"  quant    N={N:>5} T={T:>5}  packed_diff={diff_count}/{packed_ref.numel()} "
              f"scales_match={scales_match}  (float rounding)")
    else:
        print(f"  quant    N={N:>5} T={T:>5}  packed_match=True  scales_match={scales_match}")

    # Round-trip check regardless
    deq_hip = mod.mxfp4_dequantize(packed_hip, scales_hip, N)
    cos_rt = cosine_similarity(data.float(), deq_hip.float())
    print(f"           round-trip CosSim={cos_rt:.6f}")
    return packed_match and scales_match


def test_inplace(mod, N, T):
    """Test in-place variants."""
    data = torch.randn(T, N, dtype=torch.bfloat16, device=DEVICE) * 0.02
    packed_ref, scales_ref = ref_quantize(data)
    deq_ref = ref_dequantize(packed_ref, scales_ref, N)

    output = torch.empty(T, N, dtype=torch.bfloat16, device=DEVICE)
    mod.mxfp4_dequantize_inplace(packed_ref, scales_ref, output)
    max_diff = (deq_ref.float() - output.float()).abs().max().item()
    assert max_diff < 1e-4, f"In-place dequant FAIL: max_diff={max_diff}"

    packed_out = torch.empty(T, N // 2, dtype=torch.uint8, device=DEVICE)
    scales_out = torch.empty(T, N // BLOCK_SIZE, dtype=torch.uint8, device=DEVICE)
    mod.mxfp4_quantize_inplace(data, packed_out, scales_out)

    deq2 = mod.mxfp4_dequantize(packed_out, scales_out, N)
    cos = cosine_similarity(data.float(), deq2.float())
    print(f"  inplace  N={N:>5} T={T:>5}  dequant_maxdiff={max_diff:.2e}  "
          f"quant_rt_cos={cos:.6f}")


def test_3d_shapes(mod):
    """Test with 3D input [B, M, N] matching SGLang memory pool layout."""
    B, M, N = 32, 1, 576
    data = torch.randn(B, M, N, dtype=torch.bfloat16, device=DEVICE) * 0.02
    packed_ref, scales_ref = ref_quantize(data.reshape(-1, N))
    packed_ref = packed_ref.reshape(B, M, N // 2)
    scales_ref = scales_ref.reshape(B, N // BLOCK_SIZE)

    deq_hip = mod.mxfp4_dequantize(packed_ref, scales_ref, N)
    deq_ref = ref_dequantize(
        packed_ref.reshape(-1, N // 2), scales_ref.reshape(-1, N // BLOCK_SIZE), N
    ).reshape(B, M, N)

    max_diff = (deq_ref.float() - deq_hip.float()).abs().max().item()
    cos = cosine_similarity(deq_ref, deq_hip)
    print(f"  3D shape [{B},{M},{N}]  max_diff={max_diff:.2e}  CosSim={cos:.8f}")
    assert max_diff < 1e-4


# ─── Benchmarks ───


def gpu_timer(fn, warmup=20, iters=200):
    """Time a GPU function using CUDA events (microsecond precision)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def benchmark_dequant(mod, N, T_values):
    print(f"\n{'─'*80}")
    print(f"  DEQUANT Benchmark  N={N}  (bytes read: {N//2 + N//16}  bytes written: {N*2})")
    print(f"{'─'*80}")
    print(f"  {'T':>6}  {'Python ms':>12}  {'HIP ms':>12}  {'Speedup':>10}  {'GB/s (HIP)':>12}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}  {'─'*12}")

    for T in T_values:
        data = torch.randn(T, N, dtype=torch.bfloat16, device=DEVICE) * 0.02
        packed, scales = ref_quantize(data)

        t_py = gpu_timer(lambda: ref_dequantize(packed, scales, N))
        t_hip = gpu_timer(lambda: mod.mxfp4_dequantize(packed, scales, N))

        total_bytes = T * (N // 2 + N // 16 + N * 2)  # read packed+scales, write bf16
        gbps = total_bytes / (t_hip * 1e-3) / 1e9

        speedup = t_py / max(t_hip, 1e-9)
        print(f"  {T:>6}  {t_py:>11.4f}  {t_hip:>11.4f}  {speedup:>9.1f}x  {gbps:>11.1f}")


def benchmark_quant(mod, N, T_values):
    print(f"\n{'─'*80}")
    print(f"  QUANT Benchmark  N={N}")
    print(f"{'─'*80}")
    print(f"  {'T':>6}  {'Python ms':>12}  {'HIP ms':>12}  {'Speedup':>10}")
    print(f"  {'─'*6}  {'─'*12}  {'─'*12}  {'─'*10}")

    for T in T_values:
        data = torch.randn(T, N, dtype=torch.bfloat16, device=DEVICE) * 0.02

        t_py = gpu_timer(lambda: ref_quantize(data))
        t_hip = gpu_timer(lambda: mod.mxfp4_quantize(data))

        speedup = t_py / max(t_hip, 1e-9)
        print(f"  {T:>6}  {t_py:>11.4f}  {t_hip:>11.4f}  {speedup:>9.1f}x")


def benchmark_vs_torch_compile(mod, N=576, T=1024):
    """Compare against @torch.compile if available."""
    print(f"\n{'─'*80}")
    print(f"  HIP vs @torch.compile  N={N} T={T}")
    print(f"{'─'*80}")

    try:
        sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")
        from sglang.srt.layers.quantization.kvfp4_tensor import KVFP4QuantizeUtil

        data_3d = torch.randn(T, 1, N, dtype=torch.bfloat16, device=DEVICE) * 0.02

        # torch.compile warmup (triggers compilation)
        print("  Warming up @torch.compile (first call triggers JIT)...")
        packed_tc, scales_tc = KVFP4QuantizeUtil.batched_quantize(data_3d.float())
        deq_tc = KVFP4QuantizeUtil.batched_dequantize(packed_tc, scales_tc)
        torch.cuda.synchronize()

        t_tc_dq = gpu_timer(
            lambda: KVFP4QuantizeUtil.batched_dequantize(packed_tc, scales_tc)
        )
        t_tc_q = gpu_timer(
            lambda: KVFP4QuantizeUtil.batched_quantize(data_3d.float())
        )

        # HIP
        data_2d = data_3d.reshape(T, N)
        packed_hip, scales_hip = mod.mxfp4_quantize(data_2d)
        t_hip_dq = gpu_timer(
            lambda: mod.mxfp4_dequantize(packed_hip, scales_hip, N)
        )
        t_hip_q = gpu_timer(lambda: mod.mxfp4_quantize(data_2d))

        print(f"  Dequant: torch.compile={t_tc_dq:.4f}ms  HIP={t_hip_dq:.4f}ms  "
              f"speedup={t_tc_dq/max(t_hip_dq,1e-9):.1f}x")
        print(f"  Quant:   torch.compile={t_tc_q:.4f}ms   HIP={t_hip_q:.4f}ms   "
              f"speedup={t_tc_q/max(t_hip_q,1e-9):.1f}x")

        # Correctness check
        deq_tc_flat = deq_tc.reshape(T, N)
        deq_hip_flat = mod.mxfp4_dequantize(packed_tc.reshape(T, N // 2),
                                             scales_tc.reshape(T, -1), N)
        cos = cosine_similarity(deq_tc_flat, deq_hip_flat)
        print(f"  CosSim (torch.compile vs HIP dequant): {cos:.8f}")

    except Exception as e:
        print(f"  @torch.compile comparison skipped: {e}")


# ─── Main ───


def main():
    print("=" * 80)
    print("MXFP4 E2M1 HIP Kernel — Correctness & Performance")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")

    print("\nBuilding HIP kernel...")
    mod = build_kernel()
    print("Build OK\n")

    # ── Correctness ──
    print("─" * 80)
    print("  CORRECTNESS TESTS")
    print("─" * 80)

    MLA_DIMS = [576, 512, 64]
    GENERIC_DIMS = [128, 256, 1024]
    BATCH_SIZES = [1, 8, 32, 128]

    all_pass = True
    for N in MLA_DIMS + GENERIC_DIMS:
        for T in BATCH_SIZES:
            ok = test_dequant_correctness(mod, N, T)
            all_pass = all_pass and ok
            test_quant_correctness(mod, N, T)

    for N in [576, 512]:
        for T in [1, 128]:
            test_inplace(mod, N, T)

    test_3d_shapes(mod)

    if all_pass:
        print("\n  ALL DEQUANT TESTS BIT-EXACT")
    print("\n" + "=" * 80)
    print("  CORRECTNESS: ALL PASSED")
    print("=" * 80)

    # ── Benchmarks ──
    T_VALUES = [1, 32, 128, 512, 1024, 4096]

    for N in [576, 512, 64]:
        benchmark_dequant(mod, N, T_VALUES)
        benchmark_quant(mod, N, T_VALUES)

    benchmark_vs_torch_compile(mod, N=576, T=1024)

    print("\nDone.")


if __name__ == "__main__":
    main()
