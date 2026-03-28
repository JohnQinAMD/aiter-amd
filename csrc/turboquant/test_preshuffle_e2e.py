"""End-to-end test: preshuffle → flatmm kernel → compare with fused kernel reference.

The fused kernel (turboquant_fused_gemm.cu) is known correct (CosSim=1.0).
This test verifies that the flatmm kernel with preshuffled data produces
matching results, proving the preshuffle ↔ distribution alignment.
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn.functional as F
import importlib.util

_spec = importlib.util.spec_from_file_location("tq_engine",
    "/shared_aig/john/semianalysis/sglang-amd/python/sglang/srt/layers/quantization/turboquant_engine.py")
_tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preshuffle_flatmm import preshuffle_weight_flatmm_fast

from torch.utils.cpp_extension import load
ext_dir = os.path.dirname(os.path.abspath(__file__))
ck_base = "/shared_aig/john/semianalysis/aiter-amd/3rdparty/composable_kernel"

print("Building flatmm kernel...")
ext_flatmm = load(name="tq_flatmm_psh",
    sources=[os.path.join(ext_dir, "turboquant_int4_flatmm.hip")],
    extra_include_paths=[ext_dir, ck_base + "/include"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
    verbose=False)
print("Flatmm build OK")

print("Building fused kernel (reference)...")
ext_fused = load(name="tq_fused_psh",
    sources=[os.path.join(ext_dir, "turboquant_fused_gemm.cu")],
    extra_include_paths=[os.path.join(ext_dir, "include")],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
    verbose=False)
print("Fused build OK\n")

device = torch.device("cuda")

def dequant_ref(x, packed, cb, norms, gs):
    N, half_K = packed.shape; K = half_K * 2
    inv_s = 1.0 / math.sqrt(gs)
    p_np = packed.cpu().numpy()
    cb_np = cb.cpu().float().numpy()
    n_np = norms.cpu().float().numpy()
    W = np.zeros((N, K), dtype=np.float32)
    for k in range(K):
        bi = k // 2
        idx = (p_np[:, bi] & 0xF) if k % 2 == 0 else ((p_np[:, bi] >> 4) & 0xF)
        g = k // gs
        nv = n_np[:, g] if norms.ndim == 2 else n_np
        W[:, k] = cb_np[idx.astype(np.int64)] * nv * inv_s
    return x.float() @ torch.from_numpy(W).to(device).T

def timeit(fn, n=200, warmup=20):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000

print("="*75)
print("PRESHUFFLE E2E TEST")
print("="*75)

for name, K, N, gs in [("small", 256, 256, 128), ("medium", 512, 512, 128), ("kv_b", 512, 1024, 128)]:
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.float32) * 0.02
    tq = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=42)
    idx = tq["indices_packed"]
    cb = tq["codebook"].float().to(device)
    norms = tq["norms"].float().to(device)

    # Preshuffle the packed indices
    idx_np = idx.numpy()
    idx_preshuffled = preshuffle_weight_flatmm_fast(idx_np, N, K)

    # Upload both to GPU
    idx_orig = idx.to(device)
    idx_T = idx_orig.T.contiguous()
    idx_flat = torch.from_numpy(idx_preshuffled).to(device)

    for B in [1, 4, 32]:
        x = torch.randn(B, K, dtype=torch.bfloat16, device=device)

        # Reference: PyTorch dequant
        y_ref = dequant_ref(x, idx_orig, cb, norms, gs)

        # Fused kernel (known correct, uses original unpreshuffled data)
        y_fused = ext_fused.turboquant_fused_gemm(
            x.float(), idx_orig, idx_T, cb, norms, gs)

        # FlatMM with unpreshuffled data
        try:
            y_flatmm_raw = ext_flatmm.turboquant_int4_flatmm_gemm(x, idx_orig, gs)
            cos_raw = F.cosine_similarity(
                y_ref.flatten().unsqueeze(0), y_flatmm_raw.float().flatten().unsqueeze(0)).item()
        except:
            cos_raw = float('nan')

        # FlatMM with preshuffled data (1D → needs reshape to (N, K/2))
        # The kernel expects (N, K/2) but preshuffle outputs 1D.
        # We need to pass it as the right shape.
        try:
            idx_flat_2d = idx_flat.reshape(N, K // 2)
            y_flatmm_psh = ext_flatmm.turboquant_int4_flatmm_gemm(x, idx_flat_2d, gs)
            cos_psh = F.cosine_similarity(
                y_ref.flatten().unsqueeze(0), y_flatmm_psh.float().flatten().unsqueeze(0)).item()
        except Exception as e:
            cos_psh = float('nan')
            print(f"  Preshuffle flatmm error: {e}")

        cos_fused = F.cosine_similarity(
            y_ref.flatten().unsqueeze(0), y_fused.float().flatten().unsqueeze(0)).item()

        print(f"  {name:<8} B={B:>3}  fused={cos_fused:.6f}  flatmm_raw={cos_raw:.6f}  flatmm_psh={cos_psh:.6f}")

print()

# Performance comparison (small shape where flatmm already shows 1.5x)
print("="*75)
print("PERFORMANCE: Fused vs FlatMM (preshuffled)")
print("="*75)
print(f"{'Layer':<10} {'B':>4} {'BF16':>10} {'Fused':>10} {'FlatMM':>10} {'Fused/BF16':>12} {'FlatMM/BF16':>12}")
print("-" * 75)

for name, K, N, gs in [("small", 256, 256, 128)]:
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.float32) * 0.02
    tq = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=42)
    idx = tq["indices_packed"]
    idx_orig = idx.to(device)
    idx_T = idx_orig.T.contiguous()
    cb = tq["codebook"].float().to(device)
    norms = tq["norms"].float().to(device)
    idx_psh_2d = torch.from_numpy(preshuffle_weight_flatmm_fast(idx.numpy(), N, K)).reshape(N, K//2).to(device)
    W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02

    for B in [1, 4, 32]:
        x = torch.randn(B, K, dtype=torch.bfloat16, device=device)
        t_bf16 = timeit(lambda: F.linear(x, W_bf16))
        t_fused = timeit(lambda: ext_fused.turboquant_fused_gemm(x.float(), idx_orig, idx_T, cb, norms, gs))
        try:
            t_flatmm = timeit(lambda: ext_flatmm.turboquant_int4_flatmm_gemm(x, idx_psh_2d, gs))
        except:
            t_flatmm = float('inf')
        print(f"  {name:<8} {B:>4} {t_bf16:>10.4f} {t_fused:>10.4f} {t_flatmm:>10.4f} {t_bf16/t_fused:>11.2f}x {t_bf16/t_flatmm:>11.2f}x")
