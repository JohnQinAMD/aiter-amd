"""GPU kernel unit tests for 2/3/4-bit TurboQuant KV compress/decompress.

Tests:
  1. Compile kernel for all bit-widths
  2. GPU compress → GPU decompress round-trip (CosSim) per bit-width
  3. GPU compress → Python decompress cross-validation
  4. Python compress → GPU decompress cross-validation
  5. CUDA graph capture for each bit-width
  6. Performance benchmark per bit-width

Run inside ROCm container:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /shared_aig:/shared_aig \
    rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260324 \
    python3 /shared_aig/john/semianalysis/aiter-amd/csrc/turboquant/test_multibit_gpu.py
"""
import sys, os, math, time
sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant_engine import (
    get_codebook, generate_rotation_matrix,
    pack_indices, unpack_indices, packed_bytes_per_dim, pad_for_packing,
)

DEVICE = torch.device("cuda")
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
KV_DIM = KV_LORA_RANK + QK_ROPE_DIM
GROUP_SIZE = 128
N_GROUPS = KV_LORA_RANK // GROUP_SIZE

# Build GPU kernel
print("Compiling GPU kernel (2/3/4-bit)...")
from torch.utils.cpp_extension import load
ext_dir = os.path.dirname(os.path.abspath(__file__))
ck_base = "/shared_aig/john/semianalysis/aiter-amd/3rdparty/composable_kernel"
ext = load(
    name="tq_multibit_test",
    sources=[os.path.join(ext_dir, "turboquant_kv_compress.hip")],
    extra_include_paths=[ext_dir, ck_base + "/include"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
    verbose=False,
)
print("GPU kernel compiled OK\n")

Pi_all = torch.stack([
    generate_rotation_matrix(GROUP_SIZE, seed=42 + g * GROUP_SIZE).to(DEVICE)
    for g in range(N_GROUPS)
])

all_pass = True

def check(name, cond):
    global all_pass
    status = "PASS" if cond else "FAIL"
    if not cond: all_pass = False
    print(f"  [{status}] {name}")


def py_compress(kv, bit_width):
    """Python reference compress."""
    T = kv.shape[0]
    centroids, boundaries = get_codebook(bit_width)
    boundaries_d = boundaries.to(DEVICE)
    n_levels = 2 ** bit_width

    all_idx, all_norms = [], []
    latent = kv[:, :KV_LORA_RANK].float()
    rope = kv[:, KV_LORA_RANK:].float()

    for g in range(N_GROUPS):
        gs = g * GROUP_SIZE
        L_g = latent[:, gs:gs+GROUP_SIZE]
        norms = L_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        all_norms.append(norms.squeeze(1))
        Pi = Pi_all[g]
        Y = (L_g / norms) @ Pi.T * math.sqrt(GROUP_SIZE)
        idx = torch.searchsorted(boundaries_d, Y.reshape(-1)).clamp(0, n_levels-1).reshape(T, GROUP_SIZE)
        all_idx.append(idx)

    full_idx = torch.cat(all_idx, dim=1)
    norms_t = torch.stack(all_norms, dim=1).half()
    padded = pad_for_packing(KV_LORA_RANK, bit_width)
    if padded > KV_LORA_RANK:
        full_idx = F.pad(full_idx, (0, padded - KV_LORA_RANK))
    packed = pack_indices(full_idx, bit_width)

    pb = packed_bytes_per_dim(KV_LORA_RANK, bit_width)
    total = pb + N_GROUPS * 2 + QK_ROPE_DIM * 2
    result = torch.zeros(T, total, dtype=torch.uint8, device=DEVICE)
    result[:, :pb] = packed
    result[:, pb:pb + N_GROUPS * 2] = norms_t.view(torch.uint8).reshape(T, -1)
    result[:, pb + N_GROUPS * 2:] = rope.half().contiguous().view(torch.uint8).reshape(T, -1)
    return result


def py_decompress(compressed, bit_width):
    """Python reference decompress."""
    T = compressed.shape[0]
    centroids, _ = get_codebook(bit_width)
    centroids_d = centroids.to(DEVICE)

    pb = packed_bytes_per_dim(KV_LORA_RANK, bit_width)
    packed = compressed[:, :pb]
    norms_raw = compressed[:, pb:pb + N_GROUPS * 2]
    rope_raw = compressed[:, pb + N_GROUPS * 2:]

    norms = norms_raw.view(torch.float16).reshape(T, N_GROUPS).float()
    rope = rope_raw.view(torch.float16).reshape(T, QK_ROPE_DIM)

    padded = pad_for_packing(KV_LORA_RANK, bit_width)
    indices = unpack_indices(packed, padded, bit_width)[:, :KV_LORA_RANK]

    latent = torch.zeros(T, KV_LORA_RANK, dtype=torch.float32, device=DEVICE)
    for g in range(N_GROUPS):
        gs = g * GROUP_SIZE
        scale = math.sqrt(GROUP_SIZE)
        Pi = Pi_all[g]
        Y_g = centroids_d[indices[:, gs:gs+GROUP_SIZE].long()] / scale
        L_g = Y_g @ Pi
        L_g = L_g * norms[:, g].unsqueeze(1)
        latent[:, gs:gs+GROUP_SIZE] = L_g

    out = torch.cat([latent.to(torch.bfloat16), rope.to(torch.bfloat16)], dim=-1)
    return out


# ════════════════════════════════════════════
print("=" * 65)
print("Multi-bit GPU Kernel Unit Tests")
print("=" * 65)

for bw in [2, 3, 4]:
    centroids, boundaries = get_codebook(bw)
    centroids_d = centroids.to(DEVICE)
    boundaries_d = boundaries.to(DEVICE)
    n_levels = 2 ** bw

    print(f"\n--- {bw}-bit ({n_levels} levels) ---")

    torch.manual_seed(42)
    kv = torch.randn(32, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.1

    # Test 1: GPU round-trip
    gpu_comp = ext.turboquant_kv_compress(kv, Pi_all, boundaries_d, N_GROUPS, GROUP_SIZE, bw)
    gpu_dec = ext.turboquant_kv_decompress(gpu_comp, Pi_all, centroids_d, N_GROUPS, GROUP_SIZE,
                                            KV_LORA_RANK, QK_ROPE_DIM, bw)
    cos_gpu = F.cosine_similarity(
        kv[:, :KV_LORA_RANK].float().flatten().unsqueeze(0),
        gpu_dec[:, :KV_LORA_RANK].float().flatten().unsqueeze(0)
    ).item()
    expected_cos = {4: 0.99, 3: 0.97, 2: 0.90}
    check(f"GPU round-trip CosSim={cos_gpu:.6f} (expect >{expected_cos[bw]})", cos_gpu > expected_cos[bw])

    # Test 2: GPU compress → Python decompress
    py_dec = py_decompress(gpu_comp, bw)
    cos_cross1 = F.cosine_similarity(
        gpu_dec[:, :KV_LORA_RANK].float().flatten().unsqueeze(0),
        py_dec[:, :KV_LORA_RANK].float().flatten().unsqueeze(0)
    ).item()
    check(f"GPU→Python cross-validate CosSim={cos_cross1:.6f}", cos_cross1 > 0.999)

    # Test 3: Python compress → GPU decompress
    py_comp = py_compress(kv, bw)
    gpu_dec2 = ext.turboquant_kv_decompress(py_comp, Pi_all, centroids_d, N_GROUPS, GROUP_SIZE,
                                             KV_LORA_RANK, QK_ROPE_DIM, bw)
    py_dec2 = py_decompress(py_comp, bw)
    cos_cross2 = F.cosine_similarity(
        gpu_dec2[:, :KV_LORA_RANK].float().flatten().unsqueeze(0),
        py_dec2[:, :KV_LORA_RANK].float().flatten().unsqueeze(0)
    ).item()
    check(f"Python→GPU cross-validate CosSim={cos_cross2:.6f}", cos_cross2 > 0.999)

    # Test 4: Compressed byte size
    expected_pb = packed_bytes_per_dim(KV_LORA_RANK, bw)
    expected_total = expected_pb + N_GROUPS * 2 + QK_ROPE_DIM * 2
    check(f"Compressed size: {gpu_comp.shape[1]} bytes (expected {expected_total})",
          gpu_comp.shape[1] == expected_total)

    # Test 5: CUDA graph capture (compress)
    try:
        kv_in = torch.randn(8, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.1
        comp_out = torch.empty(8, expected_total, dtype=torch.uint8, device=DEVICE)
        for _ in range(3):
            ext.turboquant_kv_compress_inplace(kv_in, Pi_all, boundaries_d, comp_out, N_GROUPS, GROUP_SIZE, bw)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            ext.turboquant_kv_compress_inplace(kv_in, Pi_all, boundaries_d, comp_out, N_GROUPS, GROUP_SIZE, bw)
        kv_in.copy_(torch.randn(8, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.1)
        g.replay()
        torch.cuda.synchronize()
        ref = ext.turboquant_kv_compress(kv_in, Pi_all, boundaries_d, N_GROUPS, GROUP_SIZE, bw)
        check(f"CUDA graph compress: {'BIT-EXACT' if torch.equal(comp_out, ref) else 'MISMATCH'}",
              torch.equal(comp_out, ref))
    except Exception as e:
        check(f"CUDA graph compress: {type(e).__name__}: {e}", False)

    # Test 6: CUDA graph capture (decompress)
    try:
        dec_out = torch.empty(8, KV_DIM, dtype=torch.bfloat16, device=DEVICE)
        for _ in range(3):
            ext.turboquant_kv_decompress_inplace(comp_out, Pi_all, centroids_d, dec_out,
                                                  N_GROUPS, GROUP_SIZE, KV_LORA_RANK, QK_ROPE_DIM, bw)
        torch.cuda.synchronize()

        g2 = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g2):
            ext.turboquant_kv_decompress_inplace(comp_out, Pi_all, centroids_d, dec_out,
                                                  N_GROUPS, GROUP_SIZE, KV_LORA_RANK, QK_ROPE_DIM, bw)
        g2.replay()
        torch.cuda.synchronize()
        ref_dec = ext.turboquant_kv_decompress(comp_out, Pi_all, centroids_d, N_GROUPS, GROUP_SIZE,
                                                KV_LORA_RANK, QK_ROPE_DIM, bw)
        cos_g = F.cosine_similarity(dec_out.float().flatten().unsqueeze(0),
                                     ref_dec.float().flatten().unsqueeze(0)).item()
        check(f"CUDA graph decompress CosSim={cos_g:.6f}", cos_g > 0.999)
    except Exception as e:
        check(f"CUDA graph decompress: {type(e).__name__}: {e}", False)

    # Test 7: Performance
    kv_perf = torch.randn(1, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.1
    comp_perf = ext.turboquant_kv_compress(kv_perf, Pi_all, boundaries_d, N_GROUPS, GROUP_SIZE, bw)

    def bench(fn, n=500, warmup=50):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n): fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n * 1e6  # microseconds

    t_comp = bench(lambda: ext.turboquant_kv_compress(kv_perf, Pi_all, boundaries_d, N_GROUPS, GROUP_SIZE, bw))
    t_dec = bench(lambda: ext.turboquant_kv_decompress(comp_perf, Pi_all, centroids_d, N_GROUPS, GROUP_SIZE,
                                                        KV_LORA_RANK, QK_ROPE_DIM, bw))
    fp16_bytes = KV_DIM * 2
    comp_ratio = fp16_bytes / expected_total
    print(f"  Perf: compress={t_comp:.1f}us  decompress={t_dec:.1f}us  compression={comp_ratio:.2f}x")


# Summary
print(f"\n{'=' * 65}")
print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
print(f"{'=' * 65}")

if not all_pass:
    sys.exit(1)
