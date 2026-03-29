"""MXFP4 vs TurboQuant KV cache quality & perf benchmark.

Tests E2M1 block-scale quantization against TurboQuant-4bit Lloyd-Max
for MLA latent vectors (DeepSeek-V3: kv_lora_rank=512, rope=64).

Run inside ROCm container:
  docker run --rm --device=/dev/kfd --device=/dev/dri --group-add video \
    -v /shared_aig:/shared_aig -v /mnt/m2m_nobackup:/mnt/m2m_nobackup \
    rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260324 \
    python3 /shared_aig/john/semianalysis/aiter-amd/csrc/turboquant/test_mxfp4_kv.py
"""

import sys
import time
import math
import torch
import numpy as np

torch.manual_seed(42)

KV_LORA_RANK = 512
QK_ROPE_DIM = 64
KV_DIM = KV_LORA_RANK + QK_ROPE_DIM  # 576
N_LAYERS = 61
BLOCK_SIZE = 16  # MXFP4 block size for scales
GROUP_SIZE = 128  # TurboQuant group size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name()}")

# ========================================================================
# MXFP4 (E2M1) quantize/dequant — from SGLang's KVFP4QuantizeUtil
# ========================================================================

E2M1_MAX = 6.0
E2M1_VALUES = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=torch.float32, device=DEVICE)
E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5], dtype=torch.float32, device=DEVICE)


def mxfp4_quantize(tensor):
    """Quantize to MXFP4 E2M1 with per-16-element block scales."""
    b, m, n = tensor.shape
    reshaped = tensor.view(b, m * n // BLOCK_SIZE, BLOCK_SIZE)
    block_max = reshaped.abs().max(dim=-1, keepdim=True).values
    scale_exp = torch.ceil(torch.log2(torch.clamp(block_max / E2M1_MAX, min=1e-10)))
    scale_factors = (scale_exp + 127).squeeze(-1).to(torch.uint8)
    scaled = reshaped / torch.exp2(scale_exp)
    sign_bits = (scaled < 0).to(torch.uint8) << 3
    abs_vals = scaled.abs()
    magnitude_bits = torch.sum(abs_vals.unsqueeze(-1) >= E2M1_BOUNDS, dim=-1)
    fp4_vals = sign_bits + magnitude_bits.to(torch.uint8)
    fp4_reshaped = fp4_vals.view(b, m, n)
    packed = (fp4_reshaped[..., 1::2] << 4) + fp4_reshaped[..., 0::2]
    return packed, scale_factors


def mxfp4_dequantize(packed, scale_factors, n):
    """Dequantize MXFP4 back to float."""
    b, m, n_half = packed.shape
    fp4_vals = torch.empty(b, m, n, dtype=torch.uint8, device=packed.device)
    fp4_vals[..., 0::2] = packed & 0x0F
    fp4_vals[..., 1::2] = (packed >> 4) & 0x0F
    sign_mask = (fp4_vals & 0x08) != 0
    magnitude_idx = fp4_vals & 0x07
    float_vals = E2M1_VALUES[magnitude_idx.long()]
    float_vals = torch.where(sign_mask, -float_vals, float_vals)
    reshaped = float_vals.view(b, m * n // BLOCK_SIZE, BLOCK_SIZE)
    scale_exp = scale_factors.float() - 127
    scaled = reshaped * torch.exp2(scale_exp.unsqueeze(-1))
    return scaled.view(b, m, n)


# ========================================================================
# TurboQuant-4bit quantize/dequant
# ========================================================================

sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")
try:
    from sglang.srt.layers.quantization.turboquant_engine import (
        get_codebook, generate_rotation_matrix, pack_4bit, unpack_4bit,
    )
    HAS_TQ = True
except ImportError:
    HAS_TQ = False
    print("WARNING: TurboQuant engine not available, skipping TQ comparison")


def tq4_quantize(latent):
    """TurboQuant-4bit quantize (latent only, not rope)."""
    T = latent.shape[0]
    n_groups = KV_LORA_RANK // GROUP_SIZE
    centroids, boundaries = get_codebook(4)
    centroids = centroids.to(DEVICE)
    boundaries = boundaries.to(DEVICE)

    all_indices = []
    all_norms = []

    for g in range(n_groups):
        g_start = g * GROUP_SIZE
        g_end = g_start + GROUP_SIZE
        L_g = latent[:, g_start:g_end].float()
        norms = L_g.norm(dim=1, keepdim=True).clamp(min=1e-8)
        L_norm = L_g / norms
        all_norms.append(norms.squeeze(1))
        Pi = generate_rotation_matrix(GROUP_SIZE, seed=42 + g_start).to(DEVICE)
        Y = L_norm @ Pi.T * math.sqrt(GROUP_SIZE)
        indices = torch.searchsorted(boundaries, Y.reshape(-1))
        indices = indices.clamp(0, 15).reshape(T, GROUP_SIZE)
        all_indices.append(indices)

    return torch.cat(all_indices, dim=1), torch.stack(all_norms, dim=1)


def tq4_dequantize(indices, norms):
    """TurboQuant-4bit dequantize."""
    T = indices.shape[0]
    centroids, _ = get_codebook(4)
    centroids = centroids.to(DEVICE)
    n_groups = KV_LORA_RANK // GROUP_SIZE
    latent = torch.zeros(T, KV_LORA_RANK, dtype=torch.float32, device=DEVICE)

    for g in range(n_groups):
        g_start = g * GROUP_SIZE
        g_end = g_start + GROUP_SIZE
        scale = math.sqrt(GROUP_SIZE)
        Pi = generate_rotation_matrix(GROUP_SIZE, seed=42 + g_start).to(DEVICE)
        Y_g = centroids[indices[:, g_start:g_end].long()] / scale
        L_g = Y_g @ Pi
        L_g = L_g * norms[:, g].unsqueeze(1)
        latent[:, g_start:g_end] = L_g

    return latent


# ========================================================================
# Compression ratio calculation
# ========================================================================

def mxfp4_compressed_size():
    """Bytes per token per layer for MXFP4 MLA KV cache."""
    latent_packed = KV_LORA_RANK // 2           # 256 bytes
    latent_scales = KV_LORA_RANK // BLOCK_SIZE  # 32 bytes (1 E8M0 per 16 elems)
    rope_packed = QK_ROPE_DIM // 2              # 32 bytes
    rope_scales = QK_ROPE_DIM // BLOCK_SIZE     # 4 bytes
    return latent_packed + latent_scales + rope_packed + rope_scales  # 324 bytes


def tq4_compressed_size():
    """Bytes per token per layer for TurboQuant-4bit MLA KV cache."""
    latent_packed = KV_LORA_RANK // 2              # 256 bytes
    norms = (KV_LORA_RANK // GROUP_SIZE) * 2       # 8 bytes (FP16)
    rope_fp16 = QK_ROPE_DIM * 2                    # 128 bytes (FP16, unquantized)
    return latent_packed + norms + rope_fp16         # 392 bytes


# ========================================================================
# Quality metrics
# ========================================================================

def cosine_similarity(a, b):
    a_flat = a.reshape(-1).float()
    b_flat = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def rmse(a, b):
    return ((a.float() - b.float()) ** 2).mean().sqrt().item()


def inner_product_error(orig, recon, n_pairs=1000):
    """Measure inner product preservation (attention score proxy)."""
    T = orig.shape[0]
    if T < 2:
        return 0, 0
    idx1 = torch.randint(0, T, (n_pairs,), device=orig.device)
    idx2 = torch.randint(0, T, (n_pairs,), device=orig.device)
    ip_orig = (orig[idx1].float() * orig[idx2].float()).sum(dim=-1)
    ip_recon = (recon[idx1].float() * recon[idx2].float()).sum(dim=-1)
    bias = (ip_recon - ip_orig).mean().item()
    mse = ((ip_recon - ip_orig) ** 2).mean().item()
    return bias, math.sqrt(mse)


# ========================================================================
# Main benchmark
# ========================================================================

def run_benchmark():
    print("=" * 70)
    print("MXFP4 vs TurboQuant-4bit KV Cache Quality & Performance Benchmark")
    print("=" * 70)

    fp16_size = KV_DIM * 2
    mx4_size = mxfp4_compressed_size()
    tq4_size = tq4_compressed_size()
    print(f"\n--- Compression Ratio (per token per layer) ---")
    print(f"FP16 baseline:    {fp16_size} bytes (1.0x)")
    print(f"MXFP4 (all dims): {mx4_size} bytes ({fp16_size/mx4_size:.2f}x)")
    print(f"TQ-4bit (latent): {tq4_size} bytes ({fp16_size/tq4_size:.2f}x)")

    # Generate realistic MLA latent vectors
    for T in [1, 32, 128, 1024]:
        print(f"\n{'='*70}")
        print(f"Batch size T={T}")
        print(f"{'='*70}")

        kv_data = torch.randn(T, 1, KV_DIM, dtype=torch.bfloat16, device=DEVICE) * 0.02
        latent = kv_data[:, 0, :KV_LORA_RANK].float()
        rope = kv_data[:, 0, KV_LORA_RANK:].float()

        # --- MXFP4: quantize full KV (latent + rope) ---
        t0 = time.perf_counter()
        for _ in range(10 if T <= 128 else 3):
            mx_packed, mx_scales = mxfp4_quantize(kv_data.float())
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t_mx_q = (time.perf_counter() - t0) / (10 if T <= 128 else 3) * 1000

        t0 = time.perf_counter()
        for _ in range(10 if T <= 128 else 3):
            mx_recon = mxfp4_dequantize(mx_packed, mx_scales, KV_DIM)
        if DEVICE == "cuda":
            torch.cuda.synchronize()
        t_mx_dq = (time.perf_counter() - t0) / (10 if T <= 128 else 3) * 1000

        mx_latent = mx_recon[:, 0, :KV_LORA_RANK]
        mx_rope = mx_recon[:, 0, KV_LORA_RANK:]

        print(f"\n  MXFP4 (E2M1, block_size=16):")
        print(f"    Latent CosSim:  {cosine_similarity(latent, mx_latent):.6f}")
        print(f"    Latent RMSE:    {rmse(latent, mx_latent):.6f}")
        print(f"    Rope CosSim:    {cosine_similarity(rope, mx_rope):.6f}")
        ip_bias, ip_rmse = inner_product_error(latent, mx_latent)
        print(f"    IP bias:        {ip_bias:.6f}")
        print(f"    IP RMSE:        {ip_rmse:.6f}")
        print(f"    Quantize:       {t_mx_q:.3f} ms")
        print(f"    Dequantize:     {t_mx_dq:.3f} ms")

        # --- TurboQuant-4bit ---
        if HAS_TQ:
            t0 = time.perf_counter()
            for _ in range(10 if T <= 128 else 3):
                tq_indices, tq_norms = tq4_quantize(latent)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t_tq_q = (time.perf_counter() - t0) / (10 if T <= 128 else 3) * 1000

            t0 = time.perf_counter()
            for _ in range(10 if T <= 128 else 3):
                tq_latent = tq4_dequantize(tq_indices, tq_norms)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t_tq_dq = (time.perf_counter() - t0) / (10 if T <= 128 else 3) * 1000

            print(f"\n  TurboQuant-4bit (Lloyd-Max, group_size=128):")
            print(f"    Latent CosSim:  {cosine_similarity(latent, tq_latent):.6f}")
            print(f"    Latent RMSE:    {rmse(latent, tq_latent):.6f}")
            print(f"    Rope:           kept in FP16 (lossless)")
            ip_bias_tq, ip_rmse_tq = inner_product_error(latent, tq_latent)
            print(f"    IP bias:        {ip_bias_tq:.6f}")
            print(f"    IP RMSE:        {ip_rmse_tq:.6f}")
            print(f"    Quantize:       {t_tq_q:.3f} ms")
            print(f"    Dequantize:     {t_tq_dq:.3f} ms")

        # --- MXFP4 latent-only (keep rope FP16 like TQ) ---
        mx_packed_l, mx_scales_l = mxfp4_quantize(latent.unsqueeze(1))
        mx_recon_l = mxfp4_dequantize(mx_packed_l, mx_scales_l, KV_LORA_RANK)[:, 0]

        mx4_latent_only = KV_LORA_RANK // 2 + KV_LORA_RANK // BLOCK_SIZE + QK_ROPE_DIM * 2
        print(f"\n  MXFP4 latent-only (rope FP16, like TQ):")
        print(f"    Latent CosSim:  {cosine_similarity(latent, mx_recon_l):.6f}")
        print(f"    Compression:    {fp16_size/mx4_latent_only:.2f}x ({mx4_latent_only} bytes)")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY (T=128)")
    print(f"{'='*70}")
    print(f"{'Method':<25} {'Compression':>12} {'CosSim':>10} {'Dequant ms':>12} {'Quantize ms':>12}")
    print(f"{'-'*25} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    print(f"{'MXFP4 (all dims)':<25} {fp16_size/mx4_size:>11.2f}x {'—':>10} {'—':>12} {'—':>12}")
    print(f"{'MXFP4 (latent+ropeFP16)':<25} {fp16_size/mx4_latent_only:>11.2f}x {'—':>10} {'—':>12} {'—':>12}")
    if HAS_TQ:
        print(f"{'TQ-4bit':<25} {fp16_size/tq4_size:>11.2f}x {'—':>10} {'—':>12} {'—':>12}")
    print("\n(Detailed per-batch results above)")


if __name__ == "__main__":
    run_benchmark()
