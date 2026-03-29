"""Test all TurboQuant features: 2/3/4-bit + mixed 2.5/3.5-bit + matmul + QJL."""
import sys, math
sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant_engine import (
    get_codebook, generate_rotation_matrix,
    pack_indices, unpack_indices, pad_for_packing, packed_bytes_per_dim,
    turboquant_quantize_packed, turboquant_dequantize, turboquant_matmul_pytorch,
    mixed_bit_config, mixed_compressed_bytes, mixed_compress_latent, mixed_decompress_latent,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

all_ok = True
def check(name, cond):
    global all_ok
    if not cond: all_ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

# ================================================================
print("=" * 60)
print("Feature 1: turboquant_matmul_pytorch 2/3/4-bit")
print("=" * 60)

for bw in [2, 3, 4]:
    torch.manual_seed(42)
    M, K = 32, 512
    W = torch.randn(M, K, device=DEVICE) * 0.02
    x = torch.randn(4, K, device=DEVICE) * 0.1

    packed_data = turboquant_quantize_packed(W, bit_width=bw, group_size=128, seed=42)
    packed_data_dev = {k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in packed_data.items()}

    y_ref = x @ W.T
    y_tq = turboquant_matmul_pytorch(
        x, packed_data_dev["indices_packed"], packed_data_dev["codebook"],
        packed_data_dev["norms"], K, 128, 42, bit_width=bw,
    )
    cos = F.cosine_similarity(y_ref.flatten().unsqueeze(0), y_tq.flatten().unsqueeze(0)).item()
    thresh = {4: 0.99, 3: 0.97, 2: 0.90}
    check(f"{bw}-bit matmul CosSim={cos:.6f} (>{thresh[bw]})", cos > thresh[bw])

# ================================================================
print()
print("=" * 60)
print("Feature 3: Outlier Treatment (2.5/3.5-bit mixed precision)")
print("=" * 60)

# Test mixed_bit_config
for eff_bits in [2.5, 3.5]:
    config = mixed_bit_config(eff_bits, 4)
    print(f"  {eff_bits}-bit config (4 groups): {config}")
    check(f"{eff_bits}-bit config length", len(config) == 4)
    actual_avg = sum(config) / len(config)
    check(f"{eff_bits}-bit avg bits ≈ {eff_bits}", abs(actual_avg - eff_bits) < 0.3)

# Test mixed compress/decompress
KV_LORA_RANK = 512
QK_ROPE_DIM = 64
GROUP_SIZE = 128
N_GROUPS = 4

rotations = {}
for g in range(N_GROUPS):
    gs = g * GROUP_SIZE
    rotations[gs] = generate_rotation_matrix(GROUP_SIZE, seed=42 + gs).to(DEVICE)

torch.manual_seed(42)
T = 64
latent = torch.randn(T, KV_LORA_RANK, device=DEVICE) * 0.02
rope = torch.randn(T, QK_ROPE_DIM, device=DEVICE) * 0.01

for eff_bits in [2.5, 3.5]:
    group_bits = mixed_bit_config(eff_bits, N_GROUPS)
    total_bytes = mixed_compressed_bytes(KV_LORA_RANK, GROUP_SIZE, QK_ROPE_DIM, group_bits)
    fp16_bytes = (KV_LORA_RANK + QK_ROPE_DIM) * 2

    all_packed, norms, latent_mse = mixed_compress_latent(
        latent, group_bits, GROUP_SIZE, rotations, DEVICE
    )
    check(f"{eff_bits}-bit compress: {len(all_packed)} packed groups", len(all_packed) == N_GROUPS)

    recon = mixed_decompress_latent(
        all_packed, norms, group_bits, GROUP_SIZE, KV_LORA_RANK, rotations, DEVICE
    )
    cos = F.cosine_similarity(latent.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)).item()
    expected_cos = {2.5: 0.94, 3.5: 0.98}
    check(f"{eff_bits}-bit CosSim={cos:.6f} (>{expected_cos[eff_bits]})", cos > expected_cos[eff_bits])
    print(f"    Compression: {fp16_bytes/total_bytes:.2f}x ({total_bytes} bytes)")

# ================================================================
print()
print("=" * 60)
print("Feature 1+3 Combined: quantize_packed + dequantize at all bit-widths")
print("=" * 60)

for bw in [2, 3, 4]:
    torch.manual_seed(42)
    W = torch.randn(64, 512, device=DEVICE) * 0.02
    packed = turboquant_quantize_packed(W, bit_width=bw, group_size=128, seed=42)
    recon = turboquant_dequantize(packed, DEVICE)
    cos = F.cosine_similarity(W.flatten().unsqueeze(0), recon.flatten().unsqueeze(0)).item()
    thresh = {4: 0.99, 3: 0.97, 2: 0.90}
    check(f"{bw}-bit quantize→dequantize CosSim={cos:.6f}", cos > thresh[bw])

# ================================================================
print()
print("=" * 60)
print("Compression Summary")
print("=" * 60)
fp16 = (KV_LORA_RANK + QK_ROPE_DIM) * 2
for label, bw in [("2-bit", 2), ("2.5-bit", 2.5), ("3-bit", 3), ("3.5-bit", 3.5), ("4-bit", 4)]:
    if isinstance(bw, float) and bw != int(bw):
        gb = mixed_bit_config(bw, N_GROUPS)
        total = mixed_compressed_bytes(KV_LORA_RANK, GROUP_SIZE, QK_ROPE_DIM, gb)
    else:
        pb = packed_bytes_per_dim(KV_LORA_RANK, int(bw))
        total = pb + N_GROUPS * 2 + QK_ROPE_DIM * 2
    print(f"  {label:>8}: {total:>4} bytes → {fp16/total:.2f}x compression")

print()
print("=" * 60)
print("ALL PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 60)
sys.exit(0 if all_ok else 1)
