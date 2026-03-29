"""Test MHATokenToKVPoolTQ for GQA/MHA models.

Tests per-head quantization with various head_dim and head_num configs
matching real models: Qwen3-235B (head_dim=128, 4 KV heads),
GPT-OSS-120B (head_dim=64, 8 KV heads), Llama-3-8B (head_dim=128, 8 KV heads).
"""
import sys, math
sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")

import torch
import torch.nn.functional as F

from sglang.srt.layers.quantization.turboquant_engine import (
    get_codebook, generate_rotation_matrix,
    pack_indices, unpack_indices, pad_for_packing, packed_bytes_per_dim,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

all_ok = True
def check(name, cond):
    global all_ok
    if not cond: all_ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def test_per_head_quantize(head_dim, head_num, T=64, bit_width=4):
    """Test per-head quantize/dequantize."""
    centroids, boundaries = get_codebook(bit_width)
    centroids_d = centroids.to(DEVICE)
    boundaries_d = boundaries.to(DEVICE)
    n_levels = 2 ** bit_width
    Pi = generate_rotation_matrix(head_dim, seed=42).to(DEVICE)
    scale = math.sqrt(head_dim)

    K = torch.randn(T, head_num, head_dim, device=DEVICE) * 0.02
    V = torch.randn(T, head_num, head_dim, device=DEVICE) * 0.02

    pb = packed_bytes_per_dim(head_dim, bit_width)
    bytes_per_head = pb + 2  # packed + FP16 norm

    # Compress per-head
    for name, data in [("K", K), ("V", V)]:
        compressed = torch.zeros(T, head_num * bytes_per_head, dtype=torch.uint8, device=DEVICE)

        for h in range(head_num):
            head_data = data[:, h, :].float()
            norms = head_data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            head_norm = head_data / norms

            Y = head_norm @ Pi.T * scale
            indices = torch.searchsorted(boundaries_d, Y.reshape(-1))
            indices = indices.clamp(0, n_levels - 1).reshape(T, head_dim)

            padded = pad_for_packing(head_dim, bit_width)
            if padded > head_dim:
                indices = F.pad(indices, (0, padded - head_dim))
            packed = pack_indices(indices, bit_width)

            off = h * bytes_per_head
            compressed[:, off:off + pb] = packed
            compressed[:, off + pb:off + bytes_per_head] = (
                norms.squeeze(1).half().view(torch.uint8).reshape(T, 2)
            )

        # Decompress
        recon = torch.zeros_like(data, dtype=torch.float32)
        for h in range(head_num):
            off = h * bytes_per_head
            packed = compressed[:, off:off + pb]
            norms_raw = compressed[:, off + pb:off + bytes_per_head]
            norms = norms_raw.view(torch.float16).reshape(T).float()

            padded = pad_for_packing(head_dim, bit_width)
            indices = unpack_indices(packed, padded, bit_width)[:, :head_dim]

            Y_hat = centroids_d[indices.long()] / scale
            L = (Y_hat @ Pi) * norms.unsqueeze(1)
            recon[:, h, :] = L

        cos = F.cosine_similarity(
            data.float().flatten().unsqueeze(0),
            recon.flatten().unsqueeze(0)
        ).item()
        return cos


print("=" * 60)
print("MHA/GQA Per-Head TurboQuant Quality Test")
print("=" * 60)

configs = [
    ("Qwen3-235B (GQA)", 128, 4),
    ("GPT-OSS-120B (GQA)", 64, 8),
    ("Llama-3-8B (GQA)", 128, 8),
    ("MQA model", 128, 1),
    ("Small head_dim", 64, 4),
]

for model, hd, hn in configs:
    print(f"\n  {model}: head_dim={hd}, head_num={hn}")
    for bw in [2, 3, 4]:
        cos = test_per_head_quantize(hd, hn, T=64, bit_width=bw)
        thresh = {4: 0.99, 3: 0.97, 2: 0.90}
        check(f"  {bw}-bit CosSim={cos:.6f} (>{thresh[bw]})", cos > thresh[bw])

# Compression ratio comparison
print()
print("=" * 60)
print("Compression Ratios for GQA/MHA")
print("=" * 60)
print(f"{'Model':<25} {'head_dim':>8} {'KV heads':>8} {'BF16 B/tok':>11} {'4-bit':>8} {'3-bit':>8} {'2-bit':>8}")
print("-" * 80)

for model, hd, hn in configs:
    fp16 = hn * hd * 2 * 2  # K + V, FP16
    for bw in [4, 3, 2]:
        pb = packed_bytes_per_dim(hd, bw)
        comp = hn * (pb + 2) * 2  # K + V
    line = f"{model:<25} {hd:>8} {hn:>8} {fp16:>11}"
    for bw in [4, 3, 2]:
        pb = packed_bytes_per_dim(hd, bw)
        comp = hn * (pb + 2) * 2
        ratio = fp16 / comp
        line += f" {ratio:>7.2f}x"
    print(line)

print()
print("=" * 60)
print("ALL PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 60)
sys.exit(0 if all_ok else 1)
