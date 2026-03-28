"""TurboQuant KV Cache Compression: correctness + memory + latency benchmark.

Tests Stage 1 (MSE-only) and Stage 1+2 (MSE+QJL) for KV cache compression
on DeepSeek-V3 MLA dimensions.

Target: AMD MI355X with aiter backend.
The aiter MLA kernel uses get_key_buffer() which returns FP16 tensors.
For Phase 1 integration: quantize on write, dequantize on read.
"""
import sys, math, time
import importlib.util
import torch
import torch.nn.functional as F

_spec = importlib.util.spec_from_file_location("tq_kv",
    "/shared_aig/john/semianalysis/sglang-amd/python/sglang/srt/layers/quantization/turboquant_kv.py")
_tq_kv = importlib.util.module_from_spec(_spec)
# Need to handle the sglang import chain
sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")
try:
    _spec.loader.exec_module(_tq_kv)
except Exception:
    # Direct import of the engine if sglang import chain fails
    pass

_spec2 = importlib.util.spec_from_file_location("tq_engine",
    "/shared_aig/john/semianalysis/sglang-amd/python/sglang/srt/layers/quantization/turboquant_engine.py")
_tq = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(_tq)

device = torch.device("cuda")

def timeit(fn, n=100, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# ================================================================
print("=" * 75)
print("TURBOQUANT KV CACHE COMPRESSION BENCHMARK")
print("Target: DeepSeek-V3 MLA on MI355X (aiter backend)")
print("=" * 75)

# DeepSeek-V3 MLA dimensions
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
KV_CACHE_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
LAYERS = 61

print(f"\nDimensions: kv_lora_rank={KV_LORA_RANK}, rope_dim={QK_ROPE_HEAD_DIM}")
print(f"KV cache dim: {KV_CACHE_DIM}, Layers: {LAYERS}")

# ================================================================
print("\n" + "=" * 75)
print("1. COMPRESSION QUALITY (Stage 1: MSE-only)")
print("=" * 75)

# Create compressor (MSE-only, no QJL — for values)
from sglang.srt.layers.quantization.turboquant_kv import TurboQuantKVCompressor

for bits in [4, 3]:
    comp = TurboQuantKVCompressor(
        kv_lora_rank=KV_LORA_RANK,
        qk_rope_head_dim=QK_ROPE_HEAD_DIM,
        bit_width=bits,
        group_size=128,
        seed=42,
        use_qjl=False,
        device="cuda",
    )

    torch.manual_seed(42)
    for ctx_len in [1024, 4096]:
        kv_data = torch.randn(ctx_len, KV_CACHE_DIM, dtype=torch.float16, device=device)

        compressed = comp.compress(kv_data)
        decompressed = comp.decompress(compressed)

        # Measure quality
        cos = F.cosine_similarity(
            kv_data.float().flatten().unsqueeze(0),
            decompressed.float().flatten().unsqueeze(0)
        ).item()

        # MSE
        mse = ((kv_data.float() - decompressed.float()) ** 2).mean().item()
        snr = 10 * math.log10(kv_data.float().var().item() / (mse + 1e-30))

        # Memory
        orig_bytes = ctx_len * KV_CACHE_DIM * 2
        comp_bytes = (compressed["indices_packed"].numel()
                     + compressed["norms"].numel() * 2
                     + compressed["rope_part"].numel() * 2)
        ratio = orig_bytes / comp_bytes

        print(f"  {bits}-bit ctx={ctx_len}: CosSim={cos:.6f} SNR={snr:.1f}dB "
              f"ratio={ratio:.2f}x ({orig_bytes//1024}KB→{comp_bytes//1024}KB)")


# ================================================================
print("\n" + "=" * 75)
print("2. COMPRESSION QUALITY (Stage 1+2: MSE+QJL)")
print("=" * 75)

comp_qjl = TurboQuantKVCompressor(
    kv_lora_rank=KV_LORA_RANK,
    qk_rope_head_dim=QK_ROPE_HEAD_DIM,
    bit_width=4,
    group_size=128,
    seed=42,
    use_qjl=True,
    device="cuda",
)

torch.manual_seed(42)
for ctx_len in [1024, 4096]:
    kv_data = torch.randn(ctx_len, KV_CACHE_DIM, dtype=torch.float16, device=device)

    compressed = comp_qjl.compress(kv_data)

    # Test asymmetric inner product quality
    n_queries = 32
    queries = torch.randn(n_queries, KV_LORA_RANK, dtype=torch.float16, device=device)

    # Reference: Q @ K_latent^T (FP16, no quantization)
    k_latent = kv_data[:, :KV_LORA_RANK]  # (ctx_len, 512)
    scores_ref = queries.float() @ k_latent.float().T  # (n_queries, ctx_len)

    # TurboQuant asymmetric: Q @ K_mse^T + QJL correction
    scores_tq = comp_qjl.asymmetric_attention_scores(
        queries.unsqueeze(0),  # (1, n_queries, 512)
        compressed,
    ).squeeze(0)  # (n_queries, ctx_len)

    # Also test MSE-only (no QJL correction)
    k_mse = compressed["k_mse"].float()
    scores_mse = queries.float() @ k_mse.T

    cos_tq = F.cosine_similarity(
        scores_ref.flatten().unsqueeze(0),
        scores_tq.flatten().unsqueeze(0)
    ).item()
    cos_mse = F.cosine_similarity(
        scores_ref.flatten().unsqueeze(0),
        scores_mse.flatten().unsqueeze(0)
    ).item()

    # Memory with QJL
    comp_bytes_qjl = (compressed["indices_packed"].numel()
                     + compressed["norms"].numel() * 2
                     + compressed["rope_part"].numel() * 2
                     + compressed["qjl_signs"].numel()  # int8
                     + compressed["residual_norm"].numel() * 2)
    # Note: k_mse stored separately for Term 1. If dequanted on-the-fly, don't count it.
    orig_bytes = ctx_len * KV_CACHE_DIM * 2
    ratio_qjl = orig_bytes / comp_bytes_qjl

    print(f"  ctx={ctx_len}: attn_score CosSim: QJL={cos_tq:.6f} MSE-only={cos_mse:.6f}")
    print(f"    QJL improves attention score accuracy by {(cos_tq - cos_mse) * 100:.2f}%")
    print(f"    Memory: {ratio_qjl:.2f}x compression (excluding k_mse)")


# ================================================================
print("\n" + "=" * 75)
print("3. QUANTIZE/DEQUANTIZE LATENCY")
print("=" * 75)

print(f"\n{'Operation':<25} {'ctx_len':>8} {'Time (ms)':>10} {'per-token':>12}")
print("-" * 60)

for ctx_len in [1, 8, 32, 128, 1024]:
    kv_data = torch.randn(ctx_len, KV_CACHE_DIM, dtype=torch.float16, device=device)

    # Quantize latency
    t_quant = timeit(lambda: comp_qjl.compress(kv_data), n=50, warmup=5)
    per_token_q = t_quant / ctx_len * 1000  # microseconds

    # Dequantize latency
    compressed = comp_qjl.compress(kv_data)
    t_deq = timeit(lambda: comp_qjl.decompress(compressed), n=50, warmup=5)
    per_token_d = t_deq / ctx_len * 1000

    print(f"  Quantize (QJL)          {ctx_len:>8} {t_quant:>9.4f} {per_token_q:>10.1f} µs")
    print(f"  Dequantize (MSE)        {ctx_len:>8} {t_deq:>9.4f} {per_token_d:>10.1f} µs")


# ================================================================
print("\n" + "=" * 75)
print("4. ATTENTION SCORE COMPUTATION LATENCY")
print("=" * 75)

print(f"\n{'Method':<25} {'ctx_len':>8} {'Time (ms)':>10}")
print("-" * 50)

for ctx_len in [1024, 4096, 8192]:
    kv_data = torch.randn(ctx_len, KV_CACHE_DIM, dtype=torch.float16, device=device)
    queries = torch.randn(1, KV_LORA_RANK, dtype=torch.float16, device=device)

    # FP16 attention score: Q @ K^T
    k_fp16 = kv_data[:, :KV_LORA_RANK]
    t_fp16 = timeit(lambda: queries.float() @ k_fp16.float().T)

    # TurboQuant asymmetric score
    compressed = comp_qjl.compress(kv_data)
    t_tq = timeit(lambda: comp_qjl.asymmetric_attention_scores(
        queries.unsqueeze(0), compressed), n=50, warmup=5)

    print(f"  FP16 Q@K^T              {ctx_len:>8} {t_fp16:>9.4f}")
    print(f"  TQ asymmetric           {ctx_len:>8} {t_tq:>9.4f}")


# ================================================================
print("\n" + "=" * 75)
print("5. MEMORY SAVINGS SUMMARY")
print("=" * 75)

mem = comp_qjl.memory_usage_per_token()
print(f"\n  Per token per layer:")
print(f"    FP16:       {mem['original_bytes']:>6} bytes")
print(f"    TQ-4bit:    {mem['compressed_bytes']:>6} bytes")
print(f"    Ratio:      {mem['compression_ratio']:.2f}x")

print(f"\n  Total KV cache ({LAYERS} layers):")
for ctx in [4096, 32768, 131072]:
    fp16_gb = ctx * mem['original_bytes'] * LAYERS / 1024**3
    tq_gb = ctx * mem['compressed_bytes'] * LAYERS / 1024**3
    print(f"    ctx={ctx:>6}: FP16={fp16_gb:.2f}GB  TQ={tq_gb:.2f}GB  saved={fp16_gb-tq_gb:.2f}GB")

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)
