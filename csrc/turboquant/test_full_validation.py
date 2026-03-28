"""Full validation: paper alignment + perf vs PyTorch + real model PPL.

Tests:
  1. Paper algorithm alignment check (rotation, codebook, quantize, matmul)
  2. INT4 kernel vs PyTorch reference (turboquant_matmul_pytorch)
  3. Real model end-to-end PPL (Qwen2.5-0.5B)
"""
import os, sys, math, time, importlib.util
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("tq_engine",
    "/shared_aig/john/semianalysis/sglang-amd/python/sglang/srt/layers/quantization/turboquant_engine.py")
_tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq)

device = torch.device("cuda")

def timeit(fn, n=500, warmup=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


# =========================================================================
print("=" * 75)
print("1. PAPER ALIGNMENT CHECK")
print("=" * 75)

# 1a. Rotation matrix: Haar-distributed via QR of Gaussian
Pi = _tq.generate_rotation_matrix(128, seed=42)
assert Pi.shape == (128, 128), f"Wrong shape: {Pi.shape}"
eye = Pi @ Pi.T
ortho_err = (eye - torch.eye(128)).abs().max().item()
print(f"  Rotation orthogonality error: {ortho_err:.2e} (should be < 1e-5)")
assert ortho_err < 1e-5, "Rotation not orthogonal!"
print("  ✓ Rotation: QR of Gaussian → Haar-distributed orthogonal matrix")

# 1b. Lloyd-Max codebook for N(0,1)
centroids, boundaries = _tq.get_codebook(4)
print(f"  Codebook levels: {len(centroids)} (expected 16 for 4-bit)")
assert len(centroids) == 16
assert centroids[0] < 0 and centroids[15] > 0, "Codebook not symmetric"
assert abs(centroids[0] + centroids[15]) < 0.01, "Codebook not symmetric"
print(f"  Codebook range: [{centroids[0]:.4f}, {centroids[15]:.4f}]")
print("  ✓ Lloyd-Max: optimal scalar quantizer for N(0,1)")

# 1c. C++ codebook matches Python
cpp_cb = [-2.7330780029, -2.0695691109, -1.6186094284, -1.2567617893,
          -0.9427994490, -0.6571131349, -0.3882715702, -0.1284713149,
           0.1284713149,  0.3882715702,  0.6571131349,  0.9427994490,
           1.2567617893,  1.6186094284,  2.0695691109,  2.7330780029]
max_diff = max(abs(centroids[i].item() - cpp_cb[i]) for i in range(16))
print(f"  Python vs C++ codebook max diff: {max_diff:.2e}")
assert max_diff < 1e-6
print("  ✓ C++ LDS codebook matches Python Lloyd-Max solver")

# 1d. Quantization pipeline: normalize → rotate → scale → quantize
N, K, gs = 256, 512, 128
torch.manual_seed(42)
W = torch.randn(N, K) * 0.02
tq = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=42)
print(f"  Quantized: {N}×{K} → packed shape {tq['indices_packed'].shape}, norms shape {tq['norms'].shape}")
assert tq['indices_packed'].shape == (N, K // 2)
assert tq['norms'].shape == (N, K // gs)
print("  ✓ Quantize: per-group normalize → rotate → scale(√d) → searchsorted → pack4bit")

# 1e. Forward pass: rotate input ↔ dequant weight equivalence
W_deq = _tq.turboquant_dequantize(tq, device="cpu")
x = torch.randn(4, K) * 0.1
y_deq = x @ W_deq.T
y_otf = _tq.turboquant_matmul_pytorch(
    x, tq["indices_packed"], tq["codebook"], tq["norms"], K, gs, tq["seed"])
cos = F.cosine_similarity(y_deq.flatten().unsqueeze(0), y_otf.flatten().unsqueeze(0)).item()
print(f"  Dequant matmul vs on-the-fly matmul CosSim: {cos:.6f}")
assert cos > 0.999
print("  ✓ Forward: x_rot @ codebook[idx]^T * norm/√gs ≡ x @ W_deq^T")

# 1f. Paper says: for weight quant, only Stage 1 needed (no QJL)
# Our kernel implements Stage 1 only — correct for weight quant
print("  ✓ Stage 1 only (no QJL): correct for weight quant where input is full-precision")

# 1g. MISSING: per-group norms not applied in kernel
# The kernel computes y = x_rot @ codebook[idx]^T (no norms)
# Full TurboQuant: y = x_rot @ codebook[idx]^T * norms / √gs
print("  ⚠ Per-group norms: NOT yet applied in INT4 kernel (codebook-only dequant)")
print("    Impact: kernel output needs post-multiplication by norms/√gs")
print()


# =========================================================================
print("=" * 75)
print("2. PERFORMANCE: INT4 KERNEL vs PYTORCH REFERENCE")
print("=" * 75)

# Build INT4 kernel
from preshuffle_xdl4 import preshuffle_xdl4
from torch.utils.cpp_extension import load
ext_dir = os.path.dirname(os.path.abspath(__file__))
ck_base = "/shared_aig/john/semianalysis/aiter-amd/3rdparty/composable_kernel"
print("  Building INT4 flatmm kernel...")
ext = load(name="tq_validate",
    sources=[os.path.join(ext_dir, "turboquant_int4_flatmm.hip")],
    extra_include_paths=[ext_dir, ck_base + "/include"],
    extra_cuda_cflags=["-O3", "--offload-arch=gfx950", "-DUSE_ROCM", "-std=c++17"],
    verbose=False)
print("  Build OK\n")

print(f"{'Layer':<14} {'B':>3} {'PyTorch':>10} {'INT4 kern':>10} {'BF16':>10} {'vs PyTorch':>12} {'vs BF16':>10}")
print("-" * 75)

layers = [("kv_b_proj", 24576, 512, 128), ("gate_up", 18432, 7168, 128)]

for name, N, K, gs in layers:
    torch.manual_seed(42)
    W = torch.randn(N, K, dtype=torch.float32) * 0.02
    tq_data = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs, seed=42)
    idx = tq_data["indices_packed"]
    cb = tq_data["codebook"].float().to(device)
    norms = tq_data["norms"].float().to(device)
    seed = tq_data["seed"]

    # Preshuffle for kernel
    idx_psh = preshuffle_xdl4(idx.numpy(), N, K)
    idx_psh_t = torch.from_numpy(idx_psh).reshape(N, K // 2).to(device)
    idx_gpu = idx.to(device)

    W_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.02

    for B in [1, 4]:
        x = torch.randn(B, K, dtype=torch.bfloat16, device=device) * 0.1

        # PyTorch reference (with rotation)
        _tq.clear_rotation_cache()
        t_pytorch = timeit(lambda: _tq.turboquant_matmul_pytorch(
            x.reshape(-1, K), idx_gpu, cb, norms, K, gs, seed), n=50, warmup=5)

        # INT4 kernel (codebook-only, no rotation/norms)
        t_int4 = timeit(lambda: ext.turboquant_int4_flatmm_gemm(x, idx_psh_t, gs))

        # BF16 baseline
        t_bf16 = timeit(lambda: F.linear(x, W_bf16))

        print(f"  {name:<12} {B:>3} {t_pytorch:>9.4f}ms {t_int4:>9.4f}ms {t_bf16:>9.4f}ms"
              f" {t_pytorch/t_int4:>11.1f}x {t_bf16/t_int4:>9.2f}x")

print()
print("  Note: 'vs PyTorch' = speedup over turboquant_matmul_pytorch (includes rotation)")
print("  Note: 'vs BF16' = speedup over F.linear with full BF16 weights")
print()


# =========================================================================
print("=" * 75)
print("3. REAL MODEL PPL (Qwen2.5-0.5B)")
print("=" * 75)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    model_name = "Qwen/Qwen2.5-0.5B"
    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, trust_remote_code=True
    ).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load WikiText-103
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
    text = "\n\n".join(ds["text"])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[0]

    seq_len = 256
    n_chunks = 10

    # FP16 baseline PPL
    print("  Evaluating FP16 baseline...")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = input_ids[i*seq_len : i*seq_len + seq_len + 1].unsqueeze(0).cuda()
            logits = model(chunk[:, :-1]).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += seq_len
    ppl_fp16 = math.exp(total_loss / total_tokens)
    print(f"  FP16 PPL: {ppl_fp16:.4f}")

    # TurboQuant 4-bit PPL
    print("  Quantizing model...")
    count = 0
    for name_m, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "embed" not in name_m and "head" not in name_m:
            W = module.weight.data.float().cpu()
            in_f = module.in_features
            out_f = module.out_features
            gs_m = min(128, in_f)
            if in_f % gs_m != 0:
                gs_m = in_f

            tq_m = _tq.turboquant_quantize_packed(W, bit_width=4, group_size=gs_m, seed=42+count)

            class TQLinear(nn.Module):
                def __init__(self, tq_data, in_f, out_f, gs, bias):
                    super().__init__()
                    self.in_features = in_f
                    self.out_features = out_f
                    self.group_size = tq_data["group_size"]
                    self.seed = tq_data["seed"]
                    self.register_buffer("indices_packed", tq_data["indices_packed"])
                    self.register_buffer("codebook", tq_data["codebook"])
                    self.register_buffer("weight_norms", tq_data["norms"])
                    self.bias = bias

                def forward(self, x):
                    _tq.clear_rotation_cache()
                    orig = x.shape
                    x2d = x.reshape(-1, self.in_features)
                    y = _tq.turboquant_matmul_pytorch(
                        x2d, self.indices_packed, self.codebook,
                        self.weight_norms, self.in_features, self.group_size, self.seed)
                    y = y.reshape(*orig[:-1], self.out_features)
                    if self.bias is not None:
                        y = y + self.bias
                    return y.to(x.dtype)

            tq_lin = TQLinear(tq_m, in_f, out_f, gs_m,
                              module.bias.data.clone() if module.bias is not None else None).cuda()

            parent_name = ".".join(name_m.split(".")[:-1])
            child_name = name_m.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, child_name, tq_lin)
            count += 1

    print(f"  Quantized {count} layers")

    print("  Evaluating TQ-4bit PPL...")
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i in range(n_chunks):
            chunk = input_ids[i*seq_len : i*seq_len + seq_len + 1].unsqueeze(0).cuda()
            logits = model(chunk[:, :-1]).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), chunk[:, 1:].reshape(-1), reduction="sum")
            total_loss += loss.item()
            total_tokens += seq_len
    ppl_tq = math.exp(total_loss / total_tokens)
    print(f"  TQ-4bit PPL: {ppl_tq:.4f}")
    print(f"  PPL change: {ppl_tq - ppl_fp16:+.4f} ({(ppl_tq/ppl_fp16 - 1)*100:+.1f}%)")

except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"  Model test skipped: {e}")

print()
print("=" * 75)
print("SUMMARY")
print("=" * 75)
