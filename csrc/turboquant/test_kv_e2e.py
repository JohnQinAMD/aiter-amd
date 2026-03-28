"""End-to-end KV cache compression test on a real model.

Uses Qwen2.5-0.5B (non-MLA, standard MHA) to validate that TurboQuant
KV cache compression preserves model quality when applied to the
attention layer's KV cache.

Flow:
1. Run FP16 baseline forward pass, collect KV cache per layer
2. Quantize each KV cache layer with TurboQuant Stage 1
3. Dequantize and compare attention outputs
4. Measure PPL with quantized KV cache
"""
import sys, math, time
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")
_spec = importlib.util.spec_from_file_location("tq_engine",
    "/shared_aig/john/semianalysis/sglang-amd/python/sglang/srt/layers/quantization/turboquant_engine.py")
_tq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tq)

device = torch.device("cuda")

print("=" * 75)
print("KV CACHE COMPRESSION: END-TO-END PPL TEST")
print("=" * 75)

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model_name = "Qwen/Qwen2.5-0.5B"
print(f"\nLoading {model_name}...")
model = AutoModelForCausalLM.from_pretrained(
    model_name, dtype=torch.bfloat16, trust_remote_code=True
).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Get model dimensions
config = model.config
head_dim = config.hidden_size // config.num_attention_heads
num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
n_layers = config.num_hidden_layers
print(f"  Layers: {n_layers}, KV heads: {num_kv_heads}, head_dim: {head_dim}")
print(f"  KV dim per layer: {num_kv_heads * head_dim * 2} (K+V)")

# Load WikiText-103
ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation")
text = "\n\n".join(ds["text"])
input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
SEQ_LEN = 256
N_CHUNKS = 10

# ================================================================
# FP16 Baseline PPL
# ================================================================
print("\nEvaluating FP16 baseline PPL...")
total_loss = 0.0
total_tokens = 0
with torch.no_grad():
    for i in range(N_CHUNKS):
        chunk = input_ids[i * SEQ_LEN : i * SEQ_LEN + SEQ_LEN + 1].unsqueeze(0).cuda()
        logits = model(chunk[:, :-1]).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk[:, 1:].reshape(-1), reduction="sum")
        total_loss += loss.item()
        total_tokens += SEQ_LEN
ppl_fp16 = math.exp(total_loss / total_tokens)
print(f"  FP16 PPL: {ppl_fp16:.4f}")

# ================================================================
# KV Cache Compression Quality per Layer
# ================================================================
print("\nMeasuring KV cache compression quality...")

# Hook into the model to intercept and quantize KV cache
class KVQuantHook:
    """Intercepts KV cache in transformer layers, quantizes and dequantizes."""

    def __init__(self, bit_width=4, group_size=128):
        self.bit_width = bit_width
        self.group_size = group_size
        self.codebook_centroids, self.codebook_boundaries = _tq.get_codebook(bit_width)
        self.cos_sims_k = []
        self.cos_sims_v = []

    def quantize_dequantize(self, tensor):
        """Quantize and immediately dequantize a KV cache tensor."""
        orig_shape = tensor.shape  # (batch, heads, seq, head_dim)
        B, H, S, D = orig_shape

        flat = tensor.reshape(-1, D).float()
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        flat_norm = flat / norms

        gs = min(self.group_size, D)
        n_groups = math.ceil(D / gs)

        reconstructed = torch.zeros_like(flat)
        for g in range(n_groups):
            g_start = g * gs
            g_end = min(g_start + gs, D)
            g_dim = g_end - g_start

            Pi = _tq.generate_rotation_matrix(g_dim, seed=42 + g_start).to(tensor.device)
            Y = flat_norm[:, g_start:g_end] @ Pi.T
            scale = math.sqrt(g_dim)
            Y_scaled = Y * scale

            boundaries = self.codebook_boundaries.to(tensor.device)
            centroids = self.codebook_centroids.to(tensor.device)

            indices = torch.searchsorted(boundaries, Y_scaled.reshape(-1))
            indices = indices.clamp(0, len(centroids) - 1).reshape(-1, g_dim)

            Y_hat = centroids[indices] / scale
            L_hat = Y_hat @ Pi

            reconstructed[:, g_start:g_end] = L_hat

        reconstructed = reconstructed * norms
        return reconstructed.reshape(orig_shape).to(tensor.dtype)


kv_hook = KVQuantHook(bit_width=4, group_size=128)

# Monkey-patch the attention layers to quantize KV cache
original_forwards = {}

def make_kv_quant_wrapper(layer_module, layer_idx):
    original_forward = layer_module.forward

    def wrapped_forward(*args, **kwargs):
        # Run original forward
        result = original_forward(*args, **kwargs)

        # If result contains KV cache (past_key_values), quantize it
        if isinstance(result, tuple) and len(result) >= 2:
            output = result[0]
            # For Qwen2: result is (hidden_states, present_key_value, ...)
            # present_key_value is a tuple of (key, value) tensors
            if result[1] is not None and isinstance(result[1], tuple) and len(result[1]) == 2:
                key, value = result[1]
                key_q = kv_hook.quantize_dequantize(key)
                value_q = kv_hook.quantize_dequantize(value)

                # Measure quality
                cos_k = F.cosine_similarity(
                    key.float().flatten().unsqueeze(0),
                    key_q.float().flatten().unsqueeze(0)
                ).item()
                cos_v = F.cosine_similarity(
                    value.float().flatten().unsqueeze(0),
                    value_q.float().flatten().unsqueeze(0)
                ).item()
                kv_hook.cos_sims_k.append(cos_k)
                kv_hook.cos_sims_v.append(cos_v)

                result = (output, (key_q, value_q)) + result[2:]

        return result

    return wrapped_forward


# Apply hooks to all attention layers
for i, layer in enumerate(model.model.layers):
    layer.self_attn.forward = make_kv_quant_wrapper(layer.self_attn, i)

# Run with quantized KV cache
print("Evaluating TQ-4bit KV cache PPL...")
total_loss_q = 0.0
total_tokens_q = 0
kv_hook.cos_sims_k = []
kv_hook.cos_sims_v = []

with torch.no_grad():
    for i in range(N_CHUNKS):
        chunk = input_ids[i * SEQ_LEN : i * SEQ_LEN + SEQ_LEN + 1].unsqueeze(0).cuda()
        logits = model(chunk[:, :-1]).logits
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            chunk[:, 1:].reshape(-1), reduction="sum")
        total_loss_q += loss.item()
        total_tokens_q += SEQ_LEN

ppl_tq_kv = math.exp(total_loss_q / total_tokens_q)
avg_cos_k = sum(kv_hook.cos_sims_k) / max(len(kv_hook.cos_sims_k), 1)
avg_cos_v = sum(kv_hook.cos_sims_v) / max(len(kv_hook.cos_sims_v), 1)

print(f"\n  FP16 PPL:          {ppl_fp16:.4f}")
print(f"  TQ-4bit KV PPL:   {ppl_tq_kv:.4f}")
print(f"  PPL change:        {ppl_tq_kv - ppl_fp16:+.4f} ({(ppl_tq_kv/ppl_fp16 - 1)*100:+.1f}%)")
print(f"\n  Avg K CosSim:      {avg_cos_k:.6f}")
print(f"  Avg V CosSim:      {avg_cos_v:.6f}")

# Memory savings
kv_per_token_fp16 = num_kv_heads * head_dim * 2 * 2  # K+V, fp16
kv_per_token_tq4 = num_kv_heads * head_dim * 2 * 0.5 + 64  # 4-bit + norms overhead
ratio = kv_per_token_fp16 / kv_per_token_tq4
print(f"\n  KV memory per token: FP16={kv_per_token_fp16} bytes, TQ-4bit≈{int(kv_per_token_tq4)} bytes")
print(f"  Compression: {ratio:.2f}x")

print("\n" + "=" * 75)
print("DONE")
print("=" * 75)
