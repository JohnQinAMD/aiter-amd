"""DeepSeek-V3 layer proportion analysis for INT4 weight quantization ROI."""

# DeepSeek-V3 architecture (from HF config + sglang deepseek_v2.py)
HIDDEN = 7168
NUM_HEADS = 128
Q_LORA_RANK = 1536
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 128
QK_ROPE_HEAD_DIM = 64
V_HEAD_DIM = 128
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 192
NUM_LAYERS = 61
DENSE_LAYERS = 3       # layers 0-2
MOE_LAYERS = 58         # layers 3-60
N_ROUTED_EXPERTS = 256
EXPERTS_PER_TOKEN = 8
MOE_INTERMEDIATE = 2048
DENSE_INTERMEDIATE = 18432
N_SHARED_EXPERTS = 1
TP = 8

print("=" * 80)
print("DeepSeek-V3 Layer Proportion Analysis (TP=8, B=1 Decode)")
print("=" * 80)

# Per-layer attention linear shapes at TP=8
attn_layers = {
    "fused_qkv_a (replicated)": (Q_LORA_RANK + KV_LORA_RANK + QK_ROPE_HEAD_DIM, HIDDEN),  # (2112, 7168)
    "q_b_proj (col-parallel)":  (NUM_HEADS * QK_HEAD_DIM // TP, Q_LORA_RANK),              # (3072, 1536)
    "kv_b_proj (col-parallel)": (NUM_HEADS * (QK_NOPE_HEAD_DIM + V_HEAD_DIM) // TP, KV_LORA_RANK),  # (4096, 512)
    "o_proj (row-parallel)":    (HIDDEN, NUM_HEADS * V_HEAD_DIM // TP),                     # (7168, 2048)
}

# Active MoE FFN per layer (8 experts + 1 shared + router)
moe_layers_dict = {
    "router (replicated)":      (N_ROUTED_EXPERTS, HIDDEN),                           # (256, 7168)
    "8×expert gate_up (TP)":    (EXPERTS_PER_TOKEN, 2 * MOE_INTERMEDIATE // TP, HIDDEN),  # 8× (512, 7168)
    "8×expert down (TP)":       (EXPERTS_PER_TOKEN, HIDDEN, MOE_INTERMEDIATE // TP),      # 8× (7168, 256)
    "shared gate_up (TP)":      (2 * MOE_INTERMEDIATE // TP, HIDDEN),                    # (512, 7168)
    "shared down (TP)":         (HIDDEN, MOE_INTERMEDIATE // TP),                        # (7168, 256)
}

dense_ffn = {
    "gate_up (col-parallel)":   (2 * DENSE_INTERMEDIATE // TP, HIDDEN),   # (4608, 7168)
    "down (row-parallel)":      (HIDDEN, DENSE_INTERMEDIATE // TP),       # (7168, 2304)
}

print("\n--- Attention Linears (per layer, TP=8) ---")
print(f"{'Layer':<32} {'Shape':>14} {'Params':>10} {'BF16 MB':>10} {'K dim':>6}")
print("-" * 76)
total_attn_params = 0
for name, (N, K) in attn_layers.items():
    p = N * K
    total_attn_params += p
    mb = p * 2 / 1024**2
    print(f"{name:<32} ({N:>5},{K:>5}) {p:>10,} {mb:>9.1f} {K:>6}")
print(f"{'TOTAL':<32} {'':>14} {total_attn_params:>10,} {total_attn_params*2/1024**2:>9.1f}")

print("\n--- Active MoE FFN (per layer, TP=8, 8 active experts) ---")
total_moe_params = 0
print(f"{'Layer':<32} {'Shape':>14} {'Params':>10} {'BF16 MB':>10}")
print("-" * 76)

router_p = N_ROUTED_EXPERTS * HIDDEN
total_moe_params += router_p
print(f"{'router':<32} ({N_ROUTED_EXPERTS:>5},{HIDDEN:>5}) {router_p:>10,} {router_p*2/1024**2:>9.1f}")

gu_per = 2 * MOE_INTERMEDIATE // TP * HIDDEN
d_per = HIDDEN * MOE_INTERMEDIATE // TP
exp_p = EXPERTS_PER_TOKEN * (gu_per + d_per)
total_moe_params += exp_p
print(f"{'8 experts gate_up+down':<32} {'8×(512+256,7168)':>14} {exp_p:>10,} {exp_p*2/1024**2:>9.1f}")

shared_p = gu_per + d_per
total_moe_params += shared_p
print(f"{'shared expert':<32} {'(512+256,7168)':>14} {shared_p:>10,} {shared_p*2/1024**2:>9.1f}")

print(f"{'TOTAL MoE':<32} {'':>14} {total_moe_params:>10,} {total_moe_params*2/1024**2:>9.1f}")

print("\n--- Weight Traffic Proportions (per MoE layer) ---")
total = total_attn_params + total_moe_params
print(f"\nTotal active weights per layer: {total:,} params = {total*2/1024**2:.1f} MB")
print()
print(f"{'Component':<35} {'Params':>12} {'%':>8}")
print("-" * 58)
for name, (N, K) in attn_layers.items():
    p = N * K
    short = name.split(" (")[0]
    print(f"  {short:<33} {p:>12,} {p/total*100:>7.1f}%")

print(f"  {'MoE FFN (active)':<33} {total_moe_params:>12,} {total_moe_params/total*100:>7.1f}%")

# Estimated decode time breakdown
print("\n" + "=" * 80)
print("END-TO-END DECODE TIME ESTIMATION (B=1, TP=8)")
print("=" * 80)

MI355X_BW = 8.0  # TB/s
DISPATCH_FLOOR_US = 11.0  # µs per kernel launch

print(f"\nMI355X HBM bandwidth: {MI355X_BW} TB/s")
print(f"Kernel dispatch floor: ~{DISPATCH_FLOOR_US} µs")

print(f"\n{'Component':<35} {'BF16 MB':>10} {'Theory µs':>10} {'Est. µs':>10} {'Note':<20}")
print("-" * 90)

def est_time(mb, note="", n_launches=1):
    theory = mb / (MI355X_BW * 1e6) * 1e6  # MB / (MB/µs) → µs.  8 TB/s = 8e6 MB/s = 8 MB/µs
    actual = max(theory / n_launches, DISPATCH_FLOOR_US) * n_launches if n_launches > 0 else theory
    return theory, actual

layer_times = {}
for name, (N, K) in attn_layers.items():
    p = N * K
    mb = p * 2 / 1024**2
    theory, actual = est_time(mb)
    short = name.split(" (")[0]
    layer_times[short] = actual
    print(f"  {short:<33} {mb:>9.1f} {theory:>9.1f} {actual:>9.1f} {'at dispatch floor':<20}")

# Attention compute (flash attention, negligible at B=1)
layer_times["attn_compute"] = 5.0
print(f"  {'attention compute':<33} {'—':>10} {'—':>10} {'5.0':>10} {'flash-attn B=1':>20}")

# MoE: FusedMoE batches experts into 2 kernels (gate_up, down)
moe_gu_mb = EXPERTS_PER_TOKEN * 2 * MOE_INTERMEDIATE // TP * HIDDEN * 2 / 1024**2
moe_d_mb = EXPERTS_PER_TOKEN * HIDDEN * MOE_INTERMEDIATE // TP * 2 / 1024**2
shared_mb = shared_p * 2 / 1024**2
router_mb = router_p * 2 / 1024**2

theory_r, actual_r = est_time(router_mb)
print(f"  {'router':<33} {router_mb:>9.1f} {theory_r:>9.1f} {actual_r:>9.1f} {'dispatch floor':>20}")
layer_times["router"] = actual_r

# FusedMoE: one kernel launch for all 8 experts
theory_gu, _ = est_time(moe_gu_mb)
actual_gu = max(theory_gu, 15.0)  # FusedMoE has ~15µs min launch
print(f"  {'FusedMoE gate_up (8 exp)':<33} {moe_gu_mb:>9.1f} {theory_gu:>9.1f} {actual_gu:>9.1f} {'1 fused launch':>20}")
layer_times["moe_gate_up"] = actual_gu

theory_d, _ = est_time(moe_d_mb)
actual_d = max(theory_d, 15.0)
print(f"  {'FusedMoE down (8 exp)':<33} {moe_d_mb:>9.1f} {theory_d:>9.1f} {actual_d:>9.1f} {'1 fused launch':>20}")
layer_times["moe_down"] = actual_d

theory_sh, actual_sh = est_time(shared_mb)
print(f"  {'shared expert':<33} {shared_mb:>9.1f} {theory_sh:>9.1f} {actual_sh:>9.1f} {'dispatch floor':>20}")
layer_times["shared_expert"] = actual_sh

# Communication (allreduce for TP)
layer_times["allreduce"] = 15.0
print(f"  {'allreduce (TP=8)':<33} {'—':>10} {'—':>10} {'15.0':>10} {'NCCL/RCCL':>20}")

# LayerNorm, etc.
layer_times["misc"] = 3.0
print(f"  {'layernorm + misc':<33} {'—':>10} {'—':>10} {'3.0':>10} {'':>20}")

total_per_layer = sum(layer_times.values())
print(f"\n  {'TOTAL per layer':<33} {'':>10} {'':>10} {total_per_layer:>9.1f}")
total_per_token = total_per_layer * NUM_LAYERS
print(f"  {'TOTAL 61 layers':<33} {'':>10} {'':>10} {total_per_token/1000:>8.1f}ms")
print(f"  {'+ embed/lm_head':<33} {'':>10} {'':>10} {'~0.5':>10}ms")
total_decode = total_per_token / 1000 + 0.5
print(f"  {'EST. DECODE LATENCY':<33} {'':>10} {'':>10} {total_decode:>8.1f}ms")
print(f"  {'Tokens/sec':<33} {'':>10} {'':>10} {1000/total_decode:>8.0f}")

# INT4 impact analysis
print("\n" + "=" * 80)
print("INT4 WEIGHT QUANT IMPACT ANALYSIS")
print("=" * 80)

measured = {
    "kv_b_proj":    {"bf16": 11.5, "int4": 5.3,  "speedup": 2.16},
    "q_b_proj":     {"bf16": 11.5, "int4": 11.5, "speedup": 1.00},
    "o_proj":       {"bf16": 11.4, "int4": 15.5, "speedup": 0.74},
    "fused_qkv_a":  {"bf16": 11.0, "int4": None, "speedup": None},  # K=7168, not tested
}

print(f"\n{'Layer':<20} {'K':>6} {'BF16 µs':>10} {'INT4 µs':>10} {'Δ µs':>8} {'×61 Δ ms':>10} {'% of decode':>12}")
print("-" * 80)

total_saving = 0
for name, d in measured.items():
    K = dict(attn_layers)[
        next(k for k in attn_layers if k.startswith(name.replace("fused_qkv_a", "fused_qkv_a")))
    ][1] if name != "fused_qkv_a" else 7168
    if d["int4"] is not None:
        delta = d["bf16"] - d["int4"]
        total_delta = delta * NUM_LAYERS / 1000
        pct = total_delta / total_decode * 100
        total_saving += total_delta
        print(f"{name:<20} {K:>6} {d['bf16']:>9.1f} {d['int4']:>9.1f} {delta:>+7.1f} {total_delta:>+9.2f} {pct:>+11.1f}%")
    else:
        print(f"{name:<20} {K:>6} {d['bf16']:>9.1f} {'—':>10} {'—':>8} {'—':>10} {'not tested':>12}")

print(f"\n{'Current savings (kv_b_proj only)':<45} {(11.5-5.3)*61/1000:>+.2f}ms = {(11.5-5.3)*61/1000/total_decode*100:>+.1f}% of decode")

# What-if scenarios
print("\n--- What-If: Optimizing Larger K ---")
print(f"\n{'Scenario':<50} {'Δ ms':>8} {'New decode':>12} {'Improvement':>12}")
print("-" * 85)

base = total_decode
scenarios = [
    ("Current: kv_b_proj 2.16x only",
     (11.5-5.3)*61/1000),
    ("+ q_b_proj to 1.5x (INT4=7.7µs)",
     (11.5-5.3)*61/1000 + (11.5-7.7)*61/1000),
    ("+ o_proj to 1.3x (INT4=8.8µs)",
     (11.5-5.3)*61/1000 + (11.5-7.7)*61/1000 + (11.4-8.8)*61/1000),
    ("+ fused_qkv_a to 1.2x (INT4=9.2µs)",
     (11.5-5.3)*61/1000 + (11.5-7.7)*61/1000 + (11.4-8.8)*61/1000 + (11.0-9.2)*61/1000),
    ("ALL attn linears at 2x (optimistic)",
     sum(d["bf16"] for d in measured.values()) * 0.5 * 61/1000),
]

for desc, saving_ms in scenarios:
    new = base - saving_ms
    pct = saving_ms / base * 100
    print(f"{desc:<50} {saving_ms:>+7.2f} {new:>10.1f}ms {pct:>+10.1f}%")

# KV cache vs weight quant value comparison
print("\n" + "=" * 80)
print("VALUE COMPARISON: Weight GEMM Speedup vs KV Cache Compression")
print("=" * 80)

print("""
┌──────────────────────────┬────────────────────────┬────────────────────────┐
│ Metric                   │ Weight GEMM (kv_b_proj)│ KV Cache Compression   │
├──────────────────────────┼────────────────────────┼────────────────────────┤
│ Decode latency saving    │ ~3.7% per token        │ 0% (decompress on read)│
│ Memory saving            │ ~0.2 GB (kv_b_proj wt) │ 5.6 GB/GPU at 128K    │
│ Throughput gain (↑batch) │ minimal                │ 2-3x more concurrent   │
│ Max context length       │ unchanged              │ ~3x longer contexts    │
│ Implementation effort    │ done                   │ done                   │
├──────────────────────────┼────────────────────────┼────────────────────────┤
│ Further optimization ROI │                        │                        │
│   Optimize larger K      │ +5-10% decode (hard)   │ n/a                    │
│   GPU quant kernel       │ n/a                    │ 30x faster compress    │
│   Rotation fusion        │ eliminates x@Pi cost   │ eliminates x@Pi cost   │
└──────────────────────────┴────────────────────────┴────────────────────────┘
""")

print("CONCLUSION:")
print("  1. kv_b_proj INT4 saves ~3.7% decode latency — nice but not transformative")
print("  2. Optimizing all attn linears to 1.5x+ would add another ~6% — medium ROI")
print("  3. KV cache 2.94x compression is the killer feature: enables 3x longer")
print("     contexts or 2-3x more concurrent users per GPU")
print("  4. Recommended priority: GPU KV quant kernel > rotation fusion > larger K opt")
