"""End-to-end test of MHATokenToKVPoolTQ simulating GQA attention flow.

Simulates the full SGLang decode loop:
1. Prefill: set_kv_buffer for multiple tokens
2. Decode: set_kv_buffer for 1 new token, get_key_buffer, get_value_buffer
3. Verify dequantized output quality vs original FP16
"""
import sys, os, math
sys.path.insert(0, "/shared_aig/john/semianalysis/sglang-amd/python")
os.environ["SGLANG_KV_CACHE_TURBOQUANT"] = "4"

import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

all_ok = True
def check(name, cond):
    global all_ok
    if not cond: all_ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


class MockRadixAttention:
    def __init__(self, layer_id):
        self.layer_id = layer_id


def test_gqa_pool(head_num, head_dim, layer_num, pool_size, bit_width):
    """Full end-to-end test of MHATokenToKVPoolTQ."""
    os.environ["SGLANG_KV_CACHE_TURBOQUANT"] = str(bit_width)

    # Import AFTER setting env var
    from importlib import reload
    import sglang.srt.mem_cache.memory_pool as mp
    reload(mp)
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPoolTQ

    class MockMemSaver:
        def region(self, x):
            from contextlib import nullcontext
            return nullcontext()

    pool = MHATokenToKVPoolTQ.__new__(MHATokenToKVPoolTQ)
    pool.size = pool_size
    pool.page_size = 1
    pool.head_num = head_num
    pool.head_dim = head_dim
    pool.v_head_dim = head_dim
    pool.layer_num = layer_num
    pool.dtype = torch.bfloat16
    pool.store_dtype = torch.bfloat16
    pool.device = DEVICE
    pool.memory_saver_adapter = MockMemSaver()
    pool.custom_mem_pool = None
    pool.enable_custom_mem_pool = False
    pool.start_layer = 0
    pool.layer_transfer_counter = None
    pool._create_buffers()

    # Generate test data
    torch.manual_seed(42)
    T_prefill = 32
    K_data = torch.randn(T_prefill, head_num, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.02
    V_data = torch.randn(T_prefill, head_num, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.02
    locs = torch.arange(T_prefill, device=DEVICE)

    # Prefill: write all tokens
    layer = MockRadixAttention(0)
    pool.set_kv_buffer(layer, locs, K_data, V_data)

    # Read back and check quality
    K_out = pool.get_key_buffer(0)
    V_out = pool.get_value_buffer(0)

    K_active = K_out[:T_prefill]
    V_active = V_out[:T_prefill]

    cos_k = F.cosine_similarity(
        K_data.float().flatten().unsqueeze(0),
        K_active.float().flatten().unsqueeze(0)
    ).item()
    cos_v = F.cosine_similarity(
        V_data.float().flatten().unsqueeze(0),
        V_active.float().flatten().unsqueeze(0)
    ).item()

    # Decode: add 1 new token
    new_k = torch.randn(1, head_num, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.02
    new_v = torch.randn(1, head_num, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.02
    new_loc = torch.tensor([T_prefill], device=DEVICE)
    pool.set_kv_buffer(layer, new_loc, new_k, new_v)

    K_out2 = pool.get_key_buffer(0)
    V_out2 = pool.get_value_buffer(0)

    # Verify the new token is there
    new_k_out = K_out2[T_prefill:T_prefill+1]
    cos_new = F.cosine_similarity(
        new_k.float().flatten().unsqueeze(0),
        new_k_out.float().flatten().unsqueeze(0)
    ).item()

    # Multi-layer test
    if layer_num > 1:
        layer1 = MockRadixAttention(1)
        K_data_l1 = torch.randn(T_prefill, head_num, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.03
        V_data_l1 = torch.randn(T_prefill, head_num, head_dim, dtype=torch.bfloat16, device=DEVICE) * 0.03
        pool.set_kv_buffer(layer1, locs, K_data_l1, V_data_l1)
        K_l1 = pool.get_key_buffer(1)[:T_prefill]
        cos_l1 = F.cosine_similarity(
            K_data_l1.float().flatten().unsqueeze(0),
            K_l1.float().flatten().unsqueeze(0)
        ).item()
    else:
        cos_l1 = 1.0

    return cos_k, cos_v, cos_new, cos_l1


print("=" * 60)
print("GQA/MHA End-to-End Pool Test")
print("=" * 60)

configs = [
    ("Qwen3-GQA (hd=128, h=4)", 4, 128, 4, 128),
    ("GPT-OSS-GQA (hd=64, h=8)", 8, 64, 4, 128),
    ("Llama-GQA (hd=128, h=8)", 8, 128, 4, 128),
    ("MQA (hd=128, h=1)", 1, 128, 4, 128),
]

for name, hn, hd, nl, ps in configs:
    print(f"\n  {name}:")
    for bw in [2, 3, 4]:
        cos_k, cos_v, cos_new, cos_l1 = test_gqa_pool(hn, hd, nl, ps, bw)
        thresh = {4: 0.99, 3: 0.97, 2: 0.90}
        check(f"{bw}-bit K={cos_k:.4f} V={cos_v:.4f} new={cos_new:.4f} L1={cos_l1:.4f}",
              cos_k > thresh[bw] and cos_v > thresh[bw] and cos_new > thresh[bw] and cos_l1 > thresh[bw])

# Memory compression test
print()
print("=" * 60)
print("Memory Compression Verification")
print("=" * 60)

os.environ["SGLANG_KV_CACHE_TURBOQUANT"] = "4"
from importlib import reload
import sglang.srt.mem_cache.memory_pool as mp
reload(mp)

for hn, hd in [(4, 128), (8, 64), (8, 128)]:
    fp16_per_kv = hn * hd * 2  # bytes for K or V
    from sglang.srt.layers.quantization.turboquant_engine import packed_bytes_per_dim
    for bw in [2, 3, 4]:
        pb = packed_bytes_per_dim(hd, bw)
        tq_per_kv = hn * (pb + 2)  # packed + norm per head
        ratio = fp16_per_kv / tq_per_kv
        print(f"  h={hn} d={hd} {bw}-bit: {fp16_per_kv}B → {tq_per_kv}B ({ratio:.2f}x) per K or V")

print()
print("=" * 60)
print("ALL PASSED" if all_ok else "SOME TESTS FAILED")
print("=" * 60)
sys.exit(0 if all_ok else 1)
