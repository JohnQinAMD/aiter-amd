# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib
import math
from dataclasses import replace

import aiter
import pytest
import torch
from aiter import dtypes
from aiter.fused_moe import fused_moe_2stages, moe_sorting
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kimi_k3_moe_handoff import (
    KimiK3MXFP4ExpertRequest,
    KimiK3MXFP4W13Layout,
    kimi_k3_mxfp4_expert_handoff,
    kimi_k3_mxfp4_expert_mode,
    supports_kimi_k3_mxfp4_expert_handoff,
)
from aiter.ops.flydsl.moe_common import GateMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 prepared-route expert handoff requires gfx950",
)

EXPERTS = 896
TOPK = 16
MODEL_DIM = 3584
INTER_DIM = 384


def _request() -> KimiK3MXFP4ExpertRequest:
    generator = torch.Generator(device="cuda").manual_seed(20260728)

    def packed(shape):
        tensor = torch.full(shape, 0x11, dtype=torch.uint8, device="cuda")
        tensor = tensor.view(dtypes.fp4x2)
        tensor.is_shuffled = True
        return tensor

    w1 = packed((EXPERTS, 2 * INTER_DIM, MODEL_DIM // 2))
    w2 = packed((EXPERTS, MODEL_DIM, INTER_DIM // 2))
    w1.kimi_k3_w13_layout = kimi_k3_mxfp4_expert_mode().w13_layout.value
    w1_scale = torch.full(
        (EXPERTS * 2 * INTER_DIM, MODEL_DIM // 32),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    ).view(dtypes.fp8_e8m0)
    w2_scale = torch.full(
        (
            math.ceil(EXPERTS * MODEL_DIM / 256) * 256,
            math.ceil((INTER_DIM // 32) / 8) * 8,
        ),
        0x7F,
        dtype=torch.uint8,
        device="cuda",
    ).view(dtypes.fp8_e8m0)
    router_logits = torch.full((1, EXPERTS), -8.0, dtype=torch.float32, device="cuda")
    router_logits[:, :32] = 2.0
    return KimiK3MXFP4ExpertRequest(
        hidden_states=torch.randn(
            (1, MODEL_DIM),
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        ),
        router_logits=router_logits,
        correction_bias=torch.zeros((EXPERTS,), dtype=torch.bfloat16, device="cuda"),
        w1=w1,
        w2=w2,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        situ_beta=4.0,
        situ_linear_beta=25.0,
    )


def test_kimi_k3_handoff_consumes_prepared_sort_once(monkeypatch):
    request = _request()
    mode = kimi_k3_mxfp4_expert_mode()
    assert supports_kimi_k3_mxfp4_expert_handoff(request)

    expected_weights = torch.empty((1, TOPK), dtype=torch.float32, device="cuda")
    expected_ids = torch.empty((1, TOPK), dtype=torch.int32, device="cuda")
    aiter.biased_grouped_topk_hip(
        request.router_logits,
        request.correction_bias.float(),
        expected_weights,
        expected_ids,
        1,
        1,
        True,
        1.0,
    )
    (
        expected_sorted_ids,
        expected_sorted_weights,
        expected_sorted_experts,
        expected_num_valid,
        expected_moe_buf,
    ) = moe_sorting(
        expected_ids,
        expected_weights,
        EXPERTS,
        MODEL_DIM,
        torch.bfloat16,
        32,
    )
    expected_output = fused_moe_2stages(
        request.hidden_states,
        request.w1,
        request.w2,
        TOPK,
        expected_sorted_ids,
        expected_sorted_weights,
        expected_sorted_experts,
        expected_num_valid,
        expected_moe_buf,
        True,
        32,
        activation=aiter.ActivationType.Situv2,
        quant_type=aiter.QuantType.per_1x32,
        q_dtype_a=mode.activation_dtype,
        q_dtype_w=dtypes.fp4x2,
        w1_scale=request.w1_scale,
        w2_scale=request.w2_scale,
        topk_ids=expected_ids,
        topk_weights=expected_weights,
        beta=request.situ_beta,
        linear_beta=request.situ_linear_beta,
        gate_mode=mode.gate_mode.value,
    )
    torch.cuda.synchronize()

    def duplicate_sort_is_a_failure(*_args, **_kwargs):
        raise AssertionError("prepared-route handoff repeated routing or sorting")

    monkeypatch.setattr("aiter.fused_moe.moe_sorting", duplicate_sort_is_a_failure)
    monkeypatch.setattr("aiter.biased_grouped_topk_hip", duplicate_sort_is_a_failure)
    quant_sorted_ids = []
    if mode.activation_dtype == dtypes.bf16:
        monkeypatch.setattr(
            "aiter.ops.quant.fused_dynamic_mx_quant_moe_sort",
            duplicate_sort_is_a_failure,
        )
    else:
        fused_moe_module = importlib.import_module("aiter.fused_moe")
        original_quant = fused_moe_module.fused_dynamic_mxfp8_quant_moe_sort

        def audited_quant(*args, **kwargs):
            quant_sorted_ids.append(kwargs["sorted_ids"].data_ptr())
            return original_quant(*args, **kwargs)

        monkeypatch.setattr(
            "aiter.fused_moe.fused_dynamic_mxfp8_quant_moe_sort",
            audited_quant,
        )
    actual = kimi_k3_mxfp4_expert_handoff(request)
    torch.cuda.synchronize()
    if mode.activation_dtype == dtypes.fp8:
        # Stage1 consumes the prepared sort for activation quantization. Its
        # tuned `_fp8` epilogue fuses the inter-stage quantization.
        assert quant_sorted_ids == [actual.stage1_sorted_token_ids.data_ptr()]

    torch.testing.assert_close(actual.expert_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual.routing_weights, expected_weights, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        actual.sorted_expert_ids, expected_sorted_experts[:TOPK], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.stage1_sorted_token_ids,
        expected_sorted_ids[: actual.stage1_sorted_token_ids.numel()],
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        actual.stage1_sorted_weights,
        expected_sorted_weights[: actual.stage1_sorted_weights.numel()],
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(actual.num_valid_ids, expected_num_valid, rtol=0, atol=0)
    expected_f32 = expected_output.float()
    actual_f32 = actual.expert_output.float()
    relative_rmse = torch.sqrt(torch.mean((actual_f32 - expected_f32).square()))
    relative_rmse /= torch.sqrt(torch.mean(expected_f32.square())).clamp_min(1e-12)
    cosine = torch.nn.functional.cosine_similarity(
        expected_f32.flatten(), actual_f32.flatten(), dim=0
    )
    assert float(relative_rmse) <= 1e-3
    assert float(cosine) >= 0.999
    assert torch.isfinite(actual.expert_output).all()
    assert actual.activation_dtype == mode.activation_dtype
    assert actual.w13_layout == mode.w13_layout


@pytest.mark.parametrize(
    ("a8w4", "activation_dtype", "gate_mode", "w13_layout"),
    [
        (
            "0",
            dtypes.bf16,
            GateMode.SEPARATED,
            KimiK3MXFP4W13Layout.GATE_UP_SEPARATED_PRESHUFFLED,
        ),
        (
            "1",
            dtypes.fp8,
            GateMode.INTERLEAVE,
            KimiK3MXFP4W13Layout.GATE_UP_INTERLEAVED_PRESHUFFLED,
        ),
    ],
)
def test_kimi_k3_handoff_mode_owns_activation_and_layout(
    monkeypatch, a8w4, activation_dtype, gate_mode, w13_layout
):
    monkeypatch.setenv("AITER_SITUV2_A8W4", a8w4)
    mode = kimi_k3_mxfp4_expert_mode()
    assert mode.activation_dtype == activation_dtype
    assert mode.gate_mode == gate_mode
    assert mode.w13_layout == w13_layout
    assert supports_kimi_k3_mxfp4_expert_handoff(_request())


def test_kimi_k3_handoff_support_is_narrow():
    request = _request()
    assert supports_kimi_k3_mxfp4_expert_handoff(request)
    assert not supports_kimi_k3_mxfp4_expert_handoff(
        replace(request, hidden_states=request.hidden_states.expand(2, -1))
    )
    assert not supports_kimi_k3_mxfp4_expert_handoff(
        replace(request, correction_bias=request.correction_bias.float())
    )
    assert not supports_kimi_k3_mxfp4_expert_handoff(replace(request, situ_beta=1.0))


def test_kimi_k3_a8w4_requires_explicit_interleaved_weight_owner(monkeypatch):
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")
    request = _request()
    request.w1.kimi_k3_w13_layout = KimiK3MXFP4W13Layout.GATE_UP_SEPARATED_PRESHUFFLED.value
    assert not supports_kimi_k3_mxfp4_expert_handoff(request)


def test_kimi_k3_handoff_honors_aiter_opt_out(monkeypatch):
    request = _request()
    monkeypatch.setenv("AITER_DISABLE", "1")
    assert not supports_kimi_k3_mxfp4_expert_handoff(request)
