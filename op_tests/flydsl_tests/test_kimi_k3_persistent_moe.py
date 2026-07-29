# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import replace

import pytest
import torch

import aiter.ops.flydsl.kimi_k3_persistent_moe as persistent_moe
from aiter.ops.flydsl.kimi_k3_persistent_moe import (
    KimiK3PersistentMoERequest,
    prepare_kimi_k3_b1_persistent_moe,
    supports_kimi_k3_b1_persistent_moe,
)
from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params


def _request() -> KimiK3PersistentMoERequest:
    hidden = torch.empty((1, 3584), dtype=torch.bfloat16)
    return KimiK3PersistentMoERequest(
        hidden_states=hidden,
        router_logits=torch.empty((1, 896), dtype=torch.float32),
        correction_bias=torch.empty(896, dtype=torch.bfloat16),
        w1=torch.empty(1),
        w2=torch.empty(1),
        w1_scale=torch.empty(1),
        w2_scale=torch.empty(1),
        situ_beta=4.0,
        situ_linear_beta=25.0,
        w13_layout="gate_up_interleaved_preshuffled",
        weights_shuffled=True,
        quantization_supported=True,
        activation="situ",
        num_experts=896,
        topk=16,
        num_expert_group=1,
        topk_group=1,
        renormalize=True,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        expert_parallel=False,
        eplb_enabled=False,
        lora_enabled=False,
        has_expert_bias=False,
        apply_router_weight_on_input=False,
        expert_map_active=False,
        routing_capture_enabled=False,
        custom_routing_active=False,
        input_ids_active=False,
        routing_method="DeepSeekV3",
    )


def test_persistent_moe_fails_closed_on_cpu():
    request = _request()

    assert not supports_kimi_k3_b1_persistent_moe(request)
    assert prepare_kimi_k3_b1_persistent_moe(request) is None


def test_persistent_moe_accepts_native_kimi_k3_weight_layout(
    monkeypatch: pytest.MonkeyPatch,
):
    packed_shapes = []
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")
    monkeypatch.setattr(
        persistent_moe,
        "supports_kimi_k3_b1_route_sort_parallel",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        persistent_moe,
        "_has_validated_stage1_kernel",
        lambda: True,
    )

    def record_packed_shape(*_args, shape, **_kwargs):
        packed_shapes.append(shape)
        return True

    monkeypatch.setattr(
        persistent_moe,
        "_is_packed_tensor",
        record_packed_shape,
    )

    assert supports_kimi_k3_b1_persistent_moe(_request())
    assert packed_shapes == [
        (896, 768, 1792),
        (896, 3584, 192),
        (688128, 112),
        (3211264, 16),
    ]


def test_persistent_moe_stage1_variant_is_registered_explicitly():
    parameters = get_flydsl_kernel_params(
        "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_gui_kw7_fp8"
    )

    assert parameters is not None
    assert {
        "stage": parameters["stage"],
        "a_dtype": parameters["a_dtype"],
        "b_dtype": parameters["b_dtype"],
        "out_dtype": parameters["out_dtype"],
        "tile_m": parameters["tile_m"],
        "tile_n": parameters["tile_n"],
        "tile_k": parameters["tile_k"],
        "waves_per_eu": parameters["waves_per_eu"],
        "b_nt": parameters["b_nt"],
        "gate_mode": parameters["gate_mode"],
        "k_wave": parameters["k_wave"],
    } == persistent_moe._STAGE1_EXPECTED


@pytest.mark.parametrize(
    ("field", "unsupported"),
    [
        ("situ_beta", 1.0),
        ("situ_linear_beta", 1.0),
        ("w13_layout", "gate_up_separated_preshuffled"),
        ("weights_shuffled", False),
        ("quantization_supported", False),
        ("activation", "silu"),
        ("num_experts", 128),
        ("topk", 8),
        ("num_expert_group", 8),
        ("topk_group", 4),
        ("renormalize", False),
        ("scoring_func", "softmax"),
        ("routed_scaling_factor", 2.5),
        ("expert_parallel", True),
        ("eplb_enabled", True),
        ("lora_enabled", True),
        ("has_expert_bias", True),
        ("apply_router_weight_on_input", True),
        ("expert_map_active", True),
        ("routing_capture_enabled", True),
        ("custom_routing_active", True),
        ("input_ids_active", True),
        ("routing_method", "Renormalize"),
    ],
)
def test_persistent_moe_contract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    unsupported: object,
):
    monkeypatch.setenv("AITER_SITUV2_A8W4", "1")
    monkeypatch.setattr(
        persistent_moe,
        "supports_kimi_k3_b1_route_sort_parallel",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        persistent_moe,
        "_has_validated_stage1_kernel",
        lambda: True,
    )
    monkeypatch.setattr(
        persistent_moe,
        "_is_packed_tensor",
        lambda *args, **kwargs: True,
    )

    assert supports_kimi_k3_b1_persistent_moe(_request())
    assert not supports_kimi_k3_b1_persistent_moe(
        replace(_request(), **{field: unsupported})
    )
