# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Typed Kimi-K3 B1 route-to-expert specialization for gfx950."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

from aiter import dtypes
from aiter.ops.flydsl.kimi_k3_moe_route_parallel import (
    kimi_k3_b1_route_sort_parallel,
    supports_kimi_k3_b1_route_sort_parallel,
)
from aiter.ops.flydsl.moe_kernels import (
    flydsl_moe_stage1,
    flydsl_moe_stage2,
    get_flydsl_kernel_params,
)

_EXPERTS = 896
_TOPK = 16
_MODEL_DIM = 3584
_INTERMEDIATE_DIM = 384
_SORT_BLOCK_M = 32
_STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_gui_kw7_fp8"
_STAGE1_EXPECTED = {
    "stage": 1,
    "a_dtype": "fp8",
    "b_dtype": "fp4",
    "out_dtype": "fp8",
    "tile_m": 32,
    "tile_n": 64,
    "tile_k": 256,
    "waves_per_eu": 1,
    "b_nt": 2,
    "gate_mode": "interleave",
    "k_wave": 7,
}
_W1_SCALE_SHAPE = (
    _EXPERTS * 2 * _INTERMEDIATE_DIM,
    _MODEL_DIM // 32,
)
_W2_SCALE_SHAPE = (
    ((_EXPERTS * _MODEL_DIM + 255) // 256) * 256,
    ((_INTERMEDIATE_DIM // 32 + 7) // 8) * 8,
)


@dataclass(frozen=True)
class KimiK3PersistentMoERequest:
    """Complete immutable contract for one specialized MoE invocation."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    correction_bias: torch.Tensor | None
    w1: torch.Tensor | None
    w2: torch.Tensor | None
    w1_scale: torch.Tensor | None
    w2_scale: torch.Tensor | None
    situ_beta: float
    situ_linear_beta: float
    w13_layout: str | None
    weights_shuffled: bool
    quantization_supported: bool
    activation: str
    num_experts: int
    topk: int
    num_expert_group: int
    topk_group: int
    renormalize: bool
    scoring_func: str
    routed_scaling_factor: float
    expert_parallel: bool
    eplb_enabled: bool
    lora_enabled: bool
    has_expert_bias: bool
    apply_router_weight_on_input: bool
    expert_map_active: bool
    routing_capture_enabled: bool
    custom_routing_active: bool
    input_ids_active: bool
    routing_method: str


@dataclass(frozen=True)
class KimiK3PersistentMoEMetadata:
    """Route metadata and prequantized stage-1 activation owned together."""

    routing_weights: torch.Tensor
    expert_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    moe_buf: torch.Tensor
    quantized_hidden_states: torch.Tensor
    quantized_scales: torch.Tensor


def _is_packed_tensor(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    dtype: torch.dtype,
    shape: tuple[int, ...],
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtype
        and tensor.is_contiguous()
        and tuple(tensor.shape) == shape
    )


def _has_validated_stage1_kernel() -> bool:
    parameters = get_flydsl_kernel_params(_STAGE1_KERNEL)
    return parameters is not None and all(
        parameters.get(name) == expected for name, expected in _STAGE1_EXPECTED.items()
    )


def _has_native_weight_layout(
    request: KimiK3PersistentMoERequest,
    device: torch.device,
) -> bool:
    return (
        _is_packed_tensor(
            request.w1,
            device=device,
            dtype=dtypes.fp4x2,
            shape=(_EXPERTS, 2 * _INTERMEDIATE_DIM, _MODEL_DIM // 2),
        )
        and _is_packed_tensor(
            request.w2,
            device=device,
            dtype=dtypes.fp4x2,
            shape=(_EXPERTS, _MODEL_DIM, _INTERMEDIATE_DIM // 2),
        )
        and _is_packed_tensor(
            request.w1_scale,
            device=device,
            dtype=dtypes.fp8_e8m0,
            shape=_W1_SCALE_SHAPE,
        )
        and _is_packed_tensor(
            request.w2_scale,
            device=device,
            dtype=dtypes.fp8_e8m0,
            shape=_W2_SCALE_SHAPE,
        )
    )


def _has_supported_quantized_activation(
    request: KimiK3PersistentMoERequest,
) -> bool:
    return (
        request.activation == "situ"
        and request.situ_beta == 4.0
        and request.situ_linear_beta == 25.0
        and request.quantization_supported
        and request.weights_shuffled
        and request.w13_layout == "gate_up_interleaved_preshuffled"
    )


def _has_supported_routing(request: KimiK3PersistentMoERequest) -> bool:
    return (
        request.num_experts == _EXPERTS
        and request.topk == _TOPK
        and request.num_expert_group == 1
        and request.topk_group == 1
        and request.renormalize
        and request.scoring_func == "sigmoid"
        and request.routed_scaling_factor == 1.0
        and request.routing_method == "DeepSeekV3"
    )


def _has_unsupported_runtime_features(
    request: KimiK3PersistentMoERequest,
) -> bool:
    return any(
        (
            request.expert_parallel,
            request.eplb_enabled,
            request.lora_enabled,
            request.has_expert_bias,
            request.apply_router_weight_on_input,
            request.expert_map_active,
            request.routing_capture_enabled,
            request.custom_routing_active,
            request.input_ids_active,
        )
    )


def supports_kimi_k3_b1_persistent_moe(
    request: KimiK3PersistentMoERequest,
) -> bool:
    """Return whether the complete fixed-shape specialization is safe."""

    hidden_states = request.hidden_states
    correction_bias = request.correction_bias
    if not isinstance(correction_bias, torch.Tensor):
        return False
    device = hidden_states.device
    return (
        os.environ.get("AITER_DISABLE", "0") != "1"
        and os.environ.get("AITER_SITUV2_A8W4", "0") == "1"
        and supports_kimi_k3_b1_route_sort_parallel(
            hidden_states,
            request.router_logits,
            correction_bias,
            model_dim=_MODEL_DIM,
        )
        and _has_validated_stage1_kernel()
        and _has_native_weight_layout(request, device)
        and _has_supported_quantized_activation(request)
        and _has_supported_routing(request)
        and not _has_unsupported_runtime_features(request)
    )


def prepare_kimi_k3_b1_persistent_moe(
    request: KimiK3PersistentMoERequest,
) -> KimiK3PersistentMoEMetadata | None:
    """Prepare route and stage-1 activation, or select the generic fallback."""

    if not supports_kimi_k3_b1_persistent_moe(request):
        return None
    assert request.correction_bias is not None
    return KimiK3PersistentMoEMetadata(
        *kimi_k3_b1_route_sort_parallel(
            request.hidden_states,
            request.router_logits,
            request.correction_bias,
            model_dim=_MODEL_DIM,
        )
    )


def consume_kimi_k3_b1_persistent_moe(
    request: KimiK3PersistentMoERequest,
    metadata: KimiK3PersistentMoEMetadata,
) -> torch.Tensor:
    """Consume prepared metadata exactly once through both expert GEMMs."""

    if not supports_kimi_k3_b1_persistent_moe(request):
        raise NotImplementedError("unsupported Kimi-K3 persistent-MoE contract")
    assert request.w1 is not None
    assert request.w2 is not None
    assert request.w1_scale is not None
    assert request.w2_scale is not None

    intermediate, intermediate_scale = flydsl_moe_stage1(
        a=metadata.quantized_hidden_states,
        w1=request.w1,
        sorted_token_ids=metadata.sorted_token_ids,
        sorted_expert_ids=metadata.sorted_expert_ids,
        num_valid_ids=metadata.num_valid_ids,
        out=None,
        topk=_TOPK,
        tile_m=32,
        tile_n=64,
        tile_k=256,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="fp8",
        act="situv2",
        situ_beta=request.situ_beta,
        situ_linear_beta=request.situ_linear_beta,
        w1_scale=request.w1_scale,
        a1_scale=metadata.quantized_scales,
        sorted_weights=None,
        use_async_copy=True,
        k_batch=1,
        waves_per_eu=1,
        b_nt=2,
        gate_mode="interleave",
        model_dim_pad=0,
        inter_dim_pad=0,
        xcd_swizzle=0,
        k_wave=7,
    )
    return flydsl_moe_stage2(
        inter_states=intermediate.view(
            1,
            _TOPK,
            _INTERMEDIATE_DIM,
        ),
        w2=request.w2,
        sorted_token_ids=metadata.sorted_token_ids,
        sorted_expert_ids=metadata.sorted_expert_ids,
        num_valid_ids=metadata.num_valid_ids,
        out=metadata.moe_buf,
        topk=_TOPK,
        tile_m=32,
        tile_n=128,
        tile_k=128,
        a_dtype="fp8",
        b_dtype="fp4",
        out_dtype="bf16",
        mode="atomic",
        w2_scale=request.w2_scale,
        a2_scale=intermediate_scale,
        sorted_weights=metadata.sorted_weights,
        sort_block_m=_SORT_BLOCK_M,
        persist=False,
        waves_per_eu=1,
        use_async_copy=False,
        cu_num_mul=1,
        b_nt=2,
        model_dim_pad=0,
        inter_dim_pad=0,
        xcd_swizzle=0,
    )


__all__ = [
    "KimiK3PersistentMoEMetadata",
    "KimiK3PersistentMoERequest",
    "consume_kimi_k3_b1_persistent_moe",
    "prepare_kimi_k3_b1_persistent_moe",
    "supports_kimi_k3_b1_persistent_moe",
]
