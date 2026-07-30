# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Typed route-to-expert ownership boundary for the gfx950 B1 specialization."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.kimi_k3_moe_route import (
    KimiK3RouteSortDispatch,
    kimi_k3_b1_route_sort,
    kimi_k3_route_sort_dispatch,
    supports_kimi_k3_b1_route_sort,
)
from aiter.ops.flydsl.moe_common import GateMode

_EXPERTS = 896
_TOPK = 16
_MODEL_DIM = 3584
_INTER_DIM = 384
_BLOCK_M = 32
_W1_SCALE_SHAPE = (_EXPERTS * 2 * _INTER_DIM, _MODEL_DIM // 32)
_W2_SCALE_SHAPE = (
    ((_EXPERTS * _MODEL_DIM + 255) // 256) * 256,
    ((_INTER_DIM // 32 + 7) // 8) * 8,
)


class KimiK3MXFP4W13Layout(str, Enum):
    """Stable stage-1 layouts established while loading weights."""

    GATE_UP_SEPARATED_PRESHUFFLED = "gate_up_separated_preshuffled"
    GATE_UP_INTERLEAVED_PRESHUFFLED = "gate_up_interleaved_preshuffled"


@dataclass(frozen=True)
class KimiK3MXFP4ExpertMode:
    """Activation and stage-1 layout selected before execution."""

    activation_dtype: torch.dtype
    gate_mode: GateMode
    w13_layout: KimiK3MXFP4W13Layout


@dataclass(frozen=True)
class KimiK3MXFP4ExpertRequest:
    """Complete immutable contract for one route-to-expert invocation."""

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
class KimiK3PreparedMoEMetadata:
    """The accepted seven-tensor AITER/Opus routing contract."""

    routing_weights: torch.Tensor
    expert_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    moe_buf: torch.Tensor


@dataclass(frozen=True)
class KimiK3MXFP4ExpertResult:
    """Prepared routing and the routed BF16 expert output."""

    metadata: KimiK3PreparedMoEMetadata
    expert_output: torch.Tensor
    activation_dtype: torch.dtype
    w13_layout: KimiK3MXFP4W13Layout


def kimi_k3_mxfp4_expert_mode() -> KimiK3MXFP4ExpertMode:
    """Own the live A8W4/A16W4 activation and weight-layout selection."""

    if os.environ.get("AITER_SITUV2_A8W4", "0") == "1":
        return KimiK3MXFP4ExpertMode(
            activation_dtype=dtypes.fp8,
            gate_mode=GateMode.INTERLEAVE,
            w13_layout=KimiK3MXFP4W13Layout.GATE_UP_INTERLEAVED_PRESHUFFLED,
        )
    return KimiK3MXFP4ExpertMode(
        activation_dtype=dtypes.bf16,
        gate_mode=GateMode.SEPARATED,
        w13_layout=KimiK3MXFP4W13Layout.GATE_UP_SEPARATED_PRESHUFFLED,
    )


def _is_packed_scale(
    tensor: torch.Tensor | None,
    *,
    device: torch.device,
    shape: tuple[int, int],
) -> bool:
    return (
        isinstance(tensor, torch.Tensor)
        and tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtypes.fp8_e8m0
        and tensor.is_contiguous()
        and tuple(tensor.shape) == shape
    )


def _supports_kimi_k3_b1_route_sort_request(
    request: KimiK3MXFP4ExpertRequest,
) -> bool:
    """Implement the typed production contract for the named predicate."""

    hidden_states = request.hidden_states
    device = hidden_states.device
    mode = kimi_k3_mxfp4_expert_mode()
    if not isinstance(request.correction_bias, torch.Tensor):
        return False
    from aiter.ops.flydsl.kimi_k3_moe_route import _supports_route_tensor_contract

    primitive_supported = _supports_route_tensor_contract(
        request.router_logits,
        request.correction_bias,
        num_experts=request.num_experts,
        topk=request.topk,
        num_expert_group=request.num_expert_group,
        topk_group=request.topk_group,
        block_size_m=_BLOCK_M,
    )
    return (
        os.environ.get("AITER_DISABLE", "0") != "1"
        and primitive_supported
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and tuple(hidden_states.shape) == (1, _MODEL_DIM)
        and request.router_logits.device == device
        and isinstance(request.w1, torch.Tensor)
        and request.w1.is_cuda
        and request.w1.device == device
        and request.w1.dtype == dtypes.fp4x2
        and request.w1.is_contiguous()
        and tuple(request.w1.shape) == (_EXPERTS, 2 * _INTER_DIM, _MODEL_DIM // 2)
        and isinstance(request.w2, torch.Tensor)
        and request.w2.is_cuda
        and request.w2.device == device
        and request.w2.dtype == dtypes.fp4x2
        and request.w2.is_contiguous()
        and tuple(request.w2.shape) == (_EXPERTS, _MODEL_DIM, _INTER_DIM // 2)
        and _is_packed_scale(
            request.w1_scale,
            device=device,
            shape=_W1_SCALE_SHAPE,
        )
        and _is_packed_scale(
            request.w2_scale,
            device=device,
            shape=_W2_SCALE_SHAPE,
        )
        and request.situ_beta == 4.0
        and request.situ_linear_beta == 25.0
        and request.w13_layout == mode.w13_layout.value
        and request.weights_shuffled
        and request.quantization_supported
        and request.activation == "situ"
        and request.renormalize
        and request.scoring_func == "sigmoid"
        and request.routed_scaling_factor == 1.0
        and not request.expert_parallel
        and not request.eplb_enabled
        and not request.lora_enabled
        and not request.has_expert_bias
        and not request.apply_router_weight_on_input
        and not request.expert_map_active
        and not request.routing_capture_enabled
        and not request.custom_routing_active
        and not request.input_ids_active
        and request.routing_method == "DeepSeekV3"
    )


def prepare_kimi_k3_b1_route_sort(
    request: KimiK3MXFP4ExpertRequest,
) -> KimiK3PreparedMoEMetadata | None:
    """Prepare the seven tensors once, or explicitly select generic fallback."""

    if (
        kimi_k3_route_sort_dispatch(request)
        is not KimiK3RouteSortDispatch.PREPARED_MXFP4_GFX950_B1
    ):
        return None
    return KimiK3PreparedMoEMetadata(
        *kimi_k3_b1_route_sort(
            request.router_logits,
            request.correction_bias,
            model_dim=_MODEL_DIM,
        )
    )


def consume_kimi_k3_b1_route_sort(
    request: KimiK3MXFP4ExpertRequest,
    metadata: KimiK3PreparedMoEMetadata,
) -> KimiK3MXFP4ExpertResult:
    """Feed prepared metadata directly into quantization and both GEMMs."""

    from aiter.fused_moe import fused_moe_2stages

    mode = kimi_k3_mxfp4_expert_mode()
    assert request.w1 is not None
    assert request.w2 is not None
    assert request.w1_scale is not None
    assert request.w2_scale is not None
    expert_output = fused_moe_2stages(
        request.hidden_states,
        request.w1,
        request.w2,
        _TOPK,
        metadata.sorted_token_ids,
        metadata.sorted_weights,
        metadata.sorted_expert_ids,
        metadata.num_valid_ids,
        metadata.moe_buf,
        True,
        _BLOCK_M,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        q_dtype_a=mode.activation_dtype,
        q_dtype_w=dtypes.fp4x2,
        w1_scale=request.w1_scale,
        w2_scale=request.w2_scale,
        topk_ids=metadata.expert_ids,
        topk_weights=metadata.routing_weights,
        beta=request.situ_beta,
        linear_beta=request.situ_linear_beta,
        gate_mode=mode.gate_mode.value,
    )
    return KimiK3MXFP4ExpertResult(
        metadata=metadata,
        expert_output=expert_output,
        activation_dtype=mode.activation_dtype,
        w13_layout=mode.w13_layout,
    )


def kimi_k3_mxfp4_expert_handoff(
    request: KimiK3MXFP4ExpertRequest,
) -> KimiK3MXFP4ExpertResult:
    """Synchronous convenience API for the typed prepare/consume boundary."""

    metadata = prepare_kimi_k3_b1_route_sort(request)
    if metadata is None:
        raise NotImplementedError("unsupported Kimi-K3 prepared-route contract")
    return consume_kimi_k3_b1_route_sort(request, metadata)


__all__ = [
    "KimiK3MXFP4ExpertMode",
    "KimiK3MXFP4ExpertRequest",
    "KimiK3MXFP4ExpertResult",
    "KimiK3MXFP4W13Layout",
    "KimiK3PreparedMoEMetadata",
    "KimiK3RouteSortDispatch",
    "consume_kimi_k3_b1_route_sort",
    "kimi_k3_mxfp4_expert_handoff",
    "kimi_k3_mxfp4_expert_mode",
    "kimi_k3_route_sort_dispatch",
    "prepare_kimi_k3_b1_route_sort",
    "supports_kimi_k3_b1_route_sort",
]
