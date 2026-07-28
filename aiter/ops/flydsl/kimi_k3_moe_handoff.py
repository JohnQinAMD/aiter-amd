# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Typed Kimi-K3 route-to-MXFP4-expert ownership boundary."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.ops.flydsl.kimi_k3_moe_route import (
    kimi_k3_b1_route_sort,
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
    _EXPERTS * _MODEL_DIM,
    ((_INTER_DIM // 32 + 7) // 8) * 8,
)


class KimiK3MXFP4W13Layout(str, Enum):
    """Stable stage-1 weight layouts selected during vLLM weight loading."""

    GATE_UP_SEPARATED_PRESHUFFLED = "gate_up_separated_preshuffled"
    GATE_UP_INTERLEAVED_PRESHUFFLED = "gate_up_interleaved_preshuffled"


@dataclass(frozen=True)
class KimiK3MXFP4ExpertMode:
    """Activation and W13 layout contract for one Kimi-K3 expert mode."""

    activation_dtype: torch.dtype
    gate_mode: GateMode
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


@dataclass(frozen=True)
class KimiK3MXFP4ExpertRequest:
    """All tensors owned by the synchronous Kimi-K3 expert handoff."""

    hidden_states: torch.Tensor
    router_logits: torch.Tensor
    correction_bias: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    w1_scale: torch.Tensor
    w2_scale: torch.Tensor
    situ_beta: float
    situ_linear_beta: float


@dataclass(frozen=True)
class KimiK3MXFP4ExpertResult:
    """Routing, Opus stage metadata, and the routed BF16 expert output."""

    expert_ids: torch.Tensor
    routing_weights: torch.Tensor
    stage1_sorted_token_ids: torch.Tensor
    stage1_sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor
    expert_output: torch.Tensor
    activation_dtype: torch.dtype
    w13_layout: KimiK3MXFP4W13Layout


def _is_packed_scale(
    tensor: torch.Tensor,
    *,
    device: torch.device,
    shape: tuple[int, int],
) -> bool:
    return (
        tensor.is_cuda
        and tensor.device == device
        and tensor.dtype == dtypes.fp8_e8m0
        and tensor.is_contiguous()
        and tuple(tensor.shape) == shape
    )


def _has_expected_w13_layout(
    tensor: torch.Tensor,
    mode: KimiK3MXFP4ExpertMode,
) -> bool:
    layout = getattr(tensor, "kimi_k3_w13_layout", None)
    if isinstance(layout, KimiK3MXFP4W13Layout):
        layout = layout.value
    if mode.gate_mode == GateMode.INTERLEAVE:
        return layout == mode.w13_layout.value
    # Preserve the accepted A16W4 API for preshuffled tensors created before
    # the explicit layout tag existed.
    return layout in (None, mode.w13_layout.value)


def supports_kimi_k3_mxfp4_expert_handoff(
    request: KimiK3MXFP4ExpertRequest,
) -> bool:
    """Return whether the exact gfx950 Kimi-K3 TP8 specialization is safe."""

    hidden_states = request.hidden_states
    device = hidden_states.device
    mode = kimi_k3_mxfp4_expert_mode()
    return (
        os.environ.get("AITER_DISABLE", "0") != "1"
        and supports_kimi_k3_b1_route_sort(
            request.router_logits,
            request.correction_bias,
            num_experts=_EXPERTS,
            topk=_TOPK,
            num_expert_group=1,
            topk_group=1,
            block_size_m=_BLOCK_M,
        )
        and hidden_states.is_cuda
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and tuple(hidden_states.shape) == (1, _MODEL_DIM)
        and request.router_logits.device == device
        and request.w1.is_cuda
        and request.w1.device == device
        and request.w1.dtype == dtypes.fp4x2
        and request.w1.is_contiguous()
        and tuple(request.w1.shape) == (_EXPERTS, 2 * _INTER_DIM, _MODEL_DIM // 2)
        and bool(getattr(request.w1, "is_shuffled", False))
        and _has_expected_w13_layout(request.w1, mode)
        and request.w2.is_cuda
        and request.w2.device == device
        and request.w2.dtype == dtypes.fp4x2
        and request.w2.is_contiguous()
        and tuple(request.w2.shape) == (_EXPERTS, _MODEL_DIM, _INTER_DIM // 2)
        and bool(getattr(request.w2, "is_shuffled", False))
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
    )


def kimi_k3_mxfp4_expert_handoff(
    request: KimiK3MXFP4ExpertRequest,
) -> KimiK3MXFP4ExpertResult:
    """Consume prepared route metadata exactly once in stage1/stage2."""

    if not supports_kimi_k3_mxfp4_expert_handoff(request):
        mode = kimi_k3_mxfp4_expert_mode()
        raise NotImplementedError(
            "Kimi-K3 prepared-route expert handoff requires contiguous gfx950 "
            "B1 BF16 activations, FP32 1x896 logits, BF16 correction bias, "
            "and preshuffled TP8 896x3584x384 MXFP4 weights matching "
            f"{mode.w13_layout.value}"
        )

    mode = kimi_k3_mxfp4_expert_mode()
    (
        routing_weights,
        expert_ids,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    ) = kimi_k3_b1_route_sort(
        request.router_logits,
        request.correction_bias,
        model_dim=_MODEL_DIM,
    )

    # Import locally to avoid a package cycle while aiter.fused_moe imports
    # FlyDSL dispatch helpers.
    from aiter.fused_moe import fused_moe_2stages

    expert_output = fused_moe_2stages(
        request.hidden_states,
        request.w1,
        request.w2,
        _TOPK,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        True,
        _BLOCK_M,
        activation=ActivationType.Situv2,
        quant_type=QuantType.per_1x32,
        q_dtype_a=mode.activation_dtype,
        q_dtype_w=dtypes.fp4x2,
        w1_scale=request.w1_scale,
        w2_scale=request.w2_scale,
        topk_ids=expert_ids,
        topk_weights=routing_weights,
        beta=request.situ_beta,
        linear_beta=request.situ_linear_beta,
        gate_mode=mode.gate_mode.value,
    )
    return KimiK3MXFP4ExpertResult(
        expert_ids=expert_ids,
        routing_weights=routing_weights,
        stage1_sorted_token_ids=sorted_token_ids,
        stage1_sorted_weights=sorted_weights,
        sorted_expert_ids=sorted_expert_ids,
        num_valid_ids=num_valid_ids,
        expert_output=expert_output,
        activation_dtype=mode.activation_dtype,
        w13_layout=mode.w13_layout,
    )


__all__ = [
    "KimiK3MXFP4ExpertMode",
    "KimiK3MXFP4ExpertRequest",
    "KimiK3MXFP4ExpertResult",
    "KimiK3MXFP4W13Layout",
    "kimi_k3_mxfp4_expert_handoff",
    "kimi_k3_mxfp4_expert_mode",
    "supports_kimi_k3_mxfp4_expert_handoff",
]
