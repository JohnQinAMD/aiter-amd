# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Narrow gfx950 dispatch for Kimi-K3 B1 route+sort metadata fusion."""

import enum
import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available


class KimiK3RouteSortDispatch(enum.Enum):
    """Dispatch decision for the fixed production-shape specialization."""

    FLYDSL_GFX950_B1 = "flydsl_gfx950_b1"
    UNSUPPORTED = "unsupported"


KimiK3RouteSortResult = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


def supports_kimi_k3_b1_route_sort(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    num_experts: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    block_size_m: int,
) -> bool:
    """Return whether the exact Kimi-K3 B1 FlyDSL specialization is safe."""

    return (
        logits.is_cuda
        and correction_bias.is_cuda
        and logits.device == correction_bias.device
        and logits.dtype == torch.float32
        and correction_bias.dtype == torch.bfloat16
        and logits.is_contiguous()
        and correction_bias.is_contiguous()
        and tuple(logits.shape) == (1, 896)
        and tuple(correction_bias.shape) == (896,)
        and num_experts == 896
        and topk == 16
        and num_expert_group == 1
        and topk_group == 1
        and block_size_m == 32
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def kimi_k3_route_sort_dispatch(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    num_experts: int,
    topk: int,
    num_expert_group: int,
    topk_group: int,
    block_size_m: int,
) -> KimiK3RouteSortDispatch:
    """Choose the named route+sort implementation without changing fallbacks."""

    if supports_kimi_k3_b1_route_sort(
        logits,
        correction_bias,
        num_experts=num_experts,
        topk=topk,
        num_expert_group=num_expert_group,
        topk_group=topk_group,
        block_size_m=block_size_m,
    ):
        return KimiK3RouteSortDispatch.FLYDSL_GFX950_B1
    return KimiK3RouteSortDispatch.UNSUPPORTED


@functools.cache
def _compiled_kimi_k3_b1_route_sort():
    from aiter.ops.flydsl.kernels.kimi_k3_b1_route_sort import (
        build_kimi_k3_b1_route_sort_module,
    )

    return build_kimi_k3_b1_route_sort_module()


def kimi_k3_b1_route_sort(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    model_dim: int,
) -> KimiK3RouteSortResult:
    """Route one token and emit the standard AITER/Opus metadata tuple."""

    decision = kimi_k3_route_sort_dispatch(
        logits,
        correction_bias,
        num_experts=896,
        topk=16,
        num_expert_group=1,
        topk_group=1,
        block_size_m=32,
    )
    if decision is not KimiK3RouteSortDispatch.FLYDSL_GFX950_B1:
        raise NotImplementedError(
            "kimi_k3_b1_route_sort only supports contiguous gfx950 FP32/BF16 "
            "B1x896, topk=16, group=1/1, block_size_m=32"
        )
    if model_dim <= 0 or model_dim % 2:
        raise ValueError(f"model_dim must be positive and even, got {model_dim}")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    device = logits.device
    topk_weights = torch.empty((1, 16), dtype=torch.float32, device=device)
    topk_ids = torch.empty((1, 16), dtype=torch.int32, device=device)
    sorted_ids = torch.empty(16 * 32, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(16 * 32, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(16, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((1, model_dim), dtype=torch.bfloat16, device=device)

    _compiled_kimi_k3_b1_route_sort()(
        ptr_arg(logits),
        ptr_arg(correction_bias),
        ptr_arg(topk_weights),
        ptr_arg(topk_ids),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_weights),
        ptr_arg(sorted_expert_ids),
        ptr_arg(num_valid_ids),
        ptr_arg(moe_buf),
        moe_buf.numel() // 2,
        stream=torch.cuda.current_stream(logits.device),
    )
    return (
        topk_weights,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    )
