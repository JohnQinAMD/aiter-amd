# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Narrow gfx950 wrapper for overlapped Kimi-K3 B1 route and MXFP8 prep."""

import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

KimiK3RouteSortResult = tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]


@functools.cache
def _compiled_kimi_k3_b1_route_sort_parallel():
    from aiter.ops.flydsl.kernels.kimi_k3_b1_route_sort_parallel import (
        build_kimi_k3_b1_route_sort_parallel_module,
    )

    return build_kimi_k3_b1_route_sort_parallel_module()


def _supports_route_contract(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
) -> bool:
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
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def _supports_hidden_contract(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    *,
    model_dim: int,
) -> bool:
    return (
        hidden_states.is_cuda
        and hidden_states.device == logits.device
        and hidden_states.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and tuple(hidden_states.shape) == (1, 3584)
        and model_dim == 3584
    )


def supports_kimi_k3_b1_route_sort_parallel(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    model_dim: int,
) -> bool:
    """Return whether the fixed B1 route/quant specialization is safe."""

    return _supports_route_contract(
        logits, correction_bias
    ) and _supports_hidden_contract(
        hidden_states,
        logits,
        model_dim=model_dim,
    )


def kimi_k3_b1_route_sort_parallel(
    hidden_states: torch.Tensor,
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    model_dim: int,
) -> KimiK3RouteSortResult:
    """Route one token and emit exact prequantized A8W4 stage-1 inputs."""

    if not _supports_route_contract(logits, correction_bias):
        raise NotImplementedError(
            "parallel route-sort only supports contiguous gfx950 FP32/BF16 "
            "B1x896, topk=16, group=1/1, block_size_m=32"
        )
    if not _supports_hidden_contract(
        hidden_states,
        logits,
        model_dim=model_dim,
    ):
        raise ValueError(
            "hidden_states must be contiguous gfx950 BF16 [1, 3584] and "
            f"model_dim must be 3584; got {hidden_states.shape=}, "
            f"{hidden_states.dtype=}, {model_dim=}"
        )

    from aiter import dtypes
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    device = logits.device
    topk_weights = torch.empty((1, 16), dtype=torch.float32, device=device)
    topk_ids = torch.empty((1, 16), dtype=torch.int32, device=device)
    sorted_ids = torch.empty(16 * 32, dtype=torch.int32, device=device)
    sorted_weights = torch.empty(16 * 32, dtype=torch.float32, device=device)
    sorted_expert_ids = torch.empty(16, dtype=torch.int32, device=device)
    num_valid_ids = torch.empty(2, dtype=torch.int32, device=device)
    moe_buf = torch.empty((1, model_dim), dtype=torch.bfloat16, device=device)
    quantized_hidden = torch.empty(
        (1, model_dim),
        dtype=dtypes.fp8,
        device=device,
    )
    quantized_scales = torch.empty(
        (16 * 32, model_dim // 32),
        dtype=dtypes.fp8_e8m0,
        device=device,
    )

    _compiled_kimi_k3_b1_route_sort_parallel()(
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
        ptr_arg(hidden_states),
        ptr_arg(quantized_hidden),
        ptr_arg(quantized_scales),
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
        quantized_hidden,
        quantized_scales,
    )
