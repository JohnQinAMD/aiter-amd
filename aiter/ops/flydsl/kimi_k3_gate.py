# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Narrow gfx950 dispatch for the Kimi-K3 B1 router projection."""

import enum
import functools

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available


class KimiK3GateProjectionDispatch(enum.Enum):
    """Dispatch decision for the fixed production-shape projection."""

    FLYDSL_GFX950_B1 = "flydsl_gfx950_b1"
    UNSUPPORTED = "unsupported"


def supports_kimi_k3_b1_gate_projection(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> bool:
    """Return whether the exact Kimi-K3 B1 projection is safe."""

    return (
        hidden_states.is_cuda
        and router_weight.is_cuda
        and hidden_states.device == router_weight.device
        and hidden_states.dtype == torch.bfloat16
        and router_weight.dtype == torch.bfloat16
        and hidden_states.is_contiguous()
        and router_weight.is_contiguous()
        and tuple(hidden_states.shape) == (1, 7168)
        and tuple(router_weight.shape) == (896, 7168)
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def kimi_k3_gate_projection_dispatch(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
) -> KimiK3GateProjectionDispatch:
    """Choose the named projection implementation without changing fallbacks."""

    if supports_kimi_k3_b1_gate_projection(hidden_states, router_weight):
        return KimiK3GateProjectionDispatch.FLYDSL_GFX950_B1
    return KimiK3GateProjectionDispatch.UNSUPPORTED


@functools.cache
def _compiled_kimi_k3_b1_gate_projection():
    from aiter.ops.flydsl.kernels.kimi_k3_b1_gate_projection import (
        build_kimi_k3_b1_gate_projection_module,
    )

    return build_kimi_k3_b1_gate_projection_module()


def kimi_k3_b1_gate_projection(
    hidden_states: torch.Tensor,
    router_weight: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project one hidden-state row to FP32 Kimi-K3 router logits."""

    decision = kimi_k3_gate_projection_dispatch(hidden_states, router_weight)
    if decision is not KimiK3GateProjectionDispatch.FLYDSL_GFX950_B1:
        raise NotImplementedError(
            "kimi_k3_b1_gate_projection only supports contiguous gfx950 BF16 "
            "B1x7168 hidden states and 896x7168 router weights"
        )
    if out is None:
        out = torch.empty((1, 896), dtype=torch.float32, device=hidden_states.device)
    elif (
        out.device != hidden_states.device
        or out.dtype != torch.float32
        or not out.is_contiguous()
        or tuple(out.shape) != (1, 896)
    ):
        raise ValueError(
            "out must be contiguous FP32 shape (1, 896) on the input device"
        )

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    _compiled_kimi_k3_b1_gate_projection()(
        ptr_arg(hidden_states),
        ptr_arg(router_weight),
        ptr_arg(out),
        stream=torch.cuda.current_stream(hidden_states.device),
    )
    return out
