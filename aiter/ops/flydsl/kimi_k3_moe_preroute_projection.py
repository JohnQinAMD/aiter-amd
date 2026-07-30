# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 gfx950 B1 MoE pre-route projection experiment.

This candidate is intentionally narrow and reviewable.  It covers the
trace-attributed cluster:

* routed_down(hidden):       [1, 7168] x [3584, 7168]^T
* shared_gate_up(hidden):    [1, 7168] x [1536, 7168]^T
* SituGLU(shared_gate_up):   [1, 1536] -> [1, 768]
* shared_down(situ_out):     [1, 768] x [7168, 768]^T

The default implementation preserves production ``wvSplitK`` math and only
varies CU counts per projection.  This gives the KernelForge campaign a safe
first branch before replacing the cluster with a true fused FlyDSL/Triton
kernel.
"""

from __future__ import annotations

import os

import torch

_BATCH = 1
_HIDDEN_SIZE = 7168
_ROUTED_SIZE = 3584
_SHARED_GATE_UP_SIZE = 1536
_SHARED_INTERMEDIATE_SIZE = 768
_SHARED_DOWN_OUTPUT_SIZE = 7168
_DEFAULT_CU_COUNT = 256
_IMPL_FLYDSL_DUAL = "flydsl_dual"
_IMPL_WVSPLITK = "wvsplitk"
_SHARED_DOWN_FLYDSL = "flydsl_fused"
_SHARED_DOWN_FLYDSL_PROJECT = "flydsl_project"


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower()


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive, got {parsed}")
    return parsed


def supports_kimi_k3_moe_preroute_projection(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
) -> bool:
    return (
        hidden.is_cuda
        and routed_weight.is_cuda
        and shared_gate_up_weight.is_cuda
        and shared_down_weight.is_cuda
        and hidden.dtype == torch.bfloat16
        and routed_weight.dtype == torch.bfloat16
        and shared_gate_up_weight.dtype == torch.bfloat16
        and shared_down_weight.dtype == torch.bfloat16
        and hidden.dim() == 2
        and hidden.shape == (_BATCH, _HIDDEN_SIZE)
        and routed_weight.dim() == 2
        and routed_weight.shape == (_ROUTED_SIZE, _HIDDEN_SIZE)
        and shared_gate_up_weight.dim() == 2
        and shared_gate_up_weight.shape == (_SHARED_GATE_UP_SIZE, _HIDDEN_SIZE)
        and shared_down_weight.dim() == 2
        and shared_down_weight.shape
        == (
            _SHARED_DOWN_OUTPUT_SIZE,
            _SHARED_INTERMEDIATE_SIZE,
        )
        and hidden.is_contiguous()
        and routed_weight.is_contiguous()
        and shared_gate_up_weight.is_contiguous()
        and shared_down_weight.is_contiguous()
    )


def _situ_and_mul(
    x: torch.Tensor,
    beta: float,
    linear_beta: float,
) -> torch.Tensor:
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "situ_and_mul"):
        try:
            import vllm._C  # noqa: F401
        except ImportError:
            pass
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, "situ_and_mul"):
        raise RuntimeError("missing production vLLM torch.ops._C.situ_and_mul")
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), dtype=x.dtype, device=x.device)
    torch.ops._C.situ_and_mul(out, x, float(beta), float(linear_beta))
    return out


def kimi_k3_moe_preroute_projection(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_gate_up_weight: torch.Tensor,
    shared_down_weight: torch.Tensor,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routed latent output and shared-expert partial output.

    Environment knobs:

    * ``KIMI_K3_PREROUTE_IMPL``: ``wvsplitk`` or ``flydsl_dual``
    * ``KIMI_K3_PREROUTE_ROUTED_CU``
    * ``KIMI_K3_PREROUTE_SHARED_GATE_UP_CU``
    * ``KIMI_K3_PREROUTE_SHARED_DOWN_CU``
    * ``KIMI_K3_PREROUTE_SHARED_DOWN_IMPL``: ``wvsplitk``,
      ``flydsl_fused``, or ``flydsl_project``
    """

    if not supports_kimi_k3_moe_preroute_projection(
        hidden,
        routed_weight,
        shared_gate_up_weight,
        shared_down_weight,
    ):
        raise ValueError("unsupported Kimi-K3 B1 pre-route projection inputs")

    impl = _env_str("KIMI_K3_PREROUTE_IMPL", _IMPL_WVSPLITK)
    if impl not in (_IMPL_FLYDSL_DUAL, _IMPL_WVSPLITK):
        raise ValueError(
            "KIMI_K3_PREROUTE_IMPL must be "
            f"{_IMPL_WVSPLITK!r} or {_IMPL_FLYDSL_DUAL!r}; got {impl!r}"
        )

    from vllm import _custom_ops as ops

    if impl == _IMPL_FLYDSL_DUAL:
        from aiter.ops.flydsl.kimi_k3_moe_dual_projection import (
            kimi_k3_moe_dual_projection,
        )

        routed, shared_gate_up = kimi_k3_moe_dual_projection(
            hidden,
            routed_weight,
            shared_gate_up_weight,
        )
    else:
        routed_cu = _env_int("KIMI_K3_PREROUTE_ROUTED_CU", _DEFAULT_CU_COUNT)
        shared_gate_up_cu = _env_int(
            "KIMI_K3_PREROUTE_SHARED_GATE_UP_CU",
            _DEFAULT_CU_COUNT,
        )
        routed = ops.wvSplitK(routed_weight, hidden, routed_cu, None)
        shared_gate_up = ops.wvSplitK(
            shared_gate_up_weight,
            hidden,
            shared_gate_up_cu,
            None,
        )

    shared_down_impl = _env_str(
        "KIMI_K3_PREROUTE_SHARED_DOWN_IMPL",
        _IMPL_WVSPLITK,
    )
    if shared_down_impl == _SHARED_DOWN_FLYDSL:
        from aiter.ops.flydsl.kimi_k3_moe_dual_projection import (
            kimi_k3_shared_down_bf16,
        )

        shared_output = kimi_k3_shared_down_bf16(
            shared_gate_up,
            shared_down_weight,
            situ_beta=situ_beta,
            situ_linear_beta=situ_linear_beta,
        )
    elif shared_down_impl == _SHARED_DOWN_FLYDSL_PROJECT:
        from aiter.ops.flydsl.kimi_k3_moe_dual_projection import (
            kimi_k3_shared_down_bf16_activated,
        )

        shared_activated = _situ_and_mul(
            shared_gate_up,
            situ_beta,
            situ_linear_beta,
        )
        shared_output = kimi_k3_shared_down_bf16_activated(
            shared_activated,
            shared_down_weight,
        )
    elif shared_down_impl == _IMPL_WVSPLITK:
        shared_down_cu = _env_int(
            "KIMI_K3_PREROUTE_SHARED_DOWN_CU",
            _DEFAULT_CU_COUNT,
        )
        shared_activated = _situ_and_mul(
            shared_gate_up,
            situ_beta,
            situ_linear_beta,
        )
        shared_output = ops.wvSplitK(
            shared_down_weight,
            shared_activated,
            shared_down_cu,
            None,
        )
    else:
        raise ValueError(
            "KIMI_K3_PREROUTE_SHARED_DOWN_IMPL must be "
            f"{_IMPL_WVSPLITK!r}, {_SHARED_DOWN_FLYDSL!r}, or "
            f"{_SHARED_DOWN_FLYDSL_PROJECT!r}; "
            f"got {shared_down_impl!r}"
        )
    return routed, shared_output
