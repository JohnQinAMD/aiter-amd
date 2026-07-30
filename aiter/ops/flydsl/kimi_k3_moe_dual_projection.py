# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kimi-K3 gfx950 B1 MoE dual-projection experiment.

This module is intentionally narrow.  It is a Wave-1 tuning scaffold for the
Kimi-K3 MoE routed-down plus shared gate-up projections:

* hidden:        [1, 7168] BF16
* routed weight: [3584, 7168] BF16
* shared weight: [1536, 7168] BF16

The default candidate keeps the production ``wvSplitK`` math and varies only
the number of participating CUs for each exact shape.  Set
``KIMI_K3_DUAL_PROJ_IMPL=flydsl_small_m`` to exercise the existing FlyDSL
small-M BF16 HGEMM path for the same two projections.
"""

from __future__ import annotations

import os
import functools
import math

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.utils import is_flydsl_available

_BATCH = 1
_HIDDEN_SIZE = 7168
_ROUTED_SIZE = 3584
_SHARED_UP_SIZE = 1536
_SHARED_INTERMEDIATE_SIZE = _SHARED_UP_SIZE // 2
_PACKED_OUTPUT_SIZE = _ROUTED_SIZE + _SHARED_UP_SIZE
_DEFAULT_CU_COUNT = 256
_DEFAULT_IMPL = "wvsplitk"
_IMPL_FLYDSL_FUSED = "flydsl_fused"
_IMPL_FLYDSL_SMALL_M = "flydsl_small_m"
_IMPL_WVSPLITK = "wvsplitk"


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


def _env_nonnegative_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative, got {parsed}")
    return parsed


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be boolean-like, got {value!r}")


def supports_kimi_k3_moe_dual_projection(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_weight: torch.Tensor,
) -> bool:
    return (
        hidden.is_cuda
        and routed_weight.is_cuda
        and shared_weight.is_cuda
        and hidden.dtype == torch.bfloat16
        and routed_weight.dtype == torch.bfloat16
        and shared_weight.dtype == torch.bfloat16
        and hidden.dim() == 2
        and hidden.shape == (_BATCH, _HIDDEN_SIZE)
        and routed_weight.dim() == 2
        and routed_weight.shape == (_ROUTED_SIZE, _HIDDEN_SIZE)
        and shared_weight.dim() == 2
        and shared_weight.shape == (_SHARED_UP_SIZE, _HIDDEN_SIZE)
        and hidden.is_contiguous()
        and routed_weight.is_contiguous()
        and shared_weight.is_contiguous()
    )


def supports_kimi_k3_moe_dual_projection_packed(
    hidden: torch.Tensor,
    packed_weight: torch.Tensor,
) -> bool:
    """Return whether the one-launch packed projection contract is supported.

    ``packed_weight`` is a model-load-time concatenation of routed-down and
    shared gate/up weights.  Requiring it as an explicit input keeps packing
    out of the decode graph and makes the additional model memory visible to
    the caller.
    """

    return (
        hidden.is_cuda
        and packed_weight.is_cuda
        and hidden.dtype == torch.bfloat16
        and packed_weight.dtype == torch.bfloat16
        and hidden.dim() == 2
        and hidden.shape == (_BATCH, _HIDDEN_SIZE)
        and packed_weight.dim() == 2
        and packed_weight.shape == (_PACKED_OUTPUT_SIZE, _HIDDEN_SIZE)
        and hidden.is_contiguous()
        and packed_weight.is_contiguous()
    )


def supports_kimi_k3_moe_dual_projection_fp8_weight(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
) -> bool:
    """Return whether the row-scaled weight-only FP8 contract is supported."""

    tensors = (
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
    )
    return (
        hidden.is_cuda
        and routed_weight.is_cuda
        and routed_scale.is_cuda
        and shared_weight.is_cuda
        and shared_scale.is_cuda
        and hidden.dtype == torch.bfloat16
        and routed_weight.dtype == torch.float8_e4m3fn
        and routed_scale.dtype == torch.float32
        and shared_weight.dtype == torch.float8_e4m3fn
        and shared_scale.dtype == torch.float32
        and hidden.shape == (_BATCH, _HIDDEN_SIZE)
        and routed_weight.shape == (_ROUTED_SIZE, _HIDDEN_SIZE)
        and routed_scale.shape == (_ROUTED_SIZE,)
        and shared_weight.shape == (_SHARED_UP_SIZE, _HIDDEN_SIZE)
        and shared_scale.shape == (_SHARED_UP_SIZE,)
        and hidden.is_contiguous()
        and routed_weight.is_contiguous()
        and routed_scale.is_contiguous()
        and shared_weight.is_contiguous()
        and shared_scale.is_contiguous()
        and len({tensor.device for tensor in tensors}) == 1
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def kimi_k3_moe_dual_projection_packed(
    hidden: torch.Tensor,
    packed_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project routed-down and shared gate/up outputs in one ``wvSplitK``.

    The packed tensor must be materialized once, after checkpoint loading.
    Packing it in this function would copy roughly 73 MiB per layer per token,
    overwhelming any launch saving.
    """

    if not supports_kimi_k3_moe_dual_projection_packed(hidden, packed_weight):
        raise ValueError("unsupported Kimi-K3 B1 packed dual-projection inputs")

    from vllm import _custom_ops as ops

    cu_count = _env_int("KIMI_K3_DUAL_PROJ_PACKED_CU", _DEFAULT_CU_COUNT)
    packed_output = ops.wvSplitK(packed_weight, hidden, cu_count, None)
    return packed_output.split((_ROUTED_SIZE, _SHARED_UP_SIZE), dim=-1)


def kimi_k3_moe_dual_projection(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return routed-down and shared gate-up projections.

    Environment knobs:

    * ``KIMI_K3_DUAL_PROJ_IMPL``: ``wvsplitk`` or ``flydsl_small_m``
    * ``KIMI_K3_DUAL_PROJ_ROUTED_CU``
    * ``KIMI_K3_DUAL_PROJ_SHARED_CU``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_TILE_N``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_TILE_K``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_SPLIT_K``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_BLOCK_N_WARPS``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_N_TILE_REPEAT``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_PERSISTENT_N_TILES``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_WAVES_PER_EU``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_B_TO_LDS_UNROLL``
    * ``KIMI_K3_DUAL_PROJ_{ROUTED,SHARED}_B_TO_LDS``

    Defaults preserve the production 256-CU dispatch.
    """

    if not supports_kimi_k3_moe_dual_projection(hidden, routed_weight, shared_weight):
        raise ValueError("unsupported Kimi-K3 B1 dual-projection inputs")

    impl = _env_str("KIMI_K3_DUAL_PROJ_IMPL", _DEFAULT_IMPL)
    if impl == _IMPL_FLYDSL_SMALL_M:
        return _flydsl_small_m(hidden, routed_weight, shared_weight)
    if impl == _IMPL_FLYDSL_FUSED:
        return _flydsl_fused(hidden, routed_weight, shared_weight)
    if impl != _IMPL_WVSPLITK:
        raise ValueError(
            "KIMI_K3_DUAL_PROJ_IMPL must be one of "
            f"{_IMPL_WVSPLITK!r}, {_IMPL_FLYDSL_SMALL_M!r}, "
            f"{_IMPL_FLYDSL_FUSED!r}; got {impl!r}"
        )

    from vllm import _custom_ops as ops

    routed_cu = _env_int("KIMI_K3_DUAL_PROJ_ROUTED_CU", _DEFAULT_CU_COUNT)
    shared_cu = _env_int("KIMI_K3_DUAL_PROJ_SHARED_CU", _DEFAULT_CU_COUNT)
    routed = ops.wvSplitK(routed_weight, hidden, routed_cu, None)
    shared = ops.wvSplitK(shared_weight, hidden, shared_cu, None)
    return routed, shared


@functools.cache
def _compiled_flydsl_fp8_weight(
    rows_per_wave: int,
    cu_count: int,
    waves_per_eu: int,
    weight_cache_modifier: int,
    hidden_to_lds: bool,
):
    from aiter.ops.flydsl.kernels.kimi_k3_dual_projection_fp8_gfx950 import (
        build_kimi_k3_b1_dual_projection_fp8_module,
    )

    return build_kimi_k3_b1_dual_projection_fp8_module(
        rows_per_wave=rows_per_wave,
        cu_count=cu_count,
        waves_per_eu=waves_per_eu,
        weight_cache_modifier=weight_cache_modifier,
        hidden_to_lds=hidden_to_lds,
    )


def kimi_k3_moe_dual_projection_fp8_weight(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    routed_scale: torch.Tensor,
    shared_weight: torch.Tensor,
    shared_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project BF16 activations with model-load-time row-scaled FP8 weights."""

    if not supports_kimi_k3_moe_dual_projection_fp8_weight(
        hidden,
        routed_weight,
        routed_scale,
        shared_weight,
        shared_scale,
    ):
        raise ValueError("unsupported Kimi-K3 B1 weight-only FP8 inputs")
    if not is_flydsl_available() or get_gfx_runtime() != "gfx950":
        raise RuntimeError("Kimi-K3 FP8-weight dual projection requires FlyDSL on gfx950")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    routed = torch.empty(
        (_BATCH, _ROUTED_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    shared = torch.empty(
        (_BATCH, _SHARED_UP_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    launcher = _compiled_flydsl_fp8_weight(
        _env_int("KIMI_K3_DUAL_PROJ_FP8_ROWS_PER_WAVE", 2),
        _env_int("KIMI_K3_DUAL_PROJ_FP8_CU", 248),
        _env_nonnegative_int("KIMI_K3_DUAL_PROJ_FP8_WAVES_PER_EU", 0),
        _env_nonnegative_int(
            "KIMI_K3_DUAL_PROJ_FP8_WEIGHT_CACHE_MODIFIER",
            0,
        ),
        _env_bool("KIMI_K3_DUAL_PROJ_FP8_HIDDEN_TO_LDS", True),
    )
    launcher(
        ptr_arg(hidden),
        ptr_arg(routed_weight),
        ptr_arg(routed_scale),
        ptr_arg(shared_weight),
        ptr_arg(shared_scale),
        ptr_arg(routed),
        ptr_arg(shared),
        stream=torch.cuda.current_stream(hidden.device),
    )
    return routed, shared


@functools.cache
def _compiled_flydsl_shared_down_fp8_weight(
    rows_per_wave: int,
    cu_count: int,
    waves_per_eu: int,
    weight_cache_modifier: int,
    situ_beta: float,
    situ_linear_beta: float,
):
    from aiter.ops.flydsl.kernels.kimi_k3_shared_down_fp8_gfx950 import (
        build_kimi_k3_b1_shared_down_fp8_module,
    )

    return build_kimi_k3_b1_shared_down_fp8_module(
        rows_per_wave=rows_per_wave,
        cu_count=cu_count,
        waves_per_eu=waves_per_eu,
        weight_cache_modifier=weight_cache_modifier,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )


def supports_kimi_k3_shared_down_fp8_weight(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> bool:
    """Return whether the fused B1 SiTU/shared-down contract is supported."""

    return (
        gate_up.is_cuda
        and weight.is_cuda
        and weight_scale.is_cuda
        and gate_up.dtype == torch.bfloat16
        and weight.dtype == torch.float8_e4m3fn
        and weight_scale.dtype == torch.float32
        and gate_up.shape == (_BATCH, _SHARED_UP_SIZE)
        and weight.shape == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
        and weight_scale.shape == (_HIDDEN_SIZE,)
        and gate_up.is_contiguous()
        and weight.is_contiguous()
        and weight_scale.is_contiguous()
        and len({gate_up.device, weight.device, weight_scale.device}) == 1
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def kimi_k3_shared_down_fp8_weight(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> torch.Tensor:
    """Apply SiTU and project with model-load-time row-scaled FP8 weights."""

    if (
        not math.isfinite(situ_beta)
        or not math.isfinite(situ_linear_beta)
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        raise ValueError("SiTU beta values must be finite and positive")
    if not supports_kimi_k3_shared_down_fp8_weight(
        gate_up,
        weight,
        weight_scale,
    ):
        raise ValueError("unsupported Kimi-K3 B1 shared-down FP8 inputs")
    if not is_flydsl_available() or get_gfx_runtime() != "gfx950":
        raise RuntimeError("Kimi-K3 FP8 shared-down projection requires FlyDSL on gfx950")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    output = torch.empty(
        (_BATCH, _HIDDEN_SIZE),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    launcher = _compiled_flydsl_shared_down_fp8_weight(
        _env_int("KIMI_K3_SHARED_DOWN_FP8_ROWS_PER_WAVE", 1),
        _env_int("KIMI_K3_SHARED_DOWN_FP8_CU", 248),
        _env_nonnegative_int("KIMI_K3_SHARED_DOWN_FP8_WAVES_PER_EU", 0),
        _env_nonnegative_int(
            "KIMI_K3_SHARED_DOWN_FP8_WEIGHT_CACHE_MODIFIER",
            0,
        ),
        float(situ_beta),
        float(situ_linear_beta),
    )
    launcher(
        ptr_arg(gate_up),
        ptr_arg(weight),
        ptr_arg(weight_scale),
        ptr_arg(output),
        stream=torch.cuda.current_stream(gate_up.device),
    )
    return output


def supports_kimi_k3_shared_down_bf16(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether the fused B1 BF16 SiTU/shared-down path is supported."""

    return (
        gate_up.is_cuda
        and weight.is_cuda
        and gate_up.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and gate_up.shape == (_BATCH, _SHARED_UP_SIZE)
        and weight.shape == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
        and gate_up.is_contiguous()
        and weight.is_contiguous()
        and gate_up.device == weight.device
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


@functools.cache
def _compiled_flydsl_shared_down_bf16(
    rows_per_wave: int,
    cu_count: int,
    waves_per_eu: int,
    weight_cache_modifier: int,
    situ_beta: float,
    situ_linear_beta: float,
    apply_situ: bool,
):
    from aiter.ops.flydsl.kernels.kimi_k3_shared_down_fp8_gfx950 import (
        build_kimi_k3_b1_shared_down_fp8_module,
    )

    return build_kimi_k3_b1_shared_down_fp8_module(
        rows_per_wave=rows_per_wave,
        cu_count=cu_count,
        waves_per_eu=waves_per_eu,
        weight_cache_modifier=weight_cache_modifier,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
        weight_is_fp8=False,
        apply_situ=apply_situ,
    )


def kimi_k3_shared_down_bf16(
    gate_up: torch.Tensor,
    weight: torch.Tensor,
    *,
    situ_beta: float = 4.0,
    situ_linear_beta: float = 25.0,
) -> torch.Tensor:
    """Fuse SiTU with the exact-shape BF16 shared-down projection."""

    if (
        not math.isfinite(situ_beta)
        or not math.isfinite(situ_linear_beta)
        or situ_beta <= 0.0
        or situ_linear_beta <= 0.0
    ):
        raise ValueError("SiTU beta values must be finite and positive")
    if not supports_kimi_k3_shared_down_bf16(gate_up, weight):
        raise ValueError("unsupported Kimi-K3 B1 BF16 shared-down inputs")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    output = torch.empty(
        (_BATCH, _HIDDEN_SIZE),
        dtype=gate_up.dtype,
        device=gate_up.device,
    )
    launcher = _compiled_flydsl_shared_down_bf16(
        _env_int("KIMI_K3_SHARED_DOWN_BF16_ROWS_PER_WAVE", 2),
        _env_int("KIMI_K3_SHARED_DOWN_BF16_CU", 256),
        _env_nonnegative_int("KIMI_K3_SHARED_DOWN_BF16_WAVES_PER_EU", 0),
        _env_nonnegative_int(
            "KIMI_K3_SHARED_DOWN_BF16_WEIGHT_CACHE_MODIFIER",
            2,
        ),
        float(situ_beta),
        float(situ_linear_beta),
        True,
    )
    launcher(
        ptr_arg(gate_up),
        ptr_arg(weight),
        ptr_arg(weight),
        ptr_arg(output),
        stream=torch.cuda.current_stream(gate_up.device),
    )
    return output


def supports_kimi_k3_shared_down_bf16_activated(
    activated: torch.Tensor,
    weight: torch.Tensor,
) -> bool:
    """Return whether the pre-activated B1 BF16 projection is supported."""

    return (
        activated.is_cuda
        and weight.is_cuda
        and activated.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and activated.shape == (_BATCH, _SHARED_INTERMEDIATE_SIZE)
        and weight.shape == (_HIDDEN_SIZE, _SHARED_INTERMEDIATE_SIZE)
        and activated.is_contiguous()
        and weight.is_contiguous()
        and activated.device == weight.device
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def kimi_k3_shared_down_bf16_activated(
    activated: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Project a production-materialized BF16 SiTU activation."""

    if not supports_kimi_k3_shared_down_bf16_activated(activated, weight):
        raise ValueError("unsupported pre-activated Kimi-K3 B1 shared-down inputs")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    output = torch.empty(
        (_BATCH, _HIDDEN_SIZE),
        dtype=activated.dtype,
        device=activated.device,
    )
    launcher = _compiled_flydsl_shared_down_bf16(
        _env_int("KIMI_K3_SHARED_DOWN_BF16_ROWS_PER_WAVE", 1),
        _env_int("KIMI_K3_SHARED_DOWN_BF16_CU", 256),
        _env_nonnegative_int("KIMI_K3_SHARED_DOWN_BF16_WAVES_PER_EU", 0),
        _env_nonnegative_int(
            "KIMI_K3_SHARED_DOWN_BF16_WEIGHT_CACHE_MODIFIER",
            2,
        ),
        4.0,
        25.0,
        False,
    )
    launcher(
        ptr_arg(activated),
        ptr_arg(weight),
        ptr_arg(weight),
        ptr_arg(output),
        stream=torch.cuda.current_stream(activated.device),
    )
    return output


@functools.cache
def _compiled_flydsl_fused(
    rows_per_wave: int,
    cu_count: int,
    waves_per_eu: int,
    weight_cache_modifier: int,
    hidden_to_lds: bool,
):
    from aiter.ops.flydsl.kernels.kimi_k3_dual_projection_gfx950 import (
        build_kimi_k3_b1_dual_projection_module,
    )

    return build_kimi_k3_b1_dual_projection_module(
        rows_per_wave=rows_per_wave,
        cu_count=cu_count,
        waves_per_eu=waves_per_eu,
        weight_cache_modifier=weight_cache_modifier,
        hidden_to_lds=hidden_to_lds,
    )


def _flydsl_fused(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not is_flydsl_available() or get_gfx_runtime() != "gfx950":
        raise RuntimeError("Kimi-K3 fused dual projection requires FlyDSL on gfx950")

    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

    routed = torch.empty(
        (_BATCH, _ROUTED_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    shared = torch.empty(
        (_BATCH, _SHARED_UP_SIZE),
        dtype=hidden.dtype,
        device=hidden.device,
    )
    launcher = _compiled_flydsl_fused(
        _env_int("KIMI_K3_DUAL_PROJ_FUSED_ROWS_PER_WAVE", 2),
        _env_int("KIMI_K3_DUAL_PROJ_FUSED_CU", _DEFAULT_CU_COUNT),
        _env_nonnegative_int("KIMI_K3_DUAL_PROJ_FUSED_WAVES_PER_EU", 0),
        _env_nonnegative_int(
            "KIMI_K3_DUAL_PROJ_FUSED_WEIGHT_CACHE_MODIFIER",
            2,
        ),
        _env_bool("KIMI_K3_DUAL_PROJ_FUSED_HIDDEN_TO_LDS", True),
    )
    launcher(
        ptr_arg(hidden),
        ptr_arg(routed_weight),
        ptr_arg(shared_weight),
        ptr_arg(routed),
        ptr_arg(shared),
        stream=torch.cuda.current_stream(hidden.device),
    )
    return routed, shared


def _flydsl_small_m(
    hidden: torch.Tensor,
    routed_weight: torch.Tensor,
    shared_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        _flydsl_small_m_projection(hidden, routed_weight, "ROUTED"),
        _flydsl_small_m_projection(hidden, shared_weight, "SHARED"),
    )


def _flydsl_small_m_projection(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    knob_prefix: str,
) -> torch.Tensor:
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

    env_prefix = f"KIMI_K3_DUAL_PROJ_{knob_prefix}_"
    out = torch.empty((hidden.shape[0], weight.shape[0]), device=hidden.device, dtype=hidden.dtype)
    return flydsl_hgemm(
        hidden,
        weight,
        out,
        kernel_family="small_m",
        tile_m=16,
        tile_n=_env_int(f"{env_prefix}TILE_N", 128),
        tile_k=_env_int(f"{env_prefix}TILE_K", 64),
        split_k=_env_int(f"{env_prefix}SPLIT_K", 1),
        block_m_warps=1,
        block_n_warps=_env_int(f"{env_prefix}BLOCK_N_WARPS", 2),
        n_tile_repeat=_env_int(f"{env_prefix}N_TILE_REPEAT", 1),
        persistent_n_tiles=_env_int(f"{env_prefix}PERSISTENT_N_TILES", 1),
        waves_per_eu=_env_int(f"{env_prefix}WAVES_PER_EU", 0),
        b_to_lds_unroll=_env_int(f"{env_prefix}B_TO_LDS_UNROLL", 0),
        b_to_lds=_env_bool(f"{env_prefix}B_TO_LDS", False),
    )
