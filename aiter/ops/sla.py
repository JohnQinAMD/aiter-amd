# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
SLA (Sparse-Linear Attention) op bindings.

Exposes the CK-tile VSA sparse attention forward and backward kernels
as `sla_fwd` and `sla_bwd`. Both support BLKQ in {64, 128}, BLKK == 64,
D == 128, bf16. The bwd is a split dkdv + dq pipeline.
"""

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from ..jit.core import compile_ops


def _cmd_gen_func_sla_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lut: Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    valid_block_num: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
):
    """Module name + JIT codegen recipe.

    The CK VSA codegen at 50_sparse_attn/generate.py emits one .cpp per kernel
    instance; the filter keeps compile time manageable by building only the
    bf16 / no-mask path that SLA actually uses. fp16 would add ~2x compile
    time for no production benefit right now.
    """
    # Dtype determines the specific kernel instance; we only ship bf16 for the
    # SLA production path. fp16 can be added later by widening the filter.
    if q.dtype == torch.bfloat16:
        md_name = "module_sla_fwd"
    elif q.dtype == torch.float16:
        md_name = "module_sla_fwd_fp16"
    else:
        raise NotImplementedError(
            f"sla_fwd: unsupported dtype {q.dtype}; expected bf16 or fp16"
        )
    return {"md_name": md_name}


def _gen_sla_fwd_fake_tensors(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lut: Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    valid_block_num: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
) -> List[Tensor]:
    """Fake-tensor kernel for torch.compile / FX tracing.

    Returns [out, lse] to mirror the C++ signature.
    """
    out_fake = out if out is not None else torch.empty_like(q)
    B, H, S, _ = q.shape
    lse_fake = torch.empty((B, H, S), dtype=torch.float32, device=q.device)
    return [out_fake, lse_fake]


def _cmd_gen_func_sla_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    lut: Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
):
    """Module name for JIT codegen."""
    return {"md_name": "module_sla_bwd"}


def _gen_sla_bwd_fake_tensors(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    lut: Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
) -> List[Tensor]:
    dq_fake = dq if dq is not None else torch.empty_like(q)
    dk_fake = dk if dk is not None else torch.empty_like(k)
    dv_fake = dv if dv is not None else torch.empty_like(v)
    return [dq_fake, dk_fake, dv_fake]


@compile_ops(
    "module_sla_bwd",
    fc_name="sla_bwd",
    gen_func=_cmd_gen_func_sla_bwd,
    gen_fake=_gen_sla_bwd_fake_tensors,
)
def sla_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    lut: Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
) -> List[Tensor]:
    """SLA sparse-attention backward via the CK-tile VSA split pipeline.

    Args mirror the forward (dout added first, out + softmax_lse passed
    through from the fwd output). Returns [dq, dk, dv] bf16 tensors
    matching the fwd input shapes.
    """
    ...


@compile_ops(
    "module_sla_fwd",
    fc_name="sla_fwd",
    gen_func=_cmd_gen_func_sla_fwd,
    gen_fake=_gen_sla_fwd_fake_tensors,
)
def sla_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    lut: Tensor,
    block_m: int,
    block_n: int,
    softmax_scale: float,
    valid_block_num: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
) -> List[Tensor]:
    """SLA sparse-attention forward via CK-tile VSA kernel.

    Args:
        q, k, v: contiguous [B, H, S, D] bf16 (preferred) or fp16.
        lut: int32 [B, H, M_BLOCKS, topk] ABSOLUTE K-block indices, ascending,
            where M_BLOCKS = ceil(S / block_m).
        valid_block_num: optional int32 [B, H, M_BLOCKS]. If omitted, defaults
            to an all-topk tensor (every Q-block attends to exactly `topk`
            K-blocks, which is SLA's production config).
        block_m: Q-tile size. Must be 128 (only tile shape in the current CK
            build).
        block_n: K-tile size. Must be 64.
        softmax_scale: usually D ** -0.5.
        out: optional preallocated output tensor [B, H, S, D] of q.dtype.

    Returns:
        [out, lse] where
          out : [B, H, S, D] of q.dtype
          lse : [B, H, S] fp32 log-sum-exp, enables downstream backward paths.
    """
    ...
