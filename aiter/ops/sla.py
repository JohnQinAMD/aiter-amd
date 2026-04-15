# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""
SLA (Sparse-Linear Attention) op bindings — inference-only release.

Exposes the CK-tile VSA sparse attention forward kernel as `sla_fwd`
for BLKQ in {64, 128}, BLKK == 64, D == 128, bf16. No backward is
included in this branch; the training-capable build lives on the
sla-ck-fwd-release branch.
"""

from typing import List, Optional

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
    """Module name + JIT codegen recipe."""
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
    """Fake-tensor kernel for torch.compile / FX tracing. Returns [out, lse]."""
    out_fake = out if out is not None else torch.empty_like(q)
    B, H, S, _ = q.shape
    lse_fake = torch.empty((B, H, S), dtype=torch.float32, device=q.device)
    return [out_fake, lse_fake]


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
        lut: int32 [B, H, M_BLOCKS, topk] absolute K-block indices, ascending,
            where M_BLOCKS = ceil(S / block_m).
        valid_block_num: optional int32 [B, H, M_BLOCKS]. If omitted, defaults
            to an all-topk tensor (every Q-block attends to exactly `topk`
            K-blocks).
        block_m: Q-tile size. 64 or 128.
        block_n: K-tile size. Must be 64.
        softmax_scale: usually D ** -0.5.
        out: optional preallocated output tensor [B, H, S, D] of q.dtype.

    Returns:
        [out, lse] where
          out : [B, H, S, D] of q.dtype
          lse : [B, H, S] fp32 log-sum-exp (log2-space).
    """
    ...
