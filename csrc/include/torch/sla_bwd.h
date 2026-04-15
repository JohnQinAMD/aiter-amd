#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// Backward pass for SLA (Sparse-Linear Attention) sparse block attention.
// Uses the CK-tile VSA split bwd pipeline (dkdv-only + dq-only) and a
// torch preprocess for δ = rowsum(dO ⊙ O). The wrapper:
//   - allocates dQ, dK, dV outputs
//   - computes δ = (dO * O).sum(-1) via torch
//   - transposes SLA's M-major LUT into the K-major form the dkdv
//     kernel expects; the dq kernel consumes the M-major LUT directly
//   - calls fmha_vsa_bwd; on negative return (kernel failure) the
//     wrapper zero-fills dq/dk/dv and the caller is expected to
//     fall back to Triton
//
// Layout contract:
//   q, k, v, o, dout : [B, H, S, D] contiguous bf16/fp16, D = 128
//   lse              : [B, H, S] contiguous fp32 (log2-space from fwd)
//   lut              : [B, H, M_BLOCKS, topk] int32 absolute K-block indices
//   block_m/block_n  : 128 / 64 only
std::vector<at::Tensor> sla_bwd(at::Tensor dout,          // [B, H, S, D]
                                at::Tensor q,             // [B, H, S, D]
                                at::Tensor k,             // [B, H, S, D]
                                at::Tensor v,             // [B, H, S, D]
                                at::Tensor out,           // [B, H, S, D]
                                at::Tensor softmax_lse,   // [B, H, S] fp32
                                at::Tensor lut,           // int32 [B, H, M_BLOCKS, topk]
                                int64_t block_m,
                                int64_t block_n,
                                double softmax_scale,
                                std::optional<at::Tensor> dq,
                                std::optional<at::Tensor> dk,
                                std::optional<at::Tensor> dv);

} // namespace torch_itfs
} // namespace aiter
