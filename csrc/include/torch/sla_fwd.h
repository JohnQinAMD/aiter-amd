#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include <torch/extension.h>

namespace aiter {
namespace torch_itfs {

// Forward pass for SLA (Sparse-Linear Attention) sparse block attention.
// Uses CK-tile VSA (Variable Sparsity Attention) kernel.
//
// Layout contract (i_perm = true in CK terminology):
//   q, k, v : [B, H, S, D] contiguous, bf16 or fp16, D = 128
//   lut     : [B, H, M_BLOCKS, topk] int32, ABSOLUTE K-block indices (ascending),
//             where M_BLOCKS = ceil(S / block_m)
//   valid_block_num : optional int32 [B, H, M_BLOCKS] — default = topk broadcast
//   out     : optional [B, H, S, D] of q.dtype; allocated if not provided
//
// Returns a 2-tensor vector: [out, lse]
//   out : [B, H, S, D] of q.dtype
//   lse : [B, H, S]    fp32 (log-sum-exp per Q row, used by backward)
//
// The wrapper converts absolute LUT -> delta LUT on device before calling
// fmha_vsa_fwd. Only the (kM0=128, kN0=64) VSA tile is supported at the moment,
// so block_m must equal 128 and block_n must equal 64.
std::vector<at::Tensor> sla_fwd(at::Tensor q,             // [B, H, S, D]
                                at::Tensor k,             // [B, H, S, D]
                                at::Tensor v,             // [B, H, S, D]
                                at::Tensor lut,           // int32 [B, H, M_BLOCKS, topk]
                                int64_t block_m,
                                int64_t block_n,
                                double softmax_scale,
                                std::optional<at::Tensor> valid_block_num, // int32 [B, H, M_BLOCKS]
                                std::optional<at::Tensor> out);

} // namespace torch_itfs
} // namespace aiter
