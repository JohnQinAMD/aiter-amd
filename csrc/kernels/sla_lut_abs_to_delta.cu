// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// LUT conversion for SLA -> CK VSA.
//
// Layout contract expected by CK's fmha_fwd_vsa_kernel.hpp:
//   delta_lut[B, H, Q_blocks, K_BLOCKS]  (NOT topk-wide!)
// CK reads `valid_block_num_ptr[row]` entries per Q-block, accumulating each
// value onto a running K-block index. Only the first `topk` entries per row
// are populated with the delta-encoded absolute indices; the remainder is
// untouched tail (CK never reads past valid_block_num).
//
// SLA's get_block_map emits a topk-wide absolute-index LUT [B, H, Q_blk, topk]
// where each row is in arbitrary order. The wrapper sorts each row ascending,
// then calls this kernel to produce the K_BLOCKS-wide delta form.
//
// Grid: B*H*Q_blocks rows; one wavefront-free thread per row scans topk
// elements serially. Cost is negligible (< 1% of the forward).

#include <hip/hip_runtime.h>
#include <cstdint>

namespace aiter {

__global__ void sla_lut_abs_to_delta_kernel(const int32_t* __restrict__ sorted_abs_lut,
                                            int32_t* __restrict__ delta_lut,
                                            int32_t topk,
                                            int32_t k_blocks,
                                            int32_t total_rows)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row >= total_rows)
        return;
    const int32_t* src = sorted_abs_lut + static_cast<int64_t>(row) * topk;
    int32_t* dst       = delta_lut + static_cast<int64_t>(row) * k_blocks;
    int32_t prev       = 0;
    for(int i = 0; i < topk; ++i)
    {
        int32_t cur = src[i];
        dst[i]      = cur - prev;
        prev        = cur;
    }
    // Leave dst[topk..k_blocks] untouched; CK never reads past valid_block_num.
}

void launch_sla_lut_abs_to_delta(const int32_t* sorted_abs_lut_ptr,
                                 int32_t* delta_lut_ptr,
                                 int32_t topk,
                                 int32_t k_blocks,
                                 int32_t total_rows,
                                 hipStream_t stream)
{
    if(total_rows <= 0 || topk <= 0 || k_blocks <= 0)
        return;
    constexpr int kThreads = 256;
    const int kBlocks      = (total_rows + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(sla_lut_abs_to_delta_kernel,
                       dim3(kBlocks),
                       dim3(kThreads),
                       0,
                       stream,
                       sorted_abs_lut_ptr,
                       delta_lut_ptr,
                       topk,
                       k_blocks,
                       total_rows);
}

} // namespace aiter
