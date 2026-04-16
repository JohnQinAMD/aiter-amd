// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Transposed-LUT builder for SLA sparse backward (Stage 7 Tier 2).
//
// Input:  m_lut [B*H, M_blocks, topk] int32 absolute K-block indices (SLA form)
// Output: kq_lut  [B*H, K_blocks, max_kn_count] int32 (padded with -1)
//         kn_count [B*H, K_blocks] int32 (# of active Q-blocks per K-block)
//
// One thread per (bh, m) row of the input LUT. Each thread iterates its
// topk entries, using atomicAdd on kn_count to claim a slot in kq_lut.
// The caller must zero kn_count before launch.

#include <hip/hip_runtime.h>
#include <cstdint>

namespace aiter {

// `q_scale` expands each SLA Q-block to q_scale CK Q-tiles. When the CK
// bwd kernel's kM0 is smaller than SLA's BLKQ (e.g. kM0=64, BLKQ=128 →
// q_scale=2), each SLA Q-block covers multiple contiguous CK Q-tiles, and
// each one separately selects the same set of K-blocks. We emit q_scale
// entries per (bh, m, i) triple.
__global__ void sla_lut_transpose_kernel(const int32_t* __restrict__ m_lut,
                                          int32_t* __restrict__ kq_lut,
                                          int32_t* __restrict__ kn_count,
                                          int32_t M_blocks,
                                          int32_t K_blocks,
                                          int32_t topk,
                                          int32_t max_kn_count,
                                          int32_t q_scale)
{
    int bh = blockIdx.x;
    int m  = blockIdx.y * blockDim.x + threadIdx.x;
    if(m >= M_blocks)
        return;

    const int32_t* src = m_lut + (int64_t)bh * M_blocks * topk + (int64_t)m * topk;
    int32_t* count_row = kn_count + (int64_t)bh * K_blocks;
    int32_t* lut_mat   = kq_lut + (int64_t)bh * K_blocks * max_kn_count;

    for(int i = 0; i < topk; ++i)
    {
        int32_t k = src[i];
        if(k < 0 || k >= K_blocks)
            continue;
        for(int q = 0; q < q_scale; ++q)
        {
            int32_t ck_m = m * q_scale + q;
            int32_t pos  = atomicAdd(&count_row[k], 1);
            if(pos < max_kn_count)
            {
                lut_mat[(int64_t)k * max_kn_count + pos] = ck_m;
            }
        }
    }
}

// Tier 2.5 Step 1: the old insertion-sort phase 2 kernel was O(n²) per row
// and scaled 16× with Config C's larger topk*q_scale rows. It's been
// replaced with `torch::sort(dim=-1)` in the wrapper, which uses GPU radix
// sort. This requires the scatter phase to initialize padding slots with
// INT_MAX (not -1), so ascending sort leaves valid entries at the front.

void launch_sla_lut_transpose(const int32_t* m_lut,
                              int32_t* kq_lut,
                              int32_t* kn_count,
                              int32_t B_H,
                              int32_t M_blocks,
                              int32_t K_blocks,
                              int32_t topk,
                              int32_t max_kn_count,
                              int32_t q_scale,
                              hipStream_t stream)
{
    if(B_H <= 0 || M_blocks <= 0 || K_blocks <= 0)
        return;

    // Zero the count buffer before the bin scatter (caller already owns it).
    hipMemsetAsync(kn_count, 0, sizeof(int32_t) * (size_t)B_H * K_blocks, stream);

    // Phase 1: scatter (bh, m, i) -> kq_lut + atomic count.
    // (Phase 2 sort has been moved to torch::sort in the wrapper.)
    constexpr int kThreads = 128;
    const int blocks_y     = (M_blocks + kThreads - 1) / kThreads;
    hipLaunchKernelGGL(sla_lut_transpose_kernel,
                       dim3(B_H, blocks_y),
                       dim3(kThreads),
                       0,
                       stream,
                       m_lut,
                       kq_lut,
                       kn_count,
                       M_blocks,
                       K_blocks,
                       topk,
                       max_kn_count,
                       q_scale);
}

} // namespace aiter
