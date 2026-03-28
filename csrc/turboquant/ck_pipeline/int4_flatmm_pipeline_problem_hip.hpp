// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// TurboQuant INT4 flatmm pipeline problem definition.
// Modeled after F16xMXF4FlatmmPipelineProblem but without scale tensor fields.
// pk_int4_t weights are dequantized via constexpr Lloyd-Max codebook (zero overhead).
#pragma once

#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem_hip.hpp"

namespace ck_tile {

template <typename ADataType_,
          typename BDataType_,  // pk_int4_t (storage type)
          typename CDataType_,
          typename BlockGemmShape_,
          typename Traits_,
          GemmPipelineScheduler Scheduler_      = GemmPipelineScheduler::Intrawave,
          bool HasHotLoop_                      = true,
          TailNumber TailNum_                   = TailNumber::Full,
          amd_buffer_coherence_enum BMemNTType_ = amd_buffer_coherence_enum::coherence_default,
          bool BPreShufflePermute_              = false,
          typename ComputeDataType_             = ADataType_>
struct F16xInt4FlatmmPipelineProblem : FlatmmPipelineProblem<ADataType_,
                                                             ADataType_,  // MFMA sees bf16, not pk_int4
                                                             CDataType_,
                                                             BlockGemmShape_,
                                                             Traits_,
                                                             Scheduler_,
                                                             HasHotLoop_,
                                                             TailNum_,
                                                             BMemNTType_,
                                                             BPreShufflePermute_,
                                                             ComputeDataType_>
{
    using BlockGemmShape = BlockGemmShape_;

    // The packed storage type for weight data (pk_int4_t: 1 byte = 2 int4 values)
    using QuantType = BDataType_;

    static constexpr index_t flatNPerWarp = BlockGemmShape::flatNPerWarp;

    // MUST match mixed_prec exactly: 32 bytes per thread, flatK = 64 * 32 = 2048
    static constexpr int ContinuousKPerThread = 32;
    static constexpr index_t flatKPerWarp     = 64 * ContinuousKPerThread; // 2048
};

} // namespace ck_tile
