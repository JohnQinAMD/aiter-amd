// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused QuickReduce AllReduce + RMSNorm kernel.
// Adds fused_allreduce_rmsnorm to the QuickReduce DeviceComms.

#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>
#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>
#include <torch/all.h>

#ifdef USE_ROCM

#include "quick_all_reduce_rmsnorm.cuh"

namespace aiter {

// Kernel launch wrapper for fused AllReduce+RMSNorm
template <typename AllReduceKernel, typename T>
__global__ __quickreduce_launch_bounds_two_shot__ static void
allreduce_rmsnorm_twoshot(T const* A,
                          T* B,                   // RMSNorm output
                          T const* residual_in,
                          T* residual_out,
                          T const* rms_weight,
                          float rms_epsilon,
                          uint32_t N_total,       // total elements
                          uint32_t N_hidden,      // hidden dim per row
                          uint32_t M,             // number of rows
                          uint32_t num_blocks,
                          int rank,
                          uint8_t** dbuffer_list,
                          uint32_t data_offset,
                          uint32_t flag_color,
                          int64_t data_size_per_phase)
{
    int block = blockIdx.x;
    int grid  = gridDim.x;

    while(block < num_blocks)
    {
        AllReduceKernel::run(
            A, B, residual_in, residual_out, rms_weight, rms_epsilon,
            N_total, N_hidden, M,
            block, rank, dbuffer_list, data_offset, flag_color, data_size_per_phase);
        block += grid;
        flag_color++;
    }
}

#define TWOSHOT_RMSNORM_DISPATCH(__codec, __cbh)                                             \
    if(world_size == 2)                                                                      \
    {                                                                                        \
        using LineCodec = __codec<T, 2>;                                                     \
        using Kernel = AllReduceTwoshotFusedRMSNorm<T, LineCodec, __cbh>;                    \
        hipLaunchKernelGGL((allreduce_rmsnorm_twoshot<Kernel, T>),                          \
                           dim3(grid), dim3(kBlockTwoShot), 0, stream,                       \
                           A, B, res_in, res_out, weight, eps,                               \
                           N, N_hidden, M, num_blocks, rank, dbuffer_list,                   \
                           data_offset, flag_color, data_size_per_phase);                    \
    }                                                                                        \
    else if(world_size == 4)                                                                  \
    {                                                                                        \
        using LineCodec = __codec<T, 4>;                                                     \
        using Kernel = AllReduceTwoshotFusedRMSNorm<T, LineCodec, __cbh>;                    \
        hipLaunchKernelGGL((allreduce_rmsnorm_twoshot<Kernel, T>),                          \
                           dim3(grid), dim3(kBlockTwoShot), 0, stream,                       \
                           A, B, res_in, res_out, weight, eps,                               \
                           N, N_hidden, M, num_blocks, rank, dbuffer_list,                   \
                           data_offset, flag_color, data_size_per_phase);                    \
    }                                                                                        \
    else if(world_size == 8)                                                                  \
    {                                                                                        \
        using LineCodec = __codec<T, 8>;                                                     \
        using Kernel = AllReduceTwoshotFusedRMSNorm<T, LineCodec, __cbh>;                    \
        hipLaunchKernelGGL((allreduce_rmsnorm_twoshot<Kernel, T>),                          \
                           dim3(grid), dim3(kBlockTwoShot), 0, stream,                       \
                           A, B, res_in, res_out, weight, eps,                               \
                           N, N_hidden, M, num_blocks, rank, dbuffer_list,                   \
                           data_offset, flag_color, data_size_per_phase);                    \
    }


// PyTorch binding
void qr_fused_allreduce_rmsnorm(fptr_t _fa,
                                 torch::Tensor& inp,
                                 torch::Tensor& out,
                                 torch::Tensor& residual_in,
                                 torch::Tensor& residual_out,
                                 torch::Tensor& rms_weight,
                                 double rms_epsilon,
                                 int64_t quant_level,
                                 bool cast_bf2half)
{
    auto fa = reinterpret_cast<DeviceComms*>(_fa);
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard(device_of(inp));
    auto stream = at::hip::getCurrentHIPStream();

    TORCH_CHECK_EQ(inp.scalar_type(), out.scalar_type());
    TORCH_CHECK_EQ(inp.numel(), out.numel());
    TORCH_CHECK_EQ(inp.dim(), 2, "Input must be 2D [M, hidden]");

    uint32_t M = inp.size(0);
    uint32_t N_hidden = inp.size(1);
    uint32_t N = inp.numel();

    // Small-M check: entire tensor must fit in one tile for block-local RMSNorm
    TORCH_CHECK(N * inp.element_size() <= static_cast<uint32_t>(kTileSize),
                "Fused AR+RMSNorm only supports small M where M*hidden*2 <= 32KB. "
                "Got M=", M, " hidden=", N_hidden, " total_bytes=", N * inp.element_size());

    int world_size = fa->get_world_size();
    int rank = fa->get_rank();
    uint32_t msg_size   = N * sizeof(half); // use half size for msg calc
    uint32_t num_blocks = divceil(msg_size, kTileSize);
    uint32_t grid       = min(kMaxNumBlocks, num_blocks);
    uint32_t data_offset = fa->data_offset;
    uint8_t** dbuffer_list = fa->dbuffer_list;
    uint32_t flag_color = fa->flag_color;
    int64_t data_size_per_phase = fa->kMaxProblemSize;

    auto quant_level_ = static_cast<QuickReduceQuantLevel>(quant_level);

    if(out.scalar_type() == at::ScalarType::BFloat16)
    {
        using T = __hip_bfloat16;
        T const* A = reinterpret_cast<T const*>(inp.data_ptr());
        T* B = reinterpret_cast<T*>(out.data_ptr());
        T const* res_in = reinterpret_cast<T const*>(residual_in.data_ptr());
        T* res_out = reinterpret_cast<T*>(residual_out.data_ptr());
        T const* weight = reinterpret_cast<T const*>(rms_weight.data_ptr());
        float eps = static_cast<float>(rms_epsilon);

        if(cast_bf2half)
        {
            switch(quant_level_)
            {
            case QuickReduceQuantLevel::INT4:
                TWOSHOT_RMSNORM_DISPATCH(CodecQ4, true) break;
            case QuickReduceQuantLevel::INT6:
                TWOSHOT_RMSNORM_DISPATCH(CodecQ6, true) break;
            case QuickReduceQuantLevel::FP8:
                TWOSHOT_RMSNORM_DISPATCH(CodecFP8, true) break;
            default:
                TWOSHOT_RMSNORM_DISPATCH(CodecFP, true) break;
            }
        }
        else
        {
            switch(quant_level_)
            {
            case QuickReduceQuantLevel::INT4:
                TWOSHOT_RMSNORM_DISPATCH(CodecQ4, false) break;
            case QuickReduceQuantLevel::INT6:
                TWOSHOT_RMSNORM_DISPATCH(CodecQ6, false) break;
            case QuickReduceQuantLevel::FP8:
                TWOSHOT_RMSNORM_DISPATCH(CodecFP8, false) break;
            default:
                TWOSHOT_RMSNORM_DISPATCH(CodecFP, false) break;
            }
        }
    }
    else if(out.scalar_type() == at::ScalarType::Half)
    {
        using T = half;
        T const* A = reinterpret_cast<T const*>(inp.data_ptr());
        T* B = reinterpret_cast<T*>(out.data_ptr());
        T const* res_in = reinterpret_cast<T const*>(residual_in.data_ptr());
        T* res_out = reinterpret_cast<T*>(residual_out.data_ptr());
        T const* weight = reinterpret_cast<T const*>(rms_weight.data_ptr());
        float eps = static_cast<float>(rms_epsilon);

        switch(quant_level_)
        {
        case QuickReduceQuantLevel::INT4:
            TWOSHOT_RMSNORM_DISPATCH(CodecQ4, false) break;
        case QuickReduceQuantLevel::INT6:
            TWOSHOT_RMSNORM_DISPATCH(CodecQ6, false) break;
        case QuickReduceQuantLevel::FP8:
            TWOSHOT_RMSNORM_DISPATCH(CodecFP8, false) break;
        default:
            TWOSHOT_RMSNORM_DISPATCH(CodecFP, false) break;
        }
    }
    else
    {
        throw std::runtime_error(
            "qr_fused_allreduce_rmsnorm only supports float16 and bfloat16");
    }

    HIP_CHECK(hipGetLastError());
    fa->flag_color += divceil(N, grid);
}

// Template instantiations
#define INSTANTIATE_RMSNORM_FOR_WORLDSIZE(T, Codec, cast_bf2half)                          \
    template struct AllReduceTwoshotFusedRMSNorm<T, Codec<T, 2>, cast_bf2half>;            \
    template struct AllReduceTwoshotFusedRMSNorm<T, Codec<T, 4>, cast_bf2half>;            \
    template struct AllReduceTwoshotFusedRMSNorm<T, Codec<T, 8>, cast_bf2half>;

INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecFP, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecQ4, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecQ6, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecFP8, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecFP, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecQ4, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecQ6, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(__hip_bfloat16, CodecFP8, true)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, CodecFP, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, CodecQ4, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, CodecQ6, false)
INSTANTIATE_RMSNORM_FOR_WORLDSIZE(half, CodecFP8, false)

#endif // USE_ROCM
} // namespace aiter
