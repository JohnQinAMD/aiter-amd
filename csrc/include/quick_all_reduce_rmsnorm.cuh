#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Fused AllReduce + RMSNorm for QuickReduce.
// After the twoshot allreduce completes with data in registers (tA[kAtoms]),
// we apply RMSNorm in-place before writing to output — zero extra HBM reads.
//
// Constraints:
// - Small-M only: hidden_dim * M * sizeof(T) <= kTileSize (32KB)
//   For bf16 hidden=3072: M <= 5 (covers decode batch=4)
// - RMSNorm reduction is block-local via LDS

#include "quick_all_reduce.cuh"

namespace aiter {

// LDS for cross-warp reduction of sum-of-squares
__shared__ float s_rmsnorm_partial[4]; // 4 wavefronts

// Compute sum of squares of bf16/fp16 elements packed in int32x4_t registers
template <typename T>
__quickreduce_device_inline__ float compute_sum_sq(int32x4_t const* data, int n_atoms)
{
    float sum = 0.0f;
    for(int i = 0; i < n_atoms; i++)
    {
        T const* elems = reinterpret_cast<T const*>(&data[i]);
        for(int j = 0; j < 8; j++) // 8 T elements per int32x4_t
        {
            float v = static_cast<float>(elems[j]);
            sum += v * v;
        }
    }
    return sum;
}

// Apply RMSNorm: out[i] = in[i] * rsqrt_val * weight[i]
template <typename T>
__quickreduce_device_inline__ void apply_rmsnorm(
    int32x4_t* data,
    T const* weight,
    float rsqrt_val,
    int n_atoms,
    uint32_t weight_offset, // offset into weight array for this thread
    uint32_t N_hidden)      // hidden dim for bounds check
{
    for(int i = 0; i < n_atoms; i++)
    {
        T* elems = reinterpret_cast<T*>(&data[i]);
        uint32_t base = weight_offset + i * kAtomStride * 8; // 8 elements per atom
        for(int j = 0; j < 8; j++)
        {
            uint32_t idx = base + j;
            if(idx < N_hidden)
            {
                float v = static_cast<float>(elems[j]) * rsqrt_val * static_cast<float>(weight[idx]);
                elems[j] = static_cast<T>(v);
            }
        }
    }
}

// Warp-reduce sum using butterfly shuffle
__quickreduce_device_inline__ float warp_reduce_sum(float val)
{
    for(int offset = 32; offset > 0; offset >>= 1)
    {
        val += __shfl_xor(val, offset, 64);
    }
    return val;
}

// Block-reduce sum via LDS (4 wavefronts)
__quickreduce_device_inline__ float block_reduce_sum(float val, int warp_id, int lane_id)
{
    // First reduce within wavefront
    val = warp_reduce_sum(val);

    // Write wavefront result to LDS
    if(lane_id == 0)
    {
        s_rmsnorm_partial[warp_id] = val;
    }
    __syncthreads();

    // Final reduction across wavefronts (thread 0)
    if(warp_id == 0 && lane_id == 0)
    {
        float total = 0.0f;
        for(int i = 0; i < 4; i++) // 4 wavefronts
        {
            total += s_rmsnorm_partial[i];
        }
        s_rmsnorm_partial[0] = total;
    }
    __syncthreads();
    return s_rmsnorm_partial[0];
}


// Fused AllReduce + RMSNorm (with residual add)
// Small-M path: entire tensor fits in one tile (hidden_dim * M <= 16384 elements)
template <typename T, class Codec, bool cast_bf2half>
struct AllReduceTwoshotFusedRMSNorm
{
    static_assert(sizeof(T) == 2);
    static constexpr int kWorldSize = Codec::kWorldSize;

    __device__ static void run(T const* __restrict__ input,
                               T* __restrict__ output,         // RMSNorm output
                               T const* __restrict__ residual_in,
                               T* __restrict__ residual_out,   // residual + allreduced
                               T const* __restrict__ rms_weight,
                               float rms_epsilon,
                               uint32_t N_total,               // total elements (M * hidden)
                               uint32_t N_hidden,              // hidden dim per row
                               uint32_t M,                     // number of rows
                               int const block,
                               int const rank,
                               uint8_t** __restrict__ buffer_list,
                               uint32_t const data_offset,
                               uint32_t flag_color,
                               int64_t data_size_per_phase)
    {
        int thread           = threadIdx.x + threadIdx.y * kWavefront;
        int warp_id          = threadIdx.y;
        int lane_id          = threadIdx.x;
        uint8_t* rank_buffer = buffer_list[rank];
        Codec codec(thread, rank);
        int block_id = blockIdx.x;
        uint8_t* buffer_ptr[kWorldSize];
        for(int i = 0; i < kWorldSize; ++i)
            buffer_ptr[i] = buffer_list[i];

        // ---- Phase 0: Read input ----
        int32x4_t tA[kAtoms];
        BufferResource src_buffer(const_cast<T*>(input), N_total * sizeof(T));
        uint32_t src_offset = block * kTileSize + thread * sizeof(int32x4_t);
        for(int i = 0; i < kAtoms; i++)
        {
            tA[i] = buffer_load_dwordx4(src_buffer.descriptor, src_offset, 0, 0);
            src_offset += kAtomStride * sizeof(int32x4_t);
            if constexpr(cast_bf2half)
            {
                const nv_bfloat162* bf_buf = reinterpret_cast<const nv_bfloat162*>(&tA[i]);
                half2 half_buf[4];
                for(int j = 0; j < 4; ++j)
                {
                    float2 f    = __bfloat1622float2(bf_buf[j]);
                    half_buf[j] = __float22half2_rn(f);
                }
                tA[i] = *reinterpret_cast<const int32x4_t*>(half_buf);
            }
        }

        // ---- Phase 1A: Broadcast segments ----
        uint32_t comm_data0_offset  = data_offset + block_id * Codec::kTransmittedTileSize;
        uint32_t comm_data1_offset  = data_size_per_phase + comm_data0_offset;
        uint32_t comm_flags0_offset = block_id * (kWorldSize * sizeof(uint32_t));
        uint32_t comm_flags1_offset = (data_offset / 2) + comm_flags0_offset;

        for(int r = 0; r < kWorldSize; r++)
        {
            int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
                buffer_ptr[r] + comm_data0_offset + rank * Codec::kRankTransmittedTileSize);
            codec.send(send_buffer, &tA[r * Codec::kRankAtoms]);
        }
        __syncthreads();
        if(thread < kWorldSize)
        {
            uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(
                buffer_ptr[thread] + comm_flags0_offset + rank * sizeof(uint32_t));
            set_sync_flag(flag_ptr, flag_color);
        }

        // ---- Phase 1B: Reduce segments ----
        int32x4_t tR[Codec::kRankAtoms] = {};
        {
            int32x4_t* recv_buffer =
                reinterpret_cast<int32x4_t*>(rank_buffer + comm_data0_offset);
            uint32_t* flag_ptr =
                reinterpret_cast<uint32_t*>(rank_buffer + comm_flags0_offset);
            for(int r = 0; r < kWorldSize; r++)
            {
                if(thread == 0)
                    wait_sync_flag(&flag_ptr[r], flag_color);
                __syncthreads();
                codec.recv(&recv_buffer, tA);
                for(int i = 0; i < Codec::kRankAtoms; i++)
                    packed_assign_add<T>(&tR[i], &tA[i]);
            }
        }

        // ---- Phase 2: Broadcast reduced result ----
        for(int r = 0; r < kWorldSize; r++)
        {
            int32x4_t* send_buffer = reinterpret_cast<int32x4_t*>(
                buffer_ptr[r] + comm_data1_offset + rank * Codec::kRankTransmittedTileSize);
            codec.send(send_buffer, tR);
        }
        __syncthreads();
        if(thread < kWorldSize)
        {
            uint32_t* flag_ptr = reinterpret_cast<uint32_t*>(
                buffer_ptr[thread] + comm_flags1_offset + rank * sizeof(uint32_t));
            set_sync_flag(flag_ptr, flag_color);
        }

        // ---- Phase 2B: Gather final allreduced result into tA ----
        {
            int32x4_t* recv_buffer =
                reinterpret_cast<int32x4_t*>(rank_buffer + comm_data1_offset);
            uint32_t* flag_ptr =
                reinterpret_cast<uint32_t*>(rank_buffer + comm_flags1_offset);
            for(int r = 0; r < kWorldSize; r++)
            {
                if(thread == 0)
                    wait_sync_flag(&flag_ptr[r], flag_color);
                __syncthreads();
                codec.recv(&recv_buffer, &tA[r * Codec::kRankAtoms]);
            }
        }

        // Convert back to bf16 if needed (before RMSNorm)
        if constexpr(cast_bf2half)
        {
            for(int i = 0; i < kAtoms; i++)
            {
                const half2* half_buf = reinterpret_cast<const half2*>(&tA[i]);
                nv_bfloat162 bf16_buf[4];
                for(int j = 0; j < 4; ++j)
                {
                    float2 f    = __half22float2(half_buf[j]);
                    bf16_buf[j] = __float22bfloat162_rn(f);
                }
                tA[i] = *reinterpret_cast<const int32x4_t*>(bf16_buf);
            }
        }

        // ============================================================
        // Phase 3: FUSED RMSNorm on tA[] (allreduced data in registers)
        // ============================================================
        // tA[kAtoms] now holds the allreduced result in original dtype (T).
        // Each thread owns 64 elements at positions:
        //   thread*8, thread*8+1, ..., thread*8+7  (atom 0)
        //   (thread+kAtomStride)*8, ...             (atom 1)
        //   etc.
        // For a flat layout: element index = block*16384 + (atom*kAtomStride + thread)*8 + j
        // Row index = element_index / N_hidden
        // Col index = element_index % N_hidden

        // Step 3a: Compute residual_out = allreduced + residual_in
        //          and store allreduced result for RMSNorm input
        BufferResource res_in_buf(const_cast<T*>(residual_in), N_total * sizeof(T));
        BufferResource res_out_buf(residual_out, N_total * sizeof(T));
        uint32_t elem_offset = block * kTileSize + thread * sizeof(int32x4_t);

        for(int i = 0; i < kAtoms; i++)
        {
            // Load residual_in
            int32x4_t res_data = buffer_load_dwordx4(res_in_buf.descriptor, elem_offset, 0, 0);
            // residual_out = allreduced + residual_in
            int32x4_t res_sum = tA[i];
            packed_assign_add<T>(&res_sum, &res_data);
            // Store residual_out
            buffer_store_dwordx4(res_sum, res_out_buf.descriptor, elem_offset, 0, 0);
            // tA[i] stays as allreduced+residual for RMSNorm input
            tA[i] = res_sum;
            elem_offset += kAtomStride * sizeof(int32x4_t);
        }

        // Step 3b: Compute per-row sum of squares
        // For small M, process each row. Row r spans elements [r*N_hidden, (r+1)*N_hidden)
        // Thread's elements: base_elem = block*16384 + thread*8 for atom 0
        for(uint32_t row = 0; row < M; row++)
        {
            uint32_t row_start = row * N_hidden;
            uint32_t row_end   = row_start + N_hidden;

            // Compute this thread's contribution to sum_sq for this row
            float local_sum_sq = 0.0f;
            for(int i = 0; i < kAtoms; i++)
            {
                uint32_t base_elem = (block * kBlockSize * kAtoms + (i * kAtomStride + thread)) * 8;
                T const* elems     = reinterpret_cast<T const*>(&tA[i]);
                for(int j = 0; j < 8; j++)
                {
                    uint32_t idx = base_elem + j;
                    if(idx >= row_start && idx < row_end)
                    {
                        float v = static_cast<float>(elems[j]);
                        local_sum_sq += v * v;
                    }
                }
            }

            // Block-reduce sum_sq
            float total_sum_sq = block_reduce_sum(local_sum_sq, warp_id, lane_id);
            float rsqrt_val    = rsqrtf(total_sum_sq / static_cast<float>(N_hidden) + rms_epsilon);

            // Step 3c: Apply RMSNorm: output = (allreduced+res) * rsqrt * weight
            BufferResource dst_buf(output, N_total * sizeof(T));
            for(int i = 0; i < kAtoms; i++)
            {
                uint32_t base_elem = (block * kBlockSize * kAtoms + (i * kAtomStride + thread)) * 8;
                T* elems           = reinterpret_cast<T*>(&tA[i]);
                bool any_in_row    = false;
                int32x4_t out_data = {};
                T* out_elems       = reinterpret_cast<T*>(&out_data);

                for(int j = 0; j < 8; j++)
                {
                    uint32_t idx = base_elem + j;
                    if(idx >= row_start && idx < row_end)
                    {
                        uint32_t col = idx - row_start;
                        float v      = static_cast<float>(elems[j]) * rsqrt_val *
                                  static_cast<float>(rms_weight[col]);
                        out_elems[j] = static_cast<T>(v);
                        any_in_row   = true;
                    }
                }

                if(any_in_row)
                {
                    uint32_t write_offset =
                        (block * kBlockSize * kAtoms + (i * kAtomStride + thread)) *
                        sizeof(int32x4_t);
                    buffer_store_dwordx4(out_data, dst_buf.descriptor, write_offset, 0, 0);
                }
            }
        }
    }
};

// Kernel launch macro for fused variant
#define TWOSHOT_RMSNORM_DISPATCH(CodecType)                                                        \
    {                                                                                              \
        using AR = AllReduceTwoshotFusedRMSNorm<T, CodecType<T, world_size>, cast_bf2half>;        \
        hipLaunchKernelGGL(                                                                        \
            (allreduce_twoshot_kernel<AR>),                                                         \
            dim3(grid),                                                                            \
            kBlockTwoShot,                                                                         \
            0,                                                                                     \
            stream,                                                                                \
            A,                                                                                     \
            B,                                                                                     \
            N,                                                                                     \
            rank,                                                                                  \
            dbuffer_list,                                                                          \
            data_offset,                                                                           \
            flag_color,                                                                            \
            data_size_per_phase);                                                                   \
    }

} // namespace aiter
