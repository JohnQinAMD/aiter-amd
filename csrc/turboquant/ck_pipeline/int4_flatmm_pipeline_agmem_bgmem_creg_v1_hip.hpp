// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// TurboQuant INT4 flatmm pipeline: bf16 A x pk_int4_t B -> f32 C
// Follows F16xMXF4FlatmmPipelineAGmemBGmemCRegV1 architecture:
//   - A: DRAM -> VGPR -> SMEM (ping-pong) -> LDS -> VGPR -> MFMA
//   - B: DRAM -> VGPR (flat preshuffle load) -> dequant (constexpr codebook) -> BWarpTensor -> MFMA
//   - NO SMEM for B. Dequant in registers before each MFMA call.
#pragma once

#include "ck_tile/core_hip.hpp"
#include "ck_tile/host/concat_hip.hpp"
#include "ck_tile/ops/gemm/pipeline/gemm_pipeline_problem_hip.hpp"
#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_hip.hpp"
#include "ck_tile/ops/flatmm/pipeline/int4_flatmm_pipeline_problem_hip.hpp"
#include "ck_tile/ops/flatmm/pipeline/int4_flatmm_pipeline_agmem_bgmem_creg_v1_policy_hip.hpp"
#include "ck_tile/ops/elementwise/turboquant_pack8.hpp"

namespace ck_tile {

template <typename Problem, typename PipelinePolicy = F16xInt4FlatmmPipelineAgBgCrPolicy>
struct F16xInt4FlatmmPipelineAGmemBGmemCRegV1
{
    using ADataType      = remove_cvref_t<typename Problem::ADataType>;
    using BDataType      = remove_cvref_t<typename Problem::QuantType>;
    using CDataType      = remove_cvref_t<typename Problem::CDataType>;
    using BlockGemmShape = remove_cvref_t<typename Problem::BlockGemmShape>;
    using ComputeType    = ADataType;

    using ALayout = remove_cvref_t<typename Problem::ALayout>;
    using BLayout = remove_cvref_t<typename Problem::BLayout>;
    using CLayout = remove_cvref_t<typename Problem::CLayout>;

    using BlockFlatmm = remove_cvref_t<decltype(PipelinePolicy::template GetBlockFlatmm<Problem>())>;
    static constexpr auto config = BlockFlatmm::BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();
    using WG = remove_cvref_t<decltype(config.template at<0>())>;

    static constexpr index_t BlockSize   = Problem::kBlockSize;
    static constexpr index_t WaveSize    = get_warp_size();
    static constexpr index_t kMPerBlock  = BlockGemmShape::kM;
    static constexpr index_t kNPerBlock  = BlockGemmShape::kN;
    static constexpr index_t kKPerBlock  = BlockGemmShape::kK;
    static constexpr index_t flatKPerWarp = Problem::flatKPerWarp;
    static constexpr index_t flatNPerWarp = Problem::flatNPerWarp;

    static constexpr index_t MWarp = config.template at<1>();
    static constexpr index_t NWarp = config.template at<2>();
    static constexpr index_t MIterPerWarp = kMPerBlock / (MWarp * WG::kM);
    static constexpr index_t NIterPerWarp = kNPerBlock / (NWarp * WG::kN);
    static constexpr index_t KIterPerWarp = kKPerBlock / WG::kK;
    static constexpr index_t KFlatPerBlockPerIter = flatKPerWarp;
    static constexpr index_t NFlatPerBlockPerIter = flatNPerWarp;
    static constexpr index_t MPerBlockPerIter = kMPerBlock / MIterPerWarp;
    static constexpr index_t KPerBlockPerIter = kKPerBlock / KIterPerWarp;

    static constexpr int INT4PackedSize  = 2;
    static constexpr int XDL_PerWeightK  = 4;
    static constexpr int INT4KPerWarp    = KIterPerWarp / XDL_PerWeightK;
    static constexpr int ContinuousKPerThread = Problem::ContinuousKPerThread;

    static constexpr bool HasHotLoop = Problem::HasHotLoop;
    static constexpr auto TailNum    = Problem::TailNum;
    static constexpr index_t m_preload = 2;
    static constexpr bool UsePersistentKernel = Problem::Traits::UsePersistentKernel;
    static constexpr index_t NumWaveGroups = Problem::NumWaveGroups;
    static constexpr bool TransposeC = Problem::TransposeC;

    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    static constexpr index_t GetVectorSizeA() { return Problem::VectorSizeA; }
    static constexpr index_t GetVectorSizeB() { return 32; }
    static constexpr index_t GetVectorSizeC() { return Problem::VectorSizeC; }
    static constexpr bool kPadM = Problem::kPadM;
    static constexpr bool kPadN = Problem::kPadN;
    static constexpr bool kPadK = Problem::kPadK;
    static constexpr auto BMemNTType = Problem::BMemNTType;
    static constexpr bool BPreShufflePermute = Problem::BPreShufflePermute;

    using Underlying = FlatmmPipelineAGmemBGmemCRegV1<Problem, PipelinePolicy>;
    CK_TILE_HOST_DEVICE static constexpr index_t GetSmemSize() { return Underlying::GetSmemSize(); }

    static constexpr auto MIter_2nd_last =
        (MIterPerWarp >= 2) ? MIterPerWarp - 2 : MIterPerWarp - 1;

    // ---- Pipeline operator (without norms) ----
    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return run_impl(a_dram_block_window_tmp, b_flat_dram_block_window_tmp,
                        static_cast<const float*>(nullptr), 0, 0, 1.0f,
                        num_loop, p_smem_ping, p_smem_pong);
    }

    // ---- Pipeline operator (with per-group norms) ----
    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                   const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                   const float* norms_ptr,
                                   index_t norms_stride_n,
                                   index_t block_n_base,
                                   float inv_sqrt_gs,
                                   index_t num_loop,
                                   void* p_smem_ping,
                                   void* p_smem_pong) const
    {
        return run_impl(a_dram_block_window_tmp, b_flat_dram_block_window_tmp,
                        norms_ptr, norms_stride_n, block_n_base, inv_sqrt_gs,
                        num_loop, p_smem_ping, p_smem_pong);
    }

    template <typename ADramBlockWindowTmp, typename BFlatBlockWindowTmp>
    CK_TILE_DEVICE auto run_impl(const ADramBlockWindowTmp& a_dram_block_window_tmp,
                                  const BFlatBlockWindowTmp& b_flat_dram_block_window_tmp,
                                  const float* norms_ptr,
                                  index_t norms_stride_n,
                                  index_t block_n_base,
                                  float inv_sqrt_gs,
                                  index_t num_loop,
                                  void* p_smem_ping,
                                  void* p_smem_pong) const
    {
        const index_t iMWarp = get_warp_id() / NWarp;

        using CWarpDstr  = typename WG::CWarpDstr;
        using CWarpTensor = typename WG::CWarpTensor;
        constexpr auto c_warp_y_lengths =
            to_sequence(CWarpDstr{}.get_ys_to_d_descriptor().get_lengths());
        constexpr auto c_warp_y_index_zeros = uniform_sequence_gen_t<CWarpDstr::NDimY, 0>{};

        // === A: SMEM setup ===
        ADataType* p_a_lds_ping = static_cast<ADataType*>(p_smem_ping);
        ADataType* p_a_lds_pong = static_cast<ADataType*>(p_smem_pong);

        constexpr auto a_lds_block_desc = PipelinePolicy::template MakeALdsBlockDescriptor<Problem>();
        auto a_lds_block_ping = make_tensor_view<address_space_enum::lds>(p_a_lds_ping, a_lds_block_desc);
        auto a_lds_block_pong = make_tensor_view<address_space_enum::lds>(p_a_lds_pong, a_lds_block_desc);

        auto a_copy_lds_window_ping = make_tile_window(a_lds_block_ping,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0},
            PipelinePolicy::template MakeADramTileDistribution<Problem>());
        auto a_copy_lds_window_pong = make_tile_window(a_lds_block_pong,
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}), {0, 0},
            PipelinePolicy::template MakeADramTileDistribution<Problem>());

        auto a_warp_window_ping_tmp = make_tile_window(a_lds_block_ping,
            make_tuple(number<WG::kM>{}, number<WG::kK>{}), {iMWarp * WG::kM, 0},
            PipelinePolicy::template MakeALDS_WarpTileDistribution<Problem>());
        auto a_warp_window_pong_tmp = make_tile_window(a_lds_block_pong,
            make_tuple(number<WG::kM>{}, number<WG::kK>{}), {iMWarp * WG::kM, 0},
            PipelinePolicy::template MakeALDS_WarpTileDistribution<Problem>());

        statically_indexed_array<statically_indexed_array<decltype(a_warp_window_ping_tmp), KIterPerWarp>, MIterPerWarp>
            a_warp_windows_ping, a_warp_windows_pong;

        static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                a_warp_windows_ping(mIter)(kIter) = a_warp_window_ping_tmp;
                a_warp_windows_pong(mIter)(kIter) = a_warp_window_pong_tmp;
                move_tile_window(a_warp_windows_ping(mIter)(kIter), {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
                move_tile_window(a_warp_windows_pong(mIter)(kIter), {mIter * MPerBlockPerIter, kIter * KPerBlockPerIter});
            });
        });

        auto a_copy_dram_window = make_tile_window(
            a_dram_block_window_tmp.get_bottom_tensor_view(),
            make_tuple(number<kMPerBlock>{}, number<kKPerBlock>{}),
            a_dram_block_window_tmp.get_window_origin(),
            PipelinePolicy::template MakeADramTileDistribution<Problem>());

        // === B: coalesced load from preshuffled flat buffer ===
        const auto* b_base = reinterpret_cast<const uint8_t*>(
            &b_flat_dram_block_window_tmp.get_bottom_tensor_view().get_buffer_view()(0));

        using V4UInt_Buffer = thread_buffer<uint32_t, XDL_PerWeightK>;
        statically_indexed_array<statically_indexed_array<V4UInt_Buffer, INT4KPerWarp>, NIterPerWarp>
            b_warp_tensor_ping, b_warp_tensor_pong;

        auto block_flatmm = BlockFlatmm();
        auto c_block_tile = block_flatmm.MakeCBlockTile();
        statically_indexed_array<typename WG::BWarpTensor, NIterPerWarp> dequant_B_n;

        // LDS codebook: 16 bf16 values = 32 bytes. Loaded once, used for all dequant.
        // LDS lookup is ~1 cycle vs constexpr's 15 v_cndmask instructions per nibble.
        __shared__ ComputeType lds_codebook[16];
        if(threadIdx.x == 0)
        {
            // TurboQuant Lloyd-Max centroids for N(0,1), 4-bit
            constexpr float tq_levels[16] = {
                -2.7330780029f, -2.0695691109f, -1.6186094284f, -1.2567617893f,
                -0.9427994490f, -0.6571131349f, -0.3882715702f, -0.1284713149f,
                 0.1284713149f,  0.3882715702f,  0.6571131349f,  0.9427994490f,
                 1.2567617893f,  1.6186094284f,  2.0695691109f,  2.7330780029f,
            };
            for(int i = 0; i < 16; i++)
                lds_codebook[i] = type_convert<ComputeType>(tq_levels[i]);
        }
        block_sync_lds();

        using ABlockTile = decltype(load_tile(a_copy_dram_window));
        ABlockTile a_block_tile;
        statically_indexed_array<decltype(load_tile(a_warp_windows_ping(number<0>{})(number<0>{}))), m_preload>
            a_warp_tensor;

        // === Per-thread N-position for norm lookup ===
        const int iNWarp = get_warp_id() % NWarp;
        const int lane_n = get_lane_id() % 16;

        // Precompute norm scale for each nIter (constant per K-tile, changes per nIter)
        // norm_scale[nIter] = norms[n_row, k_group] * inv_sqrt_gs
        // n_row = block_n_base + nIter * NWarp * 16 + iNWarp * 16 + lane_n
        auto get_norm_scale = [&](auto nIter, index_t k_group) -> float {
            if(norms_ptr == nullptr) return 1.0f;
            int n_row = block_n_base + nIter * NWarp * 16 + iNWarp * 16 + lane_n;
            return norms_ptr[n_row * norms_stride_n + k_group] * inv_sqrt_gs;
        };

        // === Dequant: codebook lookup only (norms applied to partial sum in Option A) ===
        auto dequant_int4 = [&](const auto& quant_weight_tensor, auto xdl_nIter, auto xdl_kIter) {
            auto quant_idx_k = xdl_kIter % number<XDL_PerWeightK>{};
            constexpr int ScalarCnt = WG::BWarpTensor::get_thread_buffer_size();
            constexpr int PackedCnt = ScalarCnt / INT4PackedSize;
            using ComputeV2Type = std::conditional_t<std::is_same_v<ComputeType, half_t>, fp16x2_t, bf16x2_t>;
            auto raw_bytes = bit_cast<thread_buffer<pk_int4_t, 4>>(quant_weight_tensor[quant_idx_k]);
            static_for<0, PackedCnt, 1>{}([&](auto i) {
                uint8_t raw = bit_cast<uint8_t>(raw_bytes.at(i));
                ComputeV2Type val;
                val.x = lds_codebook[raw & 0xF];
                val.y = lds_codebook[(raw >> 4) & 0xF];
                dequant_B_n[xdl_nIter].get_thread_buffer().template set_as<ComputeV2Type>(i, val);
            });
        };

        // === A prefetch helpers (MXFP4 pattern) ===
        auto prefill_lds_a_stage1 = [&](auto, auto dram_tile_a, auto) {
            a_block_tile = load_tile(dram_tile_a);
        };
        auto prefill_lds_a_stage2 = [&](auto lds_tile_a) {
            store_tile(lds_tile_a, a_block_tile);
        };

        // === B prefetch: COALESCED 16-byte load from XDL4-preshuffled buffer ===
        // Preshuffle layout: (n0, kIter_pack, k_group, n_lane, 16_bytes)
        // where 16_bytes = kIter_in_pack(4) × byte_in_group(4)
        // Thread i reads 16 contiguous bytes: k_group*NLane*16 + n_lane*16
        // 64 threads × 16 bytes = 1024 bytes per (nIter, kIter_pack) — COALESCED!
        const int lane_id_v = get_lane_id();
        const int n1_v = lane_id_v % 16;
        const int k_group_v = lane_id_v / 16;
        const int warp_id_v = get_warp_id();
        const auto b_origin_n_view = b_flat_dram_block_window_tmp.get_window_origin()[number<0>{}];

        constexpr int NLane_c = 16;
        constexpr int KLane_c = 4;
        const int total_K_half = kKPerBlock * num_loop / 2;
        const int total_kIter = total_K_half / (WG::kK / 2);
        const int total_kIter_packs = total_kIter / XDL_PerWeightK;
        constexpr int bytes_per_warp_load = KLane_c * NLane_c * 16;  // 1024
        const int bytes_per_n0 = total_kIter_packs * bytes_per_warp_load;
        const int thread_offset = k_group_v * NLane_c * 16 + n1_v * 16;

        index_t kIter_pack_global = 0;

        auto prefetch_b = [&](auto& b_dst) {
            static_for<0, INT4KPerWarp, 1>{}([&](auto kIter_load) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter_) {
                    int n0 = b_origin_n_view + nIter_ * NWarp + warp_id_v;
                    int pack = kIter_pack_global + kIter_load;

                    long long addr = (long long)n0 * bytes_per_n0
                                   + pack * bytes_per_warp_load
                                   + thread_offset;

                    V4UInt_Buffer tmp;
                    __builtin_memcpy(&tmp, b_base + addr, sizeof(V4UInt_Buffer));
                    b_dst(nIter_)(kIter_load) = tmp;
                });
            });
            kIter_pack_global += INT4KPerWarp;
        };

        index_t k_group_idx = 0;

        // Option A: local C accumulator per K-tile, scale partial sum by norm
        auto c_local_tile = block_flatmm.MakeCBlockTile();

        // === GEMM helper: MFMA into local C, scale by norm, add to global C ===
        auto do_gemm = [&](auto& b_tensor, auto& a_warp_wins) {
            // Zero local accumulator
            tile_elementwise_inout([](auto& c) { c = 0; }, c_local_tile);

            // MFMA into local C (codebook-only, no norm)
            static_for<0, KIterPerWarp, 1>{}([&](auto kIter) {
                static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                    constexpr auto AwarpIter = (kIter * MIterPerWarp + mIter) % m_preload;
                    static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                        CWarpTensor c_warp_tensor;
                        c_warp_tensor.get_thread_buffer() = c_local_tile.get_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                        if constexpr(mIter == 0)
                            dequant_int4(b_tensor(nIter)(kIter / number<XDL_PerWeightK>{}), nIter, kIter);
                        WG{}(c_warp_tensor, a_warp_tensor(number<AwarpIter>{}), dequant_B_n[nIter]);
                        c_local_tile.set_y_sliced_thread_data(
                            merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                            merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                            c_warp_tensor.get_thread_buffer());
                    });
                    if constexpr((kIter * MIterPerWarp + mIter) < (KIterPerWarp * MIterPerWarp - m_preload))
                    {
                        constexpr auto AmIter = (mIter + m_preload) % MIterPerWarp;
                        constexpr auto AkIter = (kIter + (mIter + m_preload) / MIterPerWarp);
                        a_warp_tensor(number<AwarpIter>{}) = load_tile(a_warp_wins(number<AmIter>{})(number<AkIter>{}));
                    }
                    __builtin_amdgcn_sched_group_barrier(0x008, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x100, 1, 0);
                    __builtin_amdgcn_sched_group_barrier(0x020, 1, 0);

                    if constexpr((kIter == KIterPerWarp - 1) && (mIter == MIter_2nd_last))
                    {
                        block_sync_lds();
                    }
                });
            });

            // Scale local C by per-(n, group) norm and accumulate into global C
            // Each thread element corresponds to a specific (m, n) output position.
            // The norm depends on n and k_group_idx.
            // For the MFMA 16×16 layout: lane_id % 16 = N-position within warp tile.
            // NIterPerWarp iterations cover different N-blocks.
            static_for<0, MIterPerWarp, 1>{}([&](auto mIter) {
                static_for<0, NIterPerWarp, 1>{}([&](auto nIter) {
                    float ns = get_norm_scale(nIter, k_group_idx);
                    auto local_data = c_local_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    auto global_data = c_block_tile.get_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths));
                    // Scale and accumulate: global += local * norm_scale
                    constexpr int n_elems = decltype(local_data)::size();
                    for(int e = 0; e < n_elems; e++)
                    {
                        global_data(e) += local_data[e] * ns;
                    }
                    c_block_tile.set_y_sliced_thread_data(
                        merge_sequences(sequence<mIter, nIter>{}, c_warp_y_index_zeros),
                        merge_sequences(sequence<1, 1>{}, c_warp_y_lengths),
                        global_data);
                });
            });

            k_group_idx++;
        };

        // === HEAD ===
        prefill_lds_a_stage1(a_copy_lds_window_ping, a_copy_dram_window, number<3>{});
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        prefetch_b(b_warp_tensor_ping);

        prefill_lds_a_stage2(a_copy_lds_window_ping);
        __builtin_amdgcn_sched_barrier(0);

        prefill_lds_a_stage1(a_copy_lds_window_pong, a_copy_dram_window, number<3>{});
        move_tile_window(a_copy_dram_window, {0, kKPerBlock});

        tile_elementwise_inout([](auto& c) { c = 0; }, c_block_tile);
        block_sync_lds();

        static_for<0, m_preload, 1>{}([&](auto li) {
            constexpr auto mI = li % MIterPerWarp;
            constexpr auto kI = li / MIterPerWarp;
            a_warp_tensor(li) = load_tile(a_warp_windows_ping(number<mI>{})(number<kI>{}));
        });
        __builtin_amdgcn_sched_barrier(0);

        // === MAIN LOOP (MXFP4 pattern: 2 K-blocks per while iteration) ===
        index_t iCounter = (num_loop - 1) / 2;
        while(iCounter > 0)
        {
            // --- Half 1: prefetch B→pong, GEMM on ping ---
            prefetch_b(b_warp_tensor_pong);
            prefill_lds_a_stage2(a_copy_lds_window_pong);
            prefill_lds_a_stage1(a_copy_lds_window_ping, a_copy_dram_window, number<1>{});

            do_gemm(b_warp_tensor_ping, a_warp_windows_ping);

            prefill_lds_a_stage1(a_copy_lds_window_ping, a_copy_dram_window, number<2>{});
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto li) {
                constexpr auto mI = li % MIterPerWarp;
                constexpr auto kI = li / MIterPerWarp;
                a_warp_tensor(li) = load_tile(a_warp_windows_pong(number<mI>{})(number<kI>{}));
            });

            // --- Half 2: prefetch B→ping, GEMM on pong ---
            prefetch_b(b_warp_tensor_ping);
            prefill_lds_a_stage2(a_copy_lds_window_ping);
            prefill_lds_a_stage1(a_copy_lds_window_pong, a_copy_dram_window, number<1>{});

            do_gemm(b_warp_tensor_pong, a_warp_windows_pong);

            prefill_lds_a_stage1(a_copy_lds_window_pong, a_copy_dram_window, number<2>{});
            move_tile_window(a_copy_dram_window, {0, kKPerBlock});

            static_for<0, m_preload, 1>{}([&](auto li) {
                constexpr auto mI = li % MIterPerWarp;
                constexpr auto kI = li / MIterPerWarp;
                a_warp_tensor(li) = load_tile(a_warp_windows_ping(number<mI>{})(number<kI>{}));
            });

            iCounter--;
        }

        // === TAIL ===
        // For num_loop=1 (no main loop iterations): A0 is already in ping LDS,
        // B0 is in b_warp_tensor_ping. Just do GEMM.
        // For num_loop>1: the main loop left data in alternating buffers.
        if(num_loop == 1)
        {
            // A0 in ping (from HEAD), B0 in ping (from HEAD). Just GEMM.
            do_gemm(b_warp_tensor_ping, a_warp_windows_ping);
        }
        else if((num_loop - 1) % 2 == 1)
        {
            // Odd remainder after main loop: GEMM on ping then pong
            prefetch_b(b_warp_tensor_pong);
            prefill_lds_a_stage2(a_copy_lds_window_pong);

            do_gemm(b_warp_tensor_ping, a_warp_windows_ping);

            static_for<0, m_preload, 1>{}([&](auto li) {
                constexpr auto mI = li % MIterPerWarp;
                constexpr auto kI = li / MIterPerWarp;
                a_warp_tensor(li) = load_tile(a_warp_windows_pong(number<mI>{})(number<kI>{}));
            });

            prefill_lds_a_stage2(a_copy_lds_window_ping);
            block_sync_lds();
            do_gemm(b_warp_tensor_pong, a_warp_windows_pong);
        }
        else
        {
            // Even remainder: GEMM last B (ping) with last A (already in ping LDS from loop)
            // The loop's last half2 stored the current A to ping and loaded next A (OOB) to a_block_tile.
            // DON'T store a_block_tile (OOB) — use A from ping LDS directly.
            static_for<0, m_preload, 1>{}([&](auto li) {
                constexpr auto mI = li % MIterPerWarp;
                constexpr auto kI = li / MIterPerWarp;
                a_warp_tensor(li) = load_tile(a_warp_windows_ping(number<mI>{})(number<kI>{}));
            });

            do_gemm(b_warp_tensor_ping, a_warp_windows_ping);
        }

        return c_block_tile;
    }
};

} // namespace ck_tile
