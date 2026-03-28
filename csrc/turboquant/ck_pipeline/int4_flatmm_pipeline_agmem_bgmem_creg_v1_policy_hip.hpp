// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// TurboQuant INT4 flatmm pipeline policy.
// Defines MakeInt4BFlatDramTileDistribution — the B tile distribution for
// pk_int4_t data in the preshuffle "flat" layout.
// Modeled after F16xMXF4FlatmmPipelineAgBgCrPolicy::MakeFp4BFlatDramTileDistribution.
#pragma once

#include "ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy_hip.hpp"

namespace ck_tile {

struct F16xInt4FlatmmPipelineAgBgCrPolicy : UniversalFlatmmPipelineAgBgCrPolicy
{
    static constexpr auto I0 = number<0>{};
    static constexpr auto I1 = number<1>{};
    static constexpr auto I2 = number<2>{};

    // Distribution N × K must equal flatNPerWarp × flatKPerWarp.
    // N_Pack=1 so NWavePerBlk × N_Pack = N_Warp × 1 = flatNPerWarp.
    // KBPerLoad=32 so 1 × 64 × 32 = 2048 = flatKPerWarp.
    static constexpr index_t KBPerLoad = 32;
    static constexpr index_t N_Pack    = 1;

    // B tile distribution for pk_int4_t in flat preshuffle layout.
    // Identical to MakeFp4BFlatDramTileDistribution — same data density (4 bits per element).
    // The preshuffle format rearranges pk_int4_t bytes for coalesced 128-bit buffer loads
    // matching the MFMA register layout.
    template <typename Problem>
    CK_TILE_HOST_DEVICE static constexpr auto MakeInt4BFlatDramTileDistribution()
    {
        using TileShape = typename Problem::BlockGemmShape;

        static_assert(TileShape::WarpTile::at(I1) == 16, "only for XDL_N == 16");

        constexpr index_t BlockSize = Problem::kBlockSize;
        constexpr index_t WaveSize  = get_warp_size();
        constexpr index_t WaveNum   = BlockSize / WaveSize;

        constexpr index_t NWavePerBlk = TileShape::BlockWarps::at(number<1>{}); // N_Warp = 4

        constexpr index_t WaveRepeat = WaveNum / TileShape::flatNPerWarp;

        return make_static_tile_distribution(
            tile_distribution_encoding<
                sequence<WaveRepeat>,
                tuple<sequence<NWavePerBlk, N_Pack>,                  // N dimension
                      sequence<1, WaveSize, KBPerLoad>>,              // K dimension
                tuple<sequence<0, 1, 2>, sequence<2>>,
                tuple<sequence<0, 0, 0>, sequence<1>>,
                sequence<2>,
                sequence<2>>{});
    }
};

} // namespace ck_tile
