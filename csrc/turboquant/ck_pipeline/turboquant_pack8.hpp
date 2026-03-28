// TurboQuant codebook dequantization op for CK pipelines.
//
// Drop-in replacement for PassThroughPack8. Uses the Lloyd-Max optimal
// reconstruction levels for N(0,1) at 4 bits (16 levels) as a constexpr
// lookup table — identical mechanism to CK's i4_to_bhalf4, zero overhead.
//
// After TurboQuant's random orthogonal rotation, all weight groups follow
// N(0,1), so this codebook is universal (model-independent).
#pragma once

#include "ck_tile/core/config.hpp"
#include "ck_tile/core/numeric/half.hpp"
#include "ck_tile/core/numeric/bfloat16.hpp"
#include "ck_tile/core/utility/bit_cast.hpp"

// Include the appropriate platform-specific version
#if defined(__HIPCC__) || defined(__HIP_PLATFORM_AMD__)
#include "ck_tile/ops/elementwise/unary_element_wise_operation_hip.hpp"
#else
#include "ck_tile/ops/elementwise/unary_element_wise_operation.hpp"
#endif

namespace ck_tile {
namespace element_wise {

struct TurboQuantPack8
{
    static constexpr const char* name = "TurboQuantPack8";

    template <typename Y, typename X>
    CK_TILE_HOST_DEVICE void operator()(Y& y, const X& x) const;

    CK_TILE_DEVICE static constexpr bf16x4_t tq_i4_to_bhalf4(int q)
    {
        // Lloyd-Max optimal reconstruction levels for N(0,1), 4-bit.
        // Symmetric: level[i] = -level[15-i].
        constexpr auto tq_table = make_lookup_table<bf16_t, 16>([](int i) {
            constexpr float levels[16] = {
                -2.7330780029f,  // 0
                -2.0695691109f,  // 1
                -1.6186094284f,  // 2
                -1.2567617893f,  // 3
                -0.9427994490f,  // 4
                -0.6571131349f,  // 5
                -0.3882715702f,  // 6
                -0.1284713149f,  // 7
                 0.1284713149f,  // 8
                 0.3882715702f,  // 9
                 0.6571131349f,  // 10
                 0.9427994490f,  // 11
                 1.2567617893f,  // 12
                 1.6186094284f,  // 13
                 2.0695691109f,  // 14
                 2.7330780029f,  // 15
            };
            return bit_cast<bf16_t>(float_to_bf16_rtn_raw(levels[i]));
        });

        // Same nibble extraction order as CK's i4_to_bhalf4 (constexpr table path)
        return bf16x4_t{tq_table[(q >> 0) & 0xf],
                        tq_table[(q >> 16) & 0xf],
                        tq_table[(q >> 4) & 0xf],
                        tq_table[(q >> 20) & 0xf]};
    }

    CK_TILE_HOST_DEVICE constexpr void operator()(bf16x8_t& y, const pk_int4x4_t& x) const
    {
        int q = bit_cast<int>(x);
        y.lo = tq_i4_to_bhalf4(q);
        y.hi = tq_i4_to_bhalf4(q >> 8);
    }
};

} // namespace element_wise

// Standalone dequant function for use in the Int4 flatmm pipeline.
// Converts one pk_int4_t byte (2 int4 values) to bf16x2_t via constexpr codebook.
// This replaces MXFP4's pk_fp4_to_bf16x2(pk_fp4, scale) — no scale needed.
CK_TILE_DEVICE bf16x2_t tq_int4_to_bf16x2(pk_int4_t packed)
{
    constexpr auto tq = element_wise::make_lookup_table<bf16_t, 16>([](int i) {
        constexpr float L[16] = {
            -2.7330780029f, -2.0695691109f, -1.6186094284f, -1.2567617893f,
            -0.9427994490f, -0.6571131349f, -0.3882715702f, -0.1284713149f,
             0.1284713149f,  0.3882715702f,  0.6571131349f,  0.9427994490f,
             1.2567617893f,  1.6186094284f,  2.0695691109f,  2.7330780029f,
        };
        return bit_cast<bf16_t>(float_to_bf16_rtn_raw(L[i]));
    });
    uint8_t raw = bit_cast<uint8_t>(packed);
    return bf16x2_t{tq[raw & 0xf], tq[(raw >> 4) & 0xf]};
}

// Half-precision variant
CK_TILE_DEVICE fp16x2_t tq_int4_to_fp16x2(pk_int4_t packed)
{
    constexpr auto tq = element_wise::make_lookup_table<half_t, 16>([](int i) {
        constexpr float L[16] = {
            -2.7330780029f, -2.0695691109f, -1.6186094284f, -1.2567617893f,
            -0.9427994490f, -0.6571131349f, -0.3882715702f, -0.1284713149f,
             0.1284713149f,  0.3882715702f,  0.6571131349f,  0.9427994490f,
             1.2567617893f,  1.6186094284f,  2.0695691109f,  2.7330780029f,
        };
        return bit_cast<half_t>(type_convert<half_t>(L[i]));
    });
    uint8_t raw = bit_cast<uint8_t>(packed);
    return fp16x2_t{tq[raw & 0xf], tq[(raw >> 4) & 0xf]};
}

} // namespace ck_tile
