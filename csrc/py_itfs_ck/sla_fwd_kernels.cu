// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// AITER wrapper around CK-tile VSA (Variable Sparsity Attention) forward
// kernel. Exposes a Torch-friendly entry point that accepts SLA's absolute
// LUT layout ([B, H, M_BLOCKS, topk]), converts it to CK's delta-encoded form
// on device, and invokes the codegen-built `fmha_vsa_fwd` dispatcher.
//
// Tile requirement: CK is currently built with the (kM0=128, kN0=64) VSA tile
// only (see 50_sparse_attn/codegen/ops/fmha_fwd_vsa.py patches). block_m must
// be 128 and block_n must be 64.

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "py_itfs_common.h"
#include "mha_common.h"

// CK sparse-attention API types and entry points. fmha_fwd_trek.hpp includes
// "01_fmha/mask.hpp" so the 01_fmha dir must be on the include path too
// (handled in aiter/jit/optCompilerConfig.json:module_sla_fwd.extra_include).
#include "fmha_fwd_trek.hpp"

#include "torch/sla_fwd.h"

namespace aiter {

// Defined in csrc/kernels/sla_lut_abs_to_delta.cu.
void launch_sla_lut_abs_to_delta(const int32_t* sorted_abs_lut_ptr,
                                 int32_t* delta_lut_ptr,
                                 int32_t topk,
                                 int32_t k_blocks,
                                 int32_t total_rows,
                                 hipStream_t stream);

namespace torch_itfs {

std::vector<at::Tensor> sla_fwd(at::Tensor q,             // [B, H, S, D]
                                at::Tensor k,             // [B, H, S, D]
                                at::Tensor v,             // [B, H, S, D]
                                at::Tensor lut,           // int32 [B, H, M_BLOCKS, topk] absolute
                                int64_t block_m,
                                int64_t block_n,
                                double softmax_scale,
                                std::optional<at::Tensor> valid_block_num, // int32 [B, H, M_BLOCKS]
                                std::optional<at::Tensor> out_)
{
    // ---- Dtype checks ----
    auto q_dtype = q.scalar_type();
    TORCH_CHECK(q_dtype == at::ScalarType::Half || q_dtype == at::ScalarType::BFloat16,
                "sla_fwd: only fp16 and bf16 are supported");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype,
                "sla_fwd: q/k/v must share dtype");
    const std::string dtype_str = (q_dtype == at::ScalarType::BFloat16) ? "bf16" : "fp16";

    // ---- Layout / shape checks ----
    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(lut);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v); CHECK_CONTIGUOUS(lut);

    TORCH_CHECK(q.dim() == 4, "sla_fwd: q must be rank 4 [B, H, S, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "sla_fwd: k shape must match q");
    TORCH_CHECK(v.sizes() == q.sizes(), "sla_fwd: v shape must match q");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);

    TORCH_CHECK(D == 128, "sla_fwd: head dim must be 128 (CK VSA only builds d=128)");
    // The codegen builds both (kM0=64, kN0=64) and (kM0=128, kN0=64)
    // tile instances; args.block_m selects at runtime via the seqtune
    // dispatcher.
    TORCH_CHECK((block_m == 64 || block_m == 128) && block_n == 64,
                "sla_fwd: block_m must be 64 or 128, block_n must be 64");

    const int64_t M_BLOCKS = (S + block_m - 1) / block_m;
    const int64_t K_BLOCKS = (S + block_n - 1) / block_n;

    TORCH_CHECK(lut.dtype() == at::ScalarType::Int, "sla_fwd: lut must be int32");
    TORCH_CHECK(lut.dim() == 4, "sla_fwd: lut must be rank 4 [B, H, M_BLOCKS, topk]");
    TORCH_CHECK(lut.size(0) == B && lut.size(1) == H && lut.size(2) == M_BLOCKS,
                "sla_fwd: lut first three dims must match [B, H, M_BLOCKS]");
    const int64_t topk = lut.size(3);
    TORCH_CHECK(topk > 0, "sla_fwd: topk must be positive");
    TORCH_CHECK(topk <= K_BLOCKS, "sla_fwd: topk must be <= K_BLOCKS");

    // valid_block_num: default to a broadcast of topk if not provided. Allocate
    // an int32 tensor filled with topk on the same device as q.
    at::Tensor vblk;
    if(valid_block_num.has_value())
    {
        vblk = valid_block_num.value();
        CHECK_DEVICE(vblk); CHECK_CONTIGUOUS(vblk);
        TORCH_CHECK(vblk.dtype() == at::ScalarType::Int, "sla_fwd: valid_block_num must be int32");
        TORCH_CHECK(vblk.sizes() == torch::IntArrayRef({B, H, M_BLOCKS}),
                    "sla_fwd: valid_block_num must have shape [B, H, M_BLOCKS]");
    }
    else
    {
        vblk = torch::full({B, H, M_BLOCKS},
                           static_cast<int32_t>(topk),
                           q.options().dtype(at::ScalarType::Int));
    }

    // Output tensor.
    at::Tensor out;
    if(out_.has_value())
    {
        out = out_.value();
        CHECK_DEVICE(out); CHECK_CONTIGUOUS(out);
        TORCH_CHECK(out.dtype() == q_dtype, "sla_fwd: out must match q dtype");
        TORCH_CHECK(out.sizes() == q.sizes(), "sla_fwd: out must have shape [B, H, S, D]");
    }
    else
    {
        out = torch::empty_like(q);
    }

    // LSE output: shape [B, H, S] fp32, contiguous. The CK codegen
    // instantiates kStoreLSE=true, so the kernel writes this tensor
    // unconditionally. Log2-space convention (m + log2(l)).
    at::Tensor lse = torch::empty({B, H, S},
                                   q.options().dtype(at::ScalarType::Float));

    // ---- LUT absolute -> delta conversion on device ----
    // CK's VSA kernel (fmha_fwd_vsa_kernel.hpp:287-292) expects a full-width
    // [B, H, M_BLOCKS, K_BLOCKS] delta LUT whose first `valid_block_num[row]`
    // entries are delta-encoded K-block offsets. SLA's get_block_map emits an
    // ABSOLUTE [B, H, M_BLOCKS, topk] LUT with rows in arbitrary order.
    //
    // Steps:
    //   1. Sort each row of the input LUT ascending.
    //   2. Allocate a K_BLOCKS-wide output buffer (topk < K_BLOCKS in general).
    //   3. Scan each row and write delta_lut[row, 0..topk) = [abs[0], abs[1]-abs[0], ...].
    //      Indices [topk..K_BLOCKS) remain unread because valid_block_num[row] = topk.
    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    at::Tensor sorted_lut = std::get<0>(torch::sort(lut, /*dim=*/-1)).contiguous();
    at::Tensor delta_lut  = torch::empty({B, H, M_BLOCKS, K_BLOCKS},
                                         q.options().dtype(at::ScalarType::Int));

    aiter::launch_sla_lut_abs_to_delta(
        reinterpret_cast<const int32_t*>(sorted_lut.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(delta_lut.data_ptr<int32_t>()),
        static_cast<int32_t>(topk),
        static_cast<int32_t>(K_BLOCKS),
        static_cast<int32_t>(B * H * M_BLOCKS),
        stream);

    // ---- Populate CK VSA args ----
    // SLA layout is BHSD contiguous (i_perm = true). Strides:
    //   q.stride(0) = H*S*D  (batch)
    //   q.stride(1) = S*D    (nhead)
    //   q.stride(2) = D      (seq)
    //   q.stride(3) = 1
    fmha_vsa_fwd_args args{};
    args.q_ptr               = q.data_ptr();
    args.k_ptr               = k.data_ptr();
    args.v_ptr               = v.data_ptr();
    args.lut_ptr             = delta_lut.data_ptr();
    args.valid_block_num_ptr = vblk.data_ptr();
    args.o_ptr               = out.data_ptr();
    args.lse_ptr             = lse.data_ptr();

    args.seqlen_q     = static_cast<ck_tile::index_t>(S);
    args.seqlen_k     = static_cast<ck_tile::index_t>(S);
    args.batch        = static_cast<ck_tile::index_t>(B);
    args.max_seqlen_q = static_cast<ck_tile::index_t>(S);
    args.hdim_q       = static_cast<ck_tile::index_t>(D);
    args.hdim_v       = static_cast<ck_tile::index_t>(D);
    args.nhead_q      = static_cast<ck_tile::index_t>(H);
    args.nhead_k      = static_cast<ck_tile::index_t>(H);

    args.scale_s = static_cast<float>(softmax_scale);

    args.stride_q = static_cast<ck_tile::index_t>(q.stride(2));
    args.stride_k = static_cast<ck_tile::index_t>(k.stride(2));
    args.stride_v = static_cast<ck_tile::index_t>(v.stride(2));
    args.stride_o = static_cast<ck_tile::index_t>(out.stride(2));

    args.nhead_stride_q = static_cast<ck_tile::index_t>(q.stride(1));
    args.nhead_stride_k = static_cast<ck_tile::index_t>(k.stride(1));
    args.nhead_stride_v = static_cast<ck_tile::index_t>(v.stride(1));
    args.nhead_stride_o = static_cast<ck_tile::index_t>(out.stride(1));

    args.batch_stride_q = static_cast<ck_tile::index_t>(q.stride(0));
    args.batch_stride_k = static_cast<ck_tile::index_t>(k.stride(0));
    args.batch_stride_v = static_cast<ck_tile::index_t>(v.stride(0));
    args.batch_stride_o = static_cast<ck_tile::index_t>(out.stride(0));

    // LSE is contiguous [B, H, S] so strides are trivial.
    args.nhead_stride_lse = static_cast<ck_tile::index_t>(lse.stride(1));
    args.batch_stride_lse = static_cast<ck_tile::index_t>(lse.stride(0));

    // Non-causal, no window, no sink -- matches SLA's production config.
    args.window_size_left  = -1;
    args.window_size_right = -1;
    args.mask_type         = 0; // mask_enum::no_mask

    // Tile-size hint for the codegen dispatcher: block_m=64 selects
    // the (kM0=64) tile, block_m=128 selects the (kM0=128) tile.
    args.block_m = static_cast<ck_tile::index_t>(block_m);

    fmha_vsa_fwd_traits traits{};
    traits.hdim_q        = static_cast<int>(D);
    traits.hdim_v        = static_cast<int>(D);
    traits.data_type     = dtype_str;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;

    ck_tile::stream_config stream_config{stream};
    float t = fmha_vsa_fwd(traits, args, stream_config);
    TORCH_CHECK(t >= 0.0f, "sla_fwd: fmha_vsa_fwd returned invalid kernel time");

    return {out, lse};
}

} // namespace torch_itfs
} // namespace aiter
