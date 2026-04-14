// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// AITER wrapper for SLA sparse backward. Stage 7 Tier 2 scaffolding —
// all plumbing is in place and the transposed-LUT builder works, but the
// underlying CK VSA bwd kernel instantiation is stubbed (returns -1.0f).
// On stub return the wrapper zero-fills dQ/dK/dV so the caller can observe
// the fallback path; real values will come once Milestone 2.4 lands a live
// CK kernel.

#include <limits>

#include <torch/all.h>
#include <ATen/hip/HIPContext.h>
#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>

#include "py_itfs_common.h"
#include "mha_common.h"

#include "fmha_fwd_trek.hpp" // fmha_vsa_bwd_args / traits / entry point

#include "torch/sla_bwd.h"

namespace aiter {

// Defined in csrc/kernels/sla_lut_transpose.cu.
void launch_sla_lut_transpose(const int32_t* m_lut,
                              int32_t* kq_lut,
                              int32_t* kn_count,
                              int32_t B_H,
                              int32_t M_blocks,
                              int32_t K_blocks,
                              int32_t topk,
                              int32_t max_kn_count,
                              int32_t q_scale,
                              hipStream_t stream);

// CK VSA bwd kernel's kM0 — must match the tile shape instantiated in
// vsa_sparse_attention_bwd.cpp (fmha_block_tile_vsa first element).
// When this differs from SLA's BLKQ the wrapper expands the LUT so that
// each SLA Q-block becomes (BLKQ / VSA_BWD_KM0) contiguous CK Q-tiles.
constexpr int VSA_BWD_KM0 = 64;

namespace torch_itfs {

std::vector<at::Tensor> sla_bwd(at::Tensor dout,
                                at::Tensor q,
                                at::Tensor k,
                                at::Tensor v,
                                at::Tensor out,
                                at::Tensor softmax_lse,
                                at::Tensor lut,
                                int64_t block_m,
                                int64_t block_n,
                                double softmax_scale,
                                std::optional<at::Tensor> dq_,
                                std::optional<at::Tensor> dk_,
                                std::optional<at::Tensor> dv_)
{
    // ---- dtype + shape checks ----
    auto q_dtype = q.scalar_type();
    TORCH_CHECK(q_dtype == at::ScalarType::BFloat16 || q_dtype == at::ScalarType::Half,
                "sla_bwd: only fp16 and bf16 supported");
    TORCH_CHECK(k.dtype() == q_dtype && v.dtype() == q_dtype && out.dtype() == q_dtype &&
                    dout.dtype() == q_dtype,
                "sla_bwd: q/k/v/out/dout must share dtype");
    TORCH_CHECK(softmax_lse.dtype() == at::ScalarType::Float,
                "sla_bwd: softmax_lse must be fp32");
    TORCH_CHECK(lut.dtype() == at::ScalarType::Int, "sla_bwd: lut must be int32");

    CHECK_DEVICE(q); CHECK_DEVICE(k); CHECK_DEVICE(v); CHECK_DEVICE(dout);
    CHECK_DEVICE(out); CHECK_DEVICE(softmax_lse); CHECK_DEVICE(lut);
    CHECK_CONTIGUOUS(q); CHECK_CONTIGUOUS(k); CHECK_CONTIGUOUS(v);
    CHECK_CONTIGUOUS(out); CHECK_CONTIGUOUS(softmax_lse); CHECK_CONTIGUOUS(lut);
    // dout may be non-contig but we make it contig below.

    TORCH_CHECK(q.dim() == 4, "sla_bwd: q must be [B, H, S, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "sla_bwd: k shape must match q");
    TORCH_CHECK(v.sizes() == q.sizes(), "sla_bwd: v shape must match q");
    TORCH_CHECK(out.sizes() == q.sizes(), "sla_bwd: out shape must match q");
    TORCH_CHECK(dout.sizes() == q.sizes(), "sla_bwd: dout shape must match q");

    const int64_t B = q.size(0);
    const int64_t H = q.size(1);
    const int64_t S = q.size(2);
    const int64_t D = q.size(3);
    TORCH_CHECK(D == 128, "sla_bwd: only D=128 supported");
    TORCH_CHECK(block_m == 128 && block_n == 64,
                "sla_bwd: only (block_m=128, block_n=64) supported");

    const int64_t M_BLOCKS = (S + block_m - 1) / block_m;
    const int64_t K_BLOCKS = (S + block_n - 1) / block_n;

    TORCH_CHECK(lut.dim() == 4, "sla_bwd: lut must be rank 4 [B, H, M_BLOCKS, topk]");
    TORCH_CHECK(lut.size(0) == B && lut.size(1) == H && lut.size(2) == M_BLOCKS,
                "sla_bwd: lut dims must match [B, H, M_BLOCKS, topk]");
    const int64_t topk = lut.size(3);
    TORCH_CHECK(topk > 0 && topk <= K_BLOCKS, "sla_bwd: topk out of range");

    TORCH_CHECK(softmax_lse.sizes() == torch::IntArrayRef({B, H, S}),
                "sla_bwd: softmax_lse must be [B, H, S]");

    // ---- Outputs ----
    auto make_like = [&](std::optional<at::Tensor>& user, const at::Tensor& ref) {
        if(user.has_value())
        {
            CHECK_DEVICE(user.value()); CHECK_CONTIGUOUS(user.value());
            TORCH_CHECK(user.value().sizes() == ref.sizes() &&
                            user.value().dtype() == ref.dtype(),
                        "sla_bwd: pre-allocated grad has wrong shape or dtype");
            return user.value();
        }
        return torch::empty_like(ref);
    };
    at::Tensor dq = make_like(dq_, q);
    at::Tensor dk = make_like(dk_, k);
    at::Tensor dv = make_like(dv_, v);

    // Ensure dout is contiguous.
    at::Tensor dout_c = dout.is_contiguous() ? dout : dout.contiguous();

    const at::hip::OptionalHIPGuardMasqueradingAsCUDA device_guard{q.device()};
    const hipStream_t stream = at::hip::getCurrentHIPStream();

    // ---- Preprocess: δ = (dO ⊙ O).sum(-1) via torch ----
    // P2.3: use bf16 multiply + fp32 accumulation instead of pre-casting
    // both inputs to fp32. Measured at Config C: 1.06 ms → 0.32 ms
    // (3.3× faster), SNR 55.6 dB vs fp32 reference — the main bwd's
    // ~54 dB is the noise floor anyway, so the precision difference is
    // not observable downstream. The `dtype=torch::kFloat` arg forces
    // the sum accumulator to fp32 while keeping the pairwise product in
    // bf16, which skips the upcast HBM round-trip for both inputs.
    at::Tensor delta = (dout_c * out)
                           .sum(-1, /*keepdim=*/false, at::ScalarType::Float)
                           .contiguous(); // [B, H, S] fp32

    // P1.4: pass log2-space LSE from the fwd directly (no torch divide).
    // The split bwd pipelines now consume log2-space LSE and do
    // `exp2(scale*s - lse)` without a per-row log2e multiply — saves
    // ~0.2 ms wall for the torch op plus a multiply per row in the hot
    // loop. Must be kept in sync with both
    // block_fmha_bwd_dkdv_only_vsa.hpp and block_fmha_bwd_dq_only_vsa.hpp
    // softmax computations.
    at::Tensor lse_natural = softmax_lse;

    // ---- Transposed-LUT construction ----
    // kq_lut   [B*H, K_BLOCKS, max_kn_count] int32 CK Q-tile indices
    // kn_count [B*H, K_BLOCKS] int32
    //
    // When VSA_BWD_KM0 < BLKQ (e.g. kM0=64, BLKQ=128), we expand each SLA
    // Q-block into q_scale CK Q-tiles. That means more entries per K-block,
    // so max_kn_count = M_BLOCKS_CK = M_BLOCKS * q_scale.
    TORCH_CHECK(block_m % VSA_BWD_KM0 == 0,
                "sla_bwd: BLKQ must be a multiple of the CK bwd kM0 tile");
    const int32_t q_scale = static_cast<int32_t>(block_m / VSA_BWD_KM0);
    const int64_t M_BLOCKS_CK = M_BLOCKS * q_scale;
    const int64_t max_kn_count = M_BLOCKS_CK;
    // Sentinel = INT_MAX so ascending torch::sort leaves valid entries first.
    at::Tensor kq_lut = torch::full(
        {B * H, K_BLOCKS, max_kn_count},
        std::numeric_limits<int32_t>::max(),
        q.options().dtype(at::ScalarType::Int));
    at::Tensor kn_count = torch::zeros({B * H, K_BLOCKS},
                                       q.options().dtype(at::ScalarType::Int));

    aiter::launch_sla_lut_transpose(
        reinterpret_cast<const int32_t*>(lut.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kq_lut.data_ptr<int32_t>()),
        reinterpret_cast<int32_t*>(kn_count.data_ptr<int32_t>()),
        static_cast<int32_t>(B * H),
        static_cast<int32_t>(M_BLOCKS),
        static_cast<int32_t>(K_BLOCKS),
        static_cast<int32_t>(topk),
        static_cast<int32_t>(max_kn_count),
        q_scale,
        stream);

    // Tier 2.5 Step 1: sort each (bh, k) row ascending via torch.sort instead
    // of the old insertion-sort kernel. torch::sort uses GPU radix sort which
    // is O(n log n) parallel — an order of magnitude faster than O(n²)
    // single-thread sort at Config C (~2.5 ms → ~0.15 ms target).
    kq_lut = std::get<0>(torch::sort(kq_lut, /*dim=*/-1));

    // Tier 2.5 Step 1c: the dq-only (Q-major) kernel iterates the input
    // M-major LUT directly, so sort each (b, h, m_row) row ascending too.
    // Unsorted entries are *correct* (gemms are order-independent and
    // `move_tile_window` accepts negative offsets) but sorted rows give
    // monotonic HBM access for K/V which matches the expected cache
    // prefetch pattern. torch::sort is cheap (~0.15 ms at Config C).
    at::Tensor lut_sorted = std::get<0>(torch::sort(lut, /*dim=*/-1));

    // Tier 2.5 Step 1c: the split bwd no longer needs a fp32 dq_accum
    // scratch. The dq-only kernel writes dq directly to the bf16 output
    // with a single store per Q-tile — no atomics, no split-K, no cast
    // pass. The old `dq_acc` allocation has been dropped.

    // ---- Fill CK VSA bwd args ----
    fmha_vsa_bwd_args args{};
    args.q_ptr       = q.data_ptr();
    args.k_ptr       = k.data_ptr();
    args.v_ptr       = v.data_ptr();
    args.o_ptr       = out.data_ptr();
    args.lse_ptr     = lse_natural.data_ptr();
    args.do_ptr      = dout_c.data_ptr();
    args.d_ptr       = delta.data_ptr();
    // The split bwd writes dq straight to bf16; the dkdv half's driver
    // still wants a non-null dq_acc_ptr in its kargs (it never dereferences
    // it because the dkdv pipeline ignores dq), so point it at the real
    // dq buffer. The dq-only kernel owns the actual writes.
    args.dq_acc_ptr  = dq.data_ptr();
    args.dq_ptr      = dq.data_ptr();
    args.dk_ptr      = dk.data_ptr();
    args.dv_ptr      = dv.data_ptr();
    args.kq_lut_ptr  = kq_lut.data_ptr();
    args.kn_count_ptr = kn_count.data_ptr();
    args.max_kn_count = static_cast<ck_tile::index_t>(max_kn_count);

    // M-major LUT (for the dq-only half). SLA's input `lut` tensor is
    // already in [B, H, M_BLOCKS, topk] layout — we just pass it through.
    // `q_scale` handles the BLKQ → VSA_BWD_KM0 granularity mismatch in
    // the kernel; the LUT rows are keyed on SLA Q-block index.
    args.kv_block_idx_ptr  = lut_sorted.data_ptr();
    args.kv_blocks_per_row = static_cast<ck_tile::index_t>(topk);
    args.q_scale           = static_cast<ck_tile::index_t>(q_scale);

    args.seqlen_q     = static_cast<ck_tile::index_t>(S);
    args.seqlen_k     = static_cast<ck_tile::index_t>(S);
    args.batch        = static_cast<ck_tile::index_t>(B);
    args.max_seqlen_q = static_cast<ck_tile::index_t>(S);
    args.hdim_q       = static_cast<ck_tile::index_t>(D);
    args.hdim_v       = static_cast<ck_tile::index_t>(D);
    args.nhead_q      = static_cast<ck_tile::index_t>(H);
    args.nhead_k      = static_cast<ck_tile::index_t>(H);
    args.scale_s      = static_cast<float>(softmax_scale);

    // BHSD contiguous strides.
    args.stride_q  = static_cast<ck_tile::index_t>(q.stride(2));
    args.stride_k  = static_cast<ck_tile::index_t>(k.stride(2));
    args.stride_v  = static_cast<ck_tile::index_t>(v.stride(2));
    args.stride_o  = static_cast<ck_tile::index_t>(out.stride(2));
    args.stride_do = static_cast<ck_tile::index_t>(dout_c.stride(2));
    args.stride_dq = static_cast<ck_tile::index_t>(dq.stride(2));
    args.stride_dk = static_cast<ck_tile::index_t>(dk.stride(2));
    args.stride_dv = static_cast<ck_tile::index_t>(dv.stride(2));
    // dq_acc is dead (split bwd writes dq direct); reuse dq strides so
    // the dkdv driver's kargs stay internally consistent even though its
    // pipeline never touches those fields.
    args.stride_dq_acc = static_cast<ck_tile::index_t>(dq.stride(2));

    args.nhead_stride_q    = static_cast<ck_tile::index_t>(q.stride(1));
    args.nhead_stride_k    = static_cast<ck_tile::index_t>(k.stride(1));
    args.nhead_stride_v    = static_cast<ck_tile::index_t>(v.stride(1));
    args.nhead_stride_o    = static_cast<ck_tile::index_t>(out.stride(1));
    args.nhead_stride_do   = static_cast<ck_tile::index_t>(dout_c.stride(1));
    args.nhead_stride_lsed = static_cast<ck_tile::index_t>(softmax_lse.stride(1));
    args.nhead_stride_dq_acc = static_cast<ck_tile::long_index_t>(dq.stride(1));
    args.nhead_stride_dk   = static_cast<ck_tile::index_t>(dk.stride(1));
    args.nhead_stride_dv   = static_cast<ck_tile::index_t>(dv.stride(1));
    args.nhead_stride_dq   = static_cast<ck_tile::index_t>(dq.stride(1));
    args.nhead_stride_kq_lut   = static_cast<ck_tile::index_t>(K_BLOCKS * max_kn_count);
    args.nhead_stride_kn_count = static_cast<ck_tile::index_t>(K_BLOCKS);
    // M-major LUT nhead stride: `lut` is [B, H, M_BLOCKS, topk], so the
    // stride between adjacent (b, h) rows is `M_BLOCKS * topk`.
    args.nhead_stride_kv_idx   = static_cast<ck_tile::index_t>(M_BLOCKS * topk);

    args.batch_stride_q    = static_cast<ck_tile::index_t>(q.stride(0));
    args.batch_stride_k    = static_cast<ck_tile::index_t>(k.stride(0));
    args.batch_stride_v    = static_cast<ck_tile::index_t>(v.stride(0));
    args.batch_stride_o    = static_cast<ck_tile::index_t>(out.stride(0));
    args.batch_stride_do   = static_cast<ck_tile::index_t>(dout_c.stride(0));
    args.batch_stride_lsed = static_cast<ck_tile::index_t>(softmax_lse.stride(0));
    args.batch_stride_dq_acc = static_cast<ck_tile::long_index_t>(dq.stride(0));
    args.batch_stride_dk   = static_cast<ck_tile::index_t>(dk.stride(0));
    args.batch_stride_dv   = static_cast<ck_tile::index_t>(dv.stride(0));
    args.batch_stride_dq   = static_cast<ck_tile::index_t>(dq.stride(0));
    args.batch_stride_kq_lut   = static_cast<ck_tile::index_t>(H * K_BLOCKS * max_kn_count);
    args.batch_stride_kn_count = static_cast<ck_tile::index_t>(H * K_BLOCKS);
    args.batch_stride_kv_idx   = static_cast<ck_tile::index_t>(H * M_BLOCKS * topk);

    const std::string dtype_str = (q_dtype == at::ScalarType::BFloat16) ? "bf16" : "fp16";
    fmha_vsa_bwd_traits traits{};
    traits.hdim_q        = static_cast<int>(D);
    traits.hdim_v        = static_cast<int>(D);
    traits.data_type     = dtype_str;
    traits.is_v_rowmajor = true;
    traits.mask_type     = mask_enum::no_mask;

    ck_tile::stream_config stream_config{stream};
    float t = fmha_vsa_bwd(traits, args, stream_config);

    if(t < 0.0f)
    {
        dq.zero_();
        dk.zero_();
        dv.zero_();
    }
    // Tier 2.5 Step 1c: dq is written directly to its bf16/fp16 buffer by
    // the dq-only kernel — no post-pass cast needed. dk/dv are written by
    // the dkdv-only kernel's Default2DEpilogue in q's dtype.

    return {dq, dk, dv};
}

} // namespace torch_itfs
} // namespace aiter
