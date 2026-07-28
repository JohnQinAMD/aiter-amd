# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 B1 route and Opus-compatible sort metadata kernel for gfx950."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_BLOCK_THREADS = 64
_EXPERTS = 896
_TOPK = 16
_BLOCK_M = 32
_SORTED_ROWS = _TOPK * _BLOCK_M
_LOG2E = 1.4426950408889634


@fx.struct
class _RouteSortStorage:
    biased_scores: fx.Array[fx.Float32, _EXPERTS, 16]
    route_scores: fx.Array[fx.Float32, _EXPERTS, 16]
    selected_ids: fx.Array[fx.Int32, _TOPK, 16]
    selected_scores: fx.Array[fx.Float32, _TOPK, 16]


def _lds_load(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, value, idx):
    fx.ptr_store(value, ptr + fx.Int64(idx))


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_b1_route_sort_module():
    """Build the fixed-shape gfx950 route+metadata launcher."""

    @flyc.kernel(
        name="kimi_k3_b1_route_sort_gfx950",
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def route_sort_kernel(
        logits: fx.Pointer,
        correction_bias: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        moe_buf: fx.Pointer,
        moe_buf_i32_elements: fx.Int32,
    ):
        i32 = T.i32
        f32 = T.f32
        tid = ArithValue(gpu.thread_idx.x)
        c_zero_i32 = arith.constant(0, type=i32)
        c_one_i32 = arith.constant(1, type=i32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_one_f32 = arith.constant(1.0, type=f32)
        c_neg_inf = arith.constant(float("-inf"), type=f32)

        logits_rsrc = ptr_rsrc(logits)
        bias_rsrc = ptr_rsrc(correction_bias)
        topk_weights_rsrc = ptr_rsrc(topk_weights)
        topk_ids_rsrc = ptr_rsrc(topk_ids)
        sorted_ids_rsrc = ptr_rsrc(sorted_ids)
        sorted_weights_rsrc = ptr_rsrc(sorted_weights)
        sorted_experts_rsrc = ptr_rsrc(sorted_expert_ids)
        nvalid_rsrc = ptr_rsrc(num_valid_ids)
        moe_buf_rsrc = ptr_rsrc(moe_buf)

        lds = fx.SharedAllocator().allocate(_RouteSortStorage).peek()
        biased_lds = lds.biased_scores.ptr
        route_lds = lds.route_scores.ptr
        selected_ids_lds = lds.selected_ids.ptr
        selected_scores_lds = lds.selected_scores.ptr

        # Match biased_grouped_topk's vec4/thread traversal. The group-selection
        # branch disappears because NUM_GRP == TOPK_GRP == 1.
        for vec_base in range_constexpr(0, _EXPERTS, _BLOCK_THREADS * 4):
            expert_base = tid * arith.constant(4, type=i32) + arith.constant(
                vec_base, type=i32
            )
            for lane_in_vec in range_constexpr(4):
                expert = expert_base + arith.constant(lane_in_vec, type=i32)
                in_range = arith.cmpi(
                    CmpIPredicate.ult,
                    expert,
                    arith.constant(_EXPERTS, type=i32),
                )
                load_if = scf.IfOp(in_range)
                with ir.InsertionPoint(load_if.then_block):
                    x = buffer_ops.buffer_load(
                        logits_rsrc, expert, vec_width=1, dtype=f32
                    )
                    bias_bf16 = buffer_ops.buffer_load(
                        bias_rsrc, expert, vec_width=1, dtype=T.bf16
                    )
                    bias_f32 = arith.extf(f32, bias_bf16)
                    exp_value = llvm.call_intrinsic(
                        f32,
                        "llvm.amdgcn.exp2.f32",
                        [ArithValue(x) * arith.constant(-_LOG2E, type=f32)],
                        [],
                        [],
                    )
                    sigmoid = llvm.call_intrinsic(
                        f32,
                        "llvm.amdgcn.rcp.f32",
                        [c_one_f32 + exp_value],
                        [],
                        [],
                    )
                    _lds_store(route_lds, sigmoid, expert)
                    _lds_store(biased_lds, ArithValue(sigmoid) + bias_f32, expert)
                    scf.YieldOp([])
        gpu.barrier()

        route_sum = ArithValue(c_zero_f32)
        for k in range_constexpr(_TOPK):
            local_max = ArithValue(c_neg_inf)
            local_id = ArithValue(arith.constant(k, type=i32))
            for vec_base in range_constexpr(0, _EXPERTS, _BLOCK_THREADS * 4):
                expert_base = tid * arith.constant(4, type=i32) + arith.constant(
                    vec_base, type=i32
                )
                for lane_in_vec in range_constexpr(4):
                    expert = expert_base + arith.constant(lane_in_vec, type=i32)
                    in_range = arith.cmpi(
                        CmpIPredicate.ult,
                        expert,
                        arith.constant(_EXPERTS, type=i32),
                    )
                    score_if = scf.IfOp(in_range, results_=[f32], has_else=True)
                    with ir.InsertionPoint(score_if.then_block):
                        score = _lds_load(biased_lds, expert)
                        scf.YieldOp([_raw(score)])
                    with ir.InsertionPoint(score_if.else_block):
                        scf.YieldOp([c_neg_inf])
                    score = ArithValue(score_if.results[0])
                    take = arith.cmpf(CmpFPredicate.OGT, score, local_max)
                    local_max = ArithValue(
                        arith.select(take, _raw(score), _raw(local_max))
                    )
                    local_id = ArithValue(
                        arith.select(take, _raw(expert), _raw(local_id))
                    )

            # Reproduce hip_reduce.h's gfx9 DPP tree exactly. Its asymmetric
            # tie behavior is observable: a fully uniform row starts at expert
            # 252, while a tied prefix starts at expert 0.
            for dpp_control in (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143):
                local_max_i32 = arith.bitcast(i32, _raw(local_max))
                remote_max_i32 = rocdl.update_dpp(
                    i32,
                    c_zero_i32,
                    local_max_i32,
                    dpp_control,
                    0xF,
                    0xF,
                    True,
                )
                remote_max = ArithValue(arith.bitcast(f32, remote_max_i32))
                remote_id = ArithValue(
                    rocdl.update_dpp(
                        i32,
                        c_zero_i32,
                        _raw(local_id),
                        dpp_control,
                        0xF,
                        0xF,
                        True,
                    )
                )
                take_remote = arith.cmpf(CmpFPredicate.OGT, remote_max, local_max)
                local_max = ArithValue(
                    arith.select(
                        take_remote,
                        _raw(remote_max),
                        _raw(local_max),
                    )
                )
                local_id = ArithValue(
                    arith.select(
                        take_remote,
                        _raw(remote_id),
                        _raw(local_id),
                    )
                )

            selected_id = ArithValue(
                rocdl.readlane(
                    i32,
                    _raw(local_id),
                    arith.constant(_BLOCK_THREADS - 1, type=i32),
                )
            )
            selected_score = ArithValue(_lds_load(route_lds, selected_id))
            _lds_store(biased_lds, c_neg_inf, selected_id)
            route_sum = route_sum + selected_score
            _lds_store(selected_ids_lds, selected_id, arith.constant(k, type=i32))
            _lds_store(
                selected_scores_lds,
                selected_score,
                arith.constant(k, type=i32),
            )

        # Initialize every padded metadata row and zero the accumulation buffer.
        sentinel = arith.constant((_TOPK << 24) | 1, type=i32)
        for row_base in range_constexpr(0, _SORTED_ROWS, _BLOCK_THREADS):
            row = tid + arith.constant(row_base, type=i32)
            buffer_ops.buffer_store(sentinel, sorted_ids_rsrc, row)
            buffer_ops.buffer_store(c_zero_f32, sorted_weights_rsrc, row)

        moe_count = ArithValue(moe_buf_i32_elements)
        loop_lower = arith.index_cast(T.index, tid)
        loop_upper = arith.index_cast(T.index, moe_count)
        loop_step = arith.index(_BLOCK_THREADS)
        zero_loop = scf.ForOp(loop_lower, loop_upper, loop_step)
        with ir.InsertionPoint(zero_loop.body):
            zero_idx = arith.index_cast(i32, zero_loop.induction_variable)
            buffer_ops.buffer_store(c_zero_i32, moe_buf_rsrc, zero_idx)
            scf.YieldOp([])
        gpu.barrier()

        active = arith.cmpi(CmpIPredicate.ult, tid, arith.constant(_TOPK, type=i32))
        active_if = scf.IfOp(active)
        with ir.InsertionPoint(active_if.then_block):
            route_id = _lds_load(selected_ids_lds, tid)
            route_score = _lds_load(selected_scores_lds, tid)
            normalized = arith.divf(_raw(route_score), _raw(route_sum))
            buffer_ops.buffer_store(route_id, topk_ids_rsrc, tid)
            buffer_ops.buffer_store(normalized, topk_weights_rsrc, tid)

            rank = ArithValue(c_zero_i32)
            for other_slot in range_constexpr(_TOPK):
                other_id = _lds_load(
                    selected_ids_lds, arith.constant(other_slot, type=i32)
                )
                is_before = arith.cmpi(CmpIPredicate.slt, other_id, route_id)
                rank = rank + ArithValue(arith.select(is_before, c_one_i32, c_zero_i32))

            sorted_base = rank * arith.constant(_BLOCK_M, type=i32)
            packed_route = tid << arith.constant(24, type=i32)
            buffer_ops.buffer_store(route_id, sorted_experts_rsrc, rank)
            buffer_ops.buffer_store(packed_route, sorted_ids_rsrc, sorted_base)
            buffer_ops.buffer_store(normalized, sorted_weights_rsrc, sorted_base)
            scf.YieldOp([])

        is_first = arith.cmpi(CmpIPredicate.eq, tid, c_zero_i32)
        first_if = scf.IfOp(is_first)
        with ir.InsertionPoint(first_if.then_block):
            buffer_ops.buffer_store(
                arith.constant(_SORTED_ROWS, type=i32),
                nvalid_rsrc,
                c_zero_i32,
            )
            buffer_ops.buffer_store(c_one_i32, nvalid_rsrc, c_one_i32)
            scf.YieldOp([])

    @flyc.jit
    def launch_route_sort(
        logits: fx.Pointer,
        correction_bias: fx.Pointer,
        topk_weights: fx.Pointer,
        topk_ids: fx.Pointer,
        sorted_ids: fx.Pointer,
        sorted_weights: fx.Pointer,
        sorted_expert_ids: fx.Pointer,
        num_valid_ids: fx.Pointer,
        moe_buf: fx.Pointer,
        moe_buf_i32_elements: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        route_sort_kernel(
            logits,
            correction_bias,
            topk_weights,
            topk_ids,
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            moe_buf_i32_elements,
        ).launch(
            grid=(arith.index(1), 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_route_sort.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_route_sort
