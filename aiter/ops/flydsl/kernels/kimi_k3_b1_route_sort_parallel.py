# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Parallel Kimi-K3 B1 route, metadata, and stage-1 MXFP8 prep for gfx950."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, vector
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.arith import ArithValue, CmpFPredicate, CmpIPredicate
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.quant_utils import emit_mx_e8m0_scale
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)
from aiter.utility.mx_types import MxDtypeInt, MxScaleRoundModeInt

_ROUTE_THREADS = 64
_BLOCK_THREADS = 256
_QUANT_THREADS = _BLOCK_THREADS - _ROUTE_THREADS
_EXPERTS = 896
_TOPK = 16
_LOCAL_ROUTE_CANDIDATES = _EXPERTS // _ROUTE_THREADS
_DPP_PRIORITY_HIGH_MASK = 0x38
_SHORT_PRIORITY_LANES = _ROUTE_THREADS // 2
_SHORT_LANE_CANDIDATES = 12
_FULL_LANE_CANDIDATES = 16
_SHORT_PRIORITY_EXPERTS = _SHORT_PRIORITY_LANES * _SHORT_LANE_CANDIDATES
_BLOCK_M = 32
_SORTED_ROWS = _TOPK * _BLOCK_M
_MODEL_DIM = 3584
_MX_GROUP_SIZE = 32
_MX_SCALE_COLS = _MODEL_DIM // _MX_GROUP_SIZE
_MX_GROUPS_PER_ITERATION = _QUANT_THREADS // 2
_MX_VALUES_PER_THREAD = _MX_GROUP_SIZE // 2
_LOG2E = 1.4426950408889634


@fx.struct
class _RouteSortStorage:
    biased_scores: fx.Array[fx.Float32, _EXPERTS, 16]
    priority_expert_ids: fx.Array[fx.Int32, _EXPERTS, 16]
    route_scores: fx.Array[fx.Float32, _EXPERTS, 16]
    selected_ids: fx.Array[fx.Int32, _TOPK, 16]
    selected_scores: fx.Array[fx.Float32, _TOPK, 16]
    route_sum: fx.Array[fx.Float32, 1, 16]
    local_route_ids: fx.Array[
        fx.Int32,
        _ROUTE_THREADS * _LOCAL_ROUTE_CANDIDATES,
        16,
    ]
    local_route_scores: fx.Array[
        fx.Float32,
        _ROUTE_THREADS * _LOCAL_ROUTE_CANDIDATES,
        16,
    ]


def _lds_load(ptr, idx):
    return fx.ptr_load(ptr + fx.Int64(idx))


def _lds_store(ptr, value, idx):
    fx.ptr_store(value, ptr + fx.Int64(idx))


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_b1_route_sort_parallel_module():
    """Build the fixed-shape overlapped route+MXFP8 preparation launcher."""

    @flyc.kernel(
        name="kimi_k3_b1_route_quant_parallel_gfx950",
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
        hidden_states: fx.Pointer,
        quantized_hidden_states: fx.Pointer,
        quantized_scales: fx.Pointer,
    ):
        i32 = T.i32
        f32 = T.f32
        tid = ArithValue(gpu.thread_idx.x)
        c_zero_i32 = arith.constant(0, type=i32)
        c_one_i32 = arith.constant(1, type=i32)
        c_zero_f32 = arith.constant(0.0, type=f32)
        c_one_f32 = arith.constant(1.0, type=f32)
        c_neg_inf = arith.constant(float("-inf"), type=f32)
        vec4_bf16 = T.vec(4, T.bf16)
        vec2_i32 = T.vec(2, i32)
        vec2_f32 = T.vec(2, f32)
        vec4_i32 = T.vec(4, i32)
        vec4_f32 = T.vec(4, f32)

        logits_rsrc = ptr_rsrc(logits)
        bias_rsrc = ptr_rsrc(correction_bias)
        topk_weights_rsrc = ptr_rsrc(topk_weights)
        topk_ids_rsrc = ptr_rsrc(topk_ids)
        sorted_ids_rsrc = ptr_rsrc(sorted_ids)
        sorted_weights_rsrc = ptr_rsrc(sorted_weights)
        sorted_experts_rsrc = ptr_rsrc(sorted_expert_ids)
        nvalid_rsrc = ptr_rsrc(num_valid_ids)
        moe_buf_rsrc = ptr_rsrc(moe_buf)
        hidden_rsrc = ptr_rsrc(hidden_states)
        quantized_hidden_rsrc = ptr_rsrc(quantized_hidden_states)
        quantized_scale_rsrc = ptr_rsrc(quantized_scales)

        lds = fx.SharedAllocator().allocate(_RouteSortStorage).peek()
        biased_lds = lds.biased_scores.ptr
        priority_expert_ids_lds = lds.priority_expert_ids.ptr
        route_lds = lds.route_scores.ptr
        selected_ids_lds = lds.selected_ids.ptr
        selected_scores_lds = lds.selected_scores.ptr
        route_sum_lds = lds.route_sum.ptr
        local_route_ids_lds = lds.local_route_ids.ptr
        local_route_scores_lds = lds.local_route_scores.ptr

        def lane_priority_rank(lane):
            """Return the accepted DPP equal-score rank for a wave64 lane."""
            lane_bit2 = (lane >> arith.constant(2, type=i32)) & c_one_i32
            lane_bit3 = (lane >> arith.constant(3, type=i32)) & c_one_i32
            rank_hi = (
                lane ^ arith.constant(_DPP_PRIORITY_HIGH_MASK, type=i32)
            ) & arith.constant(_DPP_PRIORITY_HIGH_MASK, type=i32)
            rank_lo = (lane & arith.constant(7, type=i32)) ^ (
                (lane_bit3 << arith.constant(2, type=i32))
                | (lane_bit2 * arith.constant(3, type=i32))
            )
            return rank_hi | rank_lo

        def expert_priority_index(expert):
            """Map an expert ID to the accepted global stable-tie order."""
            original_lane = (expert >> arith.constant(2, type=i32)) & arith.constant(
                63, type=i32
            )
            rank = lane_priority_rank(original_lane)
            local_order = (
                (expert >> arith.constant(8, type=i32)) << arith.constant(2, type=i32)
            ) | (expert & arith.constant(3, type=i32))
            short_lane = arith.cmpi(
                CmpIPredicate.uge,
                original_lane,
                arith.constant(_SHORT_PRIORITY_LANES, type=i32),
            )
            short_index = (
                rank * arith.constant(_SHORT_LANE_CANDIDATES, type=i32) + local_order
            )
            full_index = (
                arith.constant(_SHORT_PRIORITY_EXPERTS, type=i32)
                + (rank - arith.constant(_SHORT_PRIORITY_LANES, type=i32))
                * arith.constant(_FULL_LANE_CANDIDATES, type=i32)
                + local_order
            )
            return arith.select(short_lane, short_index, full_index)

        # Four waves compute sigmoid+bias in parallel. Biased scores are
        # staged in the accepted global tie order so the route wave can split
        # all 896 experts into 64 balanced, contiguous 14-score partitions.
        for vec_base in range_constexpr(0, _EXPERTS, _BLOCK_THREADS * 4):
            expert_base = tid * arith.constant(4, type=i32) + arith.constant(
                vec_base, type=i32
            )
            vector_in_range = arith.cmpi(
                CmpIPredicate.ult,
                expert_base,
                arith.constant(_EXPERTS, type=i32),
            )
            load_if = scf.IfOp(vector_in_range)
            with ir.InsertionPoint(load_if.then_block):
                logits_vec = buffer_ops.buffer_load(
                    logits_rsrc,
                    expert_base,
                    vec_width=4,
                    dtype=f32,
                )
                bias_i32 = buffer_ops.buffer_load(
                    bias_rsrc,
                    expert_base // arith.constant(2, type=i32),
                    vec_width=2,
                    dtype=i32,
                )
                bias_vec = vector.bitcast(vec4_bf16, bias_i32)
                priority_base = expert_priority_index(expert_base)
                sigmoid_values = []
                biased_values = []
                expert_ids = []
                for lane_in_vec in range_constexpr(4):
                    expert = expert_base + arith.constant(
                        lane_in_vec,
                        type=i32,
                    )
                    x = vector.extract(
                        logits_vec,
                        static_position=[lane_in_vec],
                        dynamic_position=[],
                    )
                    bias_bf16 = vector.extract(
                        bias_vec,
                        static_position=[lane_in_vec],
                        dynamic_position=[],
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
                    sigmoid_values.append(sigmoid)
                    biased_values.append(ArithValue(sigmoid) + bias_f32)
                    expert_ids.append(expert)
                fx.ptr_store(
                    vector.from_elements(vec4_f32, sigmoid_values),
                    route_lds + fx.Int64(expert_base),
                )
                fx.ptr_store(
                    vector.from_elements(vec4_f32, biased_values),
                    biased_lds + fx.Int64(priority_base),
                )
                fx.ptr_store(
                    vector.from_elements(vec4_i32, expert_ids),
                    priority_expert_ids_lds + fx.Int64(priority_base),
                )
                scf.YieldOp([])
        gpu.barrier()

        route_active = arith.cmpi(
            CmpIPredicate.ult,
            tid,
            arith.constant(_ROUTE_THREADS, type=i32),
        )
        route_if = scf.IfOp(route_active)
        with ir.InsertionPoint(route_if.then_block):
            local_scores = [
                ArithValue(c_neg_inf) for _ in range(_LOCAL_ROUTE_CANDIDATES)
            ]
            local_ids = [ArithValue(c_zero_i32) for _ in range(_LOCAL_ROUTE_CANDIDATES)]

            def insert_candidate(score, expert, position):
                local_scores[position] = ArithValue(score)
                local_ids[position] = ArithValue(expert)
                for sort_offset in range_constexpr(position):
                    right_position = position - sort_offset
                    left_position = right_position - 1
                    right_score = local_scores[right_position]
                    left_score = local_scores[left_position]
                    right_id = local_ids[right_position]
                    left_id = local_ids[left_position]
                    swap = arith.cmpf(
                        CmpFPredicate.OGT,
                        right_score,
                        left_score,
                    )
                    local_scores[left_position] = ArithValue(
                        arith.select(
                            swap,
                            _raw(right_score),
                            _raw(left_score),
                        )
                    )
                    local_scores[right_position] = ArithValue(
                        arith.select(
                            swap,
                            _raw(left_score),
                            _raw(right_score),
                        )
                    )
                    local_ids[left_position] = ArithValue(
                        arith.select(swap, _raw(right_id), _raw(left_id))
                    )
                    local_ids[right_position] = ArithValue(
                        arith.select(swap, _raw(left_id), _raw(right_id))
                    )

            local_position = 0
            priority_lane_base = lane_priority_rank(tid) * arith.constant(
                _LOCAL_ROUTE_CANDIDATES, type=i32
            )
            for vec_base in range_constexpr(0, 12, 4):
                priority_base = priority_lane_base + arith.constant(
                    vec_base,
                    type=i32,
                )
                score_vector = fx.ptr_load(
                    biased_lds + fx.Int64(priority_base),
                    result_type=vec4_f32,
                )
                expert_vector = fx.ptr_load(
                    priority_expert_ids_lds + fx.Int64(priority_base),
                    result_type=vec4_i32,
                )
                for lane_in_vec in range_constexpr(4):
                    expert = vector.extract(
                        expert_vector,
                        static_position=[lane_in_vec],
                        dynamic_position=[],
                    )
                    score = vector.extract(
                        score_vector,
                        static_position=[lane_in_vec],
                        dynamic_position=[],
                    )
                    insert_candidate(score, expert, local_position)
                    local_position += 1

            priority_base = priority_lane_base + arith.constant(12, type=i32)
            score_vector = fx.ptr_load(
                biased_lds + fx.Int64(priority_base),
                result_type=vec2_f32,
            )
            expert_vector = fx.ptr_load(
                priority_expert_ids_lds + fx.Int64(priority_base),
                result_type=vec2_i32,
            )
            for lane_in_vec in range_constexpr(2):
                expert = vector.extract(
                    expert_vector,
                    static_position=[lane_in_vec],
                    dynamic_position=[],
                )
                score = vector.extract(
                    score_vector,
                    static_position=[lane_in_vec],
                    dynamic_position=[],
                )
                insert_candidate(score, expert, local_position)
                local_position += 1

            local_route_base = tid * arith.constant(
                _LOCAL_ROUTE_CANDIDATES,
                type=i32,
            )
            for position in range_constexpr(_LOCAL_ROUTE_CANDIDATES):
                local_route_offset = local_route_base + arith.constant(
                    position,
                    type=i32,
                )
                _lds_store(
                    local_route_scores_lds,
                    local_scores[position],
                    local_route_offset,
                )
                _lds_store(
                    local_route_ids_lds,
                    local_ids[position],
                    local_route_offset,
                )

            route_sum = ArithValue(c_zero_f32)
            local_rank = ArithValue(c_zero_i32)
            for k in range_constexpr(_TOPK):
                local_route_offset = local_route_base + local_rank
                local_max = ArithValue(
                    _lds_load(local_route_scores_lds, local_route_offset)
                )
                local_id = ArithValue(
                    _lds_load(local_route_ids_lds, local_route_offset)
                )
                lane_candidate_id = local_id

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
                    take_remote = arith.cmpf(
                        CmpFPredicate.OGT,
                        remote_max,
                        local_max,
                    )
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
                        arith.constant(_ROUTE_THREADS - 1, type=i32),
                    )
                )

                selected_score = ArithValue(_lds_load(route_lds, selected_id))
                is_local_winner = arith.cmpi(
                    CmpIPredicate.eq,
                    lane_candidate_id,
                    selected_id,
                )
                local_rank = local_rank + ArithValue(
                    arith.select(
                        is_local_winner,
                        c_one_i32,
                        c_zero_i32,
                    )
                )
                route_sum = route_sum + selected_score
                _lds_store(
                    selected_ids_lds,
                    selected_id,
                    arith.constant(k, type=i32),
                )
                _lds_store(
                    selected_scores_lds,
                    selected_score,
                    arith.constant(k, type=i32),
                )

            is_route_writer = arith.cmpi(
                CmpIPredicate.eq,
                tid,
                c_zero_i32,
            )
            route_writer_if = scf.IfOp(is_route_writer)
            with ir.InsertionPoint(route_writer_if.then_block):
                _lds_store(route_sum_lds, route_sum, c_zero_i32)
                scf.YieldOp([])

            scf.YieldOp([])

        # Waves 1-3 prepare the one B1 activation row for A8W4 stage 1 while
        # wave 0 performs the exact 16-step route selection above. Two adjacent
        # lanes own one 32-value MX group. The 192 quant lanes cover 96 groups
        # in the first iteration and the remaining 16 groups in the second.
        # This preserves the accepted RoundUp E8M0 and OCP FP8 E4M3 bytes while
        # removing the standalone activation-quant launch.
        quant_active = arith.cmpi(
            CmpIPredicate.uge,
            tid,
            arith.constant(_ROUTE_THREADS, type=i32),
        )
        quant_if = scf.IfOp(quant_active)
        with ir.InsertionPoint(quant_if.then_block):
            quant_tid = tid - arith.constant(_ROUTE_THREADS, type=i32)
            lane_in_group = quant_tid % arith.constant(2, type=i32)
            group_in_iteration = quant_tid // arith.constant(2, type=i32)
            c_scale_exp = arith.constant(254, type=i32)
            c_exp_shift = arith.constant(23, type=i32)
            c_amax_floor = arith.constant(1.0e-10, type=f32)
            for group_iteration in range_constexpr(
                (_MX_SCALE_COLS + _MX_GROUPS_PER_ITERATION - 1)
                // _MX_GROUPS_PER_ITERATION
            ):
                group = group_in_iteration + arith.constant(
                    group_iteration * _MX_GROUPS_PER_ITERATION,
                    type=i32,
                )
                group_in_range = arith.cmpi(
                    CmpIPredicate.ult,
                    group,
                    arith.constant(_MX_SCALE_COLS, type=i32),
                )
                group_if = scf.IfOp(group_in_range)
                with ir.InsertionPoint(group_if.then_block):
                    element_base = group * arith.constant(
                        _MX_GROUP_SIZE,
                        type=i32,
                    ) + lane_in_group * arith.constant(
                        _MX_VALUES_PER_THREAD,
                        type=i32,
                    )
                    values = []
                    local_amax = c_amax_floor
                    for element_offset in range_constexpr(_MX_VALUES_PER_THREAD):
                        value_bf16 = buffer_ops.buffer_load(
                            hidden_rsrc,
                            element_base + arith.constant(element_offset, type=i32),
                            vec_width=1,
                            dtype=T.bf16,
                        )
                        value = arith.extf(f32, value_bf16)
                        values.append(value)
                        local_amax = arith.maximumf(
                            local_amax,
                            fmath.absf(value),
                        )

                    peer_amax = ArithValue(local_amax).shuffle_xor(
                        arith.constant(1, type=i32),
                        arith.constant(_ROUTE_THREADS, type=i32),
                    )
                    group_amax = arith.maximumf(local_amax, _raw(peer_amax))
                    e8m0 = emit_mx_e8m0_scale(
                        group_amax,
                        mode=MxScaleRoundModeInt.RoundUp,
                        dtype=MxDtypeInt.FP8_E4M3,
                    )
                    quant_scale = ((c_scale_exp - e8m0) << c_exp_shift).bitcast(f32)

                    for pack_index in range_constexpr(_MX_VALUES_PER_THREAD // 4):
                        value_base = pack_index * 4
                        packed = arith.constant(0, type=i32)
                        packed = rocdl.cvt_pk_fp8_f32(
                            i32,
                            arith.mulf(
                                values[value_base],
                                quant_scale,
                            ),
                            arith.mulf(
                                values[value_base + 1],
                                quant_scale,
                            ),
                            packed,
                            0,
                        )
                        packed = rocdl.cvt_pk_fp8_f32(
                            i32,
                            arith.mulf(
                                values[value_base + 2],
                                quant_scale,
                            ),
                            arith.mulf(
                                values[value_base + 3],
                                quant_scale,
                            ),
                            packed,
                            1,
                        )
                        output_byte = element_base + arith.constant(
                            pack_index * 4,
                            type=i32,
                        )
                        buffer_ops.buffer_store(
                            packed,
                            quantized_hidden_rsrc,
                            output_byte,
                            offset_is_bytes=True,
                        )

                    is_scale_writer = arith.cmpi(
                        CmpIPredicate.eq,
                        lane_in_group,
                        c_zero_i32,
                    )
                    scale_if = scf.IfOp(is_scale_writer)
                    with ir.InsertionPoint(scale_if.then_block):
                        scale_tile = group // arith.constant(8, type=i32)
                        scale_lane4 = group % arith.constant(4, type=i32)
                        scale_half8 = (
                            group % arith.constant(8, type=i32)
                        ) // arith.constant(4, type=i32)
                        within_rank = (
                            scale_tile * arith.constant(256, type=i32)
                            + scale_lane4 * arith.constant(64, type=i32)
                            + scale_half8 * arith.constant(2, type=i32)
                        )
                        e8m0_i8 = arith.trunci(T.i8, _raw(e8m0))
                        for rank in range_constexpr(_TOPK):
                            scale_offset = within_rank + arith.constant(
                                rank * _MX_SCALE_COLS * _BLOCK_M,
                                type=i32,
                            )
                            buffer_ops.buffer_store(
                                e8m0_i8,
                                quantized_scale_rsrc,
                                scale_offset,
                                offset_is_bytes=True,
                            )
                        scf.YieldOp([])
                    scf.YieldOp([])
            scf.YieldOp([])
        gpu.barrier()

        # Initialize every padded metadata row and zero the atomic-output
        # buffer while the selected routes remain resident.
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

        active = arith.cmpi(
            CmpIPredicate.ult,
            tid,
            arith.constant(_TOPK, type=i32),
        )
        active_if = scf.IfOp(active)
        with ir.InsertionPoint(active_if.then_block):
            route_id = _lds_load(selected_ids_lds, tid)
            route_score = _lds_load(selected_scores_lds, tid)
            route_sum = _lds_load(route_sum_lds, c_zero_i32)
            normalized = arith.divf(_raw(route_score), _raw(route_sum))
            buffer_ops.buffer_store(route_id, topk_ids_rsrc, tid)
            buffer_ops.buffer_store(normalized, topk_weights_rsrc, tid)

            rank = ArithValue(c_zero_i32)
            for other_slot in range_constexpr(_TOPK):
                other_id = _lds_load(
                    selected_ids_lds,
                    arith.constant(other_slot, type=i32),
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
        hidden_states: fx.Pointer,
        quantized_hidden_states: fx.Pointer,
        quantized_scales: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
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
            hidden_states,
            quantized_hidden_states,
            quantized_scales,
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
