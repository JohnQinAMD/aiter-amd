# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fixed-shape Kimi-K3 BF16 router projection for gfx950."""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue, CmpIPredicate
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

_HIDDEN_SIZE = 7168
_EXPERTS = 896
_WAVE_SIZE = 64
_BLOCK_THREADS = 64
_WORKGROUPS = _EXPERTS


def _raw(value):
    return value.ir_value() if hasattr(value, "ir_value") else value


def build_kimi_k3_b1_gate_projection_module():
    """Build the fixed B1x7168 by 896x7168 projection launcher."""

    @flyc.kernel(
        name="kimi_k3_b1_gate_projection_gfx950",
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def projection_kernel(
        hidden: fx.Pointer,
        router_weight: fx.Pointer,
        logits: fx.Pointer,
    ):
        i32 = T.i32
        f32 = T.f32
        lane = ArithValue(gpu.thread_idx.x)
        wave = ArithValue(gpu.block_idx.x)
        expert = wave

        hidden_rsrc = ptr_rsrc(hidden)
        weight_rsrc = ptr_rsrc(router_weight)
        logits_rsrc = ptr_rsrc(logits)
        zero_i32 = arith.constant(0, type=i32)
        zero_f32 = arith.constant(0.0, type=f32)

        accumulator = ArithValue(zero_f32)
        for k_base in range_constexpr(0, _HIDDEN_SIZE, _WAVE_SIZE):
            k = lane + arith.constant(k_base, type=i32)
            hidden_bf16 = buffer_ops.buffer_load(
                hidden_rsrc,
                k,
                vec_width=1,
                dtype=T.bf16,
            )
            hidden_f32 = ArithValue(arith.extf(f32, hidden_bf16))
            weight_index = expert * arith.constant(_HIDDEN_SIZE, type=i32) + k
            weight_bf16 = buffer_ops.buffer_load(
                weight_rsrc,
                weight_index,
                vec_width=1,
                dtype=T.bf16,
            )
            weight_f32 = ArithValue(arith.extf(f32, weight_bf16))
            accumulator = accumulator + hidden_f32 * weight_f32

        # The same gfx9 DPP tree used by the accepted route kernel reduces a
        # full 64-lane wave and leaves the total in lane 63.
        for dpp_control in (0xB1, 0x4E, 0x141, 0x140, 0x142, 0x143):
            remote_i32 = rocdl.update_dpp(
                i32,
                zero_i32,
                arith.bitcast(i32, _raw(accumulator)),
                dpp_control,
                0xF,
                0xF,
                True,
            )
            remote = ArithValue(arith.bitcast(f32, remote_i32))
            accumulator = accumulator + remote

        is_last_lane = arith.cmpi(
            CmpIPredicate.eq,
            lane,
            arith.constant(_WAVE_SIZE - 1, type=i32),
        )
        store_if = scf.IfOp(is_last_lane)
        with ir.InsertionPoint(store_if.then_block):
            # GateLinear's production contract is BF16 linear followed by an
            # FP32 cast. Round once to BF16 before materializing FP32.
            rounded = arith.trunc_f(T.bf16, _raw(accumulator))
            projected = arith.extf(f32, rounded)
            buffer_ops.buffer_store(projected, logits_rsrc, expert)
            scf.YieldOp([])

    @flyc.jit
    def launch_projection(
        hidden: fx.Pointer,
        router_weight: fx.Pointer,
        logits: fx.Pointer,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            pass
        projection_kernel(hidden, router_weight, logits).launch(
            grid=(arith.index(_WORKGROUPS), 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    launch_projection.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    return launch_projection
