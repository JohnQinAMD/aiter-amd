# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kimi_k3_gate import (
    KimiK3GateProjectionDispatch,
    kimi_k3_b1_gate_projection,
    kimi_k3_gate_projection_dispatch,
    supports_kimi_k3_b1_gate_projection,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 gate projection specialization requires gfx950",
)


def _route(logits, bias):
    weights = torch.empty((1, 16), dtype=torch.float32, device=logits.device)
    expert_ids = torch.empty((1, 16), dtype=torch.int32, device=logits.device)
    aiter.biased_grouped_topk_hip(
        logits,
        bias.float(),
        weights,
        expert_ids,
        1,
        1,
        True,
        1.0,
    )
    return weights, expert_ids


@pytest.mark.parametrize("seed", [1, 17, 20260728])
def test_kimi_k3_b1_gate_projection_matches_gate_linear(seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden = torch.randn((1, 7168), generator=generator).bfloat16().cuda()
    weight = (
        torch.randn((896, 7168), generator=generator).mul_(7168**-0.5).bfloat16().cuda()
    )
    bias = torch.randn(896, generator=generator).mul_(0.01).bfloat16().cuda()

    expected_logits = F.linear(hidden, weight).float()
    expected_weights, expected_ids = _route(expected_logits, bias)
    actual_logits = kimi_k3_b1_gate_projection(hidden, weight)
    actual_weights, actual_ids = _route(actual_logits, bias)
    torch.cuda.synchronize()

    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(actual_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=0)


def test_kimi_k3_gate_projection_dispatch_is_narrow():
    hidden = torch.empty((1, 7168), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((896, 7168), dtype=torch.bfloat16, device="cuda")
    noncontiguous_weight = torch.empty(
        (7168, 896), dtype=torch.bfloat16, device="cuda"
    ).T

    assert supports_kimi_k3_b1_gate_projection(hidden, weight)
    assert (
        kimi_k3_gate_projection_dispatch(hidden, weight)
        is KimiK3GateProjectionDispatch.FLYDSL_GFX950_B1
    )
    assert (
        kimi_k3_gate_projection_dispatch(hidden.expand(2, -1), weight)
        is KimiK3GateProjectionDispatch.UNSUPPORTED
    )
    assert (
        kimi_k3_gate_projection_dispatch(hidden, weight.float())
        is KimiK3GateProjectionDispatch.UNSUPPORTED
    )
    assert (
        kimi_k3_gate_projection_dispatch(hidden, noncontiguous_weight)
        is KimiK3GateProjectionDispatch.UNSUPPORTED
    )


def test_kimi_k3_gate_projection_reuses_valid_output():
    hidden = torch.randn((1, 7168), dtype=torch.bfloat16, device="cuda")
    weight = torch.randn((896, 7168), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((1, 896), dtype=torch.float32, device="cuda")

    actual = kimi_k3_b1_gate_projection(hidden, weight, out=out)
    torch.cuda.synchronize()

    assert actual is out
    torch.testing.assert_close(actual, F.linear(hidden, weight).float(), rtol=0, atol=0)


@pytest.mark.parametrize(
    "shape,dtype",
    [
        ((896,), torch.float32),
        ((1, 896), torch.bfloat16),
    ],
)
def test_kimi_k3_gate_projection_rejects_invalid_output(shape, dtype):
    hidden = torch.empty((1, 7168), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((896, 7168), dtype=torch.bfloat16, device="cuda")
    out = torch.empty(shape, dtype=dtype, device="cuda")

    with pytest.raises(ValueError, match="out must be contiguous FP32"):
        kimi_k3_b1_gate_projection(hidden, weight, out=out)
