# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kimi_k3_moe_route import (
    KimiK3RouteSortDispatch,
    kimi_k3_b1_route_sort,
    kimi_k3_route_sort_dispatch,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx950",
    reason="Kimi-K3 route+sort specialization requires gfx950",
)


def _incumbent_route(logits, bias):
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


@pytest.mark.parametrize("case", ["random_bias", "uniform", "tied_prefix"])
def test_kimi_k3_b1_route_sort_matches_incumbent(case):
    generator = torch.Generator(device="cpu").manual_seed(20260728)
    logits = torch.randn((1, 896), generator=generator).cuda()
    bias = (torch.randn(896, generator=generator) * 0.01).bfloat16().cuda()
    if case == "uniform":
        logits.zero_()
        bias.zero_()
    elif case == "tied_prefix":
        logits.fill_(-16.0)
        logits[:, :18] = 2.0
        bias.zero_()

    expected_weights, expected_ids = _incumbent_route(logits, bias)
    (
        weights,
        expert_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
    ) = kimi_k3_b1_route_sort(logits, bias, model_dim=128)
    torch.cuda.synchronize()

    torch.testing.assert_close(expert_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(
        num_valid_ids, torch.tensor([512, 1], dtype=torch.int32, device="cuda")
    )
    torch.testing.assert_close(
        sorted_expert_ids, torch.sort(expert_ids.flatten()).values
    )
    assert torch.count_nonzero(moe_buf).item() == 0

    sentinel = (16 << 24) | 1
    for rank, expert in enumerate(sorted_expert_ids.tolist()):
        slot = torch.nonzero(expert_ids[0] == expert).item()
        base = rank * 32
        assert sorted_ids[base].item() == slot << 24
        assert torch.all(sorted_ids[base + 1 : base + 32] == sentinel)
        torch.testing.assert_close(sorted_weights[base], weights[0, slot])
        assert torch.count_nonzero(sorted_weights[base + 1 : base + 32]).item() == 0


def test_kimi_k3_route_sort_dispatch_is_narrow():
    logits = torch.empty((1, 896), dtype=torch.float32, device="cuda")
    bias = torch.empty(896, dtype=torch.bfloat16, device="cuda")
    common = {
        "num_experts": 896,
        "topk": 16,
        "num_expert_group": 1,
        "topk_group": 1,
        "block_size_m": 32,
    }
    assert (
        kimi_k3_route_sort_dispatch(logits, bias, **common)
        is KimiK3RouteSortDispatch.FLYDSL_GFX950_B1
    )
    assert (
        kimi_k3_route_sort_dispatch(logits.expand(2, -1), bias, **common)
        is KimiK3RouteSortDispatch.UNSUPPORTED
    )
    assert (
        kimi_k3_route_sort_dispatch(logits, bias.float(), **common)
        is KimiK3RouteSortDispatch.UNSUPPORTED
    )
