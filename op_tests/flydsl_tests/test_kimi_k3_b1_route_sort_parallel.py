# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.kimi_k3_moe_route_parallel import (
    kimi_k3_b1_route_sort_parallel,
    supports_kimi_k3_b1_route_sort_parallel,
)
from aiter.ops.flydsl.utils import is_flydsl_available


def _gfx950_flydsl_available() -> bool:
    return (
        torch.cuda.is_available()
        and is_flydsl_available()
        and get_gfx_runtime() == "gfx950"
    )


def test_parallel_route_support_predicate_fails_closed_on_cpu():
    hidden = torch.empty((1, 3584), dtype=torch.bfloat16)
    logits = torch.empty((1, 896), dtype=torch.float32)
    bias = torch.empty(896, dtype=torch.bfloat16)

    assert not supports_kimi_k3_b1_route_sort_parallel(
        hidden,
        logits,
        bias,
        model_dim=3584,
    )


def _make_case(case: str) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(20260729)
    logits = torch.randn((1, 896), generator=generator)
    bias = (
        torch.empty(896, dtype=torch.float32)
        .uniform_(-0.125, 0.125, generator=generator)
        .to(torch.bfloat16)
    )
    if case == "all_equal":
        logits.zero_()
        bias.zero_()
    elif case == "repeating_bias":
        logits.zero_()
        bias = (
            torch.arange(896, dtype=torch.int32)
            .remainder_(8)
            .to(torch.float32)
            .sub_(4.0)
            .mul_(0.03125)
            .to(torch.bfloat16)
        )
    elif case == "partition_boundary_ties":
        logits.fill_(-5.0)
        tied = torch.tensor(
            [
                0,
                13,
                14,
                63,
                64,
                127,
                128,
                255,
                256,
                447,
                448,
                511,
                512,
                767,
                768,
                895,
            ],
            dtype=torch.long,
        )
        logits[0, tied] = 5.0
        bias.zero_()
    return logits.cuda(), bias.cuda()


def _expected_metadata(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sentinel = (16 << 24) | 1
    sorted_ids = torch.full(
        (16 * 32,),
        sentinel,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    sorted_weights = torch.zeros(
        16 * 32,
        dtype=torch.float32,
        device=topk_ids.device,
    )
    sorted_expert_ids, slots = torch.sort(topk_ids.flatten())
    rows = torch.arange(16, device=topk_ids.device) * 32
    sorted_ids[rows] = slots.to(torch.int32) << 24
    sorted_weights[rows] = topk_weights.flatten()[slots]
    return sorted_ids, sorted_weights, sorted_expert_ids


@pytest.mark.skipif(
    not _gfx950_flydsl_available(),
    reason="requires FlyDSL on gfx950",
)
@pytest.mark.parametrize(
    "case",
    [
        "random_nonzero_bias",
        "all_equal",
        "repeating_bias",
        "partition_boundary_ties",
    ],
)
def test_parallel_route_matches_exact_metadata_and_quantization(case: str):
    logits, bias = _make_case(case)
    hidden = torch.randn(
        (1, 3584),
        dtype=torch.bfloat16,
        device="cuda",
    )
    expected_weights = torch.empty(
        (1, 16),
        dtype=torch.float32,
        device="cuda",
    )
    expected_ids = torch.empty(
        (1, 16),
        dtype=torch.int32,
        device="cuda",
    )
    aiter.biased_grouped_topk_hip(
        logits,
        bias.float(),
        expected_weights,
        expected_ids,
        1,
        1,
        True,
        1.0,
    )
    (
        expected_sorted_ids,
        expected_sorted_weights,
        expected_sorted_expert_ids,
    ) = _expected_metadata(expected_weights, expected_ids)
    expected_quantized, expected_scales = aiter.fused_dynamic_mxfp8_quant_moe_sort(
        hidden,
        sorted_ids=expected_sorted_ids,
        num_valid_ids=torch.tensor(
            [512, 1],
            dtype=torch.int32,
            device="cuda",
        ),
        token_num=1,
        topk=16,
        block_size=32,
        sorted_weights=expected_sorted_weights,
    )

    (
        topk_weights,
        topk_ids,
        sorted_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        moe_buf,
        quantized_hidden,
        quantized_scales,
    ) = kimi_k3_b1_route_sort_parallel(
        hidden,
        logits,
        bias,
        model_dim=3584,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(topk_ids, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        topk_weights,
        expected_weights,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(sorted_ids, expected_sorted_ids, rtol=0, atol=0)
    torch.testing.assert_close(
        sorted_weights,
        expected_sorted_weights,
        rtol=1e-6,
        atol=1e-7,
    )
    torch.testing.assert_close(
        sorted_expert_ids,
        expected_sorted_expert_ids,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        num_valid_ids,
        torch.tensor([512, 1], dtype=torch.int32, device="cuda"),
    )
    assert torch.count_nonzero(moe_buf).item() == 0
    torch.testing.assert_close(
        quantized_hidden.view(torch.uint8),
        expected_quantized.view(torch.uint8),
        rtol=0,
        atol=0,
    )

    active_scale_offsets = torch.tensor(
        [
            rank * 112 * 32
            + (group // 8) * 256
            + (group % 4) * 64
            + ((group % 8) // 4) * 2
            for rank in range(16)
            for group in range(112)
        ],
        dtype=torch.int64,
        device="cuda",
    )
    torch.testing.assert_close(
        quantized_scales.view(torch.uint8).flatten()[active_scale_offsets],
        expected_scales.view(torch.uint8).flatten()[active_scale_offsets],
        rtol=0,
        atol=0,
    )
