# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def _communicator(**overrides):
    values = {
        "disabled": False,
        "_ops_all_reduce_dual": object(),
        "_car_min_size": 0,
        "_car_max_size": 64 * 1024 * 1024,
        "world_size": 8,
        "fully_connected": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_should_custom_ar_dual_accepts_kimi_k3_boundary():
    communicator = _communicator()
    left = torch.empty((1, 3584), dtype=torch.bfloat16)
    right = torch.empty((1, 7168), dtype=torch.bfloat16)

    assert CustomAllreduce.should_custom_ar_dual(communicator, left, right)


@pytest.mark.parametrize(
    "left,right",
    [
        (
            torch.empty(0, dtype=torch.bfloat16),
            torch.empty(8, dtype=torch.bfloat16),
        ),
        (
            torch.empty(8, dtype=torch.bfloat16),
            torch.empty(8, dtype=torch.float16),
        ),
        (
            torch.empty(8, dtype=torch.int32),
            torch.empty(8, dtype=torch.int32),
        ),
        (
            torch.empty(7, dtype=torch.bfloat16),
            torch.empty(8, dtype=torch.bfloat16),
        ),
        (
            torch.empty((8, 16), dtype=torch.bfloat16)[:, ::2],
            torch.empty(8, dtype=torch.bfloat16),
        ),
    ],
)
def test_should_custom_ar_dual_rejects_unsupported_inputs(left, right):
    communicator = _communicator()

    assert not CustomAllreduce.should_custom_ar_dual(communicator, left, right)


def test_should_custom_ar_dual_rejects_transport_and_size_mismatches():
    left = torch.empty(8, dtype=torch.bfloat16)
    right = torch.empty(8, dtype=torch.bfloat16)

    assert not CustomAllreduce.should_custom_ar_dual(
        _communicator(disabled=True), left, right
    )
    assert not CustomAllreduce.should_custom_ar_dual(
        _communicator(_ops_all_reduce_dual=None), left, right
    )
    assert not CustomAllreduce.should_custom_ar_dual(
        _communicator(fully_connected=False), left, right
    )
    assert not CustomAllreduce.should_custom_ar_dual(
        _communicator(_car_min_size=32), left, right
    )

    left_at_limit = torch.empty(20 * 1024, dtype=torch.float32)
    assert not CustomAllreduce.should_custom_ar_dual(
        _communicator(), left_at_limit, right
    )


def test_all_reduce_dual_passes_explicit_staging_contract():
    calls = []

    def op(*args):
        calls.append(args)

    staging = SimpleNamespace(data_ptr=0x1234, max_size=64 * 1024 * 1024)
    communicator = _communicator(
        _ops_all_reduce_dual=op,
        _ptr=0x5678,
        _pool={"input": staging},
    )
    left = torch.empty((1, 3584), dtype=torch.bfloat16)
    right = torch.empty((1, 7168), dtype=torch.bfloat16)

    left_out, right_out = CustomAllreduce.all_reduce_dual(
        communicator,
        left,
        right,
    )

    assert left_out.shape == left.shape
    assert right_out.shape == right.shape
    assert calls == [
        (
            communicator._ptr,
            left,
            right,
            left_out,
            right_out,
            staging.data_ptr,
            staging.max_size,
        )
    ]
