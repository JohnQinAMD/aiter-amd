# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce


def test_custom_all_reduce_forwards_caller_owned_output():
    calls = []
    output = torch.empty(8, dtype=torch.bfloat16)

    def all_reduce(input_, **kwargs):
        calls.append((input_, kwargs))
        return kwargs["out"]

    communicator = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda input_: True,
        _IS_CAPTURING=False,
        all_reduce=all_reduce,
        _validate_all_reduce_output=lambda input_, out: None,
    )
    input_ = torch.empty_like(output)

    result = CustomAllreduce.custom_all_reduce(
        communicator,
        input_,
        out=output,
    )

    assert result is output
    assert calls == [
        (
            input_,
            {
                "out": output,
                "use_new": True,
                "open_fp8_quant": False,
                "registered_input": False,
            },
        )
    ]


def test_custom_all_reduce_warmup_preserves_and_zeros_output(monkeypatch):
    output = torch.full((8,), 7.0, dtype=torch.bfloat16)
    communicator = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda input_: True,
        _IS_CAPTURING=True,
        _validate_all_reduce_output=lambda input_, out: None,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    result = CustomAllreduce.custom_all_reduce(
        communicator,
        torch.empty_like(output),
        out=output,
    )

    assert result is output
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_custom_all_reduce_rejects_without_touching_output():
    output = torch.full((8,), 7.0, dtype=torch.bfloat16)
    communicator = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda input_: False,
    )

    result = CustomAllreduce.custom_all_reduce(
        communicator,
        torch.empty_like(output),
        out=output,
    )

    assert result is None
    torch.testing.assert_close(output, torch.full_like(output, 7.0))


@pytest.mark.parametrize("capturing", [False, True])
def test_custom_all_reduce_rejects_mismatched_output(monkeypatch, capturing):
    communicator = SimpleNamespace(
        disabled=False,
        should_custom_ar=lambda input_: True,
        _IS_CAPTURING=capturing,
        _validate_all_reduce_output=CustomAllreduce._validate_all_reduce_output,
    )
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: False,
    )

    with pytest.raises(ValueError, match="custom allreduce output"):
        CustomAllreduce.custom_all_reduce(
            communicator,
            torch.empty(8, dtype=torch.bfloat16),
            out=torch.empty(7, dtype=torch.bfloat16),
        )


def test_custom_all_reduce_rejects_aliased_output():
    tensor = torch.empty(8, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must not alias"):
        CustomAllreduce._validate_all_reduce_output(tensor, tensor)
