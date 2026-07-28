# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression coverage for the Kimi-K3 gfx950 B1 stage-1 configuration."""

import csv
from pathlib import Path

from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

STAGE1_KERNEL = "flydsl_moe1_abf16_wfp4_bf16_t16x64x256_w3_xcd4"
CONFIG_PATH = (
    Path(__file__).parents[2] / "aiter" / "configs" / "model_configs" / "kimik3_fp4_tuned_fmoe.csv"
)


def test_kimi_k3_b1_stage1_variant_is_narrowly_registered():
    params = get_flydsl_kernel_params(STAGE1_KERNEL)
    assert params is not None
    assert params["stage"] == 1
    assert params["a_dtype"] == "bf16"
    assert params["b_dtype"] == "fp4"
    assert params["out_dtype"] == "bf16"
    assert params["tile_m"] == 16
    assert params["tile_n"] == 64
    assert params["tile_k"] == 256
    assert params["gate_mode"] == "separated"


def test_kimi_k3_b1_selects_stage1_variant_through_tuned_config():
    with CONFIG_PATH.open(newline="", encoding="utf-8") as config_file:
        rows = list(csv.DictReader(config_file))

    matching = [
        row
        for row in rows
        if row["gfx"] == "gfx950"
        and row["cu_num"] == "256"
        and row["token"] == "1"
        and row["model_dim"] == "3584"
        and row["inter_dim"] == "384"
        and row["expert"] == "896"
        and row["topk"] == "16"
    ]
    assert len(matching) == 1
    assert matching[0]["kernelName1"] == STAGE1_KERNEL

    other_batches = [
        row
        for row in rows
        if row["gfx"] == "gfx950"
        and row["model_dim"] == "3584"
        and row["inter_dim"] == "384"
        and row["expert"] == "896"
        and row["topk"] == "16"
        and row["token"] != "1"
    ]
    assert all(row["kernelName1"] != STAGE1_KERNEL for row in other_batches)
