# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression coverage for the Kimi-K3 gfx950 B1 stage-2 configuration."""

import csv
from pathlib import Path

STAGE2_KERNEL = "flydsl_moe2_abf16_wfp4_bf16_t32x128x128_atomic_bnt2"
CONFIG_PATH = (
    Path(__file__).parents[2] / "aiter" / "configs" / "model_configs" / "kimik3_fp4_tuned_fmoe.csv"
)


def test_kimi_k3_b1_selects_atomic_stage2_through_tuned_config():
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
    assert matching[0]["kernelName2"] == STAGE2_KERNEL

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
    assert all(row["kernelName2"] != STAGE2_KERNEL for row in other_batches)
