// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#include "rocm_ops.hpp"
#include "torch/sla_fwd.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    SLA_FWD_PYBIND;
}
