"""Verbatim copy of student ``onpolicy/algorithms/utils/util.py`` —
subset used by the MAPPO no-attention path only.
"""

from __future__ import annotations

import numpy as np
import torch


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def check(input):
    return torch.from_numpy(input) if type(input) is np.ndarray else input


def calculate_conv_params(input_size):
    height, width, channels = input_size
    if height > 100 or width > 100:
        kernel_size = 5
    else:
        kernel_size = 3
    stride = 1
    padding = (kernel_size - 1) // 2
    return kernel_size, stride, padding
