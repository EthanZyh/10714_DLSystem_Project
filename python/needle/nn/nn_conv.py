"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module
import math


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        self.device = device
        self.dtype = dtype
        self.use_bias = bias
        self.padding = (kernel_size - 1) // 2

        weight_shape = (kernel_size, kernel_size, in_channels, out_channels)
        fan_in = in_channels * kernel_size * kernel_size
        fan_out = out_channels * kernel_size * kernel_size

        self.weight = Parameter(
            init.kaiming_uniform(
                fan_in,
                fan_out,
                shape=weight_shape,
                device=device,
                dtype=dtype,
            )
        )

        if bias:
            bound = 1.0 / math.sqrt(fan_in)
            b = init.rand(out_channels, device=device, dtype=dtype) * (2 * bound) - bound
            self.bias = Parameter(b)
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(x, (1, 2))   # (N, H, C, W)
        x = ops.transpose(x, (2, 3))   # (N, H, W, C)
        out = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            b = ops.reshape(self.bias, (1, 1, 1, self.out_channels))
            b = ops.broadcast_to(b, out.shape)
            out = out + b
        out = ops.transpose(out, (2, 3))  # (N, H, C, W)
        out = ops.transpose(out, (1, 2))  # (N, C, H, W)
        return out
        ### END YOUR SOLUTION