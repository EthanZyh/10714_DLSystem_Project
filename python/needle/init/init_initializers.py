import math
from .init_basic import *
from typing import Any


def xavier_uniform(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    shape = kwargs.pop("shape", (fan_in, fan_out))
    bound = gain * math.sqrt(6.0 / (fan_in + fan_out))
    return rand(*shape, **kwargs) * (2 * bound) - bound
    ### END YOUR SOLUTION


def xavier_normal(fan_in: int, fan_out: int, gain: float = 1.0, **kwargs: Any) -> "Tensor":
    ### BEGIN YOUR SOLUTION
    shape = kwargs.pop("shape", (fan_in, fan_out))
    std = gain * math.sqrt(2.0 / (fan_in + fan_out))
    return randn(*shape, **kwargs) * std
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    shape = kwargs.pop("shape", (fan_in, fan_out))
    bound = math.sqrt(6.0 / fan_in)
    return rand(*shape, **kwargs) * (2 * bound) - bound
    ### END YOUR SOLUTION

def kaiming_normal(fan_in: int, fan_out: int, nonlinearity: str = "relu", **kwargs: Any) -> "Tensor":
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    shape = kwargs.pop("shape", (fan_in, fan_out))
    std = math.sqrt(2.0 / fan_in)
    return randn(*shape, **kwargs) * std
    ### END YOUR SOLUTION