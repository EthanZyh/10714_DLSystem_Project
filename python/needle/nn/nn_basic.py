"""The module.
"""
from typing import Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> list[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> list["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self) -> None:
        self.training = True

    def parameters(self) -> list[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> list["Module"]:
        return _child_modules(self.__dict__)

    def eval(self) -> None:
        self.training = False
        for m in self._children():
            m.training = False

    def train(self) -> None:
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
            init.kaiming_uniform(fan_in=in_features, fan_out=out_features, device=device, dtype=dtype)
        )
        if bias:
            self.bias = Parameter(
                init.kaiming_uniform(fan_in=out_features, fan_out=1, device=device, dtype=dtype).transpose()
            )
        else:
            self.bias = None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        y = X @ self.weight
        if self.bias is not None:
            y += ops.broadcast_to(self.bias, y.shape)
        return y
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch_size = X.shape[0]
        feat_dim = 1
        for d in X.shape[1:]:
            feat_dim *= d
        return ops.reshape(X, (batch_size, feat_dim))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, C = logits.shape
        log_norm = ops.logsumexp(logits, axes=(1,))
        y_encoded = init.one_hot(C, y, device=logits.device, dtype=logits.dtype)
        correct_scores = ops.summation(logits * y_encoded, axes=(1,)) 
        loss = ops.summation(log_norm - correct_scores) / N
        return loss
        ### END YOUR SOLUTION


class MaskedSoftmaxLoss(Module):
    """
    Masked softmax loss for handling multiple masks.
    """
    def forward(self, logits: Tensor, y: Tensor, masks: tuple[Tensor]) -> Tensor:
        N, C = logits.shape
        log_norm = ops.logsumexp(logits, axes=(1,))
        y_encoded = init.one_hot(C, y, device=logits.device, dtype=logits.dtype)
        correct_scores = ops.summation(logits * y_encoded, axes=(1,)) 
        losses = log_norm - correct_scores
        return tuple(ops.summation(losses * mask) / ops.summation(mask) for mask in masks)


class BatchNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.1, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        self.running_mean = init.zeros(dim, device=device, dtype=dtype, requires_grad=False)
        self.running_var = init.ones(dim, device=device, dtype=dtype, requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        N, D = x.shape

        if self.training:
            mean = ops.summation(x, axes=(0,)) / N
            mean_b = ops.broadcast_to(mean.reshape((1, D)), x.shape)
            centered = x - mean_b
            var = ops.summation(centered ** 2, axes=(0,)) / N
            var_b = ops.broadcast_to(var.reshape((1, D)), x.shape)
            std = (var_b + self.eps) ** 0.5
            x_hat = centered / std
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.data
        else:
            mean = self.running_mean
            var = self.running_var
            mean_b = ops.broadcast_to(mean.reshape((1, D)), x.shape)
            var_b = ops.broadcast_to(var.reshape((1, D)), x.shape)
            x_hat = (x - mean_b) / (var_b + self.eps) ** 0.5

        w = ops.broadcast_to(self.weight.reshape((1, D)), x.shape)
        b = ops.broadcast_to(self.bias.reshape((1, D)), x.shape)
        return w * x_hat + b
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float = 1e-5, device: Any | None = None, dtype: str = "float32") -> None:
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, device=device, dtype=dtype, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, device=device, dtype=dtype, requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch, features = x.shape
        mean = ops.summation(x, axes=(1,)) / features
        mean = ops.reshape(mean, (batch, 1)).broadcast_to(x.shape)
        var = ops.summation((x - mean) ** 2, axes=(1,)) / features
        var = ops.reshape(var, (batch, 1)).broadcast_to(x.shape)
        norm = (x - mean) / (var + self.eps) ** 0.5
        w = ops.reshape(self.weight, (1, features)).broadcast_to(x.shape)
        b = ops.reshape(self.bias, (1, features)).broadcast_to(x.shape)
        return w * norm + b
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=1 - self.p, device=x.device, dtype="float32")
            return x * mask / (1 - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION