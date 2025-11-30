from typing import Optional, Any, Union
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        Z -= Z.max(len(Z.shape)-1, keepdims=True).broadcast_to(Z.shape)
        return Z - array_api.log(array_api.exp(Z).sum(len(Z.shape)-1, keepdims=True).broadcast_to(Z.shape))
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        y = node.realize_cached_data()  # LogSoftmax output in forward
        y = node  # need to recompute in Tensor ops

        Z = node.inputs[0]
        log_softmax = node  # output tensor y
        softmax = exp(log_softmax)
        sum_grad = summation(out_grad, axes=(1,))
        sum_grad = reshape(sum_grad, (out_grad.shape[0], 1))
        grad = out_grad - broadcast_to(sum_grad, out_grad.shape) * softmax
        return grad
        ### END YOUR SOLUTION


def logsoftmax(a: Tensor) -> Tensor:
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None) -> None:
        if isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

    def compute(self, Z: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        self.max_Z = Z.max(axis=self.axes, keepdims=True)
        diff = Z - self.max_Z.broadcast_to(Z.shape)
        e = array_api.exp(diff)
        se = array_api.sum(e, axis=self.axes)
        lse = array_api.log(se)
        return lse + self.max_Z.reshape(lse.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        shape = list(Z.shape)
        axes = range(len(shape)) if self.axes is None else self.axes
        for ax in axes:
            shape[ax] = 1
        diff = Z - self.max_Z.broadcast_to(Z.shape)
        softmax = exp(diff) / summation(exp(diff), axes=self.axes).reshape(shape).broadcast_to(Z.shape)
        out_grad = out_grad.reshape(shape).broadcast_to(Z.shape)
        return out_grad * softmax
        ### END YOUR SOLUTION


def logsumexp(a: Tensor, axes: Optional[tuple] = None) -> Tensor:
    return LogSumExp(axes=axes)(a)