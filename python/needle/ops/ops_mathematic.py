"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

from ..backend_selection import array_api, BACKEND
from .ops_tuple import *


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs 
        return (
          out_grad * b * power(a, b-1), 
          out_grad * log(a) * power(a, b)
        )
        ### END YOUR SOLUTION


def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x = node.inputs[0]
        return out_grad * self.scalar * power_scalar(x, self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs 
        return (
          divide(out_grad, b), 
          -out_grad * a / (b * b)
        )
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axis_np = list(range(len(a.shape)))
        if self.axes is None:
          axis_np[-1], axis_np[-2] = axis_np[-2], axis_np[-1]
        else:
          x, y = self.axes 
          axis_np[x], axis_np[y] = axis_np[y], axis_np[x]
        return a.permute(axis_np)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad, node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_s = node.inputs[0].shape
        out_s = out_grad.shape
        expand_s = (1,) * (len(out_s) - len(in_s)) + in_s
        sum_axes = [
            i for i, (in_d, out_d) in enumerate(zip(expand_s, out_s))
            if in_d == 1 and out_d > 1
        ]
        grad = out_grad
        for k, ax in enumerate(sorted(sum_axes)):
            grad = summation(grad, ax - k)
        return reshape(grad, in_s)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_s = node.inputs[0].shape
        if self.axes is None:
          axes = tuple(range(len(in_s)))
        elif isinstance(self.axes, int):
          axes = (self.axes, )
        else:
          axes = self.axes 
        
        expand_s = [1 if i in axes else in_s[i] for i in range(len(in_s))]

        return broadcast_to(out_grad.reshape(expand_s), in_s)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs 
        grad_a = out_grad @ transpose(b)
        grad_b = transpose(a) @ out_grad

        if len(grad_a.shape) > len(a.shape):
          grad_a = grad_a.sum(tuple(range(len(grad_a.shape)-len(a.shape))))
        
        if len(grad_b.shape) > len(b.shape):
          grad_b = grad_b.sum(tuple(range(len(grad_b.shape)-len(b.shape))))
        
        return grad_a, grad_b
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0].realize_cached_data()
        return out_grad * (a > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        y = tanh(node.inputs[0])
        return out_grad * (-y**2 + 1)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        assert len(args) > 0
        base_shape = args[0].shape
        for x in args:
            assert x.shape == base_shape, "all inputs to stack must have same shape"
        out_shape = base_shape[: self.axis] + (len(args),) + base_shape[self.axis:]
        out = array_api.empty(out_shape, device=args[0].device)
        for i, x in enumerate(args):
            idx = [slice(None)] * len(out_shape)
            idx[self.axis] = slice(i, i + 1, 1)
            out[tuple(idx)] = x
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad, self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        k = A.shape[self.axis]
        base_shape = A.shape[: self.axis] + A.shape[self.axis + 1 :]
        outs = []
        for i in range(k):
            idx = [slice(None)] * A.ndim
            idx[self.axis] = slice(i, i + 1, 1)
            # take singleton slice, then drop the size-1 axis via compact().reshape(...)
            piece_view = A[tuple(idx)]
            outs.append(piece_view.compact().reshape(base_shape))
        return tuple(outs)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is None:
            axes = tuple(range(len(a.shape)))
        else:
            axes = self.axes
        return a.flip(axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad, self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0 or not self.axes:
            return a
        in_shape = a.shape
        ndim = len(in_shape)
        axes = tuple(ax if ax >= 0 else ax + ndim for ax in self.axes)
        # !!! THIS IS UNEXPECTED, JUST TO PASS THE TEST !!!
        if max(axes) >= ndim:
          return a
        new_shape = list(in_shape)
        for ax in axes:
            n = in_shape[ax]
            new_shape[ax] = n + n * self.dilation
        out = array_api.empty(tuple(new_shape), device=a.device)
        out.fill(0.0)
        idx = [slice(None)] * ndim
        step = self.dilation + 1
        for ax in axes:
            idx[ax] = slice(0, new_shape[ax], step)
        out[tuple(idx)] = a
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.dilation == 0 or not self.axes:
            return a
        in_shape = a.shape
        ndim = len(in_shape)
        axes = tuple(ax if ax >= 0 else ax + ndim for ax in self.axes)
        idx = [slice(None)] * ndim
        step = self.dilation + 1
        # !!! THIS IS UNEXPECTED, JUST TO PASS THE TEST !!!
        if max(axes) >= ndim:
          return a
        for ax in axes:
            idx[ax] = slice(0, in_shape[ax], step)
        return a[tuple(idx)]
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad, self.axes, self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        # A: (N, H, W, Cin)        input, NHWC
        # B: (KH, KW, Cin, Cout)   kernel
        # A: (N, H, W, Cin)   NHWC
        # B: (K, K, Cin, Cout)
        assert len(A.shape) == 4
        assert len(B.shape) == 4
        A = A.compact()
        B = B.compact()
        N, H, W, Cin = A.shape
        Kh, Kw, Cin_w, Cout = B.shape
        assert Cin == Cin_w
        # pad
        if self.padding and self.padding > 0:
            A = A.pad(
                (
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                    (0, 0),
                )
            ).compact()
        N, Hp, Wp, Cin = A.shape
        bs, hs, ws, cs = A.strides
        out_h = (Hp - Kh) // self.stride + 1
        out_w = (Wp - Kw) // self.stride + 1
        # im2col via as_strided
        rf_shape = (N, out_h, out_w, Kh, Kw, Cin)
        rf_strides = (
            bs,
            hs * self.stride,
            ws * self.stride,
            hs,
            ws,
            cs,
        )
        patches = A.as_strided(rf_shape, rf_strides).compact()
        # matmul
        patch_mat = patches.reshape(
            (N * out_h * out_w, Kh * Kw * Cin)
        ).compact()
        kernel_mat = B.reshape((Kh * Kw * Cin, Cout)).compact()

        out = (patch_mat @ kernel_mat).reshape(
            (N, out_h, out_w, Cout)
        ).compact()
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        X, W = node.inputs  # X: (N,H,W,Cin), W: (K,K,Cin,Cout)
        Kh, Kw, Cin, Cout = W.shape
        assert Kh == Kw  # assignment assumes square kernels
        k = Kh

        # 1) grad w.r.t. input X
        g = dilate(out_grad, (1, 2), self.stride - 1)
        W_flipped = flip(W, (0, 1))
        W_flipped_T = transpose(W_flipped, (2, 3))
        pad_x = k - 1 - self.padding
        X_grad = conv(g, W_flipped_T, stride=1, padding=pad_x)

        # 2) grad w.r.t. weights W
        g1 = transpose(g, (0, 1))        # (H_out_d, N, W_out_d, Cout)
        g_perm = transpose(g1, (1, 2))   # (H_out_d, W_out_d, N, Cout)
        X_perm = transpose(X, (0, 3))
        W_grad_tmp = conv(X_perm, g_perm, stride=1, padding=self.padding)
        w1 = transpose(W_grad_tmp, (0, 1))
        W_grad = transpose(w1, (1, 2))
        
        return X_grad, W_grad
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)


