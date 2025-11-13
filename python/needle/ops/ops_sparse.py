# needle/ops/ops_sparse.py
"""Sparse matrix operations"""

from typing import Optional
from needle.autograd import TensorOp, Tensor, Value
from needle import init
import needle
from needle.sparse import NDArrayInt32, SparseCOO2D
from needle.autograd import array_api, NDArray, default_device


# [SPARSE] Sparse-Dense Matrix Multiplication
class SparseDenseMatMul(TensorOp):
    """Multiply sparse COO matrix by dense matrix: sparse @ dense"""
    
    def compute(self, sparse_tensor, dense_tensor):
        """
        Args:
            sparse_tensor: Tensor with SparseCOO2D cached_data
            dense_tensor: Dense Tensor/NDArray
        """
        # Get the actual data
        if isinstance(sparse_tensor, SparseCOO2D):
            sparse = sparse_tensor
        else:
            # It's a cached_data from Tensor
            sparse = sparse_tensor
        
        # Get dense array
        if isinstance(dense_tensor, NDArray):
            dense = dense_tensor
        else:
            dense = dense_tensor
        
        assert sparse.shape[1] == dense.shape[0], f"Shape mismatch: {sparse.shape} @ {dense.shape}"
        
        out_shape = (sparse.shape[0], dense.shape[1])
        out = array_api.empty(out_shape, device=dense.device)
        
        # Call backend
        if dense.device.name == "cpu":
            dense.device.sparse_dense_matmul(
                sparse.row_indices._handle,
                sparse.col_indices._handle,
                sparse.values._handle,
                dense._handle,
                out._handle,
                sparse.shape[0],
                sparse.shape[1],
                dense.shape[1]
            )
        else:
            raise NotImplementedError("Only CPU backend supported")
        
        return out
    
    def gradient(self, out_grad, node):
        """
        Compute gradients for sparse @ dense
        
        d(loss)/d(dense) = sparse.T @ out_grad
        """
        sparse_tensor, dense_tensor = node.inputs
        
        # Gradient w.r.t. dense: sparse^T @ out_grad
        # For simplicity, convert sparse to dense for backward pass
        sparse_data = sparse_tensor.realize_cached_data()
        dense_sparse_ndarray = sparse_data.to_dense()
        
        # Wrap NDArray in Tensor to use transpose
        dense_sparse_tensor = Tensor.make_const(dense_sparse_ndarray)
        
        grad_dense = needle.ops.MatMul()(dense_sparse_tensor.transpose(), out_grad)
        
        # Gradient w.r.t sparse values not implemented yet (return None)
        return None, grad_dense


# [SPARSE] Convert sparse to dense
class SparseToDense(TensorOp):
    """Convert sparse COO tensor to dense tensor"""
    
    def compute(self, sparse_tensor):
        from needle.sparse import SparseCOO2D
        
        if isinstance(sparse_tensor, SparseCOO2D):
            sparse = sparse_tensor
        else:
            sparse = sparse_tensor
        
        return sparse.to_dense()
    
    def gradient(self, out_grad, node):
        # Gradient is just the out_grad (identity operation)
        # But we don't support gradients for sparse values yet
        return None


def sparse_dense_matmul(sparse_tensor, dense_tensor):
    """Convenience function for sparse @ dense"""
    return SparseDenseMatMul()(sparse_tensor, dense_tensor)


def sparse_to_dense(sparse_tensor):
    """Convert sparse tensor to dense"""
    return SparseToDense()(sparse_tensor)