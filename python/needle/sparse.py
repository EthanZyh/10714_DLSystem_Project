# needle/sparse.py
import numpy as np
from typing import Tuple
from .backend_ndarray import NDArray, default_device


# [SPARSE] Integer NDArray for sparse indices
class NDArrayInt32:
    """Device-native int32 array for storing sparse indices"""
    
    def __init__(self, shape, device=None):
        """Create integer array on device"""
        device = device if device else default_device()
        self._init(shape, device=device)
    
    def _init(self, shape, device):
        self._shape = tuple(shape) if isinstance(shape, (list, tuple)) else (shape,)
        self._device = device
        
        # Allocate on device backend
        size = int(np.prod(self._shape))
        if device.name == "cpu":
            self._handle = device.ArrayInt32(size)
        else:
            raise NotImplementedError("Only CPU backend supported for now")
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def device(self):
        return self._device
    
    @property
    def size(self):
        return int(np.prod(self._shape))
    
    def numpy(self):
        """Convert to numpy int32 array"""
        if self._device.name == "cpu":
            return self._device.to_numpy_int32(self._handle, list(self._shape))
        else:
            raise NotImplementedError()
    
    @staticmethod
    def make(shape, device):
        """Create empty int32 array"""
        return NDArrayInt32(shape, device=device)
    
    @staticmethod
    def make_from_numpy(numpy_array, device):
        """Create from numpy int32 array"""
        numpy_array = np.array(numpy_array, dtype=np.int32)
        arr = NDArrayInt32(numpy_array.shape, device=device)
        
        if device.name == "cpu":
            device.from_numpy_int32(
                np.ascontiguousarray(numpy_array), 
                arr._handle
            )
        else:
            raise NotImplementedError()
        
        return arr

# [SPARSE] COO Sparse Matrix
class SparseCOO2D:
    """Sparse matrix in COO format with device-native storage"""
    
    def __init__(self, row_indices, col_indices, values, shape):
        """
        Args:
            row_indices: NDArrayInt32 (device integer array)
            col_indices: NDArrayInt32 (device integer array)
            values: NDArray (device float array)
            shape: (nrows, ncols)
        """
        # Import here to avoid circular dependency
        
        assert isinstance(row_indices, NDArrayInt32), "row_indices must be NDArrayInt32"
        assert isinstance(col_indices, NDArrayInt32), "col_indices must be NDArrayInt32"
        assert isinstance(values, NDArray), "values must be NDArray"
        
        assert row_indices.shape == col_indices.shape == values.shape, "Indices and values must have same shape"
        assert row_indices.device == col_indices.device == values.device, "All arrays must be on same device"
        
        self.row_indices = row_indices
        self.col_indices = col_indices
        self.values = values
        self.shape = shape
        self.nnz = values.size
        self.device = values.device
    
    @staticmethod
    def make(row_np, col_np, val_np, shape, device=None):
        """Create SparseCOO2D from numpy arrays"""
        device = device if device else default_device()
        
        row_indices = NDArrayInt32.make_from_numpy(row_np, device)
        col_indices = NDArrayInt32.make_from_numpy(col_np, device)
        values = NDArray(val_np, device=device)
        
        return SparseCOO2D(row_indices, col_indices, values, shape)
    
    def to_dense(self):
        """Convert to dense NDArray"""
        out = NDArray.make(self.shape, device=self.device)
        
        if self.device.name == "cpu":
            self.device.sparse_to_dense(
                self.row_indices._handle,
                self.col_indices._handle,
                self.values._handle,
                out._handle,
                self.shape[0],
                self.shape[1]
            )
        else:
            raise NotImplementedError()
        
        return out
    
    def __repr__(self):
        return f"SparseCOO2D(shape={self.shape}, nnz={self.nnz}, device={self.device})\n" \
                f"{self.to_dense().numpy()}"