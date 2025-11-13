# coo_demo.py

import sys 
sys.path.append('./python')

import numpy as np

import needle as ndl 
from needle.sparse import SparseCOO2D
from needle.autograd import SparseTensor


def test_coo_from_numpy():
    """Test creating COO matrix from numpy arrays"""
    # Matrix:
    # [[1. 0. 2.]
    #  [0. 3. 0.]
    #  [0. 0. 4.]]
    print("===== Testing COO from numpy =====")
    rows = np.array([0, 0, 1, 2], dtype=np.int32)
    cols = np.array([0, 2, 1, 2], dtype=np.int32)
    vals = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shape = (3, 3)
    
    coo = SparseCOO2D.make(rows, cols, vals, shape, device=ndl.cpu())
    print(f"  Shape: {coo.shape}")
    print(f"  NNZ: {coo.nnz}")
    print(f"  Device: {coo.device}")
    print(f"  COO: {coo}")
    print()


def test_coo_to_dense():
    """Test converting sparse to dense"""
    print("===== Testing COO to Dense =====")
    
    # Identity-like matrix
    rows = np.array([0, 1, 2], dtype=np.int32)
    cols = np.array([0, 1, 2], dtype=np.int32)
    vals = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    shape = (3, 3)
    
    coo = SparseCOO2D.make(rows, cols, vals, shape, device=ndl.cpu())
    dense = coo.to_dense()
    
    print("  Sparse COO:")
    print(f"    rows: {rows}")
    print(f"    cols: {cols}")
    print(f"    vals: {vals}")
    print("  Dense result:")
    print(dense.numpy())
    
    # Verify correctness
    expected = np.diag([1.0, 2.0, 3.0])
    assert np.allclose(dense.numpy(), expected), "to_dense() failed!"
    print("  ✓ Conversion correct!")
    print()


def test_sparse_dense_matmul():
    """Test sparse @ dense matrix multiplication"""
    print("===== Testing Sparse @ Dense Matmul =====")
    
    # Sparse matrix A (3x3):
    # [[1. 0. 2.]
    #  [0. 3. 0.]
    #  [4. 0. 5.]]
    rows_A = np.array([0, 0, 1, 2, 2], dtype=np.int32)
    cols_A = np.array([0, 2, 1, 0, 2], dtype=np.int32)
    vals_A = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    
    # Dense matrix B (3x2)
    B = np.array([[1.0, 2.0],
                  [3.0, 4.0],
                  [5.0, 6.0]], dtype=np.float32)
    
    # Create sparse and dense
    sparse_A = SparseCOO2D.make(rows_A, cols_A, vals_A, (3, 3), device=ndl.cpu())
    dense_B = ndl.NDArray(B, device=ndl.cpu())
    
    # Compute using backend
    result = ndl.NDArray.make((3, 2), device=ndl.cpu())
    ndl.cpu().sparse_dense_matmul(
        sparse_A.row_indices._handle,
        sparse_A.col_indices._handle,
        sparse_A.values._handle,
        dense_B._handle,
        result._handle,
        3, 3, 2
    )
    
    # Expected result: A @ B
    A_dense = sparse_A.to_dense().numpy()
    expected = A_dense @ B
    
    print("  Sparse A (as dense):")
    print(A_dense)
    print("  Dense B:")
    print(B)
    print("  Result (A @ B):")
    print(result.numpy())
    print("  Expected:")
    print(expected)
    
    assert np.allclose(result.numpy(), expected), "Matmul failed!"
    print("  ✓ Matmul correct!")
    print()


def test_sparse_tensor_creation():
    """Test creating SparseTensor"""
    print("===== Testing SparseTensor Creation =====")
    
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [1.0, 2.0, 3.0]
    shape = (3, 3)
    
    sparse = SparseTensor(rows, cols, vals, shape, device=ndl.cpu())
    
    print(f"  SparseTensor: {sparse}")
    print(f"  Shape: {sparse.shape}")
    print(f"  NNZ: {sparse.nnz}")
    print(f"  Device: {sparse.device}")
    print()


def test_sparse_tensor_matmul():
    """Test SparseTensor @ Tensor matmul"""
    print("===== Testing SparseTensor @ Tensor =====")
    
    # Sparse matrix (2x3):
    # [[1. 2. 0.]
    #  [0. 3. 4.]]
    rows = [0, 0, 1, 1]
    cols = [0, 1, 1, 2]
    vals = [1.0, 2.0, 3.0, 4.0]
    
    sparse = SparseTensor(rows, cols, vals, shape=(2, 3), device=ndl.cpu())
    
    # Dense matrix (3x2)
    dense = ndl.Tensor([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]], device=ndl.cpu())
    
    # Multiply
    result = sparse @ dense
    
    # Expected: [[7, 10], [29, 36]]
    # Sparse as dense: [[1, 2, 0], [0, 3, 4]]
    # Result: [1*1 + 2*3, 1*2 + 2*4] = [7, 10]
    #         [0*1 + 3*3 + 4*5, 0*2 + 3*4 + 4*6] = [29, 36]
    
    sparse_dense = sparse.to_dense().numpy()
    expected = sparse_dense @ dense.numpy()
    
    print("  Sparse matrix (as dense):")
    print(sparse_dense)
    print("  Dense matrix:")
    print(dense.numpy())
    print("  Result:")
    print(result.numpy())
    print("  Expected:")
    print(expected)
    
    assert np.allclose(result.numpy(), expected), "SparseTensor matmul failed!"
    print("  ✓ SparseTensor matmul correct!")
    print()


def test_sparse_tensor_to_dense():
    """Test SparseTensor.to_dense()"""
    print("===== Testing SparseTensor.to_dense() =====")
    
    rows = [0, 1, 2, 2]
    cols = [1, 0, 2, 0]
    vals = [5.0, 6.0, 7.0, 8.0]
    shape = (3, 3)
    
    sparse = SparseTensor(rows, cols, vals, shape, device=ndl.cpu())
    dense = sparse.to_dense()
    
    print("  Sparse tensor:")
    print(f"    rows: {rows}")
    print(f"    cols: {cols}")
    print(f"    vals: {vals}")
    print("  Dense result:")
    print(dense.numpy())
    
    # Verify
    expected = np.zeros((3, 3), dtype=np.float32)
    for r, c, v in zip(rows, cols, vals):
        expected[r, c] = v
    
    assert np.allclose(dense.numpy(), expected), "to_dense() failed!"
    print("  ✓ Conversion correct!")
    print()


def test_gradient_computation():
    """Test gradient computation with sparse @ dense"""
    print("===== Testing Gradient Computation =====")
    
    # Sparse matrix (2x2):
    # [[2. 0.]
    #  [0. 3.]]
    rows = [0, 1]
    cols = [0, 1]
    vals = [2.0, 3.0]
    
    sparse = SparseTensor(rows, cols, vals, shape=(2, 2), 
                         device=ndl.cpu(), requires_grad=False)
    
    # Dense matrix (2x2) with requires_grad=True
    dense = ndl.Tensor([[1.0, 2.0],
                        [3.0, 4.0]], 
                       device=ndl.cpu(), requires_grad=True)
    
    # Forward
    result = sparse @ dense
    loss = result.sum()
    
    print("  Sparse matrix (as dense):")
    print(sparse.to_dense().numpy())
    print("  Dense matrix:")
    print(dense.numpy())
    print("  Result (sparse @ dense):")
    print(result.numpy())
    print(f"  Loss (sum): {loss.numpy()}")
    
    # Backward
    loss.backward()
    
    print("  Gradient w.r.t. dense:")
    print(dense.grad.numpy())
    
    # Expected gradient: sparse.T @ ones(2, 2)
    # sparse.T = [[2, 0], [0, 3]]
    # ones = [[1, 1], [1, 1]]
    # grad = [[2, 2], [3, 3]]
    expected_grad = np.array([[2.0, 2.0],
                             [3.0, 3.0]], dtype=np.float32)
    
    print("  Expected gradient:")
    print(expected_grad)
    
    assert np.allclose(dense.grad.numpy(), expected_grad), "Gradient incorrect!"
    print("  ✓ Gradient correct!")
    print()


def test_large_sparse_matrix():
    """Test with larger sparse matrix"""
    print("===== Testing Large Sparse Matrix =====")
    
    # Create random sparse matrix (100x100 with 500 non-zeros)
    np.random.seed(42)
    n = 100
    nnz = 500
    
    rows = np.random.randint(0, n, size=nnz, dtype=np.int32)
    cols = np.random.randint(0, n, size=nnz, dtype=np.int32)
    vals = np.random.randn(nnz).astype(np.float32)
    
    sparse = SparseTensor(rows, cols, vals, shape=(n, n), device=ndl.cpu())
    
    print(f"  Matrix shape: {sparse.shape}")
    print(f"  Number of non-zeros: {sparse.nnz}")
    print(f"  Sparsity: {100 * (1 - sparse.nnz / (n * n)):.1f}%")
    
    # Multiply with small dense matrix
    dense = ndl.Tensor(np.random.randn(n, 5).astype(np.float32), device=ndl.cpu())
    result = sparse @ dense
    
    print(f"  Dense matrix shape: {dense.shape}")
    print(f"  Result shape: {result.shape}")
    
    # Verify shape
    assert result.shape == (n, 5), "Shape mismatch!"
    print("  ✓ Large matrix works!")
    print()


def test_empty_sparse_matrix():
    """Test edge case: empty sparse matrix"""
    print("===== Testing Empty Sparse Matrix =====")
    
    rows = np.array([], dtype=np.int32)
    cols = np.array([], dtype=np.int32)
    vals = np.array([], dtype=np.float32)
    shape = (3, 3)
    
    sparse = SparseTensor(rows, cols, vals, shape, device=ndl.cpu())
    
    print(f"  Empty sparse matrix: {sparse}")
    print(f"  NNZ: {sparse.nnz}")
    
    # Convert to dense (should be all zeros)
    dense = sparse.to_dense()
    print("  Dense result:")
    print(dense.numpy())
    
    expected = np.zeros((3, 3), dtype=np.float32)
    assert np.allclose(dense.numpy(), expected), "Empty sparse failed!"
    print("  ✓ Empty sparse works!")
    print()


def test_single_element():
    """Test sparse matrix with single element"""
    print("===== Testing Single Element =====")
    
    rows = [1]
    cols = [2]
    vals = [42.0]
    shape = (3, 4)
    
    sparse = SparseTensor(rows, cols, vals, shape, device=ndl.cpu())
    dense = sparse.to_dense()
    
    print(f"  Single element at ({rows[0]}, {cols[0]}) = {vals[0]}")
    print("  Dense result:")
    print(dense.numpy())
    
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1, 2] = 42.0
    
    assert np.allclose(dense.numpy(), expected), "Single element failed!"
    print("  ✓ Single element works!")
    print()


def test_vector_multiplication():
    """Test sparse matrix @ vector"""
    print("===== Testing Sparse @ Vector =====")
    
    # Sparse matrix (3x3):
    # [[1. 0. 0.]
    #  [0. 2. 0.]
    #  [0. 0. 3.]]
    rows = [0, 1, 2]
    cols = [0, 1, 2]
    vals = [1.0, 2.0, 3.0]
    
    sparse = SparseTensor(rows, cols, vals, shape=(3, 3), device=ndl.cpu())
    
    # Vector (3x1)
    vec = ndl.Tensor([[4.0], [5.0], [6.0]], device=ndl.cpu())
    
    result = sparse @ vec
    
    print("  Sparse diagonal matrix:")
    print(sparse.to_dense().numpy())
    print("  Vector:")
    print(vec.numpy())
    print("  Result:")
    print(result.numpy())
    
    # Expected: [4, 10, 18]
    expected = np.array([[4.0], [10.0], [18.0]], dtype=np.float32)
    
    assert np.allclose(result.numpy(), expected), "Vector multiplication failed!"
    print("  ✓ Vector multiplication correct!")
    print()


def test_rectangular_sparse():
    """Test rectangular sparse matrices"""
    print("===== Testing Rectangular Sparse Matrix =====")
    
    # 2x4 sparse matrix:
    # [[1. 0. 2. 0.]
    #  [0. 3. 0. 4.]]
    rows = [0, 0, 1, 1]
    cols = [0, 2, 1, 3]
    vals = [1.0, 2.0, 3.0, 4.0]
    
    sparse = SparseTensor(rows, cols, vals, shape=(2, 4), device=ndl.cpu())
    
    # 4x3 dense matrix
    dense = ndl.Tensor(np.arange(12, dtype=np.float32).reshape(4, 3), 
                       device=ndl.cpu())
    
    result = sparse @ dense
    
    print("  Sparse matrix (2x4, as dense):")
    print(sparse.to_dense().numpy())
    print("  Dense matrix (4x3):")
    print(dense.numpy())
    print("  Result (2x3):")
    print(result.numpy())
    
    # Verify
    expected = sparse.to_dense().numpy() @ dense.numpy()
    assert np.allclose(result.numpy(), expected), "Rectangular matmul failed!"
    print("  ✓ Rectangular matmul correct!")
    print()


def test_multiple_operations():
    """Test chaining operations"""
    print("===== Testing Multiple Operations =====")
    
    # Create sparse
    rows = [0, 1]
    cols = [0, 1]
    vals = [2.0, 3.0]
    sparse = SparseTensor(rows, cols, vals, shape=(2, 2), device=ndl.cpu())
    
    # Create dense
    dense1 = ndl.Tensor([[1.0, 2.0],
                         [3.0, 4.0]], 
                        device=ndl.cpu(), requires_grad=True)
    
    dense2 = ndl.Tensor([[5.0],
                         [6.0]], 
                        device=ndl.cpu(), requires_grad=True)
    
    # Chain: sparse @ dense1 @ dense2
    result1 = sparse @ dense1
    result2 = result1 @ dense2
    loss = result2.sum()
    
    print("  Sparse @ Dense1:")
    print(result1.numpy())
    print("  (Sparse @ Dense1) @ Dense2:")
    print(result2.numpy())
    print(f"  Loss: {loss.numpy()}")
    
    # Backward
    loss.backward()
    
    print("  Gradient w.r.t. dense1:")
    print(dense1.grad.numpy())
    print("  Gradient w.r.t. dense2:")
    print(dense2.grad.numpy())
    
    print("  ✓ Multiple operations work!")
    print()


def main():
    print("=" * 60)
    print("  SPARSE COO MATRIX TESTS")
    print("=" * 60)
    print()
    
    test_coo_from_numpy()
    test_coo_to_dense()
    test_sparse_dense_matmul()
    test_sparse_tensor_creation()
    test_sparse_tensor_matmul()
    test_sparse_tensor_to_dense()
    test_gradient_computation()
    test_large_sparse_matrix()
    test_empty_sparse_matrix()
    test_single_element()
    test_vector_multiplication()
    test_rectangular_sparse()
    test_multiple_operations()
    
    print("=" * 60)
    print("  ✓ ALL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    main()