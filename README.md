# CMU 10714 Deep Learning Systems Course Project (Fall 2025) (Under Construction)

[[Course Page](https://dlsyscourse.org/)]

## Needle-COO: Autodiff SpMM Kernels Powering GCN


### TODO-list:
- ✅ Implement the COO in cpu backend
- ✅ Implement the spMM forward/backward in cpu backend
- Implement the COO in cuda backend
- Implement the spMM forward/backward in cuda backend
- Implement GCN 
- Prepare the dataset 
- Implement the training loop
- Train the GCN and save the checkpoint 
- Evaluate the GCN on the test set
- Write the report


### Code Structure:
Here we use `[]` to mark the files related to the project. 
```
10714_DLSystem_Project
├── apps
│   ├── models.py
│   └── simple_ml.py
├── python
│   └── needle
│       ├── __init__.py
│       ├── autograd.py [Sparse Tensor] 
│       ├── backend_ndarray
│       │   ├── __init__.py
│       │   ├── ndarray.py
│       │   └── ndarray_backend_numpy.py
│       ├── backend_numpy.py
│       ├── backend_selection.py
│       ├── data
│       │   ├── ...
│       ├── init
│       │   ├── ...
│       ├── nn
│       │   ├── ...
│       ├── ops
│       │   ├── ops_sparse.py [SpMM forward/backward]
│       │   └── ...
│       ├── optim.py
│       └── sparse.py [NDArrayInt32(wrapper), SparseCOO2D]
└── src
    ├── ndarray_backend_cpu.cc [NDArrayInt32(impl), SpMM(impl)]
    └── ndarray_backend_cuda.cu
```







