import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import numpy as np
from ..data_basic import Dataset

class CIFAR10Dataset(Dataset):
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        if train:
            batch_files = [f"data_batch_{i}" for i in range(1, 6)]
        else:
            batch_files = ["test_batch"]
        data_parts = []
        label_parts = []
        for fname in batch_files:
            path = os.path.join(base_folder, fname)
            with open(path, "rb") as f:
                batch = pickle.load(f, encoding="bytes")
            data_parts.append(batch[b"data"])
            label_parts.append(batch[b"labels"])
        X = np.concatenate(data_parts, axis=0).astype(np.float32)
        y = np.concatenate(label_parts, axis=0).astype(np.int64)
        # scale to [0, 1]
        X /= 255.0
        # reshape to (N, 3, 32, 32)
        num = X.shape[0]
        X = X.reshape(num, 3, 32, 32)
        self.X = X
        self.y = y
        self.n = num
        self.p = p
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        ### BEGIN YOUR SOLUTION
        x = self.apply_transforms(self.X[index])
        y = self.y[index]
        return x, y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        ### BEGIN YOUR SOLUTION
        return self.n
        ### END YOUR SOLUTION
