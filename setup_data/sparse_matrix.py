import torch

class SparseMatrix:
    def __init__(self, dense_tensor: torch.Tensor):
        if dense_tensor.dim() != 2:
            raise ValueError("Input tensor must be 2D.")
        self.shape = dense_tensor.shape
        nz = dense_tensor.nonzero(as_tuple=False) # (nnz, 2)
        if nz.numel() == 0:
            self.indices = torch.empty((2, 0), dtype=torch.long)
            self.values = torch.empty((0,), dtype=dense_tensor.dtype)
        else:
            self.indices = nz.t()
            self.values = dense_tensor[nz[:, 0], nz[:, 1]]

    def to_dense(self):
        dense = torch.zeros(self.shape, dtype=self.values.dtype, device=self.values.device)
        if self.indices.numel() > 0:
            dense[self.indices[0], self.indices[1]] = self.values
        return dense
