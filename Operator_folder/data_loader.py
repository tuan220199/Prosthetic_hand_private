
from torch.utils.data import Dataset
class CustomSignalData(Dataset):
    """
    Custom dataset class for signal data.

    Args:
    - x (torch.Tensor or numpy.ndarray): Input data.
    - y (torch.Tensor or numpy.ndarray): Target labels.
    - z (torch.Tensor or numpy.ndarray, optional): Additional data (default is None, in which case it is set to y).
    - a (torch.Tensor or numpy.ndarray, optional): Additional data (default is None, in which case it is set to y).
    - transform (callable, optional): Optional transform to be applied to the input data (default is None).

    Note:
    - This dataset assumes that the input data `x`, target labels `y`, and any additional data `z` and `a` are provided.
    - If `z` or `a` is not provided, it defaults to `y`.
    - If a transform is provided, it will be applied to the input data.
    """
    def __init__(self, x, y,z=None, a= None, transform= None):
        self.x = x
        self.y = y
        self.z = z if z is not None else y
        self.a = a if a is not None else y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.x[idx]), self.y[idx], self.z[idx], self.a[idx]
        else:
            return self.x[idx], self.y[idx], self.z[idx], self.a[idx]



class CustomSignalData1(Dataset):
    def __init__(self, x, y,z=None, a= None, b=None,c=None, transform= None):
        self.x = x
        self.y = y
        self.z = z if z is not None else y
        self.a = a if a is not None else y
        self.b = b if b is not None else y
        self.c = c if c is not None else y
        self.transform = transform

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.transform:
            return self.transform(self.x[idx]), self.transform(self.y[idx]), self.z[idx]
        else:
            return self.x[idx], self.y[idx], self.z[idx], self.a[idx], self.b[idx], self.c[idx]

