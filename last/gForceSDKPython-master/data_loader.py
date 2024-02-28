
from torch.utils.data import Dataset
class CustomSignalData(Dataset):
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

