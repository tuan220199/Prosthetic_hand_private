import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder, self).__init__()
        """
        Args:
          in_features (int): Number of input features which should be equal to xsize.
          out_features (out): Number of output features which should be equal to ysize.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=False),
            #nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, in_features): Inputs.
        
        Returns:
          y of shape (batch_size, out_features): Representations.
        """
        return self.encoder(x)


class Encoder2(nn.Module):
    def __init__(self, in_features, out_features):
        super(Encoder2, self).__init__()
        """
        Args:
          in_features (int): Number of input features which should be equal to xsize.
          out_features (out): Number of output features which should be equal to ysize.
        """
        self.in_features = in_features
        self.out_features = out_features
        self.encoder = nn.Sequential(
            nn.Linear(self.in_features, self.out_features, bias=False)
        )

    def forward(self, x):
        """
        Args:
          x of shape (batch_size, in_features): Inputs.
        
        Returns:
          y of shape (batch_size, out_features): Representations.
        """
        return self.encoder(x)
