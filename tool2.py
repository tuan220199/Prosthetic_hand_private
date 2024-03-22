import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import display

import torch
import torchvision.utils as utils
import torch.nn.functional as F
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def getFeatureMatrix(rawDataMatrix, features, window_size, overlap_factor):
    nChannels, nSamples = rawDataMatrix.shape
    stride = int(window_size * overlap_factor)
    num_stride = int(np.floor(nSamples/(window_size-stride)))
    featMatrix = np.zeros((nChannels*len(features),num_stride))
    # Define a dictionary that maps feature names to functions that calculate those features
    feature_functions = {
        'RMS': lambda x: np.sqrt(np.mean(x ** 2, axis=1)),
        'MAV': lambda x: np.mean(np.abs(x), axis=1),
        'SSC': lambda x: np.mean(((x[:, 1:-1] - x[:, :-2]) * (x[:, 2:] - x[:, 1:-1])) < 0, axis=1).reshape(-1, 1),
        'ZC': lambda x: np.mean((x[:, :-1] * x[:, 1:] < 0) & (np.abs(x[:, :-1] - x[:, 1:]) > 0), axis=1).reshape(-1, 1),
        'WL': lambda x: np.mean(np.abs(x[:, :-1] - x[:, 1:]), axis=1)
    }
    # Loop over the features 
    featIndex = 0
    for feature in features:
        if feature in feature_functions:
            featFunc = feature_functions[feature]
            for i in range(num_stride):
                wdwStrtIdx = i*(window_size-stride)
                if i == num_stride:
                    sigWin = rawDataMatrix[:, wdwStrtIdx:nSamples] 
                else:
                    sigWin = rawDataMatrix[:, wdwStrtIdx:(wdwStrtIdx+window_size-1)] 
                
                featValues = featFunc(sigWin)
                featValues = featValues.flatten() # Flatten featValues before assigning it to featMatrix
                featMatrix[featIndex:featIndex + nChannels, i] = featValues     
            featIndex += nChannels
    return featMatrix

def add_awgn_tensor(signal, SNR_db,signal_power, tv_noise_gain, batch_size, y_test,features, window_size, overlap_factor, mean_train, std_train): 
    """
    Add Additive White Gaussian Noise (AWGN) to the input signal and prepare the data for training.

    Args:
        signal (numpy.ndarray): Input signal data.
        SNR_db (float): Signal-to-Noise Ratio (SNR) in decibels.
        signal_power (float): Power of the original signal.
        tv_noise_gain (float): Gain factor for time-varying noise.
        batch_size (int): Size of batches for data loaders.
        y_test (torch.Tensor): Labels for the data.
        features (list): List of features to compute.
        window_size (int): Size of the window for feature computation.
        overlap_factor (float): Overlap factor for feature computation.
        mean_train (float): Mean value of the training dataset (for normalization).
        std_train (float): Standard deviation of the training dataset (for normalization).

    Returns:
        tuple: Tuple containing two PyTorch DataLoader objects:
            - DataLoader for noisy constant signal data
            - DataLoader for noisy time-variant signal data

    Raises:
        None

    """

    # Noisy constant signal data
    # Convert SNR from decibels to linear scale.
    reqSNR = 10**(SNR_db/10)
    noise_power = signal_power / reqSNR
    noise = np.sqrt(noise_power)* np.random.normal(0, 1, signal.shape)
    noisy_signal_constant = signal + noise
    # Compute feature matrix for the noisy constant signal using the getFeatureMatrix function.
    signal_Nis_constant_getFeatMat = getFeatureMatrix(noisy_signal_constant, features, window_size, overlap_factor)
    
    # Noisy time-variant signal data
    tv_noise = tv_noise_gain* noise
    noisy_signal_time_variant = signal + tv_noise
    # Compute feature matrix for the noisy time-variant signal using the getFeatureMatrix function.
    signal_Nis_time_variant_getFeatMat= getFeatureMatrix(noisy_signal_time_variant, features, window_size, overlap_factor)
    
    # Convert feature matrices to PyTorch tensors
    DataNSC = torch.tensor(signal_Nis_constant_getFeatMat.T, dtype=torch.float32)
    DataNSTV = torch.tensor(signal_Nis_time_variant_getFeatMat.T, dtype=torch.float32)
 
    # Normalize the data using the mean and standard deviation of the training set.
    # Create TensorDataset objects (tenDataNSC and tenDataNSTV) combining normalized data with labels.
    tenDataNSC = TensorDataset((DataNSC-mean_train)/std_train, y_test)
    tenDataNSTV = TensorDataset((DataNSTV-mean_train)/std_train, y_test)

    # Create PyTorch DataLoader objects for loading data in batches during training.
    dataloader_noisy_signal_constant = DataLoader(tenDataNSC, batch_size=batch_size, drop_last=True)
    dataloader_noisy_signal_time_variant = DataLoader(tenDataNSTV, batch_size=batch_size, drop_last=True)

    return dataloader_noisy_signal_constant, dataloader_noisy_signal_time_variant
   
def shift_channels(TsCln, rawShift, features, window_size, overlap_factor, y_test, batch_size, mean_train, std_train):
    """
    Shift the channels of the input signal by a specified amount and prepare the data for training.

    Args:
        TsCln (numpy.ndarray): Input signal data.
        rawShift (int): Number of positions to shift the channels.
        features (list): List of features to compute.
        window_size (int): Size of the window for feature computation.
        overlap_factor (float): Overlap factor for feature computation.
        y_test (torch.Tensor): Labels for the data.
        batch_size (int): Size of batches for data loaders.
        mean_train (float): Mean value of the training dataset (for normalization).
        std_train (float): Standard deviation of the training dataset (for normalization).

    Returns:
        torch.utils.data.DataLoader: DataLoader for the shifted channel data.

    Raises:
        None

    """
    testShift = np.roll(TsCln,rawShift*8, axis=0)
    ch_shift_getFeatMat= getFeatureMatrix(testShift, features, window_size, overlap_factor)
    DataNSC = torch.tensor(ch_shift_getFeatMat.T, dtype=torch.float32)
    tenData_ch_shift = TensorDataset((DataNSC-mean_train)/std_train, y_test)
    dataloader_ch_shift = DataLoader(tenData_ch_shift, batch_size=batch_size, drop_last=True)
    return dataloader_ch_shift


def remove_channels(TsCln, RemovalPercentage, features, window_size, overlap_factor, y_test, batch_size, mean_train, std_train):
    """
    Remove a percentage of channels from the input signal and prepare the data for training.

    Args:
        TsCln (numpy.ndarray): Input signal data.
        RemovalPercentage (float): Percentage of channels to remove.
        features (list): List of features to compute.
        window_size (int): Size of the window for feature computation.
        overlap_factor (float): Overlap factor for feature computation.
        y_test (torch.Tensor): Labels for the data.
        batch_size (int): Size of batches for data loaders.
        mean_train (float): Mean value of the training dataset (for normalization).
        std_train (float): Standard deviation of the training dataset (for normalization).

    Returns:
        torch.utils.data.DataLoader: DataLoader for the signal data with removed channels.

    Raises:
        None

    """
    nChannel = TsCln.shape[0]
    #Randomly selects indices of channels to remove based on the specified removal percentage.
    indRmv = np.random.choice(nChannel, size=int(RemovalPercentage / 100.0 * nChannel), replace=False)
    
    testEMG4MC = TsCln.copy()
    testEMG4MC[indRmv, :] = 0
    
    miss = testEMG4MC
    ch_loss_getFeatMat= getFeatureMatrix(miss, features, window_size, overlap_factor)
    DataNSC = torch.tensor(ch_loss_getFeatMat.T, dtype=torch.float32)
    tenData_ch_loss = TensorDataset((DataNSC-mean_train)/std_train, y_test)
    dataloader_ch_loss = DataLoader(tenData_ch_loss, batch_size=batch_size, drop_last=True)
    
    return dataloader_ch_loss

class LDA(nn.Module):
    """
    Linear Discriminant Analysis (LDA) classifier.

    Args:
        num_classes (int): Number of classes in the classification problem.
        input_size (int): Size of the input features.

    Attributes:
        num_classes (int): Number of classes.
        input_size (int): Size of the input features.
        W (torch.Parameter): Weight parameter of the linear transformation.
        b (torch.Parameter): Bias parameter of the linear transformation.

    Methods:
        forward(x): Forward pass of the classifier.
        predict(x): Predicts the class labels for input data.

    """
    def __init__(self, num_classes, input_size):
        super(LDA, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.W = nn.Parameter(torch.randn(num_classes, input_size))
        self.b = nn.Parameter(torch.randn(num_classes))

    def forward(self, x):
        """
        Forward pass of the LDA classifier.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted scores for each class.

        """
        y = torch.matmul(x, torch.t(self.W)) + self.b
        return y

    def predict(self, x):
        """
        Predicts the class labels for input data.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted class labels.

        """
        y = self.forward(x)
        return torch.argmax(y, dim=1)
    
class RNN(nn.Module):
    """
    Recurrent Neural Network (RNN) classifier.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden state.
        num_classes (int): Number of classes in the classification problem.

    Attributes:
        gru1 (torch.nn.LSTM): First GRU layer.
        gru2 (torch.nn.LSTM): Second GRU layer.
        ReLU (torch.nn.ReLU): ReLU activation function.
        fc (torch.nn.Linear): Fully connected layer for classification.

    Methods:
        forward(x): Forward pass of the classifier.

    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.gru1 = nn.LSTM(input_size, hidden_size, batch_first=True)    # LSTM GRU
        self.gru2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)    # LSTM GRU
        self.ReLU = nn.ReLU() # ReLU LeakyReLU Sigmoid
        self.fc = nn.Linear(hidden_size , num_classes)
        
    def forward(self, x): 
        """
        Forward pass of the RNN classifier.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted scores for each class.

        """     
        x, _ = self.gru1(x) 
        x, _ = self.gru2(x) 
        x = self.ReLU(x) 
        x = x.reshape(x.size(0), -1) 
        x = self.fc(x) 
        return x
    
class MyCNN(nn.Module):
    """
    Convolutional Neural Network (CNN) classifier.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        num_classes (int): Number of classes in the classification problem.

    Attributes:
        conv1 (torch.nn.Conv1d): Convolutional layer.
        pool (torch.nn.MaxPool1d): Max pooling layer.
        fc1 (torch.nn.Linear): First fully connected layer.
        fc2 (torch.nn.Linear): Second fully connected layer.

    Methods:
        forward(x): Forward pass of the classifier.

    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(MyCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 5, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(5 * 48*2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
    def forward(self, x):
        """
        Convolutional Neural Network (CNN) classifier.

        Args:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden layer.
            num_classes (int): Number of classes in the classification problem.

        Attributes:
            conv1 (torch.nn.Conv1d): Convolutional layer.
            pool (torch.nn.MaxPool1d): Max pooling layer.
            fc1 (torch.nn.Linear): First fully connected layer.
            fc2 (torch.nn.Linear): Second fully connected layer.

        Methods:
            forward(x): Forward pass of the classifier.

        """
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 5 * 48*2)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
class NN(nn.Module):
    """
    Feedforward Neural Network (FNN) classifier.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.
        num_classes (int): Number of classes in the classification problem.

    Attributes:
        model (torch.nn.Sequential): Sequential container for the neural network layers.

    Methods:
        forward(x): Forward pass of the classifier.

    """
    def __init__(self, input_size, hidden_size, num_classes):
        super(NN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(), # Tanh
            nn.Linear(hidden_size, num_classes), #nn.Softmax(dim=1)
        )
    def forward(self, x):
        """
        Forward pass of the FNN classifier.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted scores for each class.

        """
        out = self.model(x)
        return out
    
def compute_accuracy(net, testloader, condition=''):
    """
    Compute the accuracy of a neural network classifier.

    Args:
        net (torch.nn.Module): Neural network classifier.
        testloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        condition (str, optional): Additional condition for the computation.

    Returns:
        float: Accuracy of the classifier on the test dataset.

    """
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for (inputs, labels) in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            if condition == 'ss':
                outputs = net(encoder(inputs))
            else:
                outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total




class Autoencoder(nn.Module):
    """
    Autoencoder neural network for feature extraction and reconstruction.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer.

    Attributes:
        encoder (torch.nn.Sequential): Encoder network.
        decoder (torch.nn.Sequential): Decoder network.

    Methods:
        forward(x): Forward pass of the autoencoder.

    """
    def __init__(self, input_size, hidden_size):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3*hidden_size),
            nn.ReLU(),
            nn.Linear(3*hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 3*hidden_size),
            nn.ReLU(),
            nn.Linear(3*hidden_size, input_size),
            nn.Sigmoid(),
        )
        
    def forward(self, x):
        """
        Forward pass of the autoencoder.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Encoded and decoded features.

        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class AutoRNN(nn.Module):
    """
    Autoencoder-RNN hybrid model for sequential data classification.

    Args:
        input_size (int): Size of the input features.
        hidden_size (int): Size of the hidden layer in RNN.
        num_classes (int): Number of classes in the classification problem.
        ae_input_size (int): Size of the input features for autoencoder.
        ae_hidden_size (int): Size of the hidden layer in the autoencoder.

    Attributes:
        ae (Autoencoder): Autoencoder model.
        gru1 (torch.nn.GRU): GRU layer.
        ReLU (torch.nn.ReLU): ReLU activation function.
        fc (torch.nn.Linear): Fully connected layer for classification.

    Methods:
        forward(x): Forward pass of the hybrid model.

    """
    def __init__(self, input_size, hidden_size, num_classes, ae_input_size, ae_hidden_size):
        super(AutoRNN, self).__init__()
        self.ae = Autoencoder(input_size, ae_hidden_size)
        self.gru1 = nn.GRU(ae_hidden_size, hidden_size, batch_first=True)    # LSTM GRU self.gru2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)    # LSTM GRU
        self.ReLU = nn.ReLU() # ReLU LeakyReLU Sigmoid # self.dropout = nn.Dropout(p=0.05)
        self.fc = nn.Linear(hidden_size , num_classes)
        
    def forward(self, x):
        """
        Forward pass of the hybrid model.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            torch.Tensor: Predicted scores for each class.

        """     
        encoded, _ = self.ae(x)
        x, _ = self.gru1(encoded) # x, _ = self.gru2(x) #  x = self.dropout(x)
        x = self.ReLU(x) 
        x = x.reshape(x.size(0), -1) 
        x = self.fc(x) 
        return x
    





__all__ = [name for name in dir() if callable(globals()[name]) and not name.startswith('_')]
