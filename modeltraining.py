import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import  load
from encoder import Encoder as E
import torch
from torch.autograd import Variable

# This is a feed forward neural network includes encoder and classifier
# Combining an encoder and a classifier into a single neural network, model can learn to
# extract relevant features from input data and use those features for making predictions or classifications 
class FFNN(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(FFNN, self).__init__()

        # encoder transform the input data into a suitable representation for classifier
        # a single linear layer followed by a sigmoid activation function
        # This layer reduces dimensionality of input data and applies a non-linear transformation
        # to learn complex pattern  
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(inputSize, 9, bias=False),
            torch.nn.Sigmoid()
        )
        self.classifer = torch.nn.Sequential(
            torch.nn.Linear(9, outputSize, bias=False),
            # torch.nn.Softmax(dim=1)
        )

    def forward(self, x, encoder=None):
        if not encoder:
            encoder = self.encoder
        z = encoder(x)
        class_z = self.classifer(z)

        return class_z
logRegres  = load('LogisticRegression1.joblib')
#RNN_model = load('RNN.joblib')

classifier = FFNN(8,3) # This indicates that the neural network expects input data with 8 features and will produce output predictions across 9 classes.
encoder = E(8,8)
encoder.load_state_dict(torch.load("encoder.pt")) # contains the learned parameters (weights and biases) of the encoder model
recovered_points_= torch.load("reference_points.pt") # These points represent reference points for inference or evaluation in the model
print(recovered_points_)
classifier.load_state_dict(torch.load("classifier.pt")) # contains the weights and biases learned during training.
classifier.eval() # sets the model to evaluation mode.
encoder.eval() # sets the model to evaluation mode.

# # This model is without opeartor. It is similiarily to the classifier model but do not have encoder
modelWOoperator = FFNN(8,3)
modelWOoperator.load_state_dict(torch.load("modelwoOperator.pt")) # loaded from the file: parameters learned during training.
modelWOoperator.eval() # evaluation mode ensures consistent behavior during inference.

DEVICE = torch.device("cpu") # operations is in CPU or GPU.
M = torch.diag(torch.ones(8)).roll(-1,1) # Create a diagnoise matrix then shift it to the right
used_bases = [torch.linalg.matrix_power(M,i).to(DEVICE) for i in range (8)] #

N_points = 1000

# This function takes raw data matrix, window length, window overlap as inputs and 
# return a feature matrix  as NumPy array
def getFeatureMatrix(rawDataMatrix, windowLength, windowOverlap):
    rms = lambda sig: np.sqrt(np.mean(sig**2)) # function to calculate the rms of signal
    nChannels,nSamples = rawDataMatrix.shape    
    I = int(np.floor(nSamples/(windowLength-windowOverlap))) # number of window 
    featMatrix = np.zeros([nChannels, I]) #Empty feature matrix size nChannel*number of window
    
    # Iterate each channel and window for calculate the rms of each window
    for channel in range(nChannels):
        for i in range (I):
            wdwStrtIdx=i*(windowLength-windowOverlap) # Get the first index of each window
            sigWin = rawDataMatrix[channel][wdwStrtIdx:(wdwStrtIdx+windowLength-1)] # Take all the signals in that window
            featMatrix[channel, i] = rms(sigWin) # Calculate the RMS value for each window
    featMatrixData = np.array(featMatrix) # Convert to NUmpy array
    return featMatrixData

# Training a logistic regression model using data from files in specific directory. 
def model_training():
    Fs = 500 # SAMpling frequency
    windowLength = int(np.floor(0.1*Fs))  #160ms
    windowOverlap =  int(np.floor(50/100 * windowLength))

    train_features = np.zeros([0,8]) # an empty NUmPY array with dimension(0,8) store features extracted from training data
    train_labels = np.zeros([0]) # empty NUmPy array store labels for training data

    for files in sorted(os.listdir('Subject_7/Shift_0/')):
        _, class_,_, rep_ = files.split('_')
        df = pd.read_csv(f'Subject_7/Shift_0/{files}',skiprows=0,sep=' ',header=None).drop(columns=[128,129])
        # For each file, convert it into pandas dataframe

        # data preprocessing 
        data_arr = np.stack([np.array(df.T[i::8]).T.flatten().astype('float32') for i in range (8)])
        data_arr -= 121
        data_arr /= 255.0


        feaData = getFeatureMatrix(data_arr, windowLength, windowOverlap)
        rms_feature = feaData.sum(0) # an array of some rms value in each samples all across channels
        baseline = 2*rms_feature[-50:].mean() #  double of mean of the last 50 elements, as a threshold 
        # value against which the signal amplitude is compared.
        start_ = np.argmax(rms_feature[::1]>baseline) # index of the first occurrence where the root mean square (RMS) value of the feature data exceeds the baseline value.
        end_  = -np.argmax(rms_feature[::-1]>baseline) # finds the index of the first occurrence where the RMS value of the feature data exceeds the baseline value
        feaData = feaData.T[start_:end_] # extract the start adn begin data which exceed the threshold
        train_features = np.concatenate([train_features,feaData])
        train_labels = np.concatenate([train_labels,np.ones_like(feaData)[:,0]*int(class_)-1])
        
    reg = LogisticRegression().fit(train_features, train_labels)

    return reg
def loadLogRegr():
    return 1

def loadwoOperator():
    return 2


def loadwithOperator():
    return 3

from scipy.stats import t

noise_std = 0.5 # standard deviation of the noise
degrees_of_freedom = 1 # degreesof freedom parameter for t-distribution

# calculate the likelihood of a measurement given a set of particles
def measurement_likelihood(measurement, particles, degrees_of_freedom):
    likelihoods = t.pdf( -particles + measurement, loc=0, scale=noise_std, df=degrees_of_freedom)
    return likelihoods
x= 0
motion_model=lambda x: x + np.random.randint(-1,2, x.shape)

num_particles = 1000
weights = np.ones(num_particles) / num_particles
particles = np.random.randint(-1,2, size=num_particles)

def get_class(modeltype, X):
    global x, motion_model, num_particles,weights, particles, degrees_of_freedom, noise_std
    if modeltype == 1: #uses logistic regression (logRegres) to predict the class.
        return int(logRegres.predict(X)[0])
    elif modeltype == 2: #ses a neural network model (modelWOoperator) to predict the class.
        with torch.no_grad(): # disable gradient computation
            return modelWOoperator(X).argmax().item()
    elif modeltype == 3: #performs particle filtering to estimate the class.
        # return int(RNN_model.predict(X)[0])
        with torch.no_grad(): # disable gradient computation
            particles = motion_model(particles)
            y1 = encoder(X)
            distances = np.zeros(8)
            for d in (range(-4,4)):
                x_rotated = y1.matmul(used_bases[d]).repeat(N_points,1)
                distances[d] = ((x_rotated-recovered_points_)**2).mean(1).topk(2, largest=False)[0].mean()
            position = distances.argmin()
            
            weights = measurement_likelihood(position, particles, degrees_of_freedom)
            weights /= np.sum(weights) # Normalize weights

            # # Resampling Step
            indices = np.random.choice(range(num_particles), size=num_particles, replace=True, p=weights.flatten())
            particles = particles[indices]
            weights = np.ones(num_particles) / num_particles

            # # State estimation
            estimated_state = np.average(particles, axis=0, weights=weights)
        
            x = estimated_state

            curr_shift = (round(x)+4)%8-4
            y_tr_est1 = y1.matmul(used_bases[curr_shift])
            y_tr1 = classifier(y_tr_est1).argmax()
            return y_tr1.item()