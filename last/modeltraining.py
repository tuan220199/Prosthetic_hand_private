import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import  load
from encoder import Encoder as E
import torch
from torch.autograd import Variable
class FFNN(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(FFNN, self).__init__()
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
logRegres  = load('LogisticRegression.joblib')

classifier = FFNN(8,9)
encoder = E(8,8)
encoder.load_state_dict(torch.load("gForceSDKPython-master/encoder.pt"))
recovered_points_= torch.load("gForceSDKPython-master/reference_points.pt")
classifier.load_state_dict(torch.load("gForceSDKPython-master/classifier.pt"))
classifier.eval()
encoder.eval()

modelWOoperator = FFNN(8,9)
modelWOoperator.load_state_dict(torch.load("gForceSDKPython-master/modelwoOperator.pt"))
modelWOoperator.eval()

DEVICE = torch.device("cpu")
M = torch.diag(torch.ones(8)).roll(-1,1)
used_bases = [torch.linalg.matrix_power(M,i).to(DEVICE) for i in range (8)]

N_points = 1000

def getFeatureMatrix(rawDataMatrix, windowLength, windowOverlap):
    rms = lambda sig: np.sqrt(np.mean(sig**2))
    nChannels,nSamples = rawDataMatrix.shape    
    I = int(np.floor(nSamples/(windowLength-windowOverlap)))
    featMatrix = np.zeros([nChannels, I])
    for channel in range(nChannels):
        for i in range (I):
            wdwStrtIdx=i*(windowLength-windowOverlap)
            sigWin = rawDataMatrix[channel][wdwStrtIdx:(wdwStrtIdx+windowLength-1)] 
            featMatrix[channel, i] = rms(sigWin)
    featMatrixData = np.array(featMatrix)
    return featMatrixData
def model_training():
    Fs = 500
    windowLength = int(np.floor(0.1*Fs))  #160ms
    windowOverlap =  int(np.floor(50/100 * windowLength))

    train_features = np.zeros([0,8])
    train_labels = np.zeros([0])

    for files in sorted(os.listdir('Subject_1/Shift_0/')):
        _, class_,_, rep_ = files.split('_')
        df = pd.read_csv(f'Subject_1/Shift_0/{files}',skiprows=0,sep=' ',header=None).drop(columns=[128,129])
        data_arr = np.stack([np.array(df.T[i::8]).T.flatten().astype('float32') for i in range (8)])
        data_arr -= 121
        data_arr /= 255.0
        feaData = getFeatureMatrix(data_arr, windowLength, windowOverlap)
        rms_feature = feaData.sum(0)
        baseline = 2*rms_feature[-50:].mean()
        start_ = np.argmax(rms_feature[::1]>baseline)
        end_  = -np.argmax(rms_feature[::-1]>baseline)
        feaData = feaData.T[start_:end_]
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

noise_std = 0.5
degrees_of_freedom = 1

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
    if modeltype == 1:
        return int(logRegres.predict(X)[0])
    elif modeltype == 2:
        with torch.no_grad():
            return modelWOoperator(X).argmax().item()
    elif modeltype == 3:
        with torch.no_grad():
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