import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from joblib import  load

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
def load_model():
    return load('LogisticRegression.joblib')