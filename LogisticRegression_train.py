import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load

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

def main():

    subject = sys.argv[1]
    No_shift = sys.argv[2]

    Fs = 500
    windowLength = int(np.floor(0.1*Fs))  #160ms
    windowOverlap =  int(np.floor(50/100 * windowLength))

    train_features = np.zeros([0,8])
    train_labels = np.zeros([0])
    test_features = np.zeros([0,8])
    test_labels = np.zeros([0])
    for shift in range(0,int(No_shift)): 
        for files in sorted(os.listdir(f'Subject_{subject}/Shift_{shift}/')):
            _, class_,_, rep_ = files.split('_')
            if int(class_) in [1,2,3]:
                df = pd.read_csv(f'Subject_{subject}/Shift_{shift}/{files}',skiprows=0,sep=' ',header=None)
                data_arr = np.stack([np.array(df.T[i::8]).T.flatten().astype('float32') for i in range (8)])
                data_arr -= 121
                data_arr /= 255.0
                feaData = getFeatureMatrix(data_arr, windowLength, windowOverlap)
                
                if not class_.startswith('9'):
                    rms_feature = feaData.sum(0)
                    baseline = 2*rms_feature[-50:].mean()
                    start_ = np.argmax(rms_feature[::1]>baseline)
                    end_  = -np.argmax(rms_feature[::-1]>baseline)
                    feaData = feaData.T[start_:end_]
                else:
                    feaData = feaData.T
                if rep_.startswith('2'):
                    test_features = np.concatenate([test_features,feaData])
                    test_labels = np.concatenate([test_labels,np.ones_like(feaData)[:,0]*int(class_)-1])
                else:
                    train_features = np.concatenate([train_features,feaData])
                    train_labels = np.concatenate([train_labels,np.ones_like(feaData)[:,0]*int(class_)-1])

    reg = LogisticRegression(penalty='l2', C=100).fit(train_features, train_labels)
    reg.score(train_features, train_labels)#, reg.score(test_features, test_labels)

    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(train_labels, reg.predict(train_features)),
                            display_labels=reg.classes_)
    disp.plot()

    dump(reg, 'LogisticRegression1.joblib')
    accuracy_list = [reg.score(test_features,test_labels)]
    print(accuracy_list)

if __name__ == "__main__":
    main()
