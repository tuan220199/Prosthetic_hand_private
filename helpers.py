
import pandas as pd
import numpy as np
import torch
import os
import seaborn as sns

packet_cnt = 0
start_time = 0



def set_cmd_cb(resp):
    print('Command result: {}'.format(resp))

rms_formuula = lambda x: np.sqrt(np.mean(x ** 2, axis=1))

DEVICE = "cpu"

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

def get_data(position= 0):
    Fs = 1000
    windowLength = int(np.floor(0.1*Fs))  #100ms
    windowOverlap =  int(np.floor(50/100 * windowLength))

    train_features = np.zeros([0,8])
    train_labels = np.zeros([0])
    test_features = np.zeros([0,8])
    test_labels = np.zeros([0])
    pathh = '/home/ros-lab/Desktop/Armband/gForceSDKPython-masterD3/Subject_5/Shift_'
    for files in sorted(os.listdir(f'{pathh}{position}/')):
        _, class_,_, rep_ = files.split('_')
        df = pd.read_csv(f'{pathh}{position}/{files}',skiprows=0,sep=' ',header=None)
        data_arr = np.stack([np.array(df.T[i::8]).T.flatten().astype('float32') for i in range (8)])
        data_arr -= 121
        data_arr /= 255.0
        feaData = getFeatureMatrix(data_arr, windowLength, windowOverlap)
        
        if not class_.startswith('9'):
            rms_feature = feaData.sum(0)
            baseline = 2.5*rms_feature[-50:].mean()
            start_ = np.argmax(rms_feature[::1]>baseline)
            end_  = -np.argmax(rms_feature[::-1]>baseline)
            feaData = feaData.T[start_:end_]
        else:
            feaData = feaData.T[50:300]
        if rep_.startswith('5') or rep_.startswith('4'):
            test_features = np.concatenate([test_features,feaData])
            test_labels = np.concatenate([test_labels,np.ones_like(feaData)[:,0]*int(class_)-1])
        else:
            train_features = np.concatenate([train_features,feaData])
            train_labels = np.concatenate([train_labels,np.ones_like(feaData)[:,0]*int(class_)-1])

    return train_features, train_labels, test_features, test_labels

def roll_data(X_0, shift):
    return np.roll(X_0,shift,1)

def get_all_data(X, y, v_shift = None):
    if not v_shift:
        all_X = np.zeros([X.shape[0],8, 8],dtype='float32')
        all_y = np.zeros([ y.shape[0],8],dtype='float32')
        all_shift = np.zeros([ y.shape[0],8],dtype='int')


        for i, shift in enumerate(range (-4,4)):
            X_i = roll_data(X, shift)
            all_X[:,i,:] = X_i
            all_y[:,i] = y
            all_shift[:,i] = shift
        return all_X.reshape(-1,8), all_y.reshape(-1,1), all_shift.reshape(-1,1)
    else:
        shapes = [X_.shape[0] for X_ in X]
        all_shift =[np.ones(shape, dtype='int')*v_shift for (v_shift, shape) in zip(v_shift,shapes)]
        return np.concatenate(X).reshape(-1,8), np.concatenate(y).reshape(-1,1), np.concatenate(all_shift).reshape(-1,1)

def get_all_data_full_feature(X, y, v_shift=None):
    num_features = 5  # Number of features in the feature matrix
    num_channels = 8  # Number of channels

    if v_shift is None:
        all_X = np.zeros([X.shape[0], 8, num_channels * num_features], dtype='float32')
        all_y = np.zeros([y.shape[0], 8], dtype='float32')
        all_shift = np.zeros([y.shape[0], 8], dtype='int')

        for i, shift in enumerate(range(-4, 4)):
            X_i = roll_data(X, shift)
            all_X[:, i, :] = X_i
            all_y[:, i] = y
            all_shift[:, i] = shift

        return all_X.reshape(-1, num_channels * num_features), all_y.reshape(-1, 1), all_shift.reshape(-1, 1)
    else:
        shapes = [X_.shape[0] for X_ in X]
        all_shift = [np.ones(shape, dtype='int') * v_shift for (v_shift, shape) in zip(v_shift, shapes)]
        return np.concatenate(X).reshape(-1, num_channels * num_features), np.concatenate(y).reshape(-1, 1), np.concatenate(all_shift).reshape(-1, 1)


def get_shift_data(all_X, all_shift, all_y):
    all_X_shift = np.concatenate([all_X, all_shift], axis=1)
    all_X1 = np.zeros_like(all_X)
    all_X2 = np.zeros_like(all_X)
    all_y_ = np.zeros_like(all_y)
    all_shift_1 = np.zeros_like(all_y)
    all_shift_2 = np.zeros_like(all_y)

    for class_label in range (9):
        class_idx = all_y.flatten() == class_label
        class_data = all_X_shift[class_idx]
        class_data_clone = class_data.copy()
        np.random.shuffle(class_data_clone)

        all_X1[class_idx] = class_data[:,:-1]
        all_X2[class_idx] = class_data_clone[:,:-1]
        all_shift_1[class_idx] = class_data[:,-1:]
        all_shift_2[class_idx] = class_data_clone[:,-1:]
        all_y_[class_idx] = class_label
    return all_X1, all_X2, all_shift_1, all_shift_2, all_y_

def get_shift_data_full_features(all_X, all_shift, all_y):
    num_features = 5  # Number of features in the feature matrix
    num_channels = 8  # Number of channels
    feature_dim = num_channels * num_features

    # Combine features with shift values
    all_X_shift = np.concatenate([all_X, all_shift], axis=1)
    
    # Initialize arrays with correct dimensions
    all_X1 = np.zeros_like(all_X)
    all_X2 = np.zeros_like(all_X)
    all_y_ = np.zeros_like(all_y)
    all_shift_1 = np.zeros_like(all_y)
    all_shift_2 = np.zeros_like(all_y)

    for class_label in range(9):
        class_idx = all_y.flatten() == class_label
        class_data = all_X_shift[class_idx]
        class_data_clone = class_data.copy()
        np.random.shuffle(class_data_clone)

        all_X1[class_idx] = class_data[:, :-1]
        all_X2[class_idx] = class_data_clone[:, :-1]
        all_shift_1[class_idx] = class_data[:, -1:]
        all_shift_2[class_idx] = class_data_clone[:, -1:]
        all_y_[class_idx] = class_label

    return all_X1, all_X2, all_shift_1, all_shift_2, all_y_

def get_shift_data1(h_X,v_X, h_shift, v_shift, h_y, v_y):
    h_X_shift = np.concatenate([h_X, h_shift], axis=1)
    v_X_shift = np.concatenate([v_X, v_shift], axis=1)

    all_X1 = np.zeros_like(h_X)
    all_X2 = np.zeros_like(h_X)

    h_shift = np.zeros_like(h_y)
    v_shift = np.zeros_like(h_y)

    for class_label in range (9):
        h_class_idx = h_y.flatten() == class_label
        v_class_idx = v_y.flatten() == class_label

        h_class_data = h_X_shift[h_class_idx].copy()
        v_class_data = v_X_shift[v_class_idx].copy()

        np.random.shuffle(h_class_data)
        np.random.shuffle(v_class_data)

        all_X1[h_class_idx] = h_class_data[:,:-1]
        all_X2[h_class_idx] = v_class_data[:,:-1].repeat(5,0)[:h_class_data.shape[0]]
        
        h_shift[h_class_idx] = h_class_data[:,-1:]
        v_shift[h_class_idx] = v_class_data[:,-1:].repeat(5,0)[:h_class_data.shape[0]]

    return all_X1, all_X2, h_shift, v_shift, h_y

def get_operators(_n_rotations, order):
    M = torch.diag(torch.ones(order)).roll(-1,1)

    phi1 = torch.zeros((8*order, 8*order))
    for i in range (8):
        phi1[order*i:order*(i+1), order*i:order*(i+1)] = M
    bases = [torch.linalg.matrix_power(phi1,i).to(DEVICE) for i in range (0, _n_rotations)]
    used_bases = bases[:17] + bases[-16:]

    return used_bases, phi1

def plot_cfs_mat(predicted, labels):
    cf_mat = np.zeros((6,6))
    for i in range(predicted.shape[0]):
        cf_mat[predicted[i], labels[i]] += 1

    return sns.heatmap(cf_mat/6, annot=True,cmap='Blues', cbar=False)

def get_centroids(encoder, loader):
    encoder.eval()
    centroids = torch.zeros(6, 192)
    first = torch.zeros(6)
    counts = torch.zeros(6,1)
    for inputs, _, labels in loader:
        inputs = inputs.to(DEVICE).flatten()
        labels = labels.flatten()[0].int()
        y_tr = encoder(inputs)
        if first[labels-1]:
            centroids[labels-1] += y_tr
        else:
            centroids[labels-1] = y_tr
            first[labels-1] = 1
        counts[labels-1] +=1
    centroids /= counts