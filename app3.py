import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader 
import mat73
from tools2 import *

def app3(FeatureNum, classifierType):
    """
    Perform experiments with different classifiers and features on a dataset.

    Args:
        FeatureNum (int): Feature number indicating the type of features to be used.
        classifierType (int): Classifier type indicator.

    """
    subNum = 3
    if FeatureNum == 1:
        features = ['RMS']
        fName = 'RMS'
    elif FeatureNum == 2:
        features = ['MAV']
        fName = 'MAV'
    elif FeatureNum == 3:
        features = ['SSC']
        fName = 'SSC'
    elif FeatureNum == 4:
        features = ['ZC']
        fName = 'ZC'
    elif FeatureNum == 5:
        features = ['WL']
        fName = 'WL'
    elif FeatureNum == 6:
        features = ['RMS','MAV','SSC','ZC','WL']
        fName = 'RMS_MAV_SSC_ZC_WL'
    elif FeatureNum == 7:
        features = ['RMS','WL']
        fName = 'RMS_WL'
        Err

    if classifierType == 1:
        Classifier = 'linear'
    elif classifierType == 2:
        Classifier = 'NN'
    elif classifierType == 3:
        Classifier = 'deep'
    elif classifierType == 4:
        Classifier = 'AutoRNN'
        
     #'linear' 'NN' 'deep' 'AutoRNN'

    print(Classifier+ ' classifier with:  ' + fName)
    # -------------------------------------------------------------

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print("\n", '---------------- Load Data ----------------',"\n")
    
    pathData = '/scratch/work/taleshm1/mansourAllSubj/Subj'+ str(subNum) + '/'
    
    if subNum == 4:
        dof1 = mat73.loadmat(pathData + 'Dof1ARep.mat')
        dof2 = mat73.loadmat(pathData + 'Dof2ARep.mat')
        dof3 = mat73.loadmat(pathData + 'Dof3ARep.mat')
    else:
        dof1 = mat73.loadmat(pathData + 'Dof1BRep.mat')
        dof2 = mat73.loadmat(pathData + 'Dof2BRep.mat')
        dof3 = mat73.loadmat(pathData + 'Dof3BRep.mat')



    K = 3 # fold number

    rangeSNR = range(1, 11)
    accuracy = np.zeros((K,len(rangeSNR)+1))
    accuracyTV = np.zeros((K,len(rangeSNR)+1))  # +1 for a clean

    rangeChLos = range(5, 51, 5)
    accuracyChLoss = np.zeros((K,len(rangeChLos)+1)) # +1 for a clean

    rangeChShif = range(-18, 19) #range(-23, 24)
    accuracyChShif = np.zeros((K,len(rangeChShif)))

    Fs = 2048
    window_size = int(np.floor(0.1*Fs))  #100ms
    overlap_factor =  0.5
    stride = int(window_size * overlap_factor)
    interOffset = window_size - stride
    batch_size = 32

    for ii in range(K):
        trSets = [x for x in range(3) if x!=ii]

        print ('----" Fold =' + str(ii +1) + ': Train Rep: ' + str(trSets) + ', Test Rep: ' + str(ii) + '  "----')
        print("\n", '---------------- divide class into train and test ----------------',"\n")
        tsM1 = dof1['Dof1Rep']['bdataPos'][ii]
        tsM2 = dof1['Dof1Rep']['bdataNeg'][ii]
        tsM3 = dof2['Dof2Rep']['bdataPos'][ii]
        tsM4 = dof2['Dof2Rep']['bdataNeg'][ii]
        tsM5 = dof3['Dof3Rep']['bdataPos'][ii]
        tsM6 = dof3['Dof3Rep']['bdataNeg'][ii]

        trM1 = np.concatenate((dof1['Dof1Rep']['bdataPos'][trSets[0]], dof1['Dof1Rep']['bdataPos'][trSets[1]]), axis=1) 
        trM2 = np.concatenate((dof1['Dof1Rep']['bdataNeg'][trSets[0]], dof1['Dof1Rep']['bdataNeg'][trSets[1]]), axis=1) 
        trM3 = np.concatenate((dof2['Dof2Rep']['bdataPos'][trSets[0]], dof2['Dof2Rep']['bdataPos'][trSets[1]]), axis=1) 
        trM4 = np.concatenate((dof2['Dof2Rep']['bdataNeg'][trSets[0]], dof2['Dof2Rep']['bdataNeg'][trSets[1]]), axis=1) 
        trM5 = np.concatenate((dof3['Dof3Rep']['bdataPos'][trSets[0]], dof3['Dof3Rep']['bdataPos'][trSets[1]]), axis=1) 
        trM6 = np.concatenate((dof3['Dof3Rep']['bdataNeg'][trSets[0]], dof3['Dof3Rep']['bdataNeg'][trSets[1]]), axis=1)


        trM = [trM1, trM2, trM3, trM4, trM5, trM6]
        tr_Cln = np.concatenate(trM, axis=1)

        tsM = [tsM1, tsM2, tsM3, tsM4, tsM5, tsM6]
        ts_Cln = np.concatenate(tsM, axis=1)

        print("\n", '---------------- test - clean  ----------------',"\n")
        ts_Cln_getFeatMat = getFeatureMatrix(ts_Cln, features, window_size, overlap_factor)

        print("\n", '---------------- Train data ----------------',"\n")
        tr_Cln_getFeatMat = getFeatureMatrix(tr_Cln, features, window_size, overlap_factor)

        print("\n", '---------------- Generate labels ----------------',"\n")
        labelsTsReal = np.concatenate((0*np.ones( (tsM1.shape[1],1)), 1* np.ones( (tsM2.shape[1],1)), 2* np.ones( (tsM3.shape[1],1)),
                        3*np.ones( (tsM4.shape[1],1)), 4*np.ones( (tsM5.shape[1],1)), 5*np.ones( (tsM6.shape[1],1))))
        labelsTrReal = np.concatenate((0* np.ones( (trM1.shape[1],1)), 1* np.ones( (trM2.shape[1],1)), 2* np.ones( (trM3.shape[1],1)),
                        3*np.ones( (trM4.shape[1],1)), 4*np.ones( (trM5.shape[1],1)), 5*np.ones( (trM6.shape[1],1))))

        labelsTs = labelsTsReal[0:ts_Cln_getFeatMat.shape[1]*interOffset:interOffset,:]
        labelsTr = labelsTrReal[0:tr_Cln_getFeatMat.shape[1]*interOffset:interOffset,:]

        # Convert the data and labels to tensors and create PyTorch datasets and dataloader for train datat
        X_train = torch.tensor(tr_Cln_getFeatMat.T, dtype=torch.float32)
        y_train = torch.tensor(labelsTr, dtype=torch.int64).long().view(-1)

        mean_train = torch.mean(X_train, dim=0)
        std_train = torch.std(X_train, dim=0)
        X_train_normalized = (X_train - mean_train) / std_train

        train_dataset = TensorDataset(X_train_normalized, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)



        # Convert the data and labels to tensors and create PyTorch datasets and dataloader for test clean
        X_test = torch.tensor(ts_Cln_getFeatMat.T, dtype=torch.float32)
        y_test = torch.tensor(labelsTs, dtype=torch.int64).long().view(-1)

        X_test_normalized = (X_test - mean_train) / std_train

        test_dataset = TensorDataset(X_test_normalized, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, drop_last=True)

        # compute the power of noise
        signal_power = np.sum(np.abs(tr_Cln)**2)/np.prod(tr_Cln.shape)

        # compute the gain of tv-noise
        gain = [(np.arange(0, X.shape[1]) / X.shape[1]).reshape(1, -1) for X in tsM]
        tv_noise_gain = np.concatenate(gain, axis=1)


        ts_awgn = [test_dataloader]    # intialize with clean data
        ts_awgn_TV = [test_dataloader] # intialize with clean data

        for SNR_db in rangeSNR:
            awgn, awgn_TV = add_awgn_tensor(ts_Cln, SNR_db, signal_power, tv_noise_gain, batch_size, y_test, features, window_size, overlap_factor,mean_train, std_train)
            ts_awgn.append(awgn)
            ts_awgn_TV.append(awgn_TV)

        ts_ch_loss = [test_dataloader] # intialize with clean data
        for ch_loss_percent in rangeChLos:
            ch_loss = remove_channels(ts_Cln, ch_loss_percent, features, window_size, overlap_factor, y_test, batch_size,mean_train, std_train)
            ts_ch_loss.append(ch_loss)

        ts_ch_shift = [] # intialize with empty
        for rawShift in rangeChShif:
            ch_shift = shift_channels(ts_Cln, rawShift, features, window_size, overlap_factor, y_test, batch_size,mean_train, std_train)
            ts_ch_shift.append(ch_shift)

        #--------------------- Define the hyperparameters ----------------
        num_epochs = 100
        input_size = X_train.shape[1]
        num_classes = len(np.unique(y_train.numpy()))
        hidden_size = int(input_size/3)

        num_epochs_ae = 25
        ae_input_size = input_size
        ae_hidden_size = int(input_size/2)


        if Classifier == 'AutoRNN':
            autoencoder = Autoencoder(input_size=input_size, hidden_size=ae_hidden_size)
            optimizer_ae = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
            criterion_ae = nn.MSELoss() # Define loss function
            # Train autoencoder
            for epoch in range(num_epochs_ae):
                for i, data in enumerate(train_dataloader, 0):
                    # Get inputs and target (same as input)
                    inputs, _ = data
                    target = inputs

                    # Forward pass
                    encoded, decoded = autoencoder(inputs)
                    loss = criterion_ae(decoded, target)

                    # Backward pass and optimize
                    optimizer_ae.zero_grad()
                    loss.backward()
                    optimizer_ae.step()

                if (epoch+1) % 5 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs_ae, i+1, len(train_dataloader), loss.item()))



        if Classifier == 'linear':
            model = LDA(num_classes, input_size)
        elif Classifier == 'NN':
            model = NN(input_size, hidden_size, num_classes)
        elif Classifier == 'deep':
            model = RNN(input_size, hidden_size, num_classes)
        elif Classifier == 'AutoRNN':
            model = AutoRNN(input_size, hidden_size, num_classes, ae_input_size, ae_hidden_size)
        else:
            raise ValueError("Invalid Classifier specified")


        # Define the loss function and optimizer
        criterion = nn.CrossEntropyLoss() 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, amsgrad=True)
        scheduler = torch.optim.lr_scheduler.OneCycleLR ( optimizer, max_lr=1e-2, anneal_strategy='linear', cycle_momentum=False, base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1e7, total_steps=num_epochs * len(train_dataloader))


        print('---------------------------- Train a classifier  ----------------------------')
        for epoch in range(num_epochs):
            running_loss = 0.0
            predicted_train = []
            correct = 0
            total = 0
            for i, (inputs, labels) in enumerate(train_dataloader, 0):
                model.train()
                optimizer.zero_grad()
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted_train =  np.concatenate((predicted_train, predicted.numpy()))
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                labels = labels.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() 
            scheduler.step()
            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = 100 * correct / total

            # Evaluate model on test dataset
            model.eval()
            correct_test = 0
            total_test = 0
            with torch.no_grad():
                for i, (inputs, labels) in enumerate(test_dataloader, 0):
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()
            epoch_acc_test = 100 * correct_test / total_test
            if (epoch+1) % 5 == 0:
                print('Epoch [{}/{}], Loss: {:.4f}, Train Acc: {:.4f}, Test Acc: {:.4f}'.format(epoch+1, num_epochs, epoch_loss, epoch_acc, epoch_acc_test))        


        for  jj  in range(len(ts_awgn)):
            accuracy[ii-1,jj]  = 100*compute_accuracy(model, ts_awgn[jj]) 
            accuracyTV[ii-1,jj]  = 100*compute_accuracy(model, ts_awgn_TV[jj])

        for jj in range(len(ts_ch_loss)):
            accuracyChLoss[ii-1,jj]  = 100*compute_accuracy(model, ts_ch_loss[jj]) 

        for jj in range(len(ts_ch_shift)):
            accuracyChShif[ii-1,jj]  = 100*compute_accuracy(model, ts_ch_shift[jj]) 

    # --------------------------------- save data --------------------------------
    savedAccuracy = {
        'accuracyAWGN': accuracy,
        'accuracyAWGNTV': accuracyTV,
        'accuracyChShif': accuracyChShif,
        'accuracyChLoss': accuracyChLoss}
    print(savedAccuracy)
    file_name = f'Sub_{subNum}_{Classifier}_{fName}.pickle'

    if Classifier == 'linear':
        folderNam = 'saveDataLin'
    elif Classifier == 'NN':
        folderNam = 'saveDataNN'
    elif Classifier == 'deep':
         folderNam = 'saveDataDeep'
    elif Classifier == 'AutoRNN':
         folderNam = 'saveDataAutoRNN'        


    file_path = os.path.join(folderNam, file_name)

    with open(file_path, 'wb') as f:
        pickle.dump(savedAccuracy, f)


if __name__=="__main__":
    print('start')
    import sys
    app3(int(sys.argv[1]),int(sys.argv[2]))