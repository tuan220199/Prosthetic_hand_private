import pandas as pd
import numpy as np
import torch
import os
import seaborn as sns


DEVICE = "cpu"

def getFeatureMatrix(rawDataMatrix, windowLength, windowOverlap):
    """
    Calculate the feature matrix (rms) from raw data matrix using a sliding window approach.

    Args:
        rawDataMatrix (numpy.ndarray): 2D array containing the raw data, where each row represents a channel and each column represents a sample.
        windowLength (int): Length of the sliding window in samples.
        windowOverlap (int): Number of samples overlapping between consecutive windows.

    Returns:
        numpy.ndarray: Feature matrix calculated from the raw data matrix using the sliding window approach.
    """
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
    """
    Load and preprocess data from CSV files and extract features for machine learning.

    Args:
        position (int, optional): Position identifier for the directory containing CSV files. Defaults to 0.

    Returns:
        tuple: A tuple containing four numpy arrays:
            - train_features: Training features extracted from the data.
            - train_labels: Labels corresponding to the training features.
            - test_features: Test features extracted from the data.
            - test_labels: Labels corresponding to the test features.

    This function loads data from CSV files located in a directory specified by 'position'.
    It preprocesses the data, calculates features using a sliding window approach, and splits the data into training and testing sets based on file names.
    Features are extracted using a sliding window approach with specified window length and overlap.
    The function returns four numpy arrays containing training features, training labels, test features, and test labels.

    Note:
        The function assumes the availability of the getFeatureMatrix function for feature extraction.

    """
    Fs = 500
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
    """
    Roll (circular shift) the columns of a 2D numpy array.

    Args:
        X_0 (numpy.ndarray): Input 2D array.
        shift (int): Number of positions to shift the columns. Positive values shift to the left, negative values shift to the right.

    Returns:
        numpy.ndarray: The input array with its columns rolled (shifted).

    Example:
        >>> X_0 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> roll_data(X_0, 1)
        array([[3, 1, 2],
               [6, 4, 5],
               [9, 7, 8]])
    """
    return np.roll(X_0,shift,1)

def get_all_data(X, y, v_shift = None):
    """
    Generate augmented dataset by rolling the input data along columns.

    Args:
        X (list of numpy.ndarray): List of 2D numpy arrays representing input data.
        y (numpy.ndarray): 1D numpy array representing target labels.
        v_shift (int or list of int, optional): Vertical shift to apply to each input array. 
            If not provided, a range of shifts from -4 to 3 will be applied.

    Returns:
        tuple: A tuple containing three numpy arrays:
            - all_X: Augmented input data with rolled columns.
            - all_y: Corresponding target labels.
            - all_shift: Shift values applied to each sample.

    If v_shift is not provided, the function applies a range of shifts from -4 to 3 to the input data X.
    It rolls each input array along its columns and concatenates them into a single augmented dataset.
    The corresponding labels y are duplicated accordingly.
    The function returns three numpy arrays: all_X containing the augmented input data, all_y containing the corresponding labels,
    and all_shift containing the applied shift values for each sample.

    If v_shift is provided, the function applies the specified vertical shift to each input array in X.
    The labels and shift values are duplicated accordingly, and the function returns the concatenated arrays.

    Examples:
        >>> X = [np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9], [10, 11, 12]])]
        >>> y = np.array([0, 1])
        >>> all_X, all_y, all_shift = get_all_data(X, y)
    """
    if not v_shift:
        all_X = np.zeros([X.shape[0],8, 8],dtype='float32') # 3D array: No. samples, 8: 8 features, 8: -4=>3 roll
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

def get_shift_data(all_X, all_shift, all_y):
    """
    Combine the original feature data with shifted feature data for each class.

    Args:
        all_X (numpy.ndarray): Original feature data matrix of shape (n_samples, n_features).
        all_shift (numpy.ndarray): Shifted feature data matrix of shape (n_samples, n_shift_features).
        all_y (numpy.ndarray): Array of class labels of shape (n_samples,).

    Returns:
        numpy.ndarray: Concatenated feature data matrix with original features for each class.
        numpy.ndarray: Concatenated feature data matrix with shifted features for each class.
        numpy.ndarray: Shift labels corresponding to the original feature data.
        numpy.ndarray: Shift labels corresponding to the shifted feature data.
        numpy.ndarray: Array of class labels for the concatenated data.

    This function combines the original feature data with shifted feature data for each class. It concatenates the
    original feature data (all_X) with the shifted feature data (all_shift) along the feature axis. It also shuffles
    the shifted feature data within each class to create a corresponding set of shifted features (all_X2).
    Additionally, it constructs arrays for the shift labels (all_shift_1 and all_shift_2) and the class labels (all_y_).
    The function returns all these arrays for further processing.

    Note:
        The function assumes that the original feature data (all_X) and the shifted feature data (all_shift) have the
        same number of samples and that each sample corresponds to a class label in the array all_y.
    """
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

def get_shift_data1(h_X,v_X, h_shift, v_shift, h_y, v_y):
    """
    Process horizontal and vertical data with shifts for classification.

    Args:
        h_X (numpy.ndarray): Horizontal feature data.
        v_X (numpy.ndarray): Vertical feature data.
        h_shift (numpy.ndarray): Horizontal shift data.
        v_shift (numpy.ndarray): Vertical shift data.
        h_y (numpy.ndarray): Labels for horizontal data.
        v_y (numpy.ndarray): Labels for vertical data.

    Returns:
        tuple: A tuple containing:
            all_X1 (numpy.ndarray): Processed horizontal feature data excluding shifts.
            all_X2 (numpy.ndarray): Processed vertical feature data excluding shifts.
            h_shift (numpy.ndarray): Shift data for horizontal data.
            v_shift (numpy.ndarray): Shift data for vertical data.
            h_y (numpy.ndarray): Labels for horizontal data.

    This function processes the horizontal and vertical feature data with their respective shifts
    for classification purposes. It concatenates the feature data with the shift data, shuffles 
    them, and then separates them back into their respective feature data arrays (excluding shifts)
    and shift arrays.

    Each class label is processed separately, and the resulting arrays contain data specific to 
    each class label.
    """
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
    """
    Generate rotation operators for quantum circuits.

    Args:
        _n_rotations (int): Total number of rotation operators to generate.
        order (int): Order of the rotation operators.

    Returns:
        tuple: A tuple containing:
            used_bases (list): List of rotation operators used in the quantum circuits.
            phi1 (torch.Tensor): Base rotation operator matrix.

    This function generates rotation operators for quantum circuits based on the specified order
    and the total number of rotation operators. It creates a base rotation operator matrix `phi1`
    with diagonal entries shifted by one position to the left. Then, it generates rotation operators
    by raising `phi1` to various powers up to `_n_rotations`. Finally, it selects a subset of these
    rotation operators (`used_bases`) for use in the quantum circuits, consisting of the first 17 and
    last 16 rotation operators from the generated list.
    """
    M = torch.diag(torch.ones(order)).roll(-1,1)

    phi1 = torch.zeros((8*order, 8*order))
    for i in range (8):
        phi1[order*i:order*(i+1), order*i:order*(i+1)] = M
    bases = [torch.linalg.matrix_power(phi1,i).to(DEVICE) for i in range (0, _n_rotations)]
    used_bases = bases[:17] + bases[-16:]

    return used_bases, phi1

def plot_cfs_mat(predicted, labels):
    """
    Plot a confusion matrix heatmap based on predicted and actual labels.

    Args:
        predicted (numpy.ndarray): Array containing predicted labels.
        labels (numpy.ndarray): Array containing actual labels.

    Returns:
        seaborn.matrix.ClusterGrid: Seaborn cluster grid object representing the confusion matrix heatmap.

    This function generates a confusion matrix heatmap based on the predicted and actual labels.
    It first initializes an empty confusion matrix `cf_mat` with dimensions (6, 6). Then, it iterates
    through each sample in the predicted and actual label arrays and increments the corresponding entry
    in the confusion matrix. Finally, it plots the confusion matrix heatmap using Seaborn's heatmap
    function, normalizing the values to the range [0, 1] and annotating each cell with the respective
    count of occurrences.
    """
    cf_mat = np.zeros((6,6))
    for i in range(predicted.shape[0]):
        cf_mat[predicted[i], labels[i]] += 1

    return sns.heatmap(cf_mat/6, annot=True,cmap='Blues', cbar=False)

def get_centroids(encoder, loader):
    """
    Calculate the centroids of each class using the provided encoder and data loader.

    Args:
        encoder (torch.nn.Module): Encoder model used to encode input data.
        loader (torch.utils.data.DataLoader): DataLoader containing the input data.

    Returns:
        torch.Tensor: Tensor containing the centroids of each class.

    This function calculates the centroids of each class using the provided encoder model and data loader.
    It first sets the encoder to evaluation mode and initializes tensors to store the centroids, counts, and
    flags indicating if the first instance of each class has been encountered. Then, it iterates through the
    data loader, encoding each input sample using the encoder and updating the corresponding centroid and
    count tensors. Finally, it divides the accumulated centroid vectors by the corresponding counts to obtain
    the centroids of each class and returns the result.
    """
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