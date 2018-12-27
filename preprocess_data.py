""" __________________________ Importing Modules ___________________________"""


import os
import numpy as np
import pandas as pd


"""_________________________________________________________________________"""


def banknote(filename):
    """___________________ Banknote Authentication Data Set ___________________
    Description: Data were extracted from images that were taken from genuine
    and forged banknote-like specimens. For digitization, an industrial camera
    usually used for print inspection was used. The final images have 400 x 400
    pixels. Due to the object lens and distance to the investigated object
    gray-scale pictures with a resolution of about 660 dpi were gained.
    Wavelet transform tools were used to extract features from images.
    There are 1372 items (images of banknotes — think Euro or dollar bill).

    Goal: Given the features/predictor variables (explained below) extracted
    from the wavelet transform of banknote-like specimen images, predict if the
    specimen is authentic (0) or forgery (1).
    There are 4 predictor variables (variance of image, skewness, kurtosis,
    entropy).

    Number of Features: 4                                                     |
    Feature Information:                                                      |
    1. Variance of wavelet transformed image (continuous)                     |
    2. Skewness of wavelet transformed image (continuous)                     |
    3. Curtosis of wavelet transformed image (continuous)                     |
    4. Entropy of image (continuous)                                          |
    Class (integer: {0,1}): Binary label; authentic = 0 forgery =1            |
    From https://archive.ics.uci.edu/ml/datasets/banknote+authentication      |
    ________________________________________________________________________"""
    """ ____________________ Preprocessing the Data ________________________"""
    cwd_ = os.getcwd()
    path_ = os.path.join(cwd_, filename)
    raw_data = pd.read_csv(path_)
    num_features = 4
    '''   _____________________ Shuffle the Data ________________________   '''
    shuffle = np.random.permutation(raw_data.shape[0])
    raw_data_ = raw_data.iloc[shuffle]
    num_train = round(0.8*raw_data_.shape[0])
    '''   _______________________ Training Set __________________________   '''
    raw_data_train = raw_data_.iloc[0: num_train-1, :]
    features_train = raw_data_train.iloc[:, 0: num_features]
    features_train = features_train.apply(pd.to_numeric, errors='coerce')
    ones_col_vec = np.ones(features_train.iloc[:, 1].shape)

    # Standardize all features.
    for f in range(num_features):
        f_mean = np.mean(features_train.iloc[:, f])
        features_train.iloc[:, f] = features_train.iloc[:, f] -\
            f_mean*ones_col_vec
        f_std = np.std(features_train.iloc[:, f])
        features_train.iloc[:, f] = features_train.iloc[:, f]/f_std
    features_train.insert(0, 'bias', ones_col_vec)
    labels_train = raw_data_train.iloc[:, -1]
    # OPT:
    '''
    features_train.to_csv(
            os.path.join(cwd_, 'banknote_auth_training_features.csv'),
            sep=',', encoding='utf-8')
    
    labels_train.to_csv(
            os.path.join(cwd_, 'banknote_auth_training_labels.csv'),
            sep=',', encoding='utf-8')
    '''
    '''   __________________________ Test Set ___________________________   '''
    raw_data_test = raw_data_.iloc[num_train:, :]
    features_test = raw_data_test.iloc[:, 0: num_features]
    features_test = features_test.apply(pd.to_numeric, errors='coerce')
    ones_col_vec = np.ones(features_test.iloc[:, 1].shape)
    # OPT: print("Size of the Test Data Set: ",features_test.shape[0])

    # Standardize all features.
    for f in range(num_features):
        f_mean = np.mean(features_test.iloc[:, f])
        features_test.iloc[:, f] = features_test.iloc[:, f]-f_mean*ones_col_vec
        f_std = np.std(features_test.iloc[:, f])
        features_test.iloc[:, f] = features_test.iloc[:, f]/f_std
    features_test.insert(0, 'bias', ones_col_vec)
    labels_test = raw_data_test.iloc[:, -1]
    labels_test_list = labels_test.values
    forged_banknote = 1*(np.isin(labels_test_list, 1))  # True +
    authentic_banknote = 1*(np.isin(labels_test_list, 0))  # True -
    # OPT:
    '''
    features_test.to_csv(os.path.join(cwd_, 'banknote_auth_test_features.csv'),
                         sep=',', encoding='utf-8')
    labels_test.to_csv('banknote_auth_test_labels.csv', sep=',',
                       encoding='utf-8')
    '''
    return features_train, labels_train, features_test, labels_test,\
        forged_banknote, authentic_banknote


"""_________________________________________________________________________"""


def diabetes(filename):
    """___________________ Banknote Authentication Data Set ___________________
    Description: The data set consist of several medical predictor
    (independent) variables (explained below) and one target (dependent)
    variable, outcome.

    Goal: Given a patient's features/predictor variables (explained below),
    predict if the patient has diabetes (1) or not (0).

    Number of Features: 8                                                     |
    Feature Information:                                                      |
    1. The number of pregnancies the patient has had (integer)                |
    2. Patient's Plasma glucose concentration                                 |
    3. Patient's Diastolic blood pressure blood pressure (integer: mm Hg)     |
    4. Patient's triceps skin fold thickness (int: mm)                        |
    5. Insulin (integer: 2-Hour serum insulin (mu U/ml))                      |
    6. Patient's Body mass index (float: weight in kg/(height in m)^2)        |
    7. Patient's Diabetes pedigree function                                   |
    8. Patient's age (integer)                                                |
    Outcome (integer: {0,1}): Binary label; 268 are 1, 500 are 0              |
    https://www.kaggle.com/uciml/pima-indians-diabetes-database/version/1#_=_ |
    ________________________________________________________________________"""
    """ ____________________ Preprocessing the Data ________________________"""
    cwd_ = os.getcwd()
    path_ = os.path.join(cwd_, filename)
    raw_data = pd.read_csv(path_)
    num_features = 8
    '''   _____________________ Shuffle the Data ________________________   '''
    shuffle = np.random.permutation(raw_data.shape[0])
    raw_data_ = raw_data.iloc[shuffle]
    num_train = round(0.8*raw_data_.shape[0])
    '''   _______________________ Training Set __________________________   '''
    raw_data_train = raw_data_.iloc[0: num_train-1, :]
    features_train = raw_data_train.iloc[:, 0: num_features]
    features_train = features_train.apply(pd.to_numeric, errors='coerce')
    ones_col_vec = np.ones(features_train.iloc[:, 1].shape)

    # Standardize all features.
    for f in range(num_features):
        f_mean = np.mean(features_train.iloc[:, f])
        features_train.iloc[:, f] = features_train.iloc[:, f] -\
            f_mean*ones_col_vec
        f_std = np.std(features_train.iloc[:, f])
        features_train.iloc[:, f] = features_train.iloc[:, f]/f_std
    features_train.insert(0, 'bias', ones_col_vec)
    labels_train = raw_data_train.iloc[:, -1]
    # OPT:
    '''
    features_train.to_csv(
            os.path.join(cwd_, 'diabetes_train_features.csv'),
            sep=',', encoding='utf-8')
    labels_train.to_csv(
            os.path.join(cwd_, 'diabetes_training_labels.csv'),
            sep=',', encoding='utf-8')
    '''
    '''   __________________________ Test Set ___________________________   '''
    raw_data_test = raw_data_.iloc[num_train:, :]
    features_test = raw_data_test.iloc[:, 0: num_features]
    features_test = features_test.apply(pd.to_numeric, errors='coerce')
    ones_col_vec = np.ones(features_test.iloc[:, 1].shape)
    # OPT: print("Size of the Test Data Set: ",features_test.shape[0])

    # Standardize all features.
    for f in range(num_features):
        f_mean = np.mean(features_test.iloc[:, f])
        features_test.iloc[:, f] = features_test.iloc[:, f]-f_mean*ones_col_vec
        f_std = np.std(features_test.iloc[:, f])
        features_test.iloc[:, f] = features_test.iloc[:, f]/f_std
    features_test.insert(0, 'bias', ones_col_vec)
    labels_test = raw_data_test.iloc[:, -1]
    labels_test_list = labels_test.values
    pos_examples = 1*(np.isin(labels_test_list, 1))  # True +
    neg_examples = 1*(np.isin(labels_test_list, 0))  # True -
    # OPT:
    '''
    features_test.to_csv(os.path.join(cwd_, 'diabetes_test_features.csv'),
                         sep=',', encoding='utf-8')
    labels_test.to_csv('diabetes_test_labels.csv', sep=',',
                       encoding='utf-8')
    '''
    return features_train, labels_train, features_test, labels_test,\
        pos_examples, neg_examples
