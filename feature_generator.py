import numpy as np
from matplotlib import pyplot as plt
import joblib
import os
from random import shuffle
from feature_modules import *
from sklearn.model_selection import KFold



# init parameters
train_data_file         = "train_data.p"
train_label_file        = "train_labels.p"
test_data_file          = "val_data.p"
test_label_file         = "val_labels.p"
cutoff_freq             = 45
compute_train_features  = True

# Feature Controllers
is_stat_features        = True
is_entropy_features     = True
is_ar_features          = True
is_fft_features         = False
is_wavelet_features     = False
is_energy_features      = True
is_diff_features        = True


# Loading Data
train_data  = joblib.load(train_data_file)
train_label = joblib.load(train_label_file)
test_data   = joblib.load(test_data_file)
test_label  = joblib.load(test_label_file)

# train_val Shuffling ...
c = list(zip(train_data, train_label))
shuffle(c)
train_data, train_label = zip(*c)
n_train = len(train_label)


if compute_train_features:
    X_train = []
    Y_train = []
    for idx, (label, data) in enumerate(zip(train_label, train_data)):
        print(idx)
        features = []
        Y_train.append(label)
        data_processed = butter_lowpass_filter(data, cutoff_freq)

        if is_stat_features:
            # Statistical features
            features += stat_mean(data_processed)
            features += stat_variance(data_processed)
            features += stat_correlation(data_processed)

        if is_entropy_features:
            # Entropy features
            features += shannon_entropy(data_processed)
            features += renyi_entropy(data_processed,alpha_list = [-5, -2, -1, 0.5, 1.5, 2, 3, 5])
            features += tsalis_entropy(data_processed, q_list=[-5, -2, -1, 0.5, 1.5, 2, 3, 5])
            # features += Approximate_entropy(data, m=2)

        if is_ar_features:
            # AR features
            features += param_ar(data_processed, max_lag_list=[4, 8, 16, 32])

        if is_fft_features:
            # DCT & DST features
            features += freq_dct(data_processed)
            features += freq_dst(data_processed)

        if is_wavelet_features:
            # Wavelet features
            features += wavelet(data_processed, family_list=['haar', 'db2', 'db3', 'db4', 'db5'])

        if is_energy_features:
            # Energy features
            features += power_bond_set1(data)
            features += power_bond_set2(data)
            features += power_bond_set3(data)
            features += power_bond_set4(data)

        if is_diff_features:
            # Differential features
            features += var_dif_channels(data_processed)
            features += mean_dif_channels(data_processed)
            features += cor_dif_channels(data_processed)

        X_train.append(features)

    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

X_test = []
Y_test = []
for idx, (data, label) in enumerate(zip(test_data, test_label)):

    features = []
    Y_test.append(label)
    data_processed = butter_lowpass_filter(data, cutoff_freq)
    print(idx, data_processed.shape)

    if is_stat_features:
        # Statistical features
        features += stat_mean(data_processed)
        features += stat_variance(data_processed)
        features += stat_correlation(data_processed)

    if is_entropy_features:
        # Entropy features
        features += shannon_entropy(data_processed)
        features += renyi_entropy(data_processed, alpha_list=[-5, -2, -1, 0.5, 1.5, 2, 3, 5])
        features += tsalis_entropy(data_processed, q_list=[-5, -2, -1, 0.5, 1.5, 2, 3, 5])
        # features += Approximate_entropy(data, m=2)

    if is_ar_features:
        # AR features
        features += param_ar(data_processed, max_lag_list=[4, 8, 16, 32])

    if is_fft_features:
        # DCT & DST features
        features += freq_dct(data_processed)
        features += freq_dst(data_processed)

    if is_wavelet_features:
        # Wavelet features
        features += wavelet(data_processed, family_list=['haar', 'db2', 'db3', 'db4', 'db5'])

    if is_energy_features:
        # Energy features
        features += power_bond_set1(data)
        features += power_bond_set2(data)
        features += power_bond_set3(data)
        features += power_bond_set4(data)

    if is_diff_features:
        # Differential features
        features += var_dif_channels(data_processed)
        features += mean_dif_channels(data_processed)
        features += cor_dif_channels(data_processed)


    X_test.append(features)

X_test = np.array(X_test)
Y_test = np.array(Y_test)
print(X_test.shape, X_train.shape, Y_test.shape, Y_train)
if compute_train_features:
    joblib.dump(X_train, "TrainFeatures.p")
    joblib.dump(Y_train, "TrainLabels.p")

joblib.dump(X_test, "TestFeatures.p")
joblib.dump(Y_test, "TestLabels.p")

