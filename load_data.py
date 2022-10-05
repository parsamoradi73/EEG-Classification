import numpy as np
from scipy.io import loadmat
import os
import joblib

disorder_train_path = "New_Shuffled_Train(disorder)"
normal_train_path = "New_Shuffled_Train(normal)"
val_path_normal = "Validation_disorder"
val_path_disorder = "Validation_normal"

labels_train = []
eeg_train_data = []
eeg_val_data = []
labels_val = []
for f in os.listdir(disorder_train_path):
    if f.endswith(".mat"):
        data_dict = loadmat(os.path.join(disorder_train_path, f), appendmat=False)
        for k in data_dict.keys():
            if k.startswith("subj"):
                data = data_dict[k]
                eeg_train_data.append(data)
                labels_train.append(1)

for f in os.listdir(normal_train_path):
    if f.endswith(".mat"):
        data_dict = loadmat(os.path.join(normal_train_path, f), appendmat=False)
        for k in data_dict.keys():
            if k.startswith("subj"):
                data = data_dict[k]
                eeg_train_data.append(data)
                labels_train.append(0)

for f in os.listdir(val_path_disorder):
    if f.endswith(".mat"):
        data_dict = loadmat(os.path.join(val_path_disorder, f), appendmat=False)
        for k in data_dict.keys():
            if k.startswith("subj"):
                data = data_dict[k]
                eeg_val_data.append(data)
                labels_val.append(1)

for f in os.listdir(val_path_normal):
    if f.endswith(".mat"):
        data_dict = loadmat(os.path.join(val_path_normal, f), appendmat=False)
        for k in data_dict.keys():
            if k.startswith("subj"):
                data = data_dict[k]
                eeg_val_data.append(data)
                labels_val.append(0)


print(len(labels_train), len(eeg_train_data), len(eeg_val_data))
joblib.dump(eeg_train_data, "train_data.p")
joblib.dump(labels_train, "train_labels.p")
joblib.dump(eeg_val_data, "val_data.p")
joblib.dump(labels_val, "val_labels.p")