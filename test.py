import numpy as np
import joblib
from sklearn.model_selection import KFold
from feature_selection import *
from Classifying import mlp_classify, rf_classify, linearsvm_classify, polysvm_classify, knn_classify, adaboost_classify,\
        ensemble_classify_hardVoting, LDA_classify
from sklearn.svm import SVR
from sklearn.metrics import f1_score

# init parameters
k_folds = 5
feature_selection_method = 'tree' # tree/RFE/KBest
n_final_feature = 10

# Load Data
X = joblib.load("TrainFeatures.p")
Y = joblib.load("TrainLabels.p")

X = np.array(X, dtype=np.float32)

n_repeat = 1000

# Start Learning
ens_accs, ens_f1   = [], []
lda_accs, lda_f1   = [], []

kf = KFold(n_splits=k_folds, shuffle=True)
for KK, (train_idx, val_idx) in enumerate(kf.split(X)):

    # print("Fold: ", KK)
    for ii in range(n_repeat):

        X_train = X[train_idx]
        Y_train = Y[train_idx]
        X_val   = X[val_idx]
        Y_val   = Y[val_idx]


        # feature normalization
        mean_train = np.mean(X_train, axis=0)
        std_train  = np.std(X_train, axis=0)

        X_train = (X_train - mean_train) / std_train
        X_val   = (X_val - mean_train) / std_train

        # feature selection
        if feature_selection_method == 'tree':
            X_train, X_val = tree_fea_selection(X_train, Y_train, X_val, n_final_feature)
        elif feature_selection_method == 'RFE':
            estimator = SVR(kernel="linear")
            X_train, X_val = RFE_selection(X_train, Y_train, X_val, estimator, n_final_feature)

        elif feature_selection_method =='KBest':
            X_train, X_val = kBest_selection(X_train, Y_train, X_val, n_final_feature)

        print(ii, X_train.shape)
        lda_cur_acc, _, lda_cur_f1, lda_result = LDA_classify(X_train, Y_train, X_val, Y_val)


        # ens_accs.append(ens_cur_acc)
        # ens_f1.append(ens_cur_f1)


        # lda_accs.append(lda_cur_acc)
        # lda_f1.append(lda_cur_f1)

        if ii > 0:
            lda_final += lda_result
        else:
            lda_final = lda_result

    y_pred_lda = np.round(lda_final / n_repeat)

    lda_accs.append(np.float16(np.sum(y_pred_lda == Y_val)) / Y_val.shape[0])
    lda_f1.append(f1_score(Y_val, y_pred_lda))

print("############################## ACCuracy")
print("Test Acc for LDA Classifier: ", sum(lda_accs)/k_folds, lda_accs)

print("############################## F1-Score")
print("Test F1 for LDA Classifier: ", sum(lda_f1)/k_folds, lda_f1)


