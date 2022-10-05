import numpy as np
import joblib
from sklearn.model_selection import KFold
from feature_selection import *
from classifying import mlp_classify, rf_classify, linearsvm_classify, polysvm_classify, knn_classify, adaboost_classify,\
        ensemble_classify_hardVoting, LDA_classify, LDA_classify_all
from sklearn.svm import SVR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score

# init parameters
k_folds = 5
feature_selection_method = 'tree' # tree/RFE/KBest
n_final_feature = 10

# Load Data
X = joblib.load("TrainFeatures.p")
Y = joblib.load("TrainLabels.p")

X = np.array(X, dtype=np.float32)
Y = np.array(Y, dtype=np.float32)

# Start Learning
rf_accs, rf_f1     = [], []
ada_accs, ada_f1   = [], []
ens_accs, ens_f1   = [], []
lda_accs, lda_f1   = [], []

kf = KFold(n_splits=k_folds, shuffle=True, random_state=12321)
for KK, (train_idx, val_idx) in enumerate(kf.split(X)):

    print("Fold: ", KK)

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
        estimator = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.015)
        X_train, X_val = RFE_selection(X_train, Y_train, X_val, estimator, n_final_feature)

    elif feature_selection_method =='KBest':
        X_train, X_val = kBest_selection(X_train, Y_train, X_val, n_final_feature)

    print(X_train.shape)

    rf_cur_acc, _, rf_cur_f1, _ = rf_classify(X_train, Y_train, X_val, Y_val)
    ada_cur_acc, _, ada_cur_f1, _ = adaboost_classify(X_train, Y_train, X_val, Y_val)
    ens_cur_acc, _, ens_cur_f1, ens_result = ensemble_classify_hardVoting(X_train, Y_train, X_val, Y_val)
    lda_cur_acc, aa_1, lda_cur_f1, lda_result = LDA_classify(X_train, Y_train, X_val, Y_val)

    rf_accs.append(rf_cur_acc)
    rf_f1.append(rf_cur_f1)

    ada_accs.append(ada_cur_acc)
    ada_f1.append(ada_cur_f1)

    ens_accs.append(ens_cur_acc)
    ens_f1.append(ens_cur_f1)

    lda_accs.append(lda_cur_acc)
    lda_f1.append(lda_cur_f1)

print("############################## ACCuracy")
print("Test Acc for ADA Classifier: ", sum(ada_accs)/k_folds, ada_accs)
print("Test Acc for ENS Classifier: ", sum(ens_accs)/k_folds, ens_accs)
print("Test Acc for RF Classifier: ", sum(rf_accs)/k_folds, rf_accs)
print("Test Acc for LDA Classifier: ", sum(lda_accs)/k_folds, lda_accs)

print("############################## F1-Score")
print("Test F1 for ADA Classifier: ", sum(ada_f1)/k_folds, ada_f1)
print("Test F1 for ENS Classifier: ", sum(ens_f1)/k_folds, ens_f1)
print("Test F1 for RF Classifier: ", sum(rf_f1)/k_folds, rf_f1)
print("Test F1 for LDA Classifier: ", sum(lda_f1)/k_folds, lda_f1, aa_1)

X_test = joblib.load("TestFeatures.p")
Y_test = joblib.load("TestLabels.p")
print(X.shape, X_test.shape)
n_repeat = 100
mean_train = np.mean(X, axis=0)
std_train = np.std(X, axis=0)

X       = (X - mean_train)/std_train
X_test  = (X_test - mean_train)/std_train

for ii in range(n_repeat):
    X_cur, X_test_cur = tree_fea_selection(X, Y, X_test, n_final_feature)

    test_prediction_ada = adaboost_classify(X_cur, Y, X_test_cur, None, is_train=False)
    test_prediction_ens = ensemble_classify_hardVoting(X_cur, Y, X_test_cur, None, is_train=False)
    test_prediction_rf  = rf_classify(X_cur, Y, X_test_cur, None, is_train=False)
    test_prediction_lda = LDA_classify(X_cur, Y, X_test_cur, None, is_train=False)
    print(ii, X_test_cur.shape)

    if ii == 0:
        lda_final = test_prediction_lda
        ens_final = test_prediction_ens
        ada_final = test_prediction_ada
        rf_final  = test_prediction_rf
    else:
        lda_final += test_prediction_lda
        ens_final += test_prediction_ens
        rf_final  += test_prediction_rf
        ada_final += test_prediction_ada

lda_final = np.uint8(np.round(lda_final/n_repeat))
ada_final = np.uint8(np.round(ada_final/n_repeat))
rf_final  = np.uint8(np.round(rf_final/n_repeat))
ens_final = np.uint8(np.round(ens_final/n_repeat))

acc_test_lda    = np.float16(np.sum(lda_final == Y_test)) / Y_test.shape[0]
f1_measure_test_lda  = f1_score(Y_test, lda_final)

acc_test_ada    = np.float16(np.sum(ada_final == Y_test)) / Y_test.shape[0]
f1_measure_test_ada  = f1_score(Y_test, ada_final)

acc_test_rf    = np.float16(np.sum(rf_final == Y_test)) / Y_test.shape[0]
f1_measure_test_rf  = f1_score(Y_test, rf_final)

acc_test_ens    = np.float16(np.sum(ens_final == Y_test)) / Y_test.shape[0]
f1_measure_test_ens  = f1_score(Y_test, ens_final)

print("############################## ACCuracy")
print("Test Acc for ADA Classifier: ", acc_test_ada)
print("Test Acc for ENS Classifier: ",  acc_test_ens)
print("Test Acc for RF Classifier: ",  acc_test_rf)
print("Test Acc for LDA Classifier: ",  acc_test_lda)

print("############################## F1-Score")
print("Test F1 for ADA Classifier: ", f1_measure_test_ada)
print("Test F1 for ENS Classifier: ", f1_measure_test_ens)
print("Test F1 for RF Classifier: ", f1_measure_test_rf)
print("Test F1 for LDA Classifier: ", f1_measure_test_lda)

