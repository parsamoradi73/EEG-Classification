import sklearn.svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from mlxtend.classifier import EnsembleVoteClassifier
from sklearn.model_selection import cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.decomposition import PCA


def mlp_classify(x_train, y_train, x_test, y_test, is_train = True):
    mlp = MLPClassifier(hidden_layer_sizes=(15, ), max_iter=2000,activation='logistic')

    # scores = cross_val_score(mlp, x_train, y_train, cv=5, scoring='accuracy')
    # print("MLP cross vval Accuracy: %0.2f ", scores)

    mlp.fit(x_train,y_train)
    prediction_train = mlp.predict(x_train)
    prediction_test = mlp.predict(x_test)

    if is_train:
        acc_test = np.float16(np.sum(prediction_test == y_test))/ y_test.shape[0]
        acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]

        f1_measure = f1_score(y_test, prediction_test)

        return acc_test, acc_train, f1_measure, prediction_test
    else:
        return prediction_test


def rf_classify(x_train, y_train, x_test, y_test, is_train=True):
    rf = RandomForestClassifier(n_estimators=20)  # initialize

    # scores = cross_val_score(rf, x_train, y_train, cv=5, scoring='accuracy')
    # print("RF cross vval Accuracy: %0.2f ", scores)

    rf.fit(x_train, y_train)  # fit the data to the algorithm
    prediction_test = rf.predict(x_test)
    prediction_train = rf.predict(x_train)

    if is_train:
        acc_test = np.float16(np.sum(prediction_test == y_test)) / y_test.shape[0]
        acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]

        f1_measure = f1_score(y_test, prediction_test)

        return acc_test, acc_train, f1_measure, prediction_test
    else:
        return prediction_test

def linearsvm_classify(x_train, y_train, x_test, y_test, is_train=True):
    kernelType = 'linear'
    model = sklearn.svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
                            decision_function_shape='ovo', gamma='auto', kernel=kernelType, degree=5,
                            max_iter=-1, probability=True, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)

    # scores = cross_val_score(model, x_train, y_train, cv=5, scoring='accuracy')
    # print("LSVM cross vval Accuracy: %0.2f ", scores)

    model.fit(x_train, y_train)
    prediction_test = model.predict(x_test)
    prediction_train = model.predict(x_train)

    #print prediction_test
    if is_train:
        acc_test = np.float16(np.sum(prediction_test == y_test))/ y_test.shape[0]
        acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]

        f1_measure = f1_score(y_test, prediction_test)

        return acc_test, acc_train, f1_measure, prediction_test
    else:
        return prediction_test

def polysvm_classify(x_train, y_train, x_test, y_test):
    model = sklearn.svm.SVC(C=1.0, cache_size=100, class_weight=None, coef0=0.0,
    decision_function_shape='ovo', gamma='auto', kernel='poly',degree=2,
    max_iter=-1, probability=False, random_state=None, shrinking=True,
    tol=0.001, verbose=False).fit(x_train, y_train)
    prediction_test = model.predict(x_test)
    prediction_train = model.predict(x_train)
    #print prediction_test
    acc_test = np.float16(np.sum(prediction_test == y_test))/ y_test.shape[0]
    acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]
    return acc_test,acc_train


def knn_classify(x_train, y_train, x_test, y_test, is_train=True):

    neighbors = KNeighborsClassifier(n_neighbors=11, p=2, algorithm='auto')
    neighbors.fit(x_train, y_train)
    prediction_test = neighbors.predict(x_test)
    prediction_train = neighbors.predict(x_train)

    if is_train:
        acc_test = np.float16(np.sum(prediction_test == y_test)) / y_test.shape[0]
        acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]

        f1_measure = f1_score(y_test, prediction_test)

        return acc_test, acc_train, f1_measure, prediction_test
    else:
        return prediction_test


def adaboost_classify(x_train, y_train, x_test, y_test, is_train=True):
    bdt = AdaBoostClassifier(algorithm="SAMME", n_estimators=50)
    # scores = cross_val_score(bdt, x_train, y_train, cv=5, scoring='accuracy')
    # print("ADA cross vval Accuracy: %0.2f ", scores)

    bdt.fit(x_train, y_train)
    prediction_train = bdt.predict(x_train)
    prediction_test = bdt.predict(x_test)

    if is_train:
        acc_test = np.float16(np.sum(prediction_test == y_test)) / y_test.shape[0]
        acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]

        f1_measure = f1_score(y_test, prediction_test)

        return acc_test, acc_train, f1_measure, prediction_test
    else:
        return prediction_test

def ensemble_classify_hardVoting(x_train, y_train, x_test, y_test, is_train = True):
    mlp     = MLPClassifier(hidden_layer_sizes=(15, ), max_iter=2000, activation='logistic')
    rf      = RandomForestClassifier(n_estimators=20)  # initialize
    lsvm    = sklearn.svm.SVC(C=0.1, cache_size=200, class_weight=None, coef0=0.0,
                            decision_function_shape='ovo', gamma='auto', kernel='linear', degree=5,
                            max_iter=-1, probability=True, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)
    neighbors = KNeighborsClassifier(n_neighbors=11, p=2, algorithm='auto')
    ada       = AdaBoostClassifier(algorithm="SAMME", n_estimators=50)
    lda       = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.015)

    clf = [lsvm,mlp,lda,ada,rf]
    eclf = EnsembleVoteClassifier(clfs=clf, voting='hard')

    # scores = cross_val_score(eclf, x_train, y_train, cv=5, scoring='accuracy')
    # print("ENS cross vval Accuracy: %0.2f ", scores)

    eclf.fit(x_train, y_train)
    prediction_train = eclf.predict(x_train)
    prediction_test = eclf.predict(x_test)
    if is_train:
        acc_test = np.float16(np.sum(prediction_test == y_test)) / y_test.shape[0]
        acc_train = np.float16(np.sum(prediction_train == y_train)) / y_train.shape[0]
        f1_measure = f1_score(y_test, prediction_test)

        return acc_test, acc_train, f1_measure, prediction_test
    else:
        return prediction_test


def LDA_classify(X_train, Y_train, X_test, Y_test, score='accuracy', k_fold_num=5, is_train=True):
    # print("# Tuning hyper-parameters for %s" % score)
    # tuned_parameters = [{'solver': ['lsqr'], 'shrinkage': [.005, .01, .015, .020, .025, .030, .035, .04, .045, 0.05]}]
    # clf = GridSearchCV(LinearDiscriminantAnalysis(), tuned_parameters, cv=k_fold_num,
    #                    scoring=score)

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.015)
    clf.fit(X_train, Y_train)

    # print("Best parameters set found on development set:")
    # print(clf.best_params_)
    y_pred_test = clf.predict(X_test)
    y_pred_train = clf.predict(X_train)

    if is_train:
        acc_test = np.float16(np.sum(y_pred_test == Y_test)) / Y_test.shape[0]
        acc_train = np.float16(np.sum(y_pred_train == Y_train)) / Y_train.shape[0]

        f1_measure = f1_score(Y_test, y_pred_test)

        return acc_test, acc_train, f1_measure, y_pred_test
    else:
        return y_pred_test


def LDA_classify_all(X,Y, k_fold_num=5, is_train=True):
    # print("# Tuning hyper-parameters for %s" % score)
    # tuned_parameters = [{'solver': ['lsqr'], 'shrinkage': [.005, .01, .015, .020, .025, .030, .035, .04, .045, 0.05]}]
    # clf = GridSearchCV(LinearDiscriminantAnalysis(), tuned_parameters, cv=k_fold_num,
    #                    scoring=score)

    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=0.015)
    eclf = make_pipeline(preprocessing.StandardScaler(),PCA(), clf)

    scores = cross_val_score(eclf, X, Y, cv=5, scoring='accuracy')
    print("LDA cross vval Accuracy: %0.2f ", np.mean(scores), scores)

