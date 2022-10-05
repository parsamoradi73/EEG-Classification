from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

def tree_fea_selection(X_train, Y_train, X_test, n_selected_feature):
    clf = ExtraTreesClassifier(n_estimators=n_selected_feature)
    clf = clf.fit(X_train, Y_train)

    model = SelectFromModel(clf, prefit=True)
    X_train_new = model.transform(X_train)

    X_test_new = model.transform(X_test)

    return X_train_new, X_test_new

def RFE_selection(X_train, Y_train, X_test, estimator, n_selected_feature):
    selector = RFE(estimator, n_selected_feature, step=1)
    selector = selector.fit(X_train, Y_train)
    X_train_new = X_train[:, selector.support_]
    X_test_new = X_test[:, selector.support_]

    return X_train_new, X_test_new

def kBest_selection(X_train, Y_train, X_test, n_selected_feature):
    selection_model = SelectKBest(f_classif, k=n_selected_feature)
    selection_model.fit(X_train, Y_train)
    X_train_new = selection_model.transform(X_train)
    X_test_new = selection_model.transform(X_test)
    return X_train_new, X_test_new



def pca_selection(X_train, Y_train, X_test, n_selected_feature):

    pca = PCA()
    ev = pca.explained_variance_ratio_
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)


