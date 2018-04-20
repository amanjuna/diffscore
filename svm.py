#!/usr/bin/env python

import sklearn as skl
import numpy as np
import pandas as pd
from sklearn.svm import SVR

import config

class Model():
    def __init__(self, config, verbose):
        pass 
        
    def initialize(self):
        self.clf = SVR(C = 1.0, epsilon=0.01)

    def format_dataset(self, dataset, n=config.NUM_CELLS_IN_DATASET):
        dsets = list(dataset.index.unique())
        X = np.zeros((len(dsets), n, self.config.n_features))
        y = np.zeros((len(dsets), n, 1))
        for i, dset in enumerate(dsets):
            data = dataset.loc[dset]
            data = data.sample(n=n, replace=True)
            X[i] = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
            y[i] = np.matrix(data["Standardized_Order"].as_matrix()).T
        return X, np.ravel(y)
        
    def fit(self, train, dev):
        data = pd.concat([train, dev])
        X = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
        y = np.matrix(data["Standardized_Order"].as_matrix()).T
        y = np.ravel(y)
        self.clf.fit(X, y)
        
    def make_pred(self, data):
        X = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
        return self.clf.predict(X)
