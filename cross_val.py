'''
File: cross_val.py
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import config
import _pickle as pickle
import os
from diffscore import Model
from sklearn.model_selection import KFold

'''
hidden_size (default 200)
lr (learning rate, default .0005)
beta (for mse, default .01)
lambd (for ls, default 1)
'''

def get_configs():
    sizes = [100, 200, 300, 400]
    learning_rates = []
    betas = []
    lambds = []
    for i in range(2, 6):
        a = 10 ** (-i)
        b = 5 * 10 ** (-i)
        learning_rates.append(a)
        learning_rates.append(b)
        betas.append(a)
        betas.append(b)
        lambds.append(a)
        lambds.append(b)

    params = []
    for size in sizes:
        for lr in learning_rates:
            for beta in betas:
                for lambd in lambds:
                    params.append(config.Config(hidden_size=size, lr=lr, beta=beta, lambd=lambd))
    print("Trying out {} models".format(len(params)))

    return params

def main():
    k = 5
    train_data = pickle.load(open("train", "rb"))
    dev_data = pickle.load(open("dev","rb"))
    test_data = pickle.load(open("test", "rb"))
    
    data = pd.concat([train_data, dev_data])
    kf = KFold(n_splits = k, shuffle = True)
    output = []
    params = get_configs()
    for param in [params[0]]:
        corr, squared = 0, 0
        i = 0
        for train_index, dev_index in kf.split(data):
            param.define_crossval(i)
            train_data = data.iloc[train_index]
            dev_data = data.iloc[dev_index]
            model = Model(param, True)
            model.initialize()
            model.fit(train_data, dev_data)
            dcorr, dsquared = model.evaluate(dev_data)
            if int(dcorr*100) != 0:
                corr += dcorr
                squared += dsquared
            model.sess.close()
            i += 1
        output.append("lr: " + str(param.lr) + " hidden_size: " + str(param.hidden_size) + " beta: " + str(param.beta) + " lambd: " + str(param.lambd) + " corr: " + str(corr/k) + " Squared loss: " + str(squared/k))
    with open("output","w+") as f:
        for item in output:
            f.write(item + "\n")
    
        

if __name__ == '__main__':
    main()
