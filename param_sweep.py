'''
File: cross_val.py
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import config
import _pickle as pickle
import os,math, csv
from Model import Model
import evaluate, visualize


'''
hidden_size (default 200)
lr (learning rate, default .0005)
beta (for mse, default .01)
lambd (for ls, default 1)
'''

def get_configs():
    n_layers = [1, 2, 3]
    sizes = [10, 50, 100, 150, 200]
    learning_rates = [0.1, 0.01, 0.001]
    alphas = [1, 2, 5]
    betas = [0.1, 0.01, 0.001, 0.0001]
    lambds = [0.1, 0.01, 0.001, 0.0001]
    n_epoch = [2, 10, 50, 100]
    #for i in range(2, 6):
    #    a = 10 ** (-i)
    #    b = 5 * 10 ** (-i)
    #    betas.append(a)
    #    betas.append(b)
    #    lambds.append(i)

    params = []
    for size in sizes:
        for lr in learning_rates:
            for alpha in alphas:
                for beta in betas:
                    for lambd in lambds:
                        for epoch in n_epoch:
                            for layer in n_layers:
                                params.append(config.Config(n_layers=layer, hidden_size=size, alpha = alpha, lr=lr, beta=beta, lambd=lambd, n_epochs = epoch))
    print("Trying out {} models".format(len(params)))

    return params

def main():
    data = pickle.load(open("data/data", "rb"))
    params = get_configs()
    random.shuffle(params)
    for i, param in enumerate(params): 
        avg_test = evaluate.evaluate_model(param)
        path = "./plots/" + param.name + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        
        visualize.plot_summary_by_dset(data, path)
        visualize.plot_aggregate_summary(data, path)
        visualize.plot_seq_summary(data, path)
                    
        if not os.path.isfile("sweep.csv"):
            with open("sweep.csv", "w") as csvfile:
                top = ["#_layers", "hidden_size", "learning_rate", "alpha", "beta", "lambds", "#_epochs"]
                for dset in config.ALLDATA_SINGLE:
                    top.append(dset + "_corr")
                    top.append(dset + "_squared")
                writer = csv.writer(csvfile)
                writer.writerow(top)
            
        with open("sweep.csv", "a+") as csvfile:
            writer = csv.writer(csvfile)
            layer = [param.n_layers, param.hidden_size, param.lr, param.alpha, param.beta, param.lambd, param.n_epochs]
            for dset in config.ALLDATA_SINGLE:
                layer.append(np.mean(avg_test[dset][0]))
                layer.append(np.mean(avg_test[dset][1]))
            writer.writerow(layer) 

if __name__ == '__main__':
    main()
