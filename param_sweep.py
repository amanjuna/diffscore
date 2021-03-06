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
import random, evaluate, visualize


'''
hidden_size (default 200)
lr (learning rate, default .0005)
beta (for mse, default .01)
lambd (for ls, default 1)
'''

def get_configs():
    sizes = [10, 50, 100]
    learning_rates = [-7, -1]
    betas = [1e-4, 1]
    lambds = [-5, 1]

    params = []
    for size in sizes:
        for rate in np.logspace(*learning_rates, num=5):
            for _ in range(6):
                params.append(config.Config(n_layers=2, hidden_size=size,
                                            n_epochs=100,
                                            lr=np.random.uniform(rate),
                                            alpha=0, beta=10**(np.random.randint(*betas) * np.random.random()),
                                            lambd=10**(np.random.randint(*lambds) * np.random.random())))
    print("Trying out {} models".format(len(params)))

    return params

def main():
    with open("data/data", "rb") as file:
        data = pickle.load(file)
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
