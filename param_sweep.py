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
    n_layers = [1, 2, 3]
    sizes = [10, 50, 100]
    learning_rates = [1e-4, .01]
    betas = [1e-5, 0.1]
    lambds = [1e-5, 0.1]

    params = []
    for size in sizes:
        for n in n_layers:
            for _ in range(5):
                params.append(config.Config(n_layers=n, hidden_size=size,
                                            n_epochs=100, 
                                            alpha=0, beta=np.random.uniform(*betas),
                                            lr=np.random.uniform(*learning_rates),
                                            lambd=np.random.uniform(*lambds)))
   
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
