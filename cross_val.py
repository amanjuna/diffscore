'''
File: cross_val.py
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import config
import _pickle as pickle
import os

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
    train_data = pickle.load(open("train", "rb"))
    dev_data = pickle.load(open("dev","rb"))
    test_data = pickle.load(open("test", "rb"))
    
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')


if __name__ == '__main__':
    main()