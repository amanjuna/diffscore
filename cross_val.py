'''
File: cross_val.py
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import config
import _pickle as pickle
import os,math
from diffscore import Model
import utils, splitPermute


'''
hidden_size (default 200)
lr (learning rate, default .0005)
beta (for mse, default .01)
lambd (for ls, default 1)
'''

def get_configs():
    sizes = [100, 150, 200]
    learning_rates = [.01]
    alphas = [1, 2, 5]
    betas = [0.1, 0.01, 0.001, 0.0001]
    lambds = [0.1, 0.01, 0.001, 0.0001]
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
                        params.append(config.Config(hidden_size=size, alpha = alpha, lr=lr, beta=beta, lambd=lambd))
    print("Trying out {} models".format(len(params)))

    return params

def main():
    k = 5
    data = pd.concat(utils.load_data())
    output = []
    params = get_configs()
    for param in params: 
        corr, squared = 0, 0
        i = 0
        # loop through k different split permutations
        for _ in range(k): 

            # get new data permutation
            train_data, dev_data, _ = splitPermute.permute(data, params.output_path)

            # build and train model with the parameters we're validating
            param.define_crossval(i)
            model = Model(param, True)
            model.initialize()
            model.fit(train_data, dev_data)

            # evaluates model performance on this dev set
            dcorr, dsquared = model.evaluate(dev_data)
            print(dcorr, dsquared)
            if dcorr*100 != 0.0 or not math.isnan(dcorr): # ignore models that tank grossly on the dev set
                corr += dcorr
                squared += dsquared
            model.sess.close()
            i += 1
        output.append("lr: " + str(param.lr) + " hidden_size: " + str(param.hidden_size) + " beta: " + str(param.beta) + " lambd: " + str(param.lambd) + " corr: " + str(corr/k) + " Squared loss: " + str(squared/k))
    
    # Save output of model so we know where to look for it later
    with open("output","w+") as f:
        for item in output:
            f.write(item + "\n")
        

if __name__ == '__main__':
    main()
