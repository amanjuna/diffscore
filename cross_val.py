'''
File: cross_val.py
'''

import numpy as np
import pandas as pd
import tensorflow as tf
import config

'''
hidden_size (default 200)
lr (learning rate, default .0005)
beta (for mse, default .01)
lambd (for ls, default 1)
'''

def get_configs():
    sizes = [100, 200, 300, 400]
    learning_rates = []
    for i in range(2, 6):
        learning_rates.append(10 ** (-i))
        learning_rates.append(5 * 10 ** (-i))
    



if __name__ == '__main__':
    main()