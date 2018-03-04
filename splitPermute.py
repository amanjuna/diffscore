'''
splitPermute.py

Randomly permutes given dataframe to return train/dev/test splits

Ensures that certain groups stay together
'''
import numpy as np
import random
import pandas as pd

KYLE = ['Kyle_Anterior', 'Kyle_Middle']
MARROW = ['Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  \
          'Marrow_plate_B',  'Marrow_plate_G']
PROTO = ['StandardProtocol', 'DirectProtocol']
REGEV = ['RegevIntestine', 'RegevDropseq']
FIBRO = ['Fibroblast_MyoF', 'Fibroblast_MFB']
INDIV = ['HumanEmbryo', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType']

TEST = REGEV + PROTO + FIBRO + ['Gottgens','GrunIntestine']

ALLDATA = INDIV + [KYLE, MARROW]
NUM_SETS = 9 # Number of dsets when treating the above blocks (except INDIV) as single dsets
NUM_TRAIN = 7
NUM_DEV = 2


def permute(data):
    '''
    Randomly puts NUM_TRAIN datasets into train set,
    NUM_DEV into dev set, and NUM_TEST into test set

    Above constant lists ensure that similar datasets travel together
    '''

    unallocated = list(ALLDATA)
    train_sets = random_pop(NUM_TRAIN, unallocated)
    dev_sets = random_pop(NUM_DEV, unallocated)

    train = data.loc[train_sets, :]
    dev = data.loc[dev_sets, :]
    test = data.loc[TEST, :]

    return train, dev, test


def random_pop(length, stack):
    sets = []
    for i in range(length):
        index = random.randrange(len(stack))
        dset = stack.pop(index)
        if type(dset) is list:
            sets += dset
        else:
            sets.append(dset)
    return sets