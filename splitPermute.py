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
INDIV = ['HumanEmbryo', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType', 'Gottgens', 'GrunIntestine']

AVOID = ['Gottgens','GrunIntestine'] + [PROTO, REGEV, FIBRO]

ALLDATA = INDIV + [KYLE, MARROW, PROTO, REGEV, FIBRO]
NUM_SETS = 14 # Number of dsets when treating the above blocks (except INDIV) as single dsets
NUM_TRAIN = 8
NUM_DEV = 3
NUM_TEST = 3


def permute(data):
    '''
    Randomly puts NUM_TRAIN datasets into train set,
    NUM_DEV into dev set, and NUM_TEST into test set

    Above constant lists ensure that similar datasets travel together
    '''

    unallocated = list(ALLDATA)
    
    train_sets = random_pop(NUM_TRAIN, unallocated)
    unallocated = [dset for dset in unallocated if dset not in train_sets]

    dev_sets = random_pop(NUM_DEV, unallocated)
    unallocated = [dset for dset in unallocated if dset not in dev_sets]
    
    test_sets = random_pop(NUM_TEST, unallocated)

    train = data.loc[train_sets, :]
    dev = data.loc[dev_sets, :]
    # test = data.loc[TEST, :]
    test = data.loc[test_sets, :]

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
