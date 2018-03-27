'''
splitPermute.py

Randomly permutes given dataframe to return train/dev/test splits

Ensures that certain groups stay together
'''
import numpy as np
import random
import pandas as pd
import os, datetime



def permute(data, path='data/'):
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

    write_split(train_sets, dev_sets, test_sets, path)

    train = data.loc[train_sets, :]
    dev = data.loc[dev_sets, :]
    # test = data.loc[TEST, :]      # We should probably keep some sort of test set sacred eventually
    test = data.loc[test_sets, :]

    return train, dev, test


def random_pop(length, stack):
    '''
    Randomly removes @length elements from the given @stack and returns what has been popped
    '''
    sets = []
    for i in range(length):
        index = random.randrange(len(stack))
        dset = stack.pop(index)
        if type(dset) is list:
            sets += dset
        else:
            sets.append(dset)
    return sets


def write_split(train, dev, test, path='data/'):
    '''
    Writes file to tell us what datasets go into what splits
    '''
    filename = path + "split_" + time_name()
    with open(filename, "w") as file:
        file.write("Training sets: \n")
        file.write(str(train) + "\n\n")
        file.write("Dev sets: \n")
        file.write(str(dev) + "\n\n")
        file.write("Test sets: \n")
        file.write(str(test) + "\n")


def time_name():
    '''
    Gets current time/date as readable string
    '''
    now = str(datetime.datetime.now())
    now_str = now[:now.find('.') + 3] # lop off most milliseconds
    filename = now_str.replace(" ",":") # hacky? but gets current time as readable filename
    return filename
