#!/usr/bin/env python

'''
Preprocssing functions and utilities for input
into the model. Main function is load_data
'''

import _pickle as pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import datetime, os, random

# Predefined data groupings
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

N_PERCENTILES = 4 # Number of percentile statistics to include

def clean_data(input_fn="data/CompiledTable_ForPaper.csv", output_fn="data/data"):
    '''
    Imports data from csv file @filename into a cleaned csv,
    "data.csv", for future examination and a binary file, data, 
    from which it can be far more quickly accessed by 
    future operations. Performs some basic cleanup operations,
    adds dataset level features (percentiles, mean, average), and
    defines indicator features. Furthermore, pads the number of cells
    ensuring each phenotype has the same number of cells
    '''
 
    continuous = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", 
                  "Entropy_0", "HUGO_MGI_GC0", "HUGO_MGI_GC1", "mtgenes", 
                  "PC1", "PC2", "C1_axis", "C2_axis"]
    ind = ["C1", "Plate", "10x", "DropSeq", "inDrop", 
           "Mouse", "Human", "nonrepeat", "repeat"]

    # Load csv and index by dataset name
    df = pd.read_csv(input_fn)
    df.fillna(value=0.0)
    datasets = df["Dataset"].unique()
    df.set_index(["Dataset"], inplace=True)
    df.sort_index(inplace=True)

    n_continuous = len(continuous)
    for i in range(n_continuous):
        feature = continuous[i]
        mean_name = feature + "_mean"
        continuous.append(mean_name)
        
        std_name = feature + "_stdev"
        continuous.append(std_name)
        
        for i in range(N_PERCENTILES): 
            feature_name = feature + "_" + str(i*100/N_PERCENTILES) 
            + "_percentile"
            continuous.append(feature_name)

    # Add dataset metadata to each of the features
    for dataset in datasets:
        for i in range(n_continuous):
            feature = continuous[i]
            mean_name = feature + "_mean"
            df.loc[dataset, mean_name] = df.loc[dataset, feature].mean()
            
            std_name = feature + "_stdev"
            df.loc[dataset, std_name] = df.loc[dataset, feature].std()
            
            for i in range(N_PERCENTILES):
                feature_name = feature + "_" + str(i*100/N_PERCENTILES) + "_percentile"
                df.loc[dataset, feature_name] = np.percentile(df.loc[dataset, feature], 
                                                              i*100/N_PERCENTILES)

    # Normalize each continuous feature to have a mean of 0 and a std of 1
    #df[continuous] = (df[continuous] - df[continuous].mean()) / df[continuous].std()
     
    # Adding indicators for sequencing types
    c1, tenX, indrops = defaultdict(float), defaultdict(float), defaultdict(float)
    dropseq, plate = defaultdict(float), defaultdict(float)  
    nonrepeat, repeat = defaultdict(float), defaultdict(float)
    human, mouse = defaultdict(float), defaultdict(float)
    
    for feature in ind:
        c1[feature] = float(feature == "C1")
        tenX[feature] = float(feature == "10x")
        indrops[feature] = float(feature == "inDrop")
        dropseq[feature] = float(feature == "DropSeq")
        plate[feature] = float(feature == "Plate")
        human[feature] = float(feature == "Human")
        mouse[feature] = float(feature == "Mouse")
        nonrepeat[feature] = float(feature=="Nonrepeat")
        repeat[feature] = float(feature == "Repeat")
    
    df["C1"] = df["SeqType"].map(c1)
    df["Plate"] = df["SeqType"].map(plate)
    df["10x"] = df["SeqType"].map(tenX)
    df["DropSeq"] = df["SeqType"].map(dropseq)
    df["inDrop"] = df["SeqType"].map(indrops)
    df["Mouse"] = df["Species"].map(mouse)
    df["Human"] = df["Species"].map(human)
    df["nonrepeat"] = df["Nonrepeat"].map(nonrepeat)
    df["repeat"] = df["Nonrepeat"].map(repeat)
 
    # Normalize to score from 0 (totipotent) to 1 (differentiated)
    min_order = df["Standardized_Order"].min()
    df["Standardized_Order"] = 1 - (df["Standardized_Order"] - min_order) / (df["Standardized_Order"] - min_order).max()

    # Pad data so all phenotypes are equally represented
    phenotypes = df.Phenotype.unique()
    median_cells = np.median([len(df.loc[df["Phenotype"]==phenotype]) for phenotype in phenotypes])
    median_cells = int(median_cells)
    normalized_df = pd.DataFrame()
    for phenotype in phenotypes:
        normalized_df = pd.concat([normalized_df, df.loc[df["Phenotype"]==phenotype].sample(n=median_cells, replace=True)])
    df = normalized_df
    
    all_features = continuous + ind
  
    final_data = df.loc[:, ["Standardized_Order"] + all_features]
    final_data.to_csv(output_fn + ".csv")
    pickle.dump(final_data, open(output_fn, "wb"))


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

    return train, dev, test, (train_sets, dev_sets, test_sets)
    
       
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
    filename = path + "split"
    with open(filename, "w") as file:
        file.write("Training sets: \n")
        file.write(str(train) + "\n\n")
        file.write("Dev sets: \n")
        file.write(str(dev) + "\n\n")
        file.write("Test sets: \n")
        file.write(str(test) + "\n")

        
def load_data(input_fn="data/CompiledTable_ForPaper.csv", output_fn="data/data", 
              model_path=""):
    '''
    Primary function to be called in order to get data for
    model. 
    '''
    if not os.path.isfile(output_fn):
        clean_data(input_fn, output_fn)
    data = pickle.load(open(output_fn, "rb"))
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    train, dev, test, dsets = permute(data, model_path)
    return train, dev, test, dsets
    
if __name__=="__main__":
    load_data(model_path="trial/")

    
    
