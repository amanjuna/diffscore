'''
unify_data.py

Compiles the distances into a unified csv file for
further processing downstream
'''
import os
import collections

import numpy as np
import pandas as pd

DATA_DIR = './data/'
MASTER = './data/NeuralnetTable.csv'
ORDER = ['ChuCellType', 'AT2', 'EPI',  'HumanEmbryo', 
         'HSMM', 'Kyle', 'GrunIntestine', 'RegevSmartseq', 
         'HSC_10x', 'Marrow_10x', 'Marrow_plate',
         'RegevDropseq', 'DirectProtocol', 'StandardProtocol']


def write_unified(data):
    '''
    Takes 2D numpy array of data to write to a .csv
    '''
    names = ['CellID,', 'DatasetLabelMark,', 'PhenotypeLabelMark,', 
             'OrderMark,', 'GCMark,', 'DiffusionMark,', 'PhenotypeMasterSheet,']
    names += ["NN_gc_val%d,"%i for i in range(50)]
    names += ["Sim%d,"%i for i in range(50)]
    names += ['ID\n']
    with open(DATA_DIR+'unified.csv', 'w') as file:
        for header in names:
            file.write(header)
        for i, line in enumerate(data):
            for entry in line:
                file.write(str(entry)+',')
            file.write(str(i)+'\n')


def label_list_count(fname='./data/IndexforDiffusionTables.txt'):
    labels = []
    count = collections.defaultdict(int)
    with open(fname) as file:
        for i, line in enumerate(file):
            if i == 0: continue
            # temp: skip fibroblasts
            entry = line.strip().split()
            if "Fibroblast" in entry[1]: continue
            labels.append(entry)
            count[entry[1]] += 1

    total = 0
    for dset, num in count.items():
        print("{} count: {}".format(dset, num))
        total += num
    print("Total according to index file: {}".format(total))
    print()


def list_order(fname='./data/IndexforDiffusionTables.txt'):
    dsets = []
    with open(fname) as file:
        for line in file:
            if line.strip().split()[1] not in dsets: 
                dsets.append(line.strip().split()[1])
    print(dsets)


def data_count():
    num_seen = 0
    total = 0
    for fname in ORDER:
        counter = 0
        with open('./data/strippedVals/'+fname+'_vals.tsv', 'r') as file:
            entries = [line.strip() for line in file]
            counter += len(entries)
        total += counter
        print("Entries for {}: {}".format(fname, counter))
    print("Total from *.tsv files: {}".format(total))
    print()


def load_sim_matrices():
    '''
    Returns list of distance matrices
    '''
    matrices = []
    neighbor_indices = []
    counter = 0
    for dset in ORDER:
        with open('./data/distanceMatrices/'+dset+'_distanceMatrix.npy', 'rb') as file:
            '''cast to list since distance matrices
               can all be different lengths - will need to 
               pad/sample/cut/whatever in later processing steps
            '''
            sim_matrix = np.load(file)
            matrices.append(sim_matrix)
            counter += sim_matrix.shape[0]
    
    print("Num cells from matrices: {}".format(counter)) # debugging
    return matrices


def load_gc_vals():
    '''
    loads smoothed gc values for each dataset file
    '''
    smoothed = []
    counter = 0
    for dset in ORDER:
        with open('./data/strippedVals/'+dset+'_vals.tsv') as file:
            vals = [float(line.strip().split()[-1]) for line in file]
        counter += len(vals)
        smoothed.append(vals)
    print("Num gc: {}".format(counter)) # Debugging
    return smoothed


def combine_gc_and_sim(matrix, gc):
    '''
    TODO: experiment with different ways of combining 
    the gc and similarity data
    '''
    combined = matrix * np.array(gc)
    return combined


def unify(data):
    entries = []
    gc_index = 5 # index of diffused gene coverage values
    with open('./data/IndexforDiffusionTables.txt') as file:
        for i, line in enumerate(file):
            if i == 0: continue # Skip header
            # temp: skip fibroblasts
            entry = line.strip().split()
            if "Fibroblast" in entry[1]: continue
            entries.append(entry)

    entries = np.array(entries)
    for i, matrix in enumerate(data):
        ids = np.argsort(-1*matrix, axis=1)[:,0:50]
        ids += 0 if i == 0 else _data.shape[0]  
        gc = np.zeros((matrix.shape[0], 50))
        for j, row in enumerate(ids):
            for k, col in enumerate(row):
                gc[j, k] = entries[col, 5].astype(float)
        sim = -1*np.sort(-1*matrix, axis=0)[0:50].T
        
        if i == 0:
            _data = np.concatenate((gc, sim), axis=1)
        else:
            temp = np.concatenate((gc, sim), axis=1)
            _data = np.concatenate((_data, temp), axis=0)
    entries = np.concatenate((entries, _data), axis=1)

    print("\nSaw {} cells, made labels for {} (these should match)\n".format(entries.shape[0], len(entries)))

    # Should be list where each entry contains the list of 
    # gc data plus the distance data for the 50 closest neighbors
    # plus the indices of those neighbors in the overall list
    return entries

def write_labels():
    df = pd.read_csv('./data/NeuralnetTable.csv')
    df.fillna(value=0.0)
    df.set_index(["Cells"], inplace=True)
    ind_var = ["C1", "Plate", "10x", "DropSeq", "inDrop", 
               "Mouse", "Human", "nonrepeat", "repeat"]
    c1, tenX, indrops = defaultdict(float), defaultdict(float), defaultdict(float)
    dropseq, plate = defaultdict(float), defaultdict(float)  
    human, mouse = defaultdict(float), defaultdict(float)

    for feature in ind_var:
        c1[feature] = float(feature == "C1")
        tenX[feature] = float(feature == "10x")
        indrops[feature] = float(feature == "inDrop")
        dropseq[feature] = float(feature == "DropSeq")
        plate[feature] = float(feature == "Plate")
        human[feature] = float(feature == "Human")
        mouse[feature] = float(feature == "Mouse")

    df["C1"] = df["SeqType"].map(c1)
    df["Plate"] = df["SeqType"].map(plate)
    df["10x"] = df["SeqType"].map(tenX)
    df["DropSeq"] = df["SeqType"].map(dropseq)
    df["inDrop"] = df["SeqType"].map(indrops)
    df["Mouse"] = df["Species"].map(mouse)
    df["Human"] = df["Species"].map(human)
    df["nonrepeat"] = df["Nonrepeat"].map(nonrepeat)
    df["repeat"] = df["Nonrepeat"].map(repeat)

    min_order = df["Standardized_Order"].min()
    df["Standardized_Order"] = 1 - (df["Standardized_Order"] - min_order) / (df["Standardized_Order"] - min_order).max()

    names = ['UniqueID'] + []
    usecols = [0] + range(7, 57)
    dists = pd.read_csv('./data/unified.tsv', delim_whitespace=True, usecols=usecols, header=0)

def annotate():
    data = pd.read_csv('./data/unified.csv')
    ord_dict = make_ord_dict()
    data["Standardized_Order"] = data["PhenotypeMasterSheet"].map(ord_dict)


def make_dicts():
    data = pd.read_csv('./data/NeuralnetTable.csv')

    # Standardized order dict
    reduced = zip(data["Phenotype"].tolist(), data["Standardized_Order"].tolist())
    ord_dict = collections.defaultdict(float)
    max_ord = data["Standardized_Order"].max()
    for phen, order in reduced:
        ord_dict[phen] = 1 - (order - 1)/(max_ord - 1) # 0 is differentiated, 1 is totipotent 

    # Species Dict


    # Platform Dict

    
    return ord_dict


def main():
    # label_list_count()
    # data_count()
    # matrices = load_sim_matrices()
    # gene_counts = load_gc_vals()
    # results = []
    # for matrix, gc in zip(matrices, gene_counts):
    #     result = combine_gc_and_sim(matrix, gc)
    #     results.append(result)
    # unified = unify(results)
    # unified = unify(matrices)
    # write_unified(unified)
    annotate()


if __name__ == '__main__':
    main()
