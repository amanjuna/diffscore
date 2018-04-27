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
    Takes 2D numpy array of data to write to a .tsv
    '''
    with open(DATA_DIR+'unified.tsv', 'w') as file:
        file.write('UniqueID\tDatasetLabelMark\tPhenotypeLabelMark\tOrderMark \
                    \tGCMark\tDiffusionMark\tPhenotypeMasterSheet\n')
        for line in data:
            for entry in line:
                file.write(str(entry)+'\t')
            file.write('\n')


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
    with open('./data/IndexforDiffusionTables.txt') as file:
        for i, line in enumerate(file):
            if i == 0: continue # Skip header
            # temp: skip fibroblasts
            entry = line.strip().split()
            if "Fibroblast" in entry[1]: continue
            entries.append(entry)

    num_seen = 0
    for matrix in data:
        counter = 0
        for i, _ in enumerate(matrix):
            entries[i + num_seen] += matrix[i,:].tolist()
            counter += 1
        num_seen += counter

    print("\nSaw {} cells, made labels for {} (these should match)\n".format(num_seen, len(entries)))
    # Should be list where each entry contains the list of 
    # gc data plus the distance data for that cell
    return entries

def main():
    label_list_count()
    data_count()
    matrices = load_sim_matrices()
    gene_counts = load_gc_vals()
    results = []
    for matrix, gc in zip(matrices, gene_counts):
        result = combine_gc_and_sim(matrix, gc)
        results.append(result)
    unified = unify(results)
    write_unified(unified)


if __name__ == '__main__':
    main()