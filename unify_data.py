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
ORDER = ['ChuCellType_vals.tsv', 'AT2_vals.tsv', 'EPI_vals.tsv',  'HumanEmbryo_vals.tsv', 
'HSMM_vals.tsv', 'Kyle_vals.tsv', 'GrunIntestine_vals.tsv', 'RegevSmartseq_vals.tsv', 
'HSC_10x_vals.tsv', 'Marrow_10x_vals.tsv', 'Marrow_plate_vals.tsv',
'RegevDropseq_vals.tsv', 'DirectProtocol_vals.tsv', 'StandardProtocol_vals.tsv']


def merge_dist_and_vals(valfile, distfile):
    '''
    @valfile is file containing the cells in the @distfile
    laid out like:
    phenotype, diff status, raw gene count, smoothed gene count

    Assumes that @distfile is a .npy distance matrix

    Returns a list, where each entry is a list of the information
    for a cell
    '''
    print(valfile)
    dists = np.load(distfile)
    cells = []
    with open(valfile) as file:
        for i, line in enumerate(file):
            entry = line.strip().split('\t')
            # info = [entry[0]] # phenotype as string
            info = [float(x) for x in entry[1:]]
            info += list(sorted(dists[i], reverse=True))
            cells.append(info)
    cells = np.array(cells)
    
    return cells


def write_unified(fname, data):
    with open(DATA_DIR+'unified/'+ str(fname), 'w') as file:
        for line in data:
            for entry in line:
                file.write(str(entry)+'\t')
            file.write('\n')


def get_filenames(dirname):
    '''
    Returns list of filenames in a given directory
    '''
    fnames = []
    for filename in os.listdir(dirname):
        if filename.endswith('.tsv') or filename.endswith('.npy'):
            fnames.append(os.path.join(dirname, filename))
    return fnames


def count_vals(fnames):
    entries = 0
    for fname in fnames:
        if "Marrow_10x" not in fname: continue
        with open(fname) as file:
            for line in file:
                if not line: continue
                entries += 1
    print(entries)


def label_list(fname='./data/IndexforDiffusionTables.txt'):
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

    for dset, num in count.items():
        print("{} count: {}".format(dset, num))
    return labels


def list_order(fname='./data/IndexforDiffusionTables.txt'):
    dsets = []
    with open(fname) as file:
        for line in file:
            if line.strip().split()[1] not in dsets: dsets.append(line.strip().split()[1])
    print(dsets)


def unify(labels):
    num_seen = 0
    for fname in ORDER:
        counter = 0
        with open('./data/strippedVals/'+fname, 'r') as file:
            for i, line in enumerate(file):
                entry = [val for val in line.strip().split()]
                labels[num_seen + i] += entry
                counter += 1
        print("Entries for {}: {}".format(fname, counter))
        num_seen += counter
    print(num_seen)
    return labels


def main():
    # val_files = get_filenames(DATA_DIR+'strippedVals/')
    # dist_files = get_filenames(DATA_DIR+'distanceMatrices/')
    # ex = merge_dist_and_vals(val_files[1], dist_files[1])
    # for val, dist in zip(val_files, dist_files):
    #     cells = merge_dist_and_vals(val, dist)
    #     write_unified(os.path.basename(val), cells)
    labels = label_list()
    print(len(labels))
    data = unify(labels)
    # with open('./data/unified_data.tsv', 'w') as file:
    #     file.write('ID\tdataset\torder\tgenecount\tsmoothed\n')
    #     for entry in data:
    #         for val in entry:
    #             file.write(str(val)+'\t')
    #         file.write('\n')

if __name__ == '__main__':
    main()