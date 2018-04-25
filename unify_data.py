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


def merge_dist_and_vals(valfile, distfile):
    '''
    @valfile is file containing the cells in the @distfile
    laid out like:
    phenotype, diff status, raw gene count, smoothed gene count

    Assumes that @distfile is a .npy distance matrix

    Returns a list, where each entry is a list of the information
    for a cell
    '''
    dists = np.load(distfile)
    cells = []
    with open(valfile) as file:
        for i, line in enumerate(file):
            entry = line.strip().split('\t')
            info = [entry[0]] # phenotype as string
            info += [float(x) for x in entry[1:]]
            info += list(sorted(dists[i]))
            cells.append(info)
    return cells


def get_filenames(dirname):
    '''
    Returns list of filenames in a given directory
    '''
    fnames = []
    for filename in os.listdir(dirname):
        if filename.endswith('.tsv') or filename.endswith('.npy'):
            fnames.append(os.path.join(dirname, filename))
    return fnames


def get_cellid_dict(fname='./data/Diffusion_Table_Indices.txt'):
    '''
    Uses index file given by Gun to construct dict from 
    dataset name to list of cell IDs in that dataset
    '''
    cell_lists = collections.defaultdict(list)
    with open(fname) as file:
        for line in file:
            entry = line.strip().split()
            if len(entry) != 2: continue
            cell_lists[entry[1]].append(entry[0])
    return cell_lists


def main():
    val_files = get_filenames(DATA_DIR+'strippedVals/')
    dist_files = get_filenames(DATA_DIR+'distanceMatrices/')
    ex = merge_dist_and_vals(val_files[1], dist_files[1])
    cell_lists = get_cellid_dict()

    print([len(cell_lists[d]) for d in cell_lists])
    for val, dist in zip(val_files, dist_files):
        print(len(merge_dist_and_vals(val, dist)))

    # unified = []
    # for val, dist in zip(val_files, dist_files):
    #     start = str(val).find('strippedVals/') + len('strippedVals/')
    #     stop = str(val).find('_')
    #     dataset_name = str(val)[start:stop]
    #     combined_vals = merge_dist_and_vals(val, dist)
    #     for i, cell_id in enumerate(cell_lists.get(dataset_name, [])):
    #         entry = [cell_id]
    #         entry += combined_vals[i]
    #         unified.append(entry)
    # print(len(unified))
    # print([len(l) for l in unified])


if __name__ == '__main__':
    main()