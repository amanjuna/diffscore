'''
unify_data.py

Compiles the distances into a unified csv file for
further processing downstream
'''
import os
import collections
import _pickle as pickle

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


def annotate_and_save():
    '''
    Reads in csv that contains phenotypes, dataset labels, gc values,
    and similarity values, adds the appropriate standardized order
    and indicators to each datapoint, then writes the resulting
    file
    '''
    data = pd.read_csv('./data/unified.csv')

    # Adds metadata - indicators for species and seqtype
    ord_dict, converters = make_dicts()
    data["Standardized_Order"] = data["PhenotypeMasterSheet"].map(ord_dict)
    for colname, converter in converters:
        data[colname] = data["DatasetLabelMark"].map(converter)

    # Pad data so all phenotypes are equally represented
    phenotypes = data["PhenotypeMasterSheet"].unique()
    num_cells = int(np.max([len(data.loc[data["PhenotypeMasterSheet"]==phenotype]) 
                           for phenotype in phenotypes]))
    normalized_df = pd.DataFrame()
    for phenotype in phenotypes:
        normalized_df = pd.concat([normalized_df, data.loc[data["PhenotypeMasterSheet"]==phenotype].sample(n=num_cells, replace=True)])
    data = normalized_df

    # Do a little column organizing and re-arranging 
    cols = data.columns.tolist()
    cols.remove("ID")
    cols.insert(0, "ID")
    ind = cols.index("NN_gc_val0")
    cols.remove("Standardized_Order")
    cols.insert(ind, "Standardized_Order")
    cols.remove("DiffusionMark")
    cols.insert(ind, "DiffusionMark")
    cols.remove("PhenotypeLabelMark")
    cols.remove("OrderMark")
    
    data = data[cols]
    data = data.rename(mapper={"DatasetLabelMark":"Dataset"}, axis='columns')
    data.set_index("Dataset", inplace=True)
    print(data.head())
    data.to_csv('./data/unified_processed.csv')
    clean = data.loc[:, "Standardized_Order":"Mouse"]
    data.info()

    pickle.dump(clean, open('./data/data', 'wb'))


def make_dicts():
    '''
    Makes dictionaries that map from dataset to standardized order, 
    dataset to species indicators, and dataset to seqtype indicators

    returns ord_dict, a dict from dset to standardized order, and
    a list of tuples, where each tuple is (featureName, correspondingDict)

    all these dicts are meant to be used with a pandas dataframe to write
    new columns using the dataset names as input
    '''
    data = pd.read_csv('./data/NeuralnetTable.csv')

    # Standardized order dict
    reduced = zip(data["Phenotype"].tolist(), data["Standardized_Order"].tolist())
    ord_dict = collections.defaultdict(float)
    max_ord = data["Standardized_Order"].max()
    for phenotype, order in reduced:
        ord_dict[phenotype] = 1 - (order - 1)/(max_ord - 1) # 0 is differentiated, 1 is totipotent 

    # Create dicts for species and sequencing type
    species_dict = collections.defaultdict(str)
    platform_dict = collections.defaultdict(str)
    reduced = zip(data["Dataset"].tolist(), data["Species"].tolist(), data["SeqType"].tolist())
    for dset, spec, seq in reduced:
        species_dict[dset] = spec
        platform_dict[dset] = seq

    # Prepare and return dicts to convert from dataset to proper indicators
    c1, tenX, indrops = collections.defaultdict(float), collections.defaultdict(float), collections.defaultdict(float)
    dropseq, plate = collections.defaultdict(float), collections.defaultdict(float)  
    human, mouse = collections.defaultdict(float), collections.defaultdict(float)
    for dset in species_dict:
        c1[dset] = float(platform_dict[dset] == "C1")
        tenX[dset] = float(platform_dict[dset] == "10x")
        indrops[dset] = float(platform_dict[dset] == "inDrop")
        dropseq[dset] = float(platform_dict[dset] == "DropSeq")
        plate[dset] = float(platform_dict[dset] == "Plate")
        human[dset] = float(species_dict[dset] == "Human")
        mouse[dset] = float(species_dict[dset] == "Mouse")
    conversion_dicts = [c1, tenX, indrops, dropseq, plate, human, mouse]
    col_names = ["C1", "10x", "inDrop", "DropSeq", "Plate", "Human", "Mouse"]

    return ord_dict, zip(col_names, conversion_dicts)


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
    annotate_and_save()


if __name__ == '__main__':
    main()
