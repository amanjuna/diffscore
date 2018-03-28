import _pickle as pickle
import config, preprocessing, math, csv
from Model import Model
import pandas as pd
import numpy as np
import scipy.stats
DATASETS = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 
            'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  'Marrow_plate_B',  
            'Marrow_plate_G', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'Camargo', 
            'ChuCellType', 'AT2', 'EPI','RegevIntestine', 'RegevDropseq', 
            'StandardProtocol', 'DirectProtocol', 'GrunIntestine',
            'Fibroblast_MyoF', 'Fibroblast_MFB']

def evaluate_model(data, param):
    avg_test = {}
    for dataset in DATASETS:
        avg_test[dataset] = [0, 0, 0]
    for i in range(10):
        param.define_crossval(i)
        train_data, dev_data, test_data = splitPermute.permute(data)
        train_datasets = pd.unique(train_data.index)
        dev_datasets = pd.unique(dev_data.index)
        test_datasets = pd.unique(test_data.index)

        model = Model(param, True)
        model.initialize()
        epoch = model.fit(train_data, dev_data)
            
        for test in test_datasets:
            tcorr, tsquared = model.evaluate(test_data.loc[test])
            avg_test[test][0] += tcorr
            avg_test[test][1] += tsquared
            avg_test[test][2] += 1
        print(avg_test)
    corr, squared, number = 0, 0, 0 
    for dataset in DATASETS:
        corr += avg_test[dataset][0]
        squared += avg_test[dataset][1]
        number += avg_test[dataset][2]
    corr /= number
    squared /= number
    with open("param_sweep.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([str(param.n_layers), str(param.hidden_size), str(param.lr), str(param.alpha), str(param.beta), str(param.lambd), str(corr), str(squared)])

def min_number(avg_test, dsets):
    return min([len(avg_test[x][0]) for x in dsets])
       
def max_number(avg_test, dsets):
    return max([len(avg_test[x][0]) for x in dsets])

def gc_corr(data):
    input_dsets = list(data.index.unique())
    corr = 0
    for dset in input_dsets:
        data_y = np.matrix(data["Standardized_Order"].as_matrix()).T
        gc = np.matrix(data["GeneCoverage_0"].as_matrix()).T
        corr += scipy.stats.pearsonr(data_y, gc)[0]
    return corr/len(input_dsets)
        
def main():
    n_replicates = 3
    param = config.Config(n_layers=2, hidden_size=200, n_epochs=100,  
                          alpha=1, lr=0.01, beta=1, lambd=1, 
                          grad_clip = False) 
    data = preprocessing.load_data(model_path=param.output_path)
    avg_test = {}
    avg_test["global"] = []
    for dataset in DATASETS:
        avg_test[dataset] = [[], []]
    i = 0
    while min_number(avg_test, DATASETS) < n_replicates:
        print([len(avg_test[x][0]) for x in DATASETS])
        i += 1
        param.define_crossval(i)
        train_data, dev_data, test_data, names = preprocessing.load_data(model_path=param.output_path)
        assert len(names[2]) <= 6
        assert len(names[1]) <= 6
        assert not set(names[0]).intersection(set(names[1]))
        assert not set(names[1]).intersection(set(names[2]))
        assert not set(names[0]).intersection(set(names[2]))
        count = 0
        while min_number(avg_test, names[2]) != min_number(avg_test, DATASETS) or max_number(avg_test, names[2]) > n_replicates-1:
            count += 1
            train_data, dev_data, test_data, names = preprocessing.load_data(model_path=param.output_path)
            if count > 200 and min_number(avg_test, names[2]) == min_number(avg_test, DATASETS):
                break
      
        model = Model(param, True)
        model.initialize()
        epoch = model.fit(train_data, dev_data)
        dev_corr, dev_squared = model.evaluate(dev_data)
        for test in names[2]:
            tcorr, tsquared = model.evaluate(test_data.loc[test])
            #if tcorr < gc_corr(test_data.loc[test]):
            #    if len(avg_test[test][0]) == min_number(avg_test, DATASETS): 
            #        if i < 20:                
            #            continue
            #        elif tcorr < gc_corr(test_data.loc[test]) - 0.2:
            #            continue
            #    else:
            #        continue
            avg_test[test][0].append(tcorr)
            avg_test[test][1].append(tsquared)
            print(test, tcorr, tsquared)
        global_corr, global_squared = model.evaluate(test_data, global_corr=True)
        avg_test["global"].append(global_corr)
    pickle.dump(avg_test, open("evaluate.data", "wb"))
    print(avg_test)
    #model = Model(param, True)
    

if __name__ == "__main__":
    main()
