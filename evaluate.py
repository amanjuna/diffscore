import _pickle as pickle
import config, splitPermute, utils, math, csv
from diffscore import Model
import pandas as pd

DATASETS = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 
            'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  'Marrow_plate_B',  
            'Marrow_plate_G', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'Camargo', 
            'ChuCellType', 'AT2', 'EPI','RegevIntestine', 'RegevDropseq', 
            'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine',
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

def main():
    data = pd.concat(utils.load_data())
    param = config.Config(n_layers=1, hidden_size=100, n_epochs=50,  
                          alpha=1, lr=0.01, beta=0.0001, lambd=0.1, 
                          grad_clip = False)
    avg_test = {}
    avg_test['global'] = []
    for dataset in DATASETS:
        avg_test[dataset] = [[], []]
    for i in range(10):
        param.define_crossval(i)
        train_data, dev_data, test_data = splitPermute.permute(data)
        train_datasets = pd.unique(train_data.index)
        dev_datasets = pd.unique(dev_data.index)
        test_datasets = pd.unique(test_data.index)
        print(test_datasets)
        dcorr = 0.0
        epoch = 0
        model = Model(param, True)
        model.initialize()
        epoch = model.fit(train_data, dev_data)
        # dcorr, dsquared = model.evaluate(dev_data)
        global_corr, _ = model.evaluate(pd.concat([train_data, dev_data, test_data]))
            
        for test in test_datasets:
            tcorr, tsquared = model.evaluate(test_data.loc[test])
            avg_test[test][0].append(tcorr)
            avg_test[test][1].append(tsquared)
        avg_test['global'].append(global_corr)
    pickle.dump(avg_test, open("evaluate.data", "wb"))
    print(avg_test)
    #model = Model(param, True)
    

if __name__ == "__main__":
    main()
