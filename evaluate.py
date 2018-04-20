import _pickle as pickle
import config, preprocessing, math, csv
from Model import Model
import pandas as pd
import numpy as np
import scipy.stats, visualize

def evaluate_model(param, n_replicates=30):
    avg_test = {}
    avg_test["global"] = []
    for dataset in config.ALLDATA_SINGLE:
        avg_test[dataset] = [[], []]
    i = 0
    while min_number(avg_test, config.ALLDATA_SINGLE) < n_replicates:
        print([len(avg_test[x][0]) for x in config.ALLDATA_SINGLE])
        i += 1
        param.define_crossval(i)
        train_data, dev_data, test_data, names = preprocessing.load_data(model_path=param.output_path)
        assert len(names[2]) <= 6
        assert len(names[1]) <= 6
        assert not set(names[0]).intersection(set(names[1]))
        assert not set(names[1]).intersection(set(names[2]))
        assert not set(names[0]).intersection(set(names[2]))
        
        while min_number(avg_test, names[2]) != min_number(avg_test, config.ALLDATA_SINGLE):
            train_data, dev_data, test_data, names = preprocessing.load_data(model_path=param.output_path)
      
        model = Model(param, True)
        model.initialize()
        epoch = model.fit(train_data, dev_data)
        for test in names[2]:
            if len(avg_test[test][0]) >= n_replicates: continue
            pred = model.make_pred(test_data.loc[test])
            model_corr, model_sq, gc_corr = get_stats(pred, test_data.loc[test])
            avg_test[test][0].append(model_corr)
            avg_test[test][1].append(model_sq)
            print(test, "Model", model_corr, "GC:", gc_corr)
        pred = model.make_pred(test_data)
        g_model_corr, g_model_sq, g_gc_corr = get_stats(pred, test_data)
        avg_test["global"].append(g_model_corr)
        model.sess.close()
        del model
    with open("evaluate.data", "wb") as file:
        pickle.dump(avg_test, file)
    return avg_test

def get_stats(pred, data):
    gc = data["GeneCoverage_0"]
    std_ord = data["Standardized_Order"]
    model_corr = scipy.stats.spearmanr(std_ord, pred)[0]
    gc_corr = scipy.stats.spearmanr(std_ord, gc)[0]
    model_sq = np.linalg.norm(std_ord - pred)
    return model_corr, model_sq, gc_corr
   
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
        corr += scipy.stats.spearmanr(data_y, gc)[0]
    return corr/len(input_dsets)
        
def main():
    param = config.Config()
    avg_test = evaluate_model(param, n_replicates=3)
    print(avg_test)
    with open("data/data", "rb") as file:
        data = pickle.load(file)
    visualize.plot_summary_by_dset(data)
    visualize.plot_aggregate_summary(data)
    visualize.plot_seq_summary(data) 

if __name__ == "__main__":
    main()
