import _pickle as pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.stats, visualize

from architecture.models.non_product.Non_product import Non_product as Model
import architecture.models.config as config


def evaluate(param, n_replicates=5):
    '''
    Implements leave-one-out cross validation
    '''
    avg_test = {}
    for dataset in config.ALLDATA_SINGLE:
        avg_test[dataset] = [[], []]
    avg_test['global'] = []
    data = pd.read_csv("./data/unified_processed.csv", index_col="Dataset")
    data = pickle.load(open('data/data', "rb"))
    for _ in range(n_replicates):
        for val_set in config.ALLDATA:
            if type(val_set) is not list:
                val_set = [val_set]
            train_indices = [name for name in config.ALLDATA_SINGLE if 
                             name not in val_set]
            train_data = data.loc[train_indices, :]
            val_data = data.loc[val_set, :]
            
            model = Model(param)
            model.fit(train_data, val_data)
            for indiv in val_set:
                indiv_data = val_data.loc[indiv, :]
                pred = model.predict(indiv_data)
                pred = np.squeeze(pred)
                corr, mse, gc_corr = get_stats(pred, indiv_data)
                avg_test[indiv][0].append(corr)
                avg_test[indiv][1].append(mse)
            global_pred = model.predict(val_data)
            global_corr, global_mse, gc_corr = get_stats(pred, val_data)
            avg_test['global'].append(global_corr)
            tf.reset_default_graph()
    with open("evaluate.data", "wb") as file:
        pickle.dump(avg_test, file)
    return avg_test


def get_stats(pred, data):
    gc = data["DiffusionMark"]
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
        gc = np.matrix(data["DiffusionMark"].as_matrix()).T
        corr += scipy.stats.spearmanr(data_y, gc)[0]
    return corr/len(input_dsets)
        

def main():
    param = config.Config("default")
    avg_test = evaluate(param, n_replicates=3)
    print(avg_test)
    with open("data/data", "rb") as file:
        data = pickle.load(file)
    visualize.plot_summary_by_dset(data)
    visualize.plot_aggregate_summary(data)
    visualize.plot_seq_summary(data) 

if __name__ == "__main__":
    main()
