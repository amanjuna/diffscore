import _pickle as pickle
import pandas as pd
import numpy as np
import tensorflow as tf
import scipy.stats, visualize

# from architecture.models.product.Product import Product as Model
from architecture.models.non_product.Non_product import Non_product as Model
import architecture.models.config as config
import architecture.models.constants as constants

def evaluate(param, n_replicates=5):
    '''
    Implements leave-one-out cross validation
    '''
    avg_test = {}
    for dataset in constants.ALLDATA_SINGLE:
        avg_test[dataset] = [[], []]
    avg_test['global'] = []
    data = pd.read_csv("./data/unified_processed.csv", index_col="Dataset")
    data = pickle.load(open('data/data', "rb"))
    for val_set in constants.ALLDATA:
        if type(val_set) is not list:
            val_set = [val_set]
        train_indices = [name for name in constants.ALLDATA_SINGLE if 
                         name not in val_set]
        train_data = data.loc[train_indices, :]
        val_data = data.loc[val_set, :]
        
        model = Model(param)
        model.fit(train_data, val_data)
        for indiv in val_set:
            indiv_data = val_data.loc[indiv, :]
            pred = model.predict(indiv_data)
            corr, mse, gc_corr = get_stats(pred, indiv_data)
            avg_test[indiv][0].append(corr)
            avg_test[indiv][1].append(mse)
        global_pred = model.predict(val_data)
        global_corr, global_mse, gc_corr = get_stats(global_pred, val_data)
        avg_test['global'].append(global_corr)
        tf.reset_default_graph()
    with open("evaluate.data", "wb") as file:
        pickle.dump(avg_test, file)
    return avg_test


def train_dev_evaluate():
    '''
    Mostly the same as normal leave-one-out cross-validation,
    except also maintains a train-dev set
    '''
    test = {}
    for dataset in constants.ALLDATA_SINGLE:
        test[dataset] = {'train':0, 'train_dev':0, 'val':0, 'gc':0}

    data = pd.read_csv("./data/unified_processed.csv", index_col="Dataset")
    data = pickle.load(open('data/data', "rb"))
    for val_set in constants.ALLDATA:
        if val_set in ['GrunIntestine']:
            continue
        if type(val_set) is not list:
            val_set = [val_set]
        train_indices = [name for name in constants.ALLDATA_SINGLE if 
                         name not in val_set]
        train_shuffled = data.loc[train_indices, :].sample(len(data.loc[train_indices]))
        train_size = int(.8*len(train_shuffled))
        train_data = train_shuffled.iloc[0:train_size,:]
        train_dev_data = train_shuffled.iloc[train_size:-1,:]
        val_data = data.loc[val_set, :]
        param = config.Config("traindev_"+val_set[0][:5], hidden_size=100, n_layers=2,
                              n_epochs=100, lambd=1e-6, dropout=.25, lr=3e-5)
        model = Model(param)
        model.fit(train_data, val_data)
        train_pred = model.predict(train_data)
        dev_pred = model.predict(train_dev_data)
        val_pred = model.predict(val_data)
        train_corr, _, _ = get_stats(train_pred, train_data)
        dev_corr, _, _ = get_stats(dev_pred, train_dev_data)

        for indiv in val_set:
            indiv_data = val_data.loc[indiv, :]
            val_pred = model.predict(indiv_data)
            corr, mse, gc_corr = get_stats(val_pred, indiv_data)
            test[indiv]['train'] = train_corr
            test[indiv]['dev'] = dev_corr
            test[indiv]['val'] = corr
            test[indiv]['gc'] = gc_corr
            print(indiv, corr)

        tf.reset_default_graph()

    with open("train_dev_evaluate.data", "wb") as file:
        pickle.dump(test, file)
    return test


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
    # param = config.Config("train_dev_test", hidden_size=100, n_layers=2,
    #                       n_epochs=500, lambd=1e-5, dropout=.25, lr=3e-6)
    # avg_test = evaluate(param, n_replicates=3)
    # print(avg_test)
    with open("data/data", "rb") as file:
        data = pickle.load(file)
    # visualize.plot_summary_by_dset(data)
    # visualize.plot_aggregate_summary(data)
    # visualize.plot_seq_summary(data) 
    test = train_dev_evaluate()
    visualize.plot_traindev_summary(data, test)


if __name__ == "__main__":
    main()
