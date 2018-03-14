'''
visualize.py
'''

import _pickle as pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy

import diffscore, config
'''
Given ground truth score and predictions, make violin plots

Access train, dev, and test from directory (pandas dataframes)

graph by dataset

Also graph gene coverage 1 vs ground truth
'''
TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  'Marrow_plate_B',  'Marrow_plate_G', 'HSC_10x']
DEV = ['HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType', 'AT2', 'EPI']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
MODEL_PATH = "./results/100_0.01_0.01_0.1/0/model.weights/weights"
NUM_NEURONS = 100

def make_predictions(data):
    x = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
    param = config.Config(hidden_size=NUM_NEURONS)
    model = diffscore.Model(param)
    model.initialize()
    pred = model.make_pred(x, MODEL_PATH)
    return pred

def load():
    with open("data/train", "rb") as f:
        train = pickle.load(f)
    with open("data/dev", "rb") as f:
        dev = pickle.load(f)
    with open("data/test", "rb") as f:
        test = pickle.load(f)

    data = pd.concat([train, dev, test])
    return data

def ground_truth(data):
    ground = data["Standardized_Order"]
    return np.array(ground)

def gc_only(data):
    gc0 = data["GeneCoverage_0"]
    return np.array(gc0)

def plot(pred, ground, title, path, gc_only=False):
    pearson, _ = scipy.stats.pearsonr(pred.ravel(), ground.ravel())
    plt.figure()
    plt.title(title + "\n" + str(pearson))
    plt.scatter(pred, ground, c='r')
    if gc_only: plt.xlabel("GeneCoverage_0")
    else: plt.xlabel("Predicted Score")
    plt.ylabel("Ground Truth Score")
    plt.savefig(path)
    plt.close()

def plot_by_dataset(data):
    for dset in TRAIN:
        setup_and_plot(data, dset, "Train")
    for dset in DEV:
        setup_and_plot(data, dset, "Dev")
    for dset in TEST:
        setup_and_plot(data, dset, "Test")

def crossval_predict(data):
    """Makes predictions for every time the data was in the test set

    For now just returns well-formatted random numbers for testing the plotting
    """
    dsets = TRAIN + DEV + TEST
    predictions = []
    for i, dset in enumerate(dsets):
        preds = []
        for j in range(10):
            preds.append(random.random() * i)
        predictions.append(preds)
    return predictions

def plot_summary(data):
    """Needs overhaul

    For every data set, plots the correlations for every time that data was in the 
    test set using box plots
    """
    title = "Summary by dataset"
    dsets = [TRAIN, DEV, TEST] # Proxy for "easy", "medium", and "hard"
    labels = TRAIN + DEV + TEST
    model_performance = crossval_predict(data) # TODO: return array of predictions on each dataset (possibly masked)
    summary_data = []
    for dset in model_performance:
        preds = [spearman for spearman in dset if spearman]
        summary_data.append(preds)

    fig, ax = plt.subplots()
    plt.title(title)
    plt.ylabel('Pearson Correlation')
    plt.xlabel('Data set')

    ax.boxplot(summary_data, labels=labels)
    plt.show()
    plt.close()




    # OBSOLETE
    #
    # for dset in dsets:
    #     y = ground_truth(data.loc[dset])
    #     gc = gc_only(data.loc[dset])
    #     predictions = make_predictions(data.loc[dset])
    #     ys.append(scipy.stats.spearmanr(predictions.ravel(), y.ravel())[0])
    #     gcs.append(scipy.stats.spearmanr(gc.ravel(), y.ravel())[0])
    # fig, ax = plt.subplots()
    # plt.title(title)
    # ind = [0, 1.25, 2.5]
    # ax.bar([x for x in ind], ys, width = 0.5, label = "Model Prediction")
    # ax.bar([x + 0.5 for x in ind], gcs, width = 0.5, label="Gene Coverage")
    # plt.ylabel('Spearman')
    # plt.xlabel('Data set')
    
    # ax.set_xticks([x + 0.25/2 for x in ind])
    # ax.set_xticklabels(("TRAIN", "DEV", "TEST"))
    # ax.legend()
    # plt.savefig("./plots/summary.png")
    # plt.close()

def plot_datasets(data):
    title = "Summary - datasets"
    ys = []
    gcs = []
    dsets = TRAIN + DEV + TEST
    for dset in dsets:
        y = ground_truth(data.loc[dset])
        gc = gc_only(data.loc[dset])
        predictions = make_predictions(data.loc[dset])
        ys.append(scipy.stats.spearmanr(predictions.ravel(), y.ravel())[0])
        gcs.append(scipy.stats.spearmanr(gc.ravel(), y.ravel())[0])
    fig, ax = plt.subplots()
    plt.title(title)
    ind = [1.25*i for i,x in enumerate(dsets)]
    ax.bar([x for x in ind], ys, width = 0.5, label = "Model Prediction")
    ax.bar([x + 0.5 for x in ind], gcs, width = 0.5, label="Gene Coverage")
    plt.ylabel('Spearman')
    plt.xlabel('Data set')
    
    ax.set_xticks([x + 0.25/2 for x in ind])
    ax.set_xticklabels(TRAIN + DEV + TEST)
    ax.legend()
    plt.savefig("./plots/summary_datasets.png")
    plt.close()
    
def setup_and_plot(data, dset, split):
    title1, path1 = make_title(dset, split)
    title2, path2 = make_title(dset, split, True)
    y = ground_truth(data.loc[dset])
    gc = gc_only(data.loc[dset])
    predictions = make_predictions(data.loc[dset])
    plot(predictions, y, title1, path1)
    plot(gc, y, title2, path2, gc_only=True)

def make_title(dset, split, gc_only=False):
    if gc_only:
        title = "Gene Coverage vs. Ground Truth on "+ dset + " (" + split + ")"
        path = "./plots/gc_only/"+ split + "_" + dset + "gc"
    else:
        title = "Predicted Score vs. Ground Truth on "+ dset + " (" + split + ")"
        path = "./plots/pred/" + split + "_" + dset
    return title, path


def main():
    data = load()
    plot_summary(data)
    # plot_by_dataset(data)
    # plot_summary(data)
    # plot_datasets(data)


if __name__ == '__main__':
    main()
