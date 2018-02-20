'''
visualize.py
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy
import _pickle as pickle
import diffscore, config
'''
Given ground truth score and predictions, make violin plots

Access train, dev, and test from directory (pandas dataframes)

graph by dataset

Also graph gene coverage 1 vs ground truth
'''
TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M', 'ChuCellType', 'HSC_10x']
DEV = ['HSMM','Marrow_plate_G','Marrow_plate_B','Camargo']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
QUESTIONABLE = ['AT2', 'EPI', "Astrocytoma"]
MODEL_PATH = "./foo"
NUM_NEURONS = 200

def make_predictions(data):
    train_x = data.ix[:, train_data.columns != "Standardized_Order"].as_matrix()
    param = config.Config(hidden_size=NUM_NEURONS)
    model = diffscore.Model(param)
    model.initialize()
    pred = model.make_pred(train_x, MODEL_PATH)

def load():
    train = pickle.load("train")
    dev = pickle.load("dev")
    test = pickle.load("test")
    data = pd.concat([train, dev, test])
    return data

def ground_truth(data):
    ground = data.loc["Standardized_Order"]
    return np.array(ground)

def gc_only(data):
    gc1 = data.loc["GeneCoverage_1"]
    return np.array(gc1)

def plot(pred, ground, title, path, gc_only=False):
    plt.figure()
    plt.title(title)
    plt.plot(ground, pred, c='r')
    if gc_only: plt.xlabel("GeneCoverage_1")
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

def setup_and_plot(data, dset, split):
    title, path = make_title(dset, split)
    y = ground_truth(data.loc[dset])
    gc = gc_only(data.loc[dset])
    predictions = make_predictions(data.loc[dset])
    plot(predictions, y, title, path)
    plot(gc, y, title, path, gc_only=True)

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
    plot_by_dataset(data)


if __name__ == '__main__':
    main()