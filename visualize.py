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
TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  'Marrow_plate_B',  'Marrow_plate_G', 'HSC_10x']
DEV = ['HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType', 'AT2', 'EPI']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
MODEL_PATH = "./0/model.weights/weights"
NUM_NEURONS = 100

def make_predictions(data):
    x = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
    param = config.Config(hidden_size=NUM_NEURONS)
    model = diffscore.Model(param)
    model.initialize()
    pred = model.make_pred(x, MODEL_PATH)
    return pred

def load():
    with open("train", "rb") as f:
        train = pickle.load(f)
    with open("dev", "rb") as f:
        dev = pickle.load(f)
    with open("test", "rb") as f:
        test = pickle.load(f)

    data = pd.concat([train, dev, test])
    return data

def ground_truth(data):
    ground = data["Standardized_Order"]
    return np.array(ground)

def gc_only(data):
    gc1 = data["GeneCoverage_1"]
    return np.array(gc1)

def plot(pred, ground, title, path, gc_only=False):
    spearman = scipy.stats.spearmanr(pred.ravel(), ground.ravel())
    plt.figure()
    plt.title(title + "\n" + str(spearman))
    plt.scatter(pred, ground, c='r')
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

def plot_summary(data):
    title = "Summary"
    ys = []
    gcs = []
    dsets = [TRAIN, DEV, TEST]
    for dset in dsets:
        y = ground_truth(data.loc[dset])
        gc = gc_only(data.loc[dset])
        predictions = make_predictions(data.loc[dset])
        ys.append(scipy.stats.spearmanr(predictions.ravel(), y.ravel())[0])
        gcs.append(scipy.stats.spearmanr(gc.ravel(), y.ravel())[0])
    plt.title(title)
    datasets = ["Train", "Dev", "Test"]
    plt.bar(datasets, ys, label = "Model Prediction")
    plt.bar(datasets, gcs, label="Gene Coverage")
    plt.ylabel('Spearman')
    plt.xlabel('Data set')
    plt.legend()
    plt.savefig("./plots/summary.png")
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
    plot_by_dataset(data)
    plot_summary(data)


if __name__ == '__main__':
    main()
