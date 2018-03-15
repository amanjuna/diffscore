"""
visualize.py

Once you have a working model, plot everything about 
your model performance
"""

import _pickle as pickle
import random

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy

import diffscore, config


TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 
         'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  'Marrow_plate_B',  
         'Marrow_plate_G', 'HSC_10x']
DEV = ['HSMM', 'Camargo', 'ChuCellType', 'AT2', 'EPI']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol',
        'Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
MODEL_PATH = "./results/100_0.01_0.01_0.1/0/model.weights/weights"
NUM_NEURONS = 100


def make_predictions(data):
    """DEPRECATED: Uses model to make standardized order predictions
    
    Sets up the model using the path defined at the top of the file
    then makes predictions on the dataframe
    """
    x = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
    param = config.Config(hidden_size=NUM_NEURONS)
    model = diffscore.Model(param)
    model.initialize()
    pred = model.make_pred(x, MODEL_PATH)
    return pred


def load():
    """Loads the data
    
    Loads from the different files then just concats them 
    all together, which is fine since we're doing the testing cross-val
    """
    with open("data/train", "rb") as f:
        train = pickle.load(f)
    with open("data/dev", "rb") as f:
        dev = pickle.load(f)
    with open("data/test", "rb") as f:
        test = pickle.load(f)

    data = pd.concat([train, dev, test])
    return data


def ground_truth(data):
    """ground_truth(DataFrame) --> np.array

    Gets the Standardized Order as a numpy array
    """
    ground = data["Standardized_Order"]
    return np.array(ground)


def gc_only(data):
    """gc_only(DataFrame) --> np.array

    Gets GeneCoverage_0 from the data frame as a numpy array
    """
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

    TEMPORARY: 
    For now just returns well-formatted random numbers for testing the plotting
    """
    dsets = TRAIN + DEV + TEST
    predictions = []
    for i, dset in enumerate(dsets):
        preds = []
        for j in range(10):
            preds.append(random.random() * (i+1))
        predictions.append(preds)
    return predictions


def gc_only_predict(data):
    """Gets correlation between GC0 and each dataset
    
    Returns a list that contains correlation for each dataset
    """
    corrs = []
    gc = gc_only(data)
    ground = ground_truth(data)
    for dset in TRAIN+DEV+TEST:
        pearson, _ = scipy.stats.pearsonr(gc, ground)
        corrs.append([pearson])
    # return corrs
    return [[random.random()*i] for i in range(len(TRAIN+DEV+TEST))]


def plot_summary_by_dset(data):
    """Plots performance of the model on each data set

    For every data set, plots the correlations for every time that data was in the 
    test set using box plots
    """
    model_performance = crossval_predict(data) # TODO: return array of predictions on each dataset (possibly masked)
    summary_data = []
    for dset in model_performance:
        preds = [spearman for spearman in dset if spearman]
        summary_data.append(preds)
    gc_points = gc_only_predict(data)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    title = "Pearson Correlation by Data Set"
    # Proxy for "easy", "medium", and "hard"
    colors = ['lightgreen']*len(TRAIN) + ['lightblue']*len(DEV) + ['pink']*len(TEST)
    labels = TRAIN + DEV + TEST

    # Create boxplots by dataset

    # Gene coverage correlations, colored dark red (will appear as thick bars on plot)
    gc_bplot = plt.boxplot(gc_points, medianprops=dict(linestyle='-',
                                                       linewidth=3, 
                                                       color='firebrick'))
    # Prediction correlations, colored so that easy is green, medium is blue, hard is pink
    pred_bplot = ax.boxplot(summary_data, patch_artist=True, labels=labels,
                            flierprops=dict(marker='o', markersize=3))
    for i, patch in enumerate(pred_bplot['boxes']):
        patch.set_facecolor(colors[i])

    # Handle labeling and formatting
    plt.title(title)
    plt.ylabel('Pearson Correlation')
    plt.xlabel('Data set')
    plt.margins(0.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.subplots_adjust(bottom=.25)

    # Format legend
    easy = mpatches.Patch(color='lightgreen', label='"Easy"')
    medium = mpatches.Patch(color='lightblue', label='"Medium"')
    hard = mpatches.Patch(color='pink', label='"Hard"')
    gc = mpatches.Patch(color='firebrick', label="GC_0 Only")
    plt.legend(handles=[easy, medium, hard, gc])

    plt.show()
    fig.savefig("./plots/summary_test.png")
    plt.close()


def plot_aggregate_summary(data):
    """Plots means of model performance vs. gene coverage on 
    easy, medium, and hard data sets
    """
    model_performance = crossval_predict(data)
    gc_points = gc_only_predict(data)

    # Calculate mean correlation for each data set
    pred_scores = get_mean_by_dataset(model_performance)
    gc_scores = get_mean_by_dataset(gc_points)

    # Group the data by difficulty for box plots
    pred_data = group_by_difficulty(pred_scores)
    gc_data = group_by_difficulty(gc_scores)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    colors = ['lightgreen', 'lightblue', 'pink']
    title = 'Mean Correlation by Difficulty'
    labels = ['"Easy"', '"Medium"', '"Hard"']

    # Gene coverage correlations, colored dark red (will appear as thick bars on plot)
    gc_bplot = plt.boxplot(gc_data, medianprops=dict(linestyle='-',
                                                       linewidth=3, 
                                                       color='firebrick'))
    # Prediction correlations, colored so that easy is green, medium is blue, hard is pink
    pred_bplot = ax.boxplot(pred_data, patch_artist=True, labels=labels,
                            flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(pred_bplot['boxes'], colors):
        patch.set_facecolor(color)

    plt.title(title)
    plt.ylabel('Pearson Correlation')

    # Format legend
    gc = mpatches.Patch(color='firebrick', label="GC_0 Only")
    plt.legend(handles=[gc])

    plt.show()
    fig.savefig('./plots/aggregate_summary_test.png')
    plt.close()


def get_mean_by_dataset(correlations):
    """Calculate mean of each dataset
    
    Given 2D list, where rows are assumed to be in 
    TRAIN DEV TEST order, returns list of mean value for each
    """
    means = []
    for i, row in enumerate(correlations):
        means.append(np.mean(correlations[i]))
    return means


def group_by_difficulty(scores):
    """Groups the input data by easy, medium, hard groups

    Given list of mean correlations assumed to be in 
    TRAIN DEV TEST order, returns 2d list where the data is 
    separated into the aforementioned groups
    """
    easy = [scores[i] for i, _ in enumerate(TRAIN)]
    medium = [scores[i+len(TRAIN)] for i, _ in enumerate(DEV)]
    hard = [scores[i+len(TRAIN+DEV)] for i, _ in enumerate(TEST)]
    return [easy, medium, hard]
    

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
    plot_summary_by_dset(data)
    plot_aggregate_summary(data)
    # plot_by_dataset(data)
    # plot_datasets(data)


if __name__ == '__main__':
    main()
