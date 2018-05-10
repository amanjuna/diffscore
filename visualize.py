"""
visualize.py

Once you have a working model, plot everything about 
your model performance
"""

import _pickle as pickle
import random
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy

import Model, config

MODEL_PATH = './results/2_200_0.01_0_1_1_100/0/model.weights/weights.ckpt'

def ground_truth(data):
    """
    ground_truth(DataFrame) --> np.array
    Gets the Standardized Order as a numpy array
    """
    ground = data["Standardized_Order"]
    return np.array(ground)


def gc_only(data):
    """
    gc_only(DataFrame) --> np.array
    Gets GeneCoverage_0 from the data frame as a numpy array
    """
    gc0 = data["GeneCoverage_0"]
    return np.array(gc0)

def plot(ground, pred, title, path, gc_only=False):
    pred = np.array(pred)
    ground = np.array(ground)
    pearson, _ = scipy.stats.pearsonr(pred, ground)
    plt.figure()
    plt.title(title + "\n" + str(pearson))
    plt.scatter(ground, pred, c='r')
    if gc_only: plt.xlabel("GeneCoverage_0")
    else: plt.xlabel("Standardized Order")
    plt.ylabel("Predicted Score")
    plt.savefig(path)
    plt.close()

def crossval_predict(data):
    """Makes predictions for every time the data was in the test set

    """
    dsets = config.ALLDATA_SINGLE
    predictions = []

    with open('evaluate.data', 'rb') as file:
        preds = pickle.load(file) # Dictionary from dset to list of lists

    for dset in dsets:
        predictions.append(preds[dset][0])

    return predictions


def gc_only_predict(data):
    """Gets correlation between GC0 and each dataset
    
    Returns a list that contains correlation for each dataset
    """
    corrs = []
    for dset in config.ALLDATA_SINGLE:
        d = data.loc[dset]
        gc = gc_only(d)
        ground = ground_truth(d)
        spearman, _ = scipy.stats.spearmanr(gc, ground)
        corrs.append([spearman])
    return corrs


def plot_summary_by_dset(data, path="./plots/"):
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

    gc_global = scipy.stats.spearmanr(gc_only(data), ground_truth(data))[0]
    with open('evaluate.data', 'rb') as file:
        preds = pickle.load(file) # Dictionary from dset to list of lists
    global_corr = preds['global']
    
    summary_data.append(global_corr)
    gc_points.append(gc_global)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    title = "Spearman Correlation by Data Set"
    # Proxy for "easy", "medium", and "hard"
    colors = ['lightgreen']*len(config.EASY) + ['lightblue']*len(config.MEDIUM) + ['pink']*len(config.HARD) + ['yellow']
    labels = config.EASY + config.MEDIUM + config.HARD + ["Global"]

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
    plt.ylabel('Spearman Correlation')
    plt.xlabel('Data set')
    plt.margins(0.2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(90)
    plt.subplots_adjust(bottom=.25)

    # Format legend
    easy = mpatches.Patch(color='lightgreen', label='"Easy"')
    medium = mpatches.Patch(color='lightblue', label='"Medium"')
    hard = mpatches.Patch(color='pink', label='"Hard"')
    glob = mpatches.Patch(color='yellow', label='"Global"')
    gc = mpatches.Patch(color='firebrick', label="GC_0 Only")
    plt.legend(handles=[easy, medium, hard, glob, gc])

    #plt.show()
    fig.savefig(path + "summary.png")
    plt.close()


def plot_aggregate_summary(data, path="./plots/"):
    """Plots means of model performance vs. gene coverage on 
    easy, medium, and hard data sets
    """
    model_performance = crossval_predict(data)
    gc_points = gc_only_predict(data)

    # Calculate mean correlation for each data set
    pred_scores = get_mean_by_dataset(model_performance)
    gc_scores = get_mean_by_dataset(gc_points)

    # Group the data by difficulty for box plots
    pred_easy, pred_medium, pred_hard = group_by_difficulty(pred_scores)
    gc_easy, gc_medium, gc_hard = group_by_difficulty(gc_scores)


    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    colors = ['lightgreen', 'lightblue']
    title = 'Mean Correlation by Difficulty'
    labels = ['"Easy"', '"Medium"', '"Hard"']

    # Easy 
    bplot1 = ax.boxplot([pred_easy, gc_easy], positions=[1,2], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # Medium
    bplot2 = ax.boxplot([pred_medium, gc_medium], positions=[4,5], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    # Hard
    bplot3 = ax.boxplot([pred_hard, gc_hard], positions=[7,8], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    # Axes and whatnot
    plt.title(title)
    plt.ylabel('Spearman Correlation')
    plt.xlim(0,9)
    # plt.ylim(-1,1)
    ax.set_xticklabels(labels)
    ax.set_xticks([1.5, 4.5, 7.5])

    # Format legend
    model = mpatches.Patch(color='lightgreen', label="Model")
    gc = mpatches.Patch(color='lightblue', label="GC_0 Only")
    plt.legend(handles=[model, gc])

    #plt.show()
    fig.savefig(path + 'aggregate_summary.png')
    plt.close()

def plot_seq_summary(data, path="./plots/"):
    """
    Plots means of model performance vs. gene coverage on 
    different sequencing technologies
    """
    model_performance = crossval_predict(data)
    gc_points = gc_only_predict(data)

    # Calculate mean correlation for each data set
    pred_scores = get_mean_by_dataset(model_performance)
    gc_scores = get_mean_by_dataset(gc_points)

    # Group the data by difficulty for box plots
    plate_pred, ten_x_pred, c1_pred, dropseq_pred, indrop_pred = group_by_seq(pred_scores)
    plate_gc, ten_x_gc, c1_gc, dropseq_gc, indrop_gc = group_by_seq(gc_scores)


    fig = plt.figure(figsize=(14,6))
    ax = fig.add_subplot(111)
    colors = ['lightgreen', 'lightblue']
    title = 'Mean Correlation by Sequencing Platform'
    labels = ['Plate', '10X', 'C1', 'DropSeq', 'inDrop']

    # Plate 
    bplot1 = ax.boxplot([plate_pred, plate_gc], positions=[1,2], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)

    # 10x
    bplot2 = ax.boxplot([ten_x_pred, ten_x_gc], positions=[3,4], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)

    # c1
    bplot3 = ax.boxplot([c1_pred, c1_gc], positions=[5,6], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)

    # dropseq
    bplot4 = ax.boxplot([dropseq_pred, dropseq_gc], positions=[7,8], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)

    # indrop
    bplot5 = ax.boxplot([indrop_pred, indrop_gc], positions=[9,10], widths=.6, patch_artist=True,
                        flierprops=dict(marker='o', markersize=3))
    for patch, color in zip(bplot5['boxes'], colors):
        patch.set_facecolor(color)
        
    # Axes and whatnot
    plt.title(title)
    plt.ylabel('Spearman Correlation')
    plt.xlim(0,11)
    # plt.ylim(-1,1)
    ax.set_xticklabels(labels)
    ax.set_xticks([1.5, 3.5, 5.5, 7.5, 9.5])

    # Format legend
    model = mpatches.Patch(color='lightgreen', label="Model")
    gc = mpatches.Patch(color='lightblue', label="GC_0 Only")
    plt.legend(handles=[model, gc])

    #plt.show()
    fig.savefig(path + 'seq_summary.png')
    plt.close()

def get_mean_by_dataset(correlations):
    """Calculate mean of each dataset
    
    Given 2D list, where rows are assumed to be in 
    TRAIN DEV TEST order, returns list of mean value for each
    """
    means = []
    for i, row in enumerate(correlations):
        clean_row = [val for val in row if not math.isnan(val)]
        if not clean_row:
            means.append(0)
            continue
        means.append(sum(clean_row)/len(clean_row))

    return means

def group_by_seq(scores):
    plate = [scores[i] for i, _ in enumerate(config.PLATE)]
    ten_x = [scores[i + len(config.PLATE)] for i, _ in enumerate(config.TEN_X)]
    c1 = [scores[i + len(config.PLATE + config.TEN_X)] for i, _ in enumerate(config.C1)]
    dropseq = [scores[i + len(config.PLATE + config.TEN_X + config.C1)] for i, _ in enumerate(config.DROPSEQ)]
    indrop = [scores[i + len(config.PLATE + config.TEN_X + config.C1 + config.DROPSEQ)] for i, _ in enumerate(config.INDROP)]
    return [plate, ten_x, c1, dropseq, indrop]

def group_by_difficulty(scores):
    """Groups the input data by easy, medium, hard groups

    Given list of mean correlations assumed to be in 
    TRAIN DEV TEST order, returns 2d list where the data is 
    separated into the aforementioned groups
    """
    easy = [scores[i] for i, _ in enumerate(config.EASY)]
    medium = [scores[i+len(config.MEDIUM)] for i, _ in enumerate(config.MEDIUM)]
    hard = [scores[i+len(config.EASY+config.MEDIUM)] for i, _ in enumerate(config.HARD)]
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


def model_prediction_plot(config, data):
    m = Model.Model(config)
    m.initialize()
    data_y = np.matrix(data["Standardized_Order"].as_matrix()).T
    data_x = data.ix[:, data.columns != "Standardized_Order"]
    preds = m.make_pred(data_x)
    preds = np.reshape(preds, (-1, 1))
    plot(data_y, preds, "Model Predictions", './plots/model_predictions.png')


def main():
    data = pickle.load(open("data/data", "rb"))
    # plot_summary_by_dset(data)
    # plot_aggregate_summary(data)
    # plot_seq_summary(data)
    # plot_by_dataset(data)
    # plot_datasets(data)

    # others = set(config.ALLDATA_SINGLE) - set(config.PLATE)
    # plate = data.loc[config.PLATE]
    # non_plate = data.loc[others]
    # plate_standardized_order = ground_truth(plate)
    # non_plate_standardized_order = ground_truth(non_plate)
    # gc_prediction = gc_only(plate)
    # non_plate_gc_prediction = gc_only(non_plate)
    # plot(plate_standardized_order, gc_prediction, "GC Plate Predictions", './plots/GC_plate_predictions.png')
    # plot(non_plate_standardized_order, non_plate_gc_prediction, "GC Non-Plate Predictions", './plots/GC_non_plate_predictions.png')

    plot_summary_by_dset(data)
    plot_aggregate_summary(data)
    plot_seq_summary(data)

if __name__ == '__main__':
    main()
