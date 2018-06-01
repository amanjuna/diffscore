"""
visualize.py

Once you have a working model, plot everything about 
your model performance

Plotting and predicting functions go here - all other
functions go in visualize_utils
"""

import _pickle as pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import tensorflow as tf
import scipy

import architecture.models.constants as constants
import architecture.models.config as config
# from architecture.models.non_product.Non_product import Non_product as Model
from architecture.models.product.Product import Product as Model
import visualize_utils as utils


def crossval_predict(filename='evaluate.data'):
    """
    Loads predictions for every time the data was in the test set
    """
    dsets = constants.ALLDATA_SINGLE
    predictions = []

    with open(filename, 'rb') as file:
        preds = pickle.load(file) # Dictionary from dset to list of lists

    for dset in dsets:
        predictions.append(preds[dset][0])

    return predictions


def gc_only_predict(data):
    """Gets correlation between GC0 and each dataset
    
    Returns a list that contains correlation for each dataset
    """
    corrs = []
    for dset in constants.ALLDATA_SINGLE:
        d = data.loc[dset]
        gc = utils.gc_only(d)
        ground = utils.ground_truth(d)
        spearman, _ = scipy.stats.spearmanr(gc, ground)
        corrs.append([spearman])
    return corrs


def plot(ground, pred, title, path):
    '''
    Simple scatterplot function that plots 
    @ground on the x axis and @pred on 
    the y axis
    '''
    pred = np.array(pred)
    ground = np.array(ground)
    spearman, _ = scipy.stats.spearmanr(pred, ground)

    plt.scatter(ground, pred, c='r')
    plt.title(title + "\n" + str(spearman))
    plt.xlabel("Standardized Order")
    plt.ylabel("Predicted Score")
    plt.savefig(path)
    plt.close()


def model_prediction_plot(param, data, path="./plots/model_predictions.png"):
    '''
    Plots the given model's predictions on the given data
    '''
    data_y = np.matrix(data["Standardized_Order"].as_matrix()).T
    data_x = data.ix[:, data.columns != "Standardized_Order"]
    model = Model(param)
    preds = model.predict(data)
    preds = np.reshape(preds, (-1, 1))
    plot(data_y, preds, "Model Predictions", path)


def plate_nonplate_plot(param, data):
    order = utils.ground_truth(data)
    gc_order = utils.gc_only(data)

    plate = data.loc[(data.Plate==1.0) | (data.C1==1.0)]
    plate_standardized_order = plate["Standardized_Order"]

    non_plate = data.loc[(data.Plate==0) & (data.C1==0)]
    non_plate_standardized_order = non_plate["Standardized_Order"]

    plate_model = Model(param)
    nonplate_model = Model(param)
    plate_pred = plate_model.predict(plate)
    non_plate_pred = nonplate_model.predict(non_plate)

    plot(plate_standardized_order, plate_pred, \
         "Model Plate Predictions", './plots/model_plate_pred.png')
    plot(non_plate_standardized_order, non_plate_pred, \
         "Model Non-Plate Predictions", './plots/model_nonplate_pred.png')


def gc_prediction_plot(data):
    '''
    Plots how gc predicts standardized order, for comparison
    '''
    order = utils.ground_truth(data)
    gc_order = utils.gc_only(data)

    plate = data.loc[(data.Plate==1.0) | (data.C1==1.0)]
    plate_standardized_order = plate["Standardized_Order"]

    non_plate = data.loc[(data.Plate==0) & (data.C1==0)]
    non_plate_standardized_order = non_plate["Standardized_Order"]

    gc_prediction = utils.gc_only(plate)
    non_plate_gc_prediction = utils.gc_only(non_plate)

    plot(plate_standardized_order, gc_prediction, \
         "GC Plate Predictions", './plots/GC_plate_predictions.png')
    plot(non_plate_standardized_order, non_plate_gc_prediction,\
         "GC Non-Plate Predictions", './plots/GC_non_plate_predictions.png')
    plot(order, gc_order, "GC Predictions", './plots/GC_predictions.png')


def plot_summary_by_dset(data, path="./plots/"):
    """
    Plots performance of the model on each data set

    For every data set, plots the correlations for every time that data was in the 
    test set using box plots
    """
    model_performance = crossval_predict()

    summary_data = []
    for dset in model_performance:
        preds = [spearman for spearman in dset if spearman]
        summary_data.append(preds)
    gc_points = gc_only_predict(data)


    gc_global = scipy.stats.spearmanr(utils.gc_only(data), utils.ground_truth(data))[0]
    with open('evaluate.data', 'rb') as file:
        preds = pickle.load(file) # Dictionary from dset to list of lists
    global_corr = preds['global']
    
    summary_data.append(global_corr)
    gc_points.append(gc_global)
    
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    title = "Spearman Correlation by Data Set"
    # colors = ['lightgreen']*len(constants.EASY) + ['lightblue']*len(constants.MEDIUM) + ['pink']*len(constants.HARD) #+ ['yellow']
    # labels = constants.EASY + constants.MEDIUM + constants.HARD #+ ["Global"]
    colors = ['lightblue']*len(constants.ALLDATA_SINGLE) + ['yellow']
    labels = constants.ALLDATA_SINGLE + ['Global']

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
    # easy = mpatches.Patch(color='lightgreen', label='"Easy"')
    # medium = mpatches.Patch(color='lightblue', label='"Medium"')
    # hard = mpatches.Patch(color='pink', label='"Hard"')
    model = mpatches.Patch(color='lightblue', label="Model")
    glob = mpatches.Patch(color='yellow', label='"Global"')
    gc = mpatches.Patch(color='firebrick', label="GC Only")
    # plt.legend(handles=[easy, medium, hard, glob, gc])
    plt.legend(handles=[model, gc, glob])

    fig.savefig(path + "summary.png")
    plt.close()


def plot_aggregate_summary(data, path="./plots/"):
    """
    Plots means of model performance vs. gene coverage on 
    easy, medium, and hard data sets
    """
    model_performance = crossval_predict()
    gc_points = gc_only_predict(data)

    # Calculate mean correlation for each data set
    pred_scores = utils.get_mean_by_dataset(model_performance)
    gc_scores = utils.get_mean_by_dataset(gc_points)

    # Group the data by difficulty for box plots
    pred_easy, pred_medium, pred_hard = utils.group_by_difficulty(pred_scores)
    gc_easy, gc_medium, gc_hard = utils.group_by_difficulty(gc_scores)


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
    gc = mpatches.Patch(color='lightblue', label="GC Only")
    plt.legend(handles=[model, gc])

    fig.savefig(path + 'aggregate_summary.png')
    plt.close()


def plot_seq_summary(data, path="./plots/"):
    """
    Plots means of model performance vs. gene coverage on 
    different sequencing technologies
    """
    model_performance = crossval_predict()
    gc_points = gc_only_predict(data)

    # Calculate mean correlation for each data set
    pred_scores = utils.get_mean_by_dataset(model_performance)
    gc_scores = utils.get_mean_by_dataset(gc_points)

    # Group the data by difficulty for box plots
    plate_pred, ten_x_pred, c1_pred, dropseq_pred, indrop_pred = utils.group_by_seq(pred_scores)
    plate_gc, ten_x_gc, c1_gc, dropseq_gc, indrop_gc = utils.group_by_seq(gc_scores)


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
    gc = mpatches.Patch(color='lightblue', label="GC Only")
    plt.legend(handles=[model, gc])

    #plt.show()
    fig.savefig(path + 'seq_summary.png')
    plt.close()


def plot_traindev_summary(data, scores=None):
    dsets = constants.ALLDATA_SINGLE

    if scores is None:
        with open('train_dev_evaluate.data', 'rb') as file:
            scores = pickle.load(file)
    
    N = len(dsets)
    fig, ax = plt.subplots()
    ind = np.arange(N)    # the x locations for the groups
    width = 0.2         # the width of the bars

    train_scores = [scores[d]['train'] for d in dsets]
    train_dev_scores = [scores[d]['train_dev'] for d in dsets]
    val_scores = [scores[d]['val'] for d in dsets]
    gc_scores = [scores[d]['gc'] for d in dsets]

    train_bars = ax.bar(ind, train_scores, width, color='g', bottom=0)
    train_dev_bars = ax.bar(ind+width, train_dev_scores, width, color='b', bottom=0)
    val_bars = ax.bar(ind+2*width, val_scores, width, color='r', bottom=0)
    gc_bars = ax.bar(ind+3*width, gc_scores, width, color='p', bottom=0)

    ax.set_title('Spearman Correlation by Data Set')
    ax.set_xticks(ind+(3*width/2)) # Centered tick for 4 bars
    ax.set_xticklabels(dsets)

    ax.legend((train_bars[0], train_dev_bars[0], val_bars[0]), ('Train', 'Train Dev', 'Val'))
    ax.set_ylabel('Spearman Correlation')

    fig.savefig('train_dev_summary.png')


def main():
    data = pickle.load(open("data/data", "rb"))
    plot_summary_by_dset(data)
    plot_aggregate_summary(data)
    plot_seq_summary(data)

    param = config.Config('default_model')
    # model_prediction_plot(param, data)
    plate_nonplate_plot(param, data)
    # gc_prediction_plot(data)


if __name__ == '__main__':
    main()
