'''
visualize_utils.py

Houses all the grouping/utility functions for
visualize.py
'''
import math 

import numpy as np

import architecture.models.constants as constants


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
    Gets DiffusionMark from the data frame as a numpy array
    """
    gc0 = data["DiffusionMark"]
    return np.array(gc0)


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
    '''
    Gathers scores into groups by sequencing platform for plotting
    '''
    plate = [scores[i] for i, _ in enumerate(constants.PLATE)]
    ten_x = [scores[i + len(constants.PLATE)] for i, _ in enumerate(constants.TEN_X)]
    c1 = [scores[i + len(constants.PLATE + constants.TEN_X)] for i, _ in enumerate(constants.C1)]
    dropseq = [scores[i + len(constants.PLATE + constants.TEN_X + constants.C1)]\
               for i, _ in enumerate(constants.DROPSEQ)]
    indrop = [scores[i+len(constants.PLATE+constants.TEN_X+constants.C1+constants.DROPSEQ)] \
              for i, _ in enumerate(constants.INDROP)]
    return [plate, ten_x, c1, dropseq, indrop]


def group_by_difficulty(scores):
    """Groups the input data by easy, medium, hard groups

    Given list of mean correlations assumed to be in 
    TRAIN DEV TEST order, returns 2d list where the data is 
    separated into the aforementioned groups
    """
    easy = [scores[i] for i, _ in enumerate(constants.EASY)]
    medium = [scores[i+len(constants.MEDIUM)] for i, _ in enumerate(constants.MEDIUM)]
    hard = [scores[i+len(constants.EASY+constants.MEDIUM)] for i, _ in enumerate(constants.HARD)]
    return [easy, medium, hard]