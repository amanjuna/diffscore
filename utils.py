import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
import splitPermute

CONTINUOUS_FEATURES = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", "GeneCoverage_1", "Entropy_0"]
CATEGORICAL_FEATURES = ["C1", "Plate", "10x", "DropSeq", "inDrop", "PC1", "PC2"]

TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M', 'ChuCellType', 'HSC_10x']
DEV = ['HSMM','Marrow_plate_G','Marrow_plate_B','Camargo']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
QUESTIONABLE = ['AT2', 'EPI', "Astrocytoma"]

def load_data():
    added_features = []

    # Load csv and index by dataset name
    df = pd.read_csv("CompiledTableNN_filtered_PCAUpdated.csv")
    datasets = df["Dataset"].unique()
    df.set_index(["Dataset"], inplace=True)
    df.sort_index(inplace=True)

    # Add features about whole datasets to individual entries (deciles, mean, std dev)
    for feature in CONTINUOUS_FEATURES:
        added_features.append(feature + " mean")
        added_features.append(feature + " stdev")
        for i in range(0, 11, 1):
            added_features.append(feature + " " + str(i*10) + " percentile")
    
    # Add dataset metadata to each of the features
    for dataset in datasets:
        for feature in CONTINUOUS_FEATURES:
            df.loc[dataset, feature + " mean"] = np.mean(df.loc[dataset, feature])
            df.loc[dataset, feature + " stdev"] = df.loc[dataset, feature].std()
            for i in range(0, 11, 1):
                df.loc[dataset, feature + " " + str(i*10) + " percentile"] = np.percentile(df.loc[dataset, feature], i*10)

    continuous = CONTINUOUS_FEATURES + added_features
    all_features = continuous + CATEGORICAL_FEATURES

    df[continuous] = (df[continuous] - df[continuous].mean()) / df[continuous].std()
    # Adding indicators for sequencing types
    c1, tenX, indrops = defaultdict(float), defaultdict(float), defaultdict(float)
    dropseq, plate = defaultdict(float), defaultdict(float)
    for feature in ["Plate", "10x", "inDrop", "DropSeq", "C1"]:
        c1[feature] = float(feature == "C1")
        tenX[feature] = float(feature == "10x")
        indrops[feature] = float(feature == "inDrop")
        dropseq[feature] = float(feature == "DropSeq")
        plate[feature] = float(feature == "Plate")

    
    df["C1"] = df["SeqType"].map(c1)
    df["Plate"] = df["SeqType"].map(plate)
    df["10x"] = df["SeqType"].map(tenX)
    df["DropSeq"] = df["SeqType"].map(dropseq)
    df["inDrop"] = df["SeqType"].map(indrops)
    
    # Normalize to score from 0 (totipotent) to 1 (differentiated)
    min_order = df["Standardized_Order"].min()
    df["Standardized_Order"] = 1 - (df["Standardized_Order"] - min_order) / (df["Standardized_Order"] - min_order).max()
    
    data = df.loc[:, ["Standardized_Order"] + all_features]
    return splitPermute.permute(data) # returns train, dev, and test datasets as 3 different DataFrames

if __name__=="__main__":
    load_data()
