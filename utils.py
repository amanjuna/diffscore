import numpy as np
import pandas as pd
import tensorflow as tf

CONTINUOUS_FEATURES = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", "GeneCoverage_1", "Entropy_0"]
CATEGORICAL_FEATURES = ["C1", "Plate", "10X", "DropSeq", "inDrop", "PC1", "PC2"]

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
    all_features = CONTINUOUS_FEATURES + added_features + CATEGORICAL_FEATURES

    df[continuous] = (df[continuous] - df[continuous].mean()) / df[continuous].std()
    # Adding indicators for sequencing types
    c1 = {"Plate": 0.0, "10X": 0.0, "inDrop": 0.0, "DropSeq": 0.0, "C1": 1.0}
    tenX = {"Plate": 0.0, "10X": 1.0, "inDrop": 0.0, "DropSeq": 0.0, "C1": 0.0}
    indrops = {"Plate": 0.0, "10X": 0.0, "inDrop": 1.0, "DropSeq": 0.0, "C1": 0.0}
    dropseq = {"Plate": 0.0, "10X": 0.0, "inDrop": 0.0, "DropSeq": 1.0, "C1": 0.0}
    plate = {"Plate": 1.0, "10X": 0.0, "inDrop": 0.0, "DropSeq": 0.0, "C1": 0.0}
    
    df["C1"] = df["SeqType"].map(c1)
    df["Plate"] = df["SeqType"].map(plate)
    df["10X"] = df["SeqType"].map(tenX)
    df["DropSeq"] = df["SeqType"].map(dropseq)
    df["inDrop"] = df["SeqType"].map(indrops)
    
    # Normalize to score from 0 (totipotent) to 1 (differentiated)
    min_order = df["Standardized_Order"].min()
    df["Standardized_Order"] = 1 - (df["Standardized_Order"] - min_order) / (df["Standardized_Order"] - min_order).max()
    
    train_data = df.loc[TRAIN, ["Standardized_Order"] + all_features]
    dev_data = df.loc[DEV, ["Standardized_Order"] + all_features]
    test_data = df.loc[TEST, ["Standardized_Order"] + all_features]
    return train_data, dev_data, test_data

if __name__=="__main__":
    loadData()
