import numpy as np
import pandas as pd
import tensorflow as tf

CONTINUOUS_FEATURES = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", "GeneCoverage_1", "Entropy_0", "Entropy_1"]
CATEGORICAL_FEATURES = ["cl1", "plate", "droplet", "PC1", "PC2"]

ADDED_FEATURES = []

FEATURES = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES

TRAIN = ['Kyle_Anterior', 'Kyle_Middle',  'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M', 'ChuCellType', 'HSC_10x']
DEV = ['HSMM','Marrow_plate_G','Marrow_plate_B','Camargo']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
QUESTIONABLE = ['AT2', 'EPI', "Astrocytoma"]

def load_data():
    df = pd.read_csv("CompiledTableNN_filtered_PCAUpdated.csv")
    datasets = df["Dataset"].unique()
    df["DatasetName"] = df["Dataset"]
    df.set_index(["Dataset"], inplace=True)
    df.sort_index(inplace=True)
    for feature in CONTINUOUS_FEATURES:
        ADDED_FEATURES.append(feature + " mean")
        ADDED_FEATURES.append(feature + " median")
        for i in range(0, 11, 1):
            ADDED_FEATURES.append(feature + " " + str(i*10) + " percentile")
    
    for dataset in datasets:
        # Add dataset metadata to each of the features
        for feature in CONTINUOUS_FEATURES:
            df.loc[dataset, feature + " mean"] = np.mean(df.loc[dataset, feature]) 
            df.loc[dataset, feature + " median"] = np.median(df.loc[dataset, feature])
            for i in range(0, 11, 1):
                df.loc[dataset, feature + " " + str(i*10) + " percentile"] = np.percentile(df.loc[dataset, feature], i*10)

    continuous = CONTINUOUS_FEATURES + ADDED_FEATURES
    FEATURES = CONTINUOUS_FEATURES + ADDED_FEATURES + CATEGORICAL_FEATURES

    df[continuous] = (df[continuous] - df[continuous].mean())/df[continuous].std()
    # Adding indicators for sequencing types
    cl1 = {"Plate": 0.0, "Droplet": 0.0, "C1": 1.0}
    droplet = {"Plate": 0.0, "Droplet": 1.0, "C1": 0.0}
    plate = {"Plate": 1.0, "Droplet": 0.0, "C1": 0.0}
    
    df["cl1"] = df["SeqType"].map(cl1)
    df["plate"] = df["SeqType"].map(plate)
    df["droplet"] = df["SeqType"].map(droplet)
    
    df["Standardized_Order"] = 1 - (df["Standardized_Order"] - df["Standardized_Order"].min())/(df["Standardized_Order"]- df["Standardized_Order"].min()).max()

    df.to_csv("um.csv")
    
    train_data = df.loc[TRAIN, ["Standardized_Order"] + FEATURES]
    dev_data = df.loc[DEV, ["Standardized_Order"] + FEATURES]
    test_data = df.loc[TEST, ["Standardized_Order"] + FEATURES]
    return train_data, dev_data, test_data

if __name__=="__main__":
    loadData()
