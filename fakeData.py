'''
fakeData.py
'''
import numpy as np
import pandas as pd
import splitPermute as perm

TRAIN = ['Kyle_Anterior', 'Kyle_Middle', 'HumanEmbryo', 'Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M', 'Marrow_plate_B', 'Marrow_plate_G', 'HSC_10x']
DEV = ['HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType', 'AT2', 'EPI']
TEST = ['RegevIntestine', 'RegevDropseq', 'StandardProtocol', 'DirectProtocol','Gottgens','GrunIntestine','Fibroblast_MyoF', 'Fibroblast_MFB']
ALL = TRAIN + DEV + TEST

def make_df():
    df = pd.read_csv("data/CompiledTableNN_filtered_PCAUpdated.csv")
    datasets = df["Dataset"].unique()
    df.set_index(["Dataset"], inplace=True)

    return df

def test_shuffle(data):
    for i in range(5):
        train, dev, test = perm.permute(data)
        print(train.info())
        # print(train.head())
        print('------------')
        print(dev.info())
        # print(dev.head())
        print('------------')
        print(test.info())
        # print(test.head())
        print()
        

if __name__ == '__main__':
    test_shuffle(make_df())