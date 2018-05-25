'''
File: config.py
'''
import os
import time
import json

# Predefined data groupings
KYLE = ['Kyle_Anterior', 'Kyle_Middle']
MARROW_10X = ['Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B']
MARROW_PLATE = ['Marrow_plate_M', 'Marrow_plate_B', 'Marrow_plate_G']
PROTO = ['StandardProtocol', 'DirectProtocol']
REGEV = ['RegevIntestine', 'RegevDropseq']
# FIBRO = ['Fibroblast_MyoF', 'Fibroblast_MFB'] # excluded as of 4/10/18
# INDIV = ['HumanEmbryo', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType', 'GrunIntestine']

# INDIV as of 4/10/18 (excludes Camargo and ChuCellType)
INDIV = ['HumanEmbryo', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'GrunIntestine']

# ALLDATA = INDIV + [KYLE, MARROW, PROTO, REGEV, FIBRO]
ALLDATA = INDIV + [KYLE, MARROW_10X, MARROW_PLATE, PROTO, REGEV]

# ALLDATA_SINGLE = INDIV + KYLE + MARROW + PROTO + REGEV + FIBRO
ALLDATA_SINGLE = ['ChuCellType', 'Kyle_Anterior', 'AT2', 'EPI', 'HumanEmbryo',
       'HSMM', 'Kyle_Middle', 'GrunIntestine', 'RegevDropseq',
       'RegevIntestine', 'HSC_10x', 'Marrow_10x_M', 'Marrow_10x_G',
       'Marrow_10x_E', 'Marrow_plate_M', 'Marrow_plate_B', 'Marrow_10x_B',
       'Marrow_plate_G', 'DirectProtocol', 'StandardProtocol']

NUM_CELLS_IN_DATASET = 1000
NUM_SETS = 10 # Number of dsets when treating the above blocks (except INDIV) as single dsets
NUM_TRAIN = 5
NUM_DEV = 2
NUM_TEST = 3
N_REPLICATES = 30
N_PERCENTILES = 2 # Number of percentile statistics to include

'''
CONTINUOUS_VAR = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", 
                  "Entropy_0", "HUGO_MGI_GC0", "HUGO_MGI_GC1", "mtgenes", 
                  "PC1", "PC2", "C1_axis", "C2_axis"]
'''
CONTINUOUS_VAR = []#["GeneCoverage_0", "PC1", "PC2"]
IND_VAR = ["C1", "Plate", "10x", "DropSeq", "inDrop", 
           "Mouse", "Human", "nonrepeat", "repeat"]

EASY = KYLE + MARROW_10X + MARROW_PLATE
MEDIUM = INDIV
HARD = REGEV + PROTO #+ FIBRO

PLATE = list(set(["Astrocytoma", "ChuCellType", "Clarke", 
                  "Fibroblast_MFB", "Fibroblast_MyoF", "Gottgens", "HumanEmbryo", 
             "Marrow_plate_B", "Marrow_plate_G", 
             "Marrow_plate_M", "RegevIntestine", "Stemnet"]).intersection(set(ALLDATA_SINGLE)))
TEN_X = list(set(["HSC_10x", "Marrow_10x_B", 
                  "Marrow_10x_E", "Marrow_10x_G", 
                  "Marrow_10x_M"]).intersection(set(ALLDATA_SINGLE)))

C1 = list(set(["AT2", "EPI", "HSMM", 
               "Kyle_Anterior", "Kyle_Middle"]).intersection(set(ALLDATA_SINGLE)))
DROPSEQ = list(set(["RegevDropseq"]).intersection(set(ALLDATA_SINGLE)))
INDROP = list(set(["Camargo", "DirectProtocol", "StandardProtocol"]).intersection(set(ALLDATA_SINGLE)))


class Config(object):
    def __init__(self, name, n_features=108, n_neighbors=50, n_classes=1, 
                 dropout=.2, n_layers=3,hidden_size=300, n_epochs=10, batch_size=256, 
                 lr=5e-5, lambd=1, grad_clip=False, clip_val=10, load=False):

        assert name is not None, "You must specify an experiment name"
        self.name = name
        home = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.output_path = os.path.join(home, 'results', self.name)

        if load:
            self.load_params(os.path.join(self.output_path, 'params.json'))
        else:
            self.n_features = n_features
            self.n_neighbors = n_neighbors
            self.n_layers = n_layers
            self.batch_size = batch_size
            self.n_classes = n_classes
            self.dropout = dropout
            self.hidden_size = hidden_size
            self.n_epochs = n_epochs
            self.lr = lr
            self.lambd = lambd
            self.grad_clip = grad_clip
            self.clip_val = clip_val
            self.tensorboard_dir = os.path.join(self.output_path, 'tensorboard/')
            self.weights_path = os.path.join(self.output_path, 'weights')
            self.model_output = os.path.join(self.weights_path, 'weights.ckpt')
            self.write_params()


    def write_params(self):
        params = {}
        for param, value in self.__dict__.items():
            params[param] = value
        with open(os.join(self.output_path, 'params.json'), 'w') as file:
            json.dump(params, file)


    def load_params(self, filename):
        with open(filename) as f:
            data = json.load(f)
            for param, val in data.items():
                self.__dict__[param] = val
