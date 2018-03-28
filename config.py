'''
File: config.py
'''

# Predefined data groupings
KYLE = ['Kyle_Anterior', 'Kyle_Middle']
MARROW = ['Marrow_10x_G', 'Marrow_10x_E','Marrow_10x_B', 'Marrow_plate_M',  \
          'Marrow_plate_B',  'Marrow_plate_G']
PROTO = ['StandardProtocol', 'DirectProtocol']
REGEV = ['RegevIntestine', 'RegevDropseq']
FIBRO = ['Fibroblast_MyoF', 'Fibroblast_MFB']
INDIV = ['HumanEmbryo', 'HSC_10x', 'HSMM', 'AT2', 'EPI', 'Camargo', 'ChuCellType', 'GrunIntestine']

AVOID = ['GrunIntestine'] + [PROTO, REGEV, FIBRO]

ALLDATA = INDIV + [KYLE, MARROW, PROTO, REGEV, FIBRO]

ALLDATA_SINGLE = INDIV + KYLE + MARROW + PROTO + REGEV + FIBRO

NUM_SETS = 13 # Number of dsets when treating the above blocks (except INDIV) as single dsets
NUM_TRAIN = 7
NUM_DEV = 3
NUM_TEST = 3
N_REPLICATES = 30
N_PERCENTILES = 100 # Number of percentile statistics to include

CONTINUOUS_VAR = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", 
                  "Entropy_0", "HUGO_MGI_GC0", "HUGO_MGI_GC1", "mtgenes", 
                  "PC1", "PC2", "C1_axis", "C2_axis"]
IND_VAR = ["C1", "Plate", "10x", "DropSeq", "inDrop", 
           "Mouse", "Human", "nonrepeat", "repeat"]

EASY = KYLE + MARROW
MEDIUM = INDIV
HARD = REGEV + PROTO + FIBRO

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
    def __init__(self, n_features=1245, n_classes=1, dropout=0.0, n_layers = 2,
                 hidden_size=100, n_epochs=100, lr=0.0005, alpha=5, beta=.01, 
                 lambd=1, grad_clip=False, clip_val=10, crossval=0):
        attributes = {}
        self.n_features = n_features
        self.n_layers = n_layers
        attributes["n_features"] = self.n_features
        self.n_classes = n_classes
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.lambd = lambd
        self.grad_clip = grad_clip
        self.clip_val = clip_val
        self.crossval = crossval
        self.name = str(n_layers) + "_" + str(hidden_size) + "_" + str(lr) + "_" + str(alpha) + "_" + str(beta) + "_" + str(lambd) + "_" + str(n_epochs)
        self.output_path = "results/" + self.name + "/" + str(crossval) + "/"
        self.train_path = self.output_path + "/train/"
        self.dev_path = self.output_path + "/dev/"
        self.model_output = self.output_path + "/model.weights/weights"
        self.log_path = self.output_path + "log.txt"

    def define_crossval(self, crossval):
        self.crossval = crossval
        self.name = str(self.n_layers) + "_" + str(self.hidden_size) + "_" + str(self.lr) + "_" + str(self.beta) + "_" + str(self.lambd)
        self.output_path = "results/" + self.name + "/" + str(crossval)
        self.train_path = self.output_path + "/train/"
        self.dev_path = self.output_path + "/dev/"
        self.model_output = self.output_path + "/model.weights/weights"
        self.log_path = self.output_path + "log.txt"
