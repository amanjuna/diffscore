'''
constants.py

All the dataset constants
'''

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
# unused features
CONTINUOUS_VAR = ["G1_mean", "G2_mean", "HK_mean", "GeneCoverage_0", 
                  "Entropy_0", "HUGO_MGI_GC0", "HUGO_MGI_GC1", "mtgenes", 
                  "PC1", "PC2", "C1_axis", "C2_axis"]
'''
CONTINUOUS_VAR = []
IND_VAR = ["C1", "Plate", "10x", "DropSeq", "inDrop", "Mouse", "Human"]

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
