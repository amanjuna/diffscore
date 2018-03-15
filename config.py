'''
File: config.py
'''
from datetime import datetime
class Config(object):


    def __init__(self, n_features=125, n_classes=1, dropout=0.0, n_layers = 2, \
                 hidden_size=100, n_epochs=100, lr=0.0005, alpha=5, beta=.01, lambd=1, grad_clip=False, clip_val=10, crossval=0):
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
        self.name = str(1) + "_" + str(hidden_size) + "_" + str(lr) + "_" + str(beta) + "_" + str(lambd)
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
