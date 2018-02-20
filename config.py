'''
File: config.py
'''
from datetime import datetime
class Config(object):

    output_path  = "results/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    train_path = output_path + "/train/"
    dev_path = output_path + "/dev/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    def __init__(self, n_features=91, n_classes=1, dropout=0.0, \
                 hidden_size=200, n_epochs=5, lr=0.0005, beta=.01, lambd=1, grad_clip=False):
        attributes = {}
        self.n_features = n_features
        attributes["n_features"] = self.n_features
        self.n_classes = n_classes
        self.dropout = dropout
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.beta = beta
        self.lambd = lambd
        self.grad_clip = grad_clip
        
