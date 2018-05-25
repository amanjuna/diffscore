'''
File: config.py
'''
import os
import time
import json


class Config(object):
    def __init__(self, name, n_features=108, n_neighbors=50, n_classes=1, 
                 dropout=0.1, n_layers=3, hidden_size=300, n_epochs=200, batch_size=256, 
                 lr=3e-3, lambd=1e-3, grad_clip=False, clip_val=10, load=False):

        assert name is not None, "You must specify an experiment name"
        self.name = name
        home = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.output_path = os.path.join(home, 'results', self.name)
        if not os.path.exists(self.output_path):
            os.mkdir(self.output_path)

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
        with open(os.path.join(self.output_path, 'params.json'), 'w') as file:
            json.dump(params, file)


    def load_params(self, filename):
        with open(filename) as f:
            data = json.load(f)
            for param, val in data.items():
                self.__dict__[param] = val
