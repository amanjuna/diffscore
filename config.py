'''
File: config.py
'''

class Config(object):

    output_path  = "results/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"

    def __init__(self, n_features=103, n_classes=1, dropout=0.0, batch_size=2000,\
                 hidden_size=200, n_epochs=1000, lr=0.0005, beta=.01, lambd=1, grad_clip=False):
        attributes = {}
        self.n_features = n_features
        attributes["n_features"] = self.n_features
        self.n_classes = n_classes
        attributes[]
        self.dropout = dropout
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_epochs = n_epochs
        self.lr = lr
        self.beta = beta
        self.lambd = lambd
        self.grad_clip = grad_clip
        