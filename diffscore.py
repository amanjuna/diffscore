import _pickle as pickle
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import os, time, scipy.stats, random
import matplotlib.pyplot as mpl
from collections import defaultdict
from datetime import datetime

GET_DATA = True
FEATURES = []

class Config(object):
    n_features = 103
    n_classes = 1
    dropout = 0
    batch_size = 2000
    hidden_size = 200
    n_epochs = 1000
    lr = 0.0005
    beta = .01
    grad_clip = False
    output_path  = "results/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    model_output = output_path + "model.weights/"
    log_path     = output_path + "log.txt"


class Model():
    def initialize(self):
        self.sess = tf.Session(graph = self.graph)
            # Tensorboard functions
        self.sess.run(self.init_op)

    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    def train_on_batch(self, inputs_batch, labels_batch, index):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss, grad_norm, summary = self.sess.run([self.train_op, self.loss, self.grad_norm, self.merged], feed_dict=feed)
        self.file_writer.add_summary(summary, index)
        return loss

    def predict_on_batch(self, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions
    
    def run_epoch(self, train_data, dev_data, index):
        train_y = np.matrix(train_data["Standardized_Order"].as_matrix()).T
        train_x = train_data.ix[:, train_data.columns != "Standardized_Order"].as_matrix()
        loss = self.train_on_batch(train_x, train_y, index)
        print(loss)

        dev_y = np.matrix(dev_data["Standardized_Order"].as_matrix()).T
        dev_x = dev_data.ix[:, dev_data.columns != "Standardized_Order"].as_matrix()
        pred = self.predict_on_batch(inputs_batch = dev_x)
        dev_loss = scipy.stats.spearmanr(pred, dev_y)[0]
        print("dev: {:.2f}".format(dev_loss))

        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)
        self.saver.save(self.sess, self.config.model_output)
        return dev_loss

    def fit(self, train_examples, dev_set):
        best_dev_UAS = 0
        for epoch in range(self.config.n_epochs):
            print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(train_examples, dev_set, epoch)
            if dev_UAS > best_dev_UAS:
                best_dev_UAS = dev_UAS
                if self.saver:
                    print("New best dev UAS! Saving model in ./data/weights/parser.weights")
                    self.saver.save(self.sess, './data/weights/parser.weights')
            print()

    # Adds variables
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.n_features], name = "input")
        self.labels_placeholder = tf.placeholder(tf.float32, shape = [None, 1], name = "output")
        self.dropout_placeholder = tf.placeholder(tf.float32)

    def add_prediction_op(self):
        x = self.input_placeholder
        layer1 = tf.contrib.layers.fully_connected(x, self.config.hidden_size)
        layer2 = tf.contrib.layers.fully_connected(layer1, self.config.hidden_size)
        pred = tf.contrib.layers.fully_connected(layer2, 1)
        return pred
        
    def add_loss_op(self, pred):
        vx = pred - tf.reduce_mean(pred, axis = 0)
        vy = self.labels_placeholder - tf.reduce_mean(self.labels_placeholder, axis = 0)
        loss = -tf.reduce_sum(tf.multiply(vx,vy))/tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vx))), tf.sqrt(tf.reduce_sum(tf.square(vy))))

        # squared distance
        loss += tf.losses.mean_squared_error(self.labels_placeholder, pred)

        # L2 regularization
        weights = [var for var in tf.trainable_variables() if 'weights' in str(var)]
        for var in weights:
            loss += tf.nn.l2_loss(var)
            
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        grads_and_vars = optimizer.compute_gradients(self.loss, vars)
        if self.config.grad_clip:
            grads_and_vars = [(tf.clip_by_norm(gv[0], self.config.clip_val), gv[1]) for gv in grads_and_vars]
        self.grad_norm = tf.global_norm(tf.trainable_variables())
        train_op = optimizer.apply_gradients(grads_and_vars)
        return train_op
    
    def build_graph(self):
        with tf.Graph().as_default() as self.graph:
            self.build()
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("grads norm", self.grad_norm)
            self.merged = tf.summary.merge_all()
            self.file_writer = tf.summary.FileWriter(self.config.output_path, 
                                                self.graph)
            self.init_op = tf.global_variables_initializer()
            self.saver=tf.train.Saver()
        self.graph.finalize()

    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)
        
        
    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        
    def __init__(self, config):
        self.config = config
        self.build_graph()
        
def main():
    if GET_DATA:
        train_data, dev_data, test_data = utils.load_data()
        pickle.dump(train_data, open("train", "wb"))
        pickle.dump(dev_data, open("dev", "wb"))
        pickle.dump(test_data, open("test", "wb"))
    
    train_data = pickle.load(open("train", "rb"))
    dev_data = pickle.load(open("dev","rb"))
    test_data = pickle.load(open("test", "rb"))
    
    if not os.path.exists('./data/weights/'):
        os.makedirs('./data/weights/')

    config = Config()
    model = Model(config)
    model.initialize()
    model.fit(train_data, dev_data)


if __name__ == "__main__":
    main()
