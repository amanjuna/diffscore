import _pickle as pickle
import utils
import numpy as np
import pandas as pd
import tensorflow as tf
import os, time, scipy.stats, random
import matplotlib.pyplot as mpl
from collections import defaultdict

import config

GET_DATA = False

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

    def train(self, inputs_batch, labels_batch, index):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch,
                                     dropout=self.config.dropout)

        _, loss, grad_norm, summary = self.sess.run([self.train_op, self.loss, self.grad_norm, self.merged], feed_dict=feed)
        print("Grad norm: {}".format(grad_norm))
        print("Loss: {}".format(loss))
        self.train_writer.add_summary(summary, index)
        return loss

    def test(self, inputs_batch, labels_batch, index):
        feed = self.create_feed_dict(inputs_batch, labels_batch=labels_batch, dropout=self.config.dropout)
        corr, summary, pred_test, squared = self.sess.run([self.corr, self.merged, self.pred_test, self.squared], \
                                                            feed_dict = feed)
        self.dev_writer.add_summary(summary, index)

        return corr, pred_test, squared

    def predict(self, inputs_batch):
        feed = self.create_feed_dict(inputs_batch)
        predictions = self.sess.run(self.pred, feed_dict=feed)
        return predictions
    
    def run_epoch(self, train_data, dev_data, index):
        batch_size = 12000

        train_y = np.matrix(train_data["Standardized_Order"].as_matrix()).T
        train_x = train_data.ix[:, train_data.columns != "Standardized_Order"].as_matrix()

        for i in range(train_y.shape[0] // batch_size):
            start = i * batch_size
            end = (i+1) * batch_size
            loss = self.train(train_x[start:end,:], train_y[start:end], index)

        # for i in range(train_y.shape[0]):
        #     single_x = np.reshape(train_x[i], (1, 91))
        #     loss = self.train(single_x, train_y[i], index)

        # loss = self.train(train_x, train_y, index)

        dev_y = np.matrix(dev_data["Standardized_Order"].as_matrix()).T
        dev_x = dev_data.ix[:, dev_data.columns != "Standardized_Order"].as_matrix()

        dev_loss, dev_pred, squared = self.test(dev_x, dev_y, index)
        if self.verbose:
            print("train loss:", loss, "dev corr:", dev_loss, "dev squared:", squared)
        return squared

    def fit(self, train_examples, dev_set):
        best_dev_UAS = float("inf")
        for epoch in range(self.config.n_epochs):
            if self.verbose:
                print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_UAS = self.run_epoch(train_examples, dev_set, epoch)
            if dev_UAS < best_dev_UAS:
                best_dev_UAS = dev_UAS
                if self.saver:
                    if self.verbose:
                        print("New best dev UAS! Saving model in ./data/weights/network.weights")
                    self.saver.save(self.sess, self.config.model_output)
            if self.verbose: print()

    def evaluate(self, dev_data):
        self.saver.restore(self.sess, self.config.model_output)
        dev_y = np.matrix(dev_data["Standardized_Order"].as_matrix()).T
        dev_x = dev_data.ix[:, dev_data.columns != "Standardized_Order"].as_matrix()
        dev_loss, dev_pred, squared = self.test(dev_x, dev_y, 0)
        return dev_loss, squared
            
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

    def corr(self, pred):
        vx = pred - tf.reduce_mean(pred, axis = 0)
        vy = self.labels_placeholder - tf.reduce_mean(self.labels_placeholder, axis = 0)
        corr = tf.reduce_sum(tf.multiply(vx,vy))/(tf.multiply(tf.sqrt(tf.reduce_sum(tf.square(vx))), tf.sqrt(tf.reduce_sum(tf.square(vy))))+ tf.constant(1e-7))
        self.pred_test = pred
        self.squared = tf.losses.mean_squared_error(self.labels_placeholder, pred)
        return corr

    def make_pred(self, x, model):
        self.saver.restore(self.sess, model)
        feed = self.create_feed_dict(x)
        pred = self.sess.run(self.pred, feed_dict=feed)
        return pred
    
    def add_loss_op(self, pred):

        #loss = (1-self.corr(pred))**2
        loss = -self.corr(pred)
        # maybe try minimizing negative log for faster training in beginning?
        # loss = -tf.log(self.corr(pred))

        # squared distance
        loss += self.config.beta * tf.losses.mean_squared_error(self.labels_placeholder, pred)

        # L2 regularization
        weights = [var for var in tf.trainable_variables() if 'weights' in str(var)]
        l2 = 0 
        for var in weights:
            l2 += tf.nn.l2_loss(var)
        loss += self.config.lambd / 2 * l2
            
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
            tf.summary.scalar("pearson correlation", self.corr)
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(self.config.train_path, self.graph)
            self.dev_writer = tf.summary.FileWriter(self.config.dev_path, self.graph)
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
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
        self.corr = self.corr(self.pred)
        
    def __init__(self, config, verbose=True):
        self.config = config
        self.build_graph()
        self.verbose = verbose
        
def main():
    if GET_DATA:
        train_data, dev_data, test_data = utils.load_data()
        pickle.dump(train_data, open("train", "wb"))
        pickle.dump(dev_data, open("dev", "wb"))
        pickle.dump(test_data, open("test", "wb"))
    
    train_data = pickle.load(open("train", "rb"))
    dev_data = pickle.load(open("dev","rb"))
    test_data = pickle.load(open("test", "rb"))

    #train_x = train_data.ix[:, train_data.columns != "Standardized_Order"].as_matrix()
    #param = config.Config()
    #model = Model(param)
    #model.initialize()
    #pred = model.make_pred(train_x, "/home/amanjuna/CloudStation/Junior/diffscore/results/100_0.01_0.01_0.01/1/model.weights/weights")
    #print(pred)
                    
    #if not os.path.exists('./data/weights/'):
    #    os.makedirs('./data/weights/')

    loss = 0
    param = config.Config(hidden_size=100, n_epochs=20, beta=.01, lambd=.5, lr=0.01)
    model = Model(param)
    model.initialize()
    model.fit(train_data, dev_data)
    loss, squared = model.evaluate(dev_data)
    print(loss, squared)
    model.sess.close()

if __name__ == "__main__":
    main()
