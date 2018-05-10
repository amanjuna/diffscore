import _pickle as pickle
import math
import os
import time
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import tensorflow as tf
import scipy.stats 

import preprocessing
import config
import visualize


class Model():

    def __init__(self, config, verbose=True):
        self.config = config
        self.build_graph()
        self.verbose = verbose


    def build_graph(self):
        with tf.Graph().as_default() as self.graph:
            self.build()
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("grads norm", self.grad_norm)
            tf.summary.scalar("pearson correlation", self.corr)
            self.merged = tf.summary.merge_all()
            self.init_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver()
        self.graph.finalize()


    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.corr = self.corr(self.pred)


    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, 
                                                shape=[None, None, self.config.n_features], 
                                                name = "input")
        self.labels_placeholder = tf.placeholder(tf.float32, 
                                                 shape=[None, None, 1],
                                                 name = "output")
        self.dropout_placeholder = tf.placeholder(tf.float32)


    def initialize(self):
        '''
        Initializes model variables
        '''
        session_conf = tf.ConfigProto(
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1,
            allow_soft_placement=True,
            device_count={'CPU': 1})
        self.sess = tf.Session(config=session_conf, graph = self.graph)
        self.sess.run(self.init_op)


    def format_dataset(self, dataset, n=config.NUM_CELLS_IN_DATASET):
        dsets = list(dataset.index.unique())
        X = np.zeros((len(dsets), n, self.config.n_features))
        y = np.zeros((len(dsets), n, 1))
        for i, dset in enumerate(dsets):
            data = dataset.loc[dset]
            data = data.sample(n=n, replace=True)
            X[i] = data.loc[:, data.columns != "Standardized_Order"].as_matrix()
            y[i] = np.matrix(data["Standardized_Order"].as_matrix()).T
        return X, y
        
        
    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        '''
        Creates the model feed dict
        '''
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict


    def add_prediction_op(self):
        '''
        Defines the computational graph then returns the prediction tensor
        '''
        x = self.combine_features()
        arr = [0]*(self.config.n_layers+1)
        arr[0] = x
        for i in range(1, self.config.n_layers+1):
            arr[i] = tf.contrib.layers.fully_connected(arr[i-1], self.config.hidden_size)
        output = tf.contrib.layers.fully_connected(arr[self.config.n_layers], 1, activation_fn=None)
        return output


    def combine_features(self):
        '''
        Takes GC and similarity values from input, combines them
        (currently by multiplying them), then outputs a new 
        tensor where each example should be:
            cell GC value (x1)
            neighbor_GC*neighbor_sim features (x50)
            Species and platform indicators (x7, for now)
        '''
        start = self.config.n_neighbors + 1
        end = start + self.config.n_neighbors
        gcs = self.input_placeholder[:, :, 1:start]
        sims = self.input_placeholder[:, :, start:end]
        combined_weight = tf.get_variable("Combination_weights", shape=(1, 1, self.config.n_neighbors))
        combined = gcs * sims * combined_weight
        temp = tf.concat([self.input_placeholder[:, :, 0:1], combined], axis=2)
        return tf.concat([temp, self.input_placeholder[:, :, end:]], axis=2)
    
    
    def train(self, inputs_batch, labels_batch):
        '''
        Performs a single training iteration on the given
        input batch.
        '''
        feed = self.create_feed_dict(inputs_batch=inputs_batch, 
                                     labels_batch=labels_batch,
                                     dropout=self.config.dropout)
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.merged], feed_dict=feed)
        return loss

    
    def test(self, data):
        '''
        Tests the model's present performance by getting correlation
        and squared error on the input batch. 
        '''
        input_dsets = list(data.index.unique())
        dataset_x, dataset_y = self.format_dataset(data)
        feed = self.create_feed_dict(dataset_x, dataset_y, dropout=self.config.dropout)
        corr, squared, pred = self.sess.run([self.corr, self.squared, self.pred], feed_dict = feed)
        return corr, squared, pred 
    

    def run_epoch(self, train_data, index):
        train_x, train_y = self.format_dataset(train_data)
        loss = self.train(train_x, train_y)
        
        train_corr, train_squared, pred = self.test(train_data)
        if self.verbose:
            print("train corr:", train_corr, "train squared:", train_squared)
        return train_corr, train_squared

    
    def fit(self, train_examples, dev_set):
        best_dev = float("inf")
        epoch = 0
        while epoch < self.config.n_epochs:
            epoch += 1
            if self.verbose:
                print("Epoch {:} out of {:}".format(epoch, self.config.n_epochs))
            dev_corr, dev_squared = self.run_epoch(pd.concat([train_examples, dev_set]), epoch)
            # 
            if dev_squared < best_dev:
                epoch = 0
                best_dev = dev_squared
                if self.saver:
                    if self.verbose:
                        print("New best dev MSE! Saving model in ./results/model.weights/weights")
                    self.saver.save(self.sess, self.config.model_output)
                    
            if self.verbose: print()
        return epoch

    
    def evaluate(self, data):
        self.saver.restore(self.sess, self.config.model_output)
        corr, squared, pred = self.test(data)
        return corr, squared, pred


    def corr(self, pred):
        vx = tf.squeeze(pred) - tf.reduce_mean(pred, axis = 1)
        vy = tf.squeeze(self.labels_placeholder) - tf.reduce_mean(self.labels_placeholder, axis = 1)
        corr_num = tf.reduce_sum(tf.multiply(vx, vy), axis=1)
        corr_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(vx), axis=1), tf.reduce_sum(tf.square(vy), axis=1)))
        corr = corr_num/corr_den
        self.squared = tf.losses.mean_squared_error(self.labels_placeholder, pred)
        corr = tf.reduce_mean(corr)
        self.pred = pred
        return corr


    def make_pred(self, data):
        self.saver.restore(self.sess, self.config.model_output)
        preds = []
        init_len = data.shape[0]
        rem = config.NUM_CELLS_IN_DATASET - data.shape[0] % config.NUM_CELLS_IN_DATASET
        zeros = pd.DataFrame(0, index=np.arange(rem), columns=data.columns)
        data = data.append(zeros)
        for i in range(int(data.shape[0]/config.NUM_CELLS_IN_DATASET)):
            _data = data[i*config.NUM_CELLS_IN_DATASET:(i+1)*config.NUM_CELLS_IN_DATASET]
            X = _data.ix[:, _data.columns != "Standardized_Order"].as_matrix()
            X = np.expand_dims(X, axis=0)
            feed = self.create_feed_dict(X)
            pred = self.sess.run(self.pred, feed)
            pred = np.squeeze(pred)
            preds.append(pred)
        pred = np.concatenate(preds)[0:init_len]
        return pred
    

    def add_loss_op(self, pred):
       
        # squared loss
        loss = self.config.beta * tf.losses.mean_squared_error(self.labels_placeholder, pred)

        # L2 regularization
        weights = [var for var in tf.trainable_variables() if 'weights' in str(var)]
        l2 = np.sum([tf.nn.l2_loss(var) for var in weights]) 
        loss += self.config.lambd / 2 * l2
            
        return loss

    def add_training_op(self, loss):
        optimizer = tf.train.AdamOptimizer(self.config.lr)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        grads = [gv[0] for gv in grads_and_vars]
        vars = [gv[1] for gv in grads_and_vars]
        if self.config.grad_clip:
            grads, _ = tf.clip_by_global_norm(grads, self.config.clip_val)
        grads_and_vars = zip(grads, vars)
        self.grad_norm = tf.global_norm(grads)
        train_op = optimizer.apply_gradients(grads_and_vars)
        return train_op
    



    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)
        
        
        
def main():
    param = config.Config(hidden_size=200, 
                          n_epochs=100, 
                          alpha=0, 
                          beta=1, 
                          lambd=1, 
                          lr=0.01)
    
    # train, dev, test, dsets = preprocessing.load_data(model_path=param.output_path)
    all_data = preprocessing.load_data(model_path=param.output_path, separate=False)
    all_data = all_data.loc[:,"Standardized_Order":"Mouse"]
    print(all_data.columns)
    print(all_data.info())

    # Fit and log model
    model = Model(param)
    model.initialize()
    model.fit(all_data, pd.DataFrame())
    visualize.model_prediction_plot(param, all_data)
    # model.sess.close()

if __name__ == "__main__":
    main()
