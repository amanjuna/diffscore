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
        self.verbose = verbose
        self.build_graph() 
        self.sess = tf.Session(graph = self.graph)
        self.sess.run(self.init_op)

    def build_graph(self):      
        with tf.Graph().as_default() as self.graph:
            self.global_step = tf.Variable(0, name='global_step', trainable=False) 
            self.add_placeholders()
            self.pred = self.add_prediction_op()
            self.loss = self.add_loss_op(self.pred)            
            self.train_op = self.add_training_op(self.loss)         
            self.init_op = tf.global_variables_initializer()
            
            # Adds tensorflow utilites
            self.train_writer = tf.summary.FileWriter('./train', self.graph) 
            self.saver = tf.train.Saver() 
            self.add_summaries()
         
        self.graph.finalize()

        
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, 
                                                shape=[None, self.config.n_features], 
                                                name = "input")
        self.labels_placeholder = tf.placeholder(tf.float32, 
                                                 shape=[None, 1],
                                                 name = "output")
        self.dropout_placeholder = tf.placeholder(tf.float32)
        self.weight_placeholder = tf.placeholder(tf.float32, shape=[None, 1])

        
    def add_prediction_op(self):
        '''
        Defines the computational graph then returns the prediction tensor
        '''
        x = self.input_placeholder#self.combine_features()
        arr = [0]*(self.config.n_layers+1)
        arr[0] = tf.contrib.layers.layer_norm(x)
        for i in range(1, self.config.n_layers+1):
            arr[i] = tf.contrib.layers.fully_connected(arr[i-1], self.config.hidden_size)
            arr[i] = tf.contrib.layers.layer_norm(arr[i])
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
        gcs = self.input_placeholder[:, 1:start]
        sims = self.input_placeholder[:, start:end]
        combined_weight = tf.get_variable("Combination_weights", shape=(1, self.config.n_neighbors))
        combined = gcs * sims * combined_weight
        temp = tf.concat([self.input_placeholder[:, 0:1], combined], axis=1)
        return tf.concat([temp, self.input_placeholder[:, end:]], axis=1)


    def add_loss_op(self, pred):
        # squared loss
        self.squared = tf.losses.mean_squared_error(self.labels_placeholder, 
                                                    pred, 
                                                    weights=self.weight_placeholder,
                                                    loss_collection=None)
        loss = self.squared

        # L2 regularization
        loss += self.config.lambd / 2 * self.weight_l2()
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
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)
        return train_op


    def fit(self, train_examples, dev_set):
        best_dev = float("inf")
        epoch = 0
        while epoch < self.config.n_epochs:
            epoch += 1
            if self.verbose:
                print("Epoch {:} out of {:}".format(epoch, self.config.n_epochs))
            dev_squared = self.run_epoch(pd.concat([train_examples, dev_set]), epoch)
            if dev_squared < best_dev:
                best_dev = dev_squared
                if self.saver:
                    if self.verbose:
                        print("New best dev MSE! Saving model in ./results/model.weights/weights")
                    self.saver.save(self.sess, self.config.model_output)
            if self.verbose: print()

        return epoch


    def run_epoch(self, train_data, index):
        train_x, train_y, weight = self.format_dataset(train_data)
        loss = self.train(train_x, train_y, weight) 
        train_squared, pred = self.test(train_data)
        if index % 100 == 0:
            corr, _ = scipy.stats.spearmanr(pred, train_y)
        else:
            corr = 0
        if self.verbose:
            print("train corr:", corr, " train squared:", train_squared)

        return train_squared

    
    def format_dataset(self, data, n=config.NUM_CELLS_IN_DATASET):
        cols = list(data.columns)
        cols.remove('Standardized_Order')
        cols.remove('weight')
        X = data.loc[:, cols].as_matrix()
        y = np.matrix(data["Standardized_Order"].as_matrix()).T
        weight = np.matrix(data["weight"].as_matrix()).T
        weight *= weight.shape[0]/np.sum(weight)
        return X, y, weight


    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0, weight=None):
        '''
        Creates the model feed dict
        '''
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        if weight is not None:
            feed_dict[self.weight_placeholder] = weight
        return feed_dict


    def train(self, inputs_batch, labels_batch, weight):
        '''
        Performs a single training iteration on the given
        input batch.
        '''
        feed = self.create_feed_dict(inputs_batch=inputs_batch, 
                                     labels_batch=labels_batch,
                                     dropout=self.config.dropout,
                                     weight=weight)
        _, loss, summary, global_step = self.sess.run([self.train_op, self.loss, self.merged, self.global_step], 
                                          feed_dict=feed)
        self.train_writer.add_summary(summary, global_step)
        return loss

    
    def test(self, data):
        '''
        Tests the model's present performance by getting correlation
        and squared error on the input batch. 
        '''
        input_dsets = list(data.index.unique())
        dataset_x, dataset_y, weight = self.format_dataset(data)
        feed = self.create_feed_dict(dataset_x, dataset_y, dropout=self.config.dropout, weight=np.ones_like(dataset_y))
        squared, pred = self.sess.run([self.squared, self.pred], feed_dict=feed)
        return squared, pred 

    
    def evaluate(self, data):
        self.saver.restore(self.sess, self.config.model_output)
        squared, pred = self.test(data)
        return squared, pred


    def make_pred(self, data):
        self.saver.restore(self.sess, self.config.model_output)
        cols = list(data.columns)
        cols.remove('weight')
        X = data.ix[:, cols].as_matrix()
        feed = self.create_feed_dict(X)
        pred = self.sess.run(self.pred, feed)
        pred = np.squeeze(pred)
        return pred 


    def save(self):
        """
        Saves session
        """
        if not os.path.exists(self.config.model_output):
            os.makedirs(self.config.model_output)

        self.saver.save(self.sess, self.config.model_output)
       
        
    def weight_l2(self):
        weights = [var for var in tf.trainable_variables() if 'weights' in str(var)]
        l2 = np.sum([tf.nn.l2_loss(var) for var in weights])
        return l2
        
    
    def add_summaries(self):
        with tf.name_scope("metrics"):
            tf.summary.scalar('Loss', self.loss)
            tf.summary.scalar('Squared Error', self.squared)
            tf.summary.scalar("Grad Norm", self.grad_norm)
            tf.summary.scalar('Weight L2', self.weight_l2())
        weights = [var for var in tf.trainable_variables()]
        for i, weight in enumerate(weights):
            tf.summary.histogram(weight.name, weight)
        self.merged = tf.summary.merge_all()
   
        
def main():
    tf.set_random_seed(1234)
    param = config.Config(hidden_size=300,
                          n_layers=3, 
                          n_epochs=1000,  
                          beta=1, 
                          lambd=0, 
                          lr=5e-5)
    
    # train, dev, test, dsets = preprocessing.load_data(model_path=param.output_path)
    all_data = preprocessing.load_data(model_path=param.output_path, separate=False)
    all_data = pd.read_csv("data/unified_processed.csv").sample(1000)
    all_data = all_data.loc[:,"Standardized_Order":"weight"]
    plate = all_data.loc[(all_data.Plate==1.0) | (all_data.C1==1.0)]
    nonplate = all_data.loc[(all_data.Plate==0) & (all_data.C1==0)]

    # Fit and log model
    model = Model(param)
    model.fit(all_data, pd.DataFrame())
    visualize.model_prediction_plot(model, all_data)
    visualize.model_prediction_plot(model, plate, './plots/model_predictions_plate.png')
    visualize.model_prediction_plot(model, nonplate, './plots/model_prediction_nonplate.png')
    # model.sess.close()

if __name__ == "__main__":
    main()
