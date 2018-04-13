import _pickle as pickle
import numpy as np
import math
import pandas as pd
import tensorflow as tf
import os, time, scipy.stats, random, preprocessing, config
import matplotlib.pyplot as mpl
from collections import defaultdict

class Model():
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

        
    def create_feed_dict(self, inputs_batch, labels_batch=None, dropout=0):
        '''
        Creates the model feed dict
        '''
        feed_dict = {self.input_placeholder: inputs_batch, self.dropout_placeholder: dropout}
        if labels_batch is not None:
            feed_dict[self.labels_placeholder] = labels_batch
        return feed_dict

    
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

    
    def test(self, data, global_corr=False):
        '''
        Tests the model's present performance by getting correlation
        and squared error on the input batch. 
        '''
        if global_corr:
            data_y = np.matrix(data["Standardized_Order"].as_matrix()).T
            data_x = data.ix[:, data.columns != "Standardized_Order"].as_matrix()
            feed = self.create_feed_dict(data_x, labels_batch=data_y, dropout=self.config.dropout)
            corr, squared = self.sess.run([self.corr, self.squared], feed_dict=feed)
        else:
            input_dsets = list(data.index.unique())
            corr, squared = 0, 0
            for dset in input_dsets:
                dataset = data.loc[dset]
                dataset_y = np.matrix(dataset["Standardized_Order"].as_matrix()).T
                dataset_x = dataset.ix[:, dataset.columns != "Standardized_Order"].as_matrix()
                feed = self.create_feed_dict(dataset_x, labels_batch=dataset_y, dropout=self.config.dropout)
                _corr, _squared, pred = self.sess.run([self.corr, self.squared, self.pred], feed_dict = feed)
                corr += _corr
                squared += _squared
            corr /= len(input_dsets)
            squared /= len(input_dsets)
        
        return corr, squared
  
    
    def run_epoch(self, train_data, dev_data, index):        
        train_dsets = list(train_data.index.unique())
        random.shuffle(train_dsets)
        dev_dsets = list(dev_data.index.unique())
        for dset in train_dsets:
            train = train_data.loc[dset]
            train_y = np.matrix(train["Standardized_Order"].as_matrix()).T
            train_x = train.ix[:, train.columns != "Standardized_Order"].as_matrix()
            loss = self.train(train_x, train_y)

        train_corr, train_squared = self.test(train_data)
        dev_corr, dev_squared = self.test(dev_data)
        if self.verbose:
            print("train corr:", train_corr, "dev corr:", dev_corr, "dev squared:", dev_squared)
        return dev_corr+train_corr, dev_squared

    
    def fit(self, train_examples, dev_set):
        best_dev_corr = 0
        for epoch in range(self.config.n_epochs):
            if self.verbose:
                print("Epoch {:} out of {:}".format(epoch + 1, self.config.n_epochs))
            dev_corr, dev_squared = self.run_epoch(train_examples, dev_set, epoch)
            # 
            if (dev_corr > best_dev_corr) or epoch == 0:
                best_dev_corr = dev_corr
                if self.saver:
                    if self.verbose:
                        print("New best dev corr! Saving model in ./results/model.weights/weights")
                    self.saver.save(self.sess, self.config.model_output)
                    
            if self.verbose: print()
        return epoch

    
    def evaluate(self, data, global_corr=False):
        self.saver.restore(self.sess, self.config.model_output)
        corr, squared = self.test(data, global_corr)
        return corr, squared
        
   
    def add_placeholders(self):
        self.input_placeholder = tf.placeholder(tf.float32, shape = [None, self.config.n_features], name = "input")
        self.labels_placeholder = tf.placeholder(tf.float32, shape = [None, 1], name = "output")
        self.dropout_placeholder = tf.placeholder(tf.float32)

        
    def add_prediction_op(self):
        x = self.input_placeholder
        arr = [0]*(self.config.n_layers+1)
        arr[0] = x
        for i in range(1, self.config.n_layers+1):
            arr[i] = tf.contrib.layers.fully_connected(arr[i-1], self.config.hidden_size)
        output = tf.contrib.layers.fully_connected(arr[self.config.n_layers], 1, activation_fn=None)
        return output


    def corr(self, pred):
        vx = pred - tf.reduce_mean(pred, axis = 0)
        vy = self.labels_placeholder - tf.reduce_mean(self.labels_placeholder, axis = 0)
        corr_num = tf.reduce_sum(tf.multiply(vx, vy))
        corr_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(vx)), tf.reduce_sum(tf.square(vy))))
        corr = corr_num/corr_den
        self.pred = pred
        self.corr_num = corr_num
        self.corr_den = corr_den
        self.squared = tf.losses.mean_squared_error(self.labels_placeholder, pred)
        return corr

    def make_pred(self, x, model):
        self.saver.restore(self.sess, model)
        feed = self.create_feed_dict(x)
        pred = self.sess.run(self.pred, feed_dict=feed)
        return pred
    
    def add_loss_op(self, pred):
        # correlation loss
        loss = self.config.alpha * (1-self.corr(pred))**2
       
        # squared loss
        loss += self.config.beta * tf.losses.mean_squared_error(self.labels_placeholder, pred)

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
    param = config.Config(hidden_size=200, 
                          n_epochs=100, 
                          alpha=2.0, 
                          beta=0.1, 
                          lambd=0.01, 
                          lr=0.001)
    
    train, dev, test, dsets = preprocessing.load_data(model_path=param.output_path)

    # Fit and log model
    model = Model(param)
    model.initialize()
    model.fit(train, dev)
    model.sess.close()

if __name__ == "__main__":
    main()
