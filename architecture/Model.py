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
    def __init__(self, config, data, is_training=True):
        self.config = config
        self.is_training = is_training
        self.build(data)
       

    def build(self, data):
        dataset = self.format_dataset(data)
        iterator = dataset.make_initializable_iterator()
        self.input_data = iterator.get_next()
        self.iter_init = iterator.initializer
        self.global_step = tf.Variable(0, trainable=False)
        
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        if self.is_training:
            self.train_op = self.add_training_op(self.loss)    


    def format_dataset(self, data, n=config.NUM_CELLS_IN_DATASET):
        cols = list(data.columns)
        cols.remove('Standardized_Order')
        cols.remove('weight')

        inputs = tf.constant(data.loc[:, cols].as_matrix(), tf.float32)
        weights = np.matrix(data["weight"].as_matrix()).T
        weights *= weights.shape[0]/np.sum(weights)
        weights = tf.constant(weights, tf.float32)
        labels = tf.constant(np.matrix(data["Standardized_Order"].as_matrix()).T, tf.float32)

        dataset = tf.data.Dataset.from_tensor_slices((inputs, weights, labels))
        if self.is_training:
            dataset = dataset.shuffle(self.train_len)
        dataset = dataset.batch(self.config.batch_size)
        return dataset
      
      
    def add_prediction_op(self):
        '''
        Defines the computational graph then returns the prediction tensor
        '''
        with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):
            x = self.input_data
            arr = [0]*(self.config.n_layers+1)
            arr[0] = tf.contrib.layers.layer_norm(x)
            for i in range(1, self.config.n_layers+1):
                arr[i] = tf.contrib.layers.fully_connected(arr[i-1], self.config.hidden_size)
                arr[i] = tf.contrib.layers.layer_norm(arr[i])
                arr[i] = tf.layers.dropout(inputs=arr[i], rate=self.config.dropout,
                                           training=self.is_training)
            output = tf.contrib.layers.fully_connected(arr[self.config.n_layers], 1,
                                                       activation_fn=tf.nn.sigmoid)
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
        gcs = self.input[0][:, 1:start]
        sims = self.input[0][:, start:end]
        combined_weight = tf.get_variable("Combination_weights", shape=(1, self.config.n_neighbors))
        combined = gcs * sims * combined_weight
        temp = tf.concat([self.input[0][:, 0:1], combined], axis=1)
        return tf.concat([temp, self.input[0][:, end:]], axis=1)


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
    

    def add_loss_op(self, pred):
        with tf.variable_scope("loss", reuse=tf.AUTO_REUSE):
            # squared loss
            self.squared = tf.losses.mean_squared_error(self.input[2], 
                                                        pred, 
                                                        weights=self.input[1],
                                                        loss_collection=None)
            loss = self.squared
        
            # L2 regularization
            loss += self.config.lambd / 2 * self.weight_l2()
        return loss 
    

    def correlation_op(self):
        vx = tf.squeeze(self.pred) - tf.reduce_mean(self.pred,)
        vy = tf.squeeze(self.input[2]) - tf.reduce_mean(self.input[2])
        corr_num = tf.reduce_sum(tf.multiply(vx, vy))
        corr_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(vx)), tf.reduce_sum(tf.square(vy))))
        self.corr = corr_num/corr_den

      
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


    def make_pred(self):
        self.saver.restore(self.sess, self.config.model_output)
        self.sess.run(self.iter_init)
        preds = []
        while True:
            try:
                pred = self.sess.run(self.pred)
                pred = np.squeeze(pred)
                preds += list(pred)
            except tf.errors.OutOfRangeError:
                break
                
        return preds


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
            self.correlation_op()
            tf.summary.scalar('Pearson Correlation', self.corr)
        
            weights = [var for var in tf.trainable_variables()]
            for i, weight in enumerate(weights):
                tf.summary.histogram(weight.name, weight)

        self.merged = tf.summary.merge_all()
        
        
def main():
    tf.set_random_seed(1234)
    
    # train, dev, test, dsets = preprocessing.load_data(model_path=param.output_path)
    all_data = pd.read_csv("data/unified_processed.csv", index_col="Dataset")
    print(all_data.columns)
    all_data = all_data.loc[:,"Standardized_Order":"weight"]
    plate = all_data.loc[(all_data.Plate==1.0) | (all_data.C1==1.0)]
    nonplate = all_data.loc[(all_data.Plate==0) & (all_data.C1==0)]
    datasets = config.ALLDATA_SINGLE[0]

    
    # Train set

    for dset in config.ALLDATA:
        if isinstance(dset, (list,)):
            val_set = dset
            dset = dset[0]
        else:
            val_set = [dset]
        param = config.Config(hidden_size=300,
                              n_layers=3, 
                              n_epochs=200,  
                              beta=1, 
                              lambd=1, 
                              lr=3e-5,
                              dropout=0.0,
                              name = dset + "_combined") 
        train_indices = [name for name in config.ALLDATA_SINGLE if 
                             name not in val_set]
        train_data = all_data.loc[train_indices, :]
        val_data = all_data.loc[val_set, :]
        if len(val_data) == 0:
            continue
        # Fit and log model
        model = Model(param, train_data, val_data, is_training=True)
        model.fit()
    
    #pred_model = Model(param, val_data, is_training=False)
    #pred = pred_model.make_pred()
    
    
    #visualize.model_prediction_plot(param, all_data)
    #visualize.model_prediction_plot(param, plate, './plots/model_predictions_plate.png')
    #visualize.model_prediction_plot(param, nonplate, './plots/model_prediction_nonplate.png')
    # model.sess.close()

if __name__ == "__main__":
    main()
