import tensorflow as tf
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "./architecture/models/")

from Model import Model
import config

class Product(Model):      
    def add_prediction_op(self):
        '''
        Defines the computational graph then returns the prediction tensor
        '''
        with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):
            x = self.combine_features() # data (self.input is a (cell, weight, label) tuple)
            arr = [0]*(self.config.n_layers+1)
            arr[0] = tf.contrib.layers.layer_norm(x)
            #arr[0] = x 
            for i in range(1, self.config.n_layers+1):
                arr[i] = tf.contrib.layers.fully_connected(arr[i-1], self.config.hidden_size)
                arr[i] = tf.contrib.layers.layer_norm(arr[i])
            output = tf.contrib.layers.fully_connected(arr[self.config.n_layers], 1, activation_fn=tf.nn.sigmoid)
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
        gcs = self.input_data[0][:, 1:start]
        sims = self.input_data[0][:, start:end]
        combined_weight = tf.get_variable("Combination_weights", shape=(1, self.config.n_neighbors))
        combined = gcs * sims * combined_weight
        temp = tf.concat([self.input_data[0][:, 0:1], combined], axis=1)
        return tf.concat([temp, self.input_data[0][:, end:]], axis=1)

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
            loss = tf.losses.mean_squared_error(self.input_data[2], 
                                                pred, 
                                                weights=self.input_data[1])
            # L2 regularization
            loss += self.config.lambd / 2 * self.weight_l2()
        return loss 

    def input_op(self, data):
        cols = list(data.columns)
        cols.remove('Standardized_Order')
        cols.remove('weight')

        inputs = tf.constant(data.loc[:, cols].as_matrix(), tf.float32)
        weights = np.matrix(data["weight"].as_matrix()).T
        weights *= weights.shape[0]/np.sum(weights)
        weights = tf.constant(weights, tf.float32)
        labels = tf.constant(np.matrix(data["Standardized_Order"].as_matrix()).T, tf.float32)
        
        dataset = tf.data.Dataset.from_tensor_slices((inputs, weights, labels))
        dataset = dataset.shuffle(len(data))
        dataset = dataset.batch(self.config.batch_size)
        return dataset
