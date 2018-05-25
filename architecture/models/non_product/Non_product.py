import tensorflow as tf
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, "./architecture/models/")

from Model import Model
import config

class Non_product(Model):      
    def add_prediction_op(self):
        '''
        Defines the computational graph then returns the prediction tensor
        '''
        with tf.variable_scope('predictions', reuse=tf.AUTO_REUSE):
            x = self.input_data[0] # data (self.input_data is a (cell, weight, label) tuple)
            arr = [0]*(self.config.n_layers+1)
            arr[0] = x#tf.contrib.layers.layer_norm(x)
            for i in range(1, self.config.n_layers+1):
                arr[i] = tf.contrib.layers.fully_connected(arr[i-1], self.config.hidden_size)
                arr[i] = tf.layers.dropout(arr[i], rate=self.config.dropout, training=self.is_training)
            output = tf.contrib.layers.fully_connected(arr[self.config.n_layers], 1, activation_fn=tf.nn.sigmoid)
        return output


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
            #loss = tf.losses.mean_squared_error(self.input_data[2], 
            #                                    pred, 
            #                                    weights=self.input_data[1])
            # L2 regularization
            loss = ((1-self.corr(pred)))**2*tf.reduce_sum(self.input_data[1])
            loss += self.config.lambd / 2 * self.weight_l2()
        return loss 

    def corr(self, pred):
        vx = tf.squeeze(pred) - tf.reduce_mean(pred,)
        vy = tf.squeeze(self.input_data[2]) - tf.reduce_mean(self.input_data[2])
        corr_num = tf.reduce_sum(tf.multiply(vx, vy))
        corr_den = tf.sqrt(tf.multiply(tf.reduce_sum(tf.square(vx)), tf.reduce_sum(tf.square(vy))))
        return corr_num/corr_den 
    
    
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
