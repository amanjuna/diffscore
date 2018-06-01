
import os

import tensorflow as tf
import scipy.stats as st
import numpy as np

class Model():
    def __init__(self, config, verbose=True):
        self.config = config
        self.verbose = verbose

        
    def build(self):
        '''
        Adds important graph functions as well as the global 
        step which is important for logging training progress
        '''
        with tf.variable_scope('overall', reuse=tf.AUTO_REUSE):
            self.global_step = tf.Variable(0, trainable=False)
            self.is_training = tf.placeholder(tf.bool)
            with tf.variable_scope('prediction'):
                self.pred = self.add_prediction_op()
            with tf.variable_scope('loss'):
                self.loss = self.add_loss_op(self.pred)
            with tf.variable_scope('optimizer'):
                self.train_op = self.add_training_op(self.loss)    

        
    def input_op(self, data):
        '''
        Returns a dataset object which is appropriate for the
        prediction, loss and training ops
        '''
        raise NotImplementedError("Do not instantiate a base Model object - implement this method in a subclass")
        
    
    def add_prediction_op(self):
        '''
        Adds the workhouse prediction structure to the graph.
        You need to set self.input_data to point to the correct
        input data before calling this function.
        '''
        raise NotImplementedError("Do not instantiate a base Model object - implement this method in a subclass")

    
    def add_loss_op(self):
        '''
        Adds the loss function to the graph
        '''
        raise NotImplementedError("Do not inbtantiate a base Model object - implement this method in a subclass")

    
    def add_train_op(self):
        '''
        Adds the training operations (optimizer and gradient
        clipping operations for instance) to the graph
        '''
        raise NotImplementedError("Do not instantiate a base Model object - implement this method in a subclass")

      
    def fit(self, train_data, val_data):
        '''
        Runs training/validation loop

        @train_data and @val_data are pandas dataframes
        '''
        train_iter_init, val_iter_init, self.input_data = self._prepare_train_val(train_data, val_data)
        self.build()
        saver = tf.train.Saver(max_to_keep=1)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            train_writer = tf.summary.FileWriter(os.path.join(self.config.tensorboard_dir, 'train/'))
            val_writer = tf.summary.FileWriter(os.path.join(self.config.tensorboard_dir, 'val/'))

            training_handle = sess.run(train_iter_init.string_handle())
            val_handle = sess.run(val_iter_init.string_handle())
            
            bar = tf.keras.utils.Progbar(target=self.config.n_epochs,\
                                         stateful_metrics=["Train_Spearman", "Train_Pearson",
                                                           "Val_Spearman", "Val_Pearson"])
            best_spear = float('-inf')
            for epoch in range(self.config.n_epochs):
                if self.verbose:
                    print("Epoch {}\n".format(epoch+1))
                sess.run(train_iter_init.initializer)        
                sess.run(val_iter_init.initializer)
                train_metrics = self.run_epoch(sess, training_handle, train_writer, "train")
                val_metrics = self.run_epoch(sess, val_handle, val_writer, "val")
                train_metrics = [("Train_" + x[0], x[1]) for x in train_metrics]
                val_metrics = [("Val_" + x[0], x[1]) for x in val_metrics]
                bar.add(1, values=train_metrics + val_metrics)
                val_spear = val_metrics[1][1] # Select based on spearman correlation
                if val_spear > best_spear:
                    best_spear = val_spear
                    if self.verbose:
                        print("\nNew best correlation! Saving model in ./results/weights/weights.ckpt")
                    saver.save(sess, self.config.model_output)
                if self.verbose: print()


    def predict(self, data):
        '''
        @data is a pandas dataframe
        '''
        inputs = self.input_op(data)
        iterator = inputs.make_initializable_iterator()
        self.input_data = iterator.get_next()
        self.build()
        self.handle = tf.placeholder(tf.string, shape=())
        with tf.Session() as sess:
            # sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(max_to_keep=1)
            saver.restore(sess, self.config.model_output)
            test_handle = sess.run(iterator.string_handle())
            sess.run(iterator.initializer)
            preds = []
            while True:
                try:
                    _pred = sess.run(self.pred, feed_dict={self.handle: test_handle,
                                                           self.is_training: False})
                    preds.append(_pred)
                except tf.errors.OutOfRangeError:
                    break
        return np.concatenate(preds).flatten()

    
    def _prepare_train_val(self, train_data, dev_data):
        train_input, val_input = self.input_op(train_data), self.input_op(dev_data)
        train_iter_init = train_input.make_initializable_iterator()
        val_iter_init = val_input.make_initializable_iterator()
        self.handle = tf.placeholder(tf.string, shape=())
        iterator = tf.data.Iterator.from_string_handle(self.handle,
                                                       train_input.output_types,
                                                       train_input.output_shapes)
        return train_iter_init, val_iter_init, iterator.get_next()

    
    def run_epoch(self, sess, handle, writer, epoch_type):
        metrics = []
        while True:
            try:
                if epoch_type == "train":
                    _, loss, pred, inputs, global_step = sess.run((self.train_op, self.loss,
                                                                  self.pred, self.input_data,
                                                                  self.global_step),
                                                                  feed_dict={self.handle: handle,
                                                                             self.is_training: True})
                else:
                    loss, pred, inputs, global_step = sess.run((self.loss,
                                                               self.pred, self.input_data,
                                                               self.global_step),
                                                               feed_dict={self.handle: handle,
                                                                          self.is_training: False})
                labels = inputs[2]
                metrics.append((loss, pred, labels))
            except tf.errors.OutOfRangeError:
                break
        summary, important_metrics = self.make_summary(metrics, epoch_type)
        writer.add_summary(summary, global_step)
        return important_metrics

    
    def make_summary(self, metrics, epoch_type):
        loss = np.mean([x[0] for x in metrics])
        pred = np.concatenate([x[1] for x in metrics])
        labels = np.concatenate([x[2] for x in metrics])
        print(np.concatenate([pred, labels], axis=1))
        squared_error = ((pred - labels)**2).mean()
        spearman_corr = st.spearmanr(pred, labels)[0]
        pearson_corr = st.pearsonr(pred, labels)[0][0]
        summary = tf.Summary(value=[tf.Summary.Value(tag='Loss', simple_value=loss),
                                    tf.Summary.Value(tag="Squared Error", simple_value=squared_error),
                                    tf.Summary.Value(tag="Spearman Correlation", simple_value=spearman_corr),
                                    tf.Summary.Value(tag="Pearson Correlation", simple_value=pearson_corr)])
        important_metrics = (("Loss", loss), ("Spearman", spearman_corr), ("Pearson", pearson_corr))

        return summary, important_metrics
    
        
    def weight_l2(self):
        weights = [var for var in tf.trainable_variables() if 'weights' in str(var)]
        l2 = np.sum([tf.nn.l2_loss(var) for var in weights])
        return l2
