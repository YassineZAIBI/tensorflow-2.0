# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:05:28 2019

@author: YZ
"""

import tensorflow as tf


class Rnn_Estimator(tf.keras.Model):
    def __init__(self,units1):
        super(Rnn_Estimator).__init()
        self.initializer1 = tf.keras.initializers.lecun_normal(seed = 73, epoch)
        self.units11 = units1 
        self.lstm_layer1 = tf.keras.layers.LSTM(units = self.units1 , return_sequences=True, kernel_initializer = self.initializer1)
        self.lstm_dropout1  = tf.keras.layers.Dropout(rate = 0.2)
        self.lstm_dense1 = tf.keras.layers.Dense(units = 1, activation = "sigmoid" )
        self.loss_object = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam()
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.MSE(name='train_accuracy')
        self.test_loss = tf.keras.losses.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.MSE(name='test_accuracy')
        self.epoch = epoch 
        
        
        def call(self):
            x = inputs
            x = tf.keras.Input(shape = [batch , timestemp, features])
            x = self.lstm_layer1(x)
            x = self.lstm_dropout1(x)
            return self.lstm_dense1(x)
        
        def train_step(self,x_train, y_train):
            with tf.GradientTape() as tape:
                predictions = self.call(x_train)
                loss = self.loss_object(y_train, predictions)
            gradients = tape.gradient(loss, self.trainable_variables)
            self.optimizer.apply_gradients(list(zip(gradients, self.trainable_variables)))
            self.train_loss(loss)
            self.train_accuracy(y_train, predictions)
            
        def test_step(self, x_test , y_test): 
            predictions = self.call(x_test)
            t_loss = self.loss_object(y_test, predictions)
            
            self.test_loss(t_loss)
            self.test_accuracy(y_test, predictions)
            self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            self.test_log_dir = '/home/tensorboard/Rnn_classifier/' + self.current_time
            self.test_summary_writer = tf.summary.create_file_writer(self.test_log_dir)
            
        def fit(self, X , y): 
            for epoch in range(self.epoch):
                x_train = X[0]
                x_test = X[1]
                y_train = y[0]
                y_test = y[1]
                
                self.train_step(x_train,y_train)
                self.test_step(x_test, y_test)
                with self.train_summary_writer.as_default():
                    self.summary_loss = tf.summary.scalar('loss',self.test_loss.result(),step = epoch)
                    self.summary_accuarcy = tf.summary.scalar('accuracy', self.test_accuracy.result(), step = epoch)
                    
if __name__ == "__main__":  
    my_model = Rnn_Estimator()
    get_result = my_model.fit(X_,Y_ )                   
            
            