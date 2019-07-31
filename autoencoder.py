# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 13:44:20 2019

@author: YZ
"""

import tensorflow as tf 


class Encoder(tf.keras.layers.Layer):
      def __init__(self, intermediate_dim):
        super(Encoder, self).__init__()
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        
      def call(self, input_features):
        activation = self.hidden_layer(input_features)
        return self.output_layer(activation)



class Decoder(tf.keras.layers.Layer):
      def __init__(self, intermediate_dim, original_dim, learning_rate , momentum):
        super(Decoder, self).__init__()
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.hidden_layer = tf.keras.layers.Dense(units=intermediate_dim, activation=tf.nn.relu)
        self.output_layer = tf.keras.layers.Dense(units=original_dim, activation=tf.nn.relu)
        self.opt = tf.optimizers.SGD(learning_rate=self.learning_rate, momentum=self.momentum)

  
      def call(self, code):
        activation = self.hidden_layer(code)
        return self.output_layer(activation)


class Autoencoder(tf.keras.Model):
      def __init__(self, intermediate_dim, original_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(intermediate_dim=intermediate_dim)
        self.decoder = Decoder(intermediate_dim=intermediate_dim, original_dim=original_dim)

  
      def call(self, input_features):
        code = self.encoder(input_features)
        reconstructed = self.decoder(code)
        return reconstructed

      def loss(self,model, original):
        self.reconstruction_error = tf.reduce_mean(tf.square(tf.subtract(model(original), original)))
        return self.reconstruction_error


      def train(self, model, original):
          with tf.GradientTape() as tape:
              gradients = tape.gradient(self.loss(model, original), model.trainable_variables)
              gradient_variables = zip(gradients, model.trainable_variables)
              self.opt.apply_gradients(gradient_variables)
