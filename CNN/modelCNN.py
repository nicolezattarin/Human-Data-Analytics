import math
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
np.random.seed(4)

import tensorflow_datasets as tfds

class modelCNN():

    def __init__(self):
        self.trained = False

    def ConvPool(self, X, conv_feature_maps=4, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu',  
                pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same'):
        """
        Help function for convolutional + max pooling layers
        
        Arguments:
        X -- imput tensor

        Returns:
        model -- a Model() instance in TensorFlow
        """
        
        # CONV -> Batch Normalization -> ReLU Block applied to X 
        X = tf.keras.layers.Conv2D(filters=conv_feature_maps, kernel_size=conv_kernel, strides=conv_strides, padding=conv_padding)(X)
        X = tf.keras.layers.BatchNormalization(axis=3)(X)
        X = tf.keras.layers.Activation(activation)(X)
        
        # MAXPOOL 
        X = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)(X)
        return X


    def SignModel(self, input_shape):
        """
        Implementation of the SignModel
        
        Arguments:
        input_shape -- shape of the images of the dataset

        Returns:
        model -- a Model() instance in TensorFlow
        """
        
        ### START CODE HERE 
        # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
        X_input = tf.keras.layers.Input(input_shape)

        # FIRST CONV + MAXPOOL BLOCK
        X = self.ConvPool(X_input, conv_feature_maps=16, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu',
                        pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same')
        
        # SECOND CONV + MAXPOOL BLOCK
        X = self.ConvPool(X, conv_feature_maps=32, conv_kernel=(3, 3), conv_strides=(1, 1), conv_padding='same', activation='relu',
                        pool_size=(2, 2), pool_strides=(2, 2), pool_padding='same')

        # FLATTEN THE TENSOR 
        X = tf.keras.layers.Flatten()(X)
        
        # FULLYCONNECTED (DENSE) LAYER WITH RELU ACTIVATION AND 16 OUTPUT NEURONS
        X = tf.keras.layers.Dense(16, activation='relu')(X)
        
        # DROPOUT LAYER (DISCARD PROBABILITY 0.4)
        X = tf.keras.layers.Dropout(0.4)(X)
        
        # DENSE LAYER WITHOUT ACTIVATION AND 3 OUTPUT NEURONS
        X = tf.keras.layers.Dense(3, activation=None)(X)
                                        
        # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
        model = tf.keras.Model(inputs = X_input, outputs = X, name='SignModel')
        
        return model

    def train(self, train_data, val_data, input_shape, learning_rate, epochs):

        self.model = self.SignModel(input_shape)
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        loss_funct = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.model.compile(optimizer=adam_optimizer, loss=loss_funct, metrics=['accuracy'])

        history = self.model.fit(train_data, epochs=10, validation_data=val_data)

        self.trained = True
        return history