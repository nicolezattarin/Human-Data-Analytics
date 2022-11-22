import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import skimage.transform
import tensorflow as tf
from cnn_utils import *

np.random.seed(4)


class SequentialCNN():

    def __init__(self):
        self.network_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(4, (3, 3), activation='relu', padding='same', strides=(1,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=(1,1)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(6)# leave out activation function for loss function
        ])


    def compute_cost(self, y_pred, y_true):
        """
        Computes the cost
        
        Arguments:
        outp -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as outp
        
        Returns:
        cost - Tensor of the cost function
        """
        losses = tf.keras.losses.categorical_crossentropy(y_pred, y_true, from_logits=True)
        cost = tf.reduce_mean(losses)
        
        return cost

    
    def train(self, X_train, Y_train, X_test, Y_test, learning_rate = 0.005,
            num_epochs = 300, minibatch_size = 64, print_cost = True):
        """
        Train a ConvNet in TensorFlow
        
        Arguments:
        network_model -- the keras Sequential model to be trained
        X_train -- training set, of shape (None, 64, 64, 3)
        Y_train -- training set, of shape (None, n_y = 6)
        X_test -- test set, of shape (None, 64, 64, 3)
        Y_test -- test set, of shape (None, n_y = 6)
        learning_rate -- learning rate of the optimization
        num_epochs -- number of epochs of the optimization loop
        minibatch_size -- size of a minibatch
        print_cost -- True to print the cost every 100 epochs
        
        Returns:
        train_accuracy -- real number, accuracy on the train set (X_train)
        validation_accuracy -- real number, testing accuracy on the validation set (X_val)
        """
        
        tf.random.set_seed(1)                             # to keep results consistent (tensorflow seed)
        seed = 3                                          # to keep results consistent (numpy seed)
        (m, n_H0, n_W0, n_C0) = X_train.shape             # set shapes on training set


        n_y = Y_train.shape[1]                            
        costs = []                                        # To keep track of the cost
        
        # Backpropagation: AdamOptimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Do the training loop
        for epoch in range(num_epochs):

            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed += 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                with tf.GradientTape() as tape:
                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch

                    # Forward propagation
                    outp = self.network_model(minibatch_X, training=True)
                    # Cost function
                    cost = self.compute_cost(outp, minibatch_Y)
                    
                # Compute the gradient
                gradients = tape.gradient(cost, self.network_model.trainable_variables)

                # Apply the optimizer
                optimizer.apply_gradients(zip(gradients, self.network_model.trainable_variables))
                minibatch_cost += cost / num_minibatches

            # Print the cost every epoch
            if print_cost == True and epoch % 5 == 0:
                print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
            if print_cost == True and epoch % 1 == 0:
                costs.append(minibatch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # Calculate accuracy on the validation set
        train_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.network_model(X_train, training=False), 1), tf.argmax(Y_train, 1)), "float"))
        test_accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.network_model(X_test, training=False), 1), tf.argmax(Y_test, 1)), "float"))

        print("Train Accuracy:", train_accuracy)
        print("Test Accuracy:", test_accuracy)


        return train_accuracy, test_accuracy, self.network_model


if __name__ == '__main__':
    # Load the dataset
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    # Normalize image vectors
    X_train = X_train_orig/255.
    X_test = X_test_orig/255.

    # Reshape
    Y_train = convert_to_one_hot(Y_train_orig, 6).T
    Y_test = convert_to_one_hot(Y_test_orig, 6).T

    # Create the model
    X_train_cast = X_train.astype(dtype=np.float32)
    X_test_cast = X_test.astype(dtype=np.float32)

    model = SequentialCNN()

    # Train the model
    _, _, network_model_trained = model.train(X_train_cast, Y_train, X_test_cast, Y_test, num_epochs=50)

