
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules.gman_submodules import *
from submodules.dcgru_cell_tf2 import *

###########################################################################

class LastRepeat(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(LastRepeat, self).__init__()
        pass
        
    def build(self, input_shape):
        pass

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        return X[:, -1:, :]



class MyGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGRU, self).__init__()
        self.D = args.D
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FCs_1 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FCs_2 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.Q*self.num_nodes)])
        self.gru = layers.GRU(self.D)


    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        X = self.FCs_1(X)
        X = self.gru(X)
        Y = self.FCs_2(X)
        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))
        return Y


class MyLSTM(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyLSTM, self).__init__()
        self.D = args.D
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FCs_1 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FCs_2 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.Q*self.num_nodes)])
        self.lstm = layers.LSTM(self.D)


    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        X = self.FCs_1(X)
        X = self.lstm(X)
        Y = self.FCs_2(X)
        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))
        return Y

##############




########################
