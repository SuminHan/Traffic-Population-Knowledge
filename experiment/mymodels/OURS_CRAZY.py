import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules import *


class MyOURS_CRAZY(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_CRAZY, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.ksize = extdata['ksize']*2+1
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']

        # self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)
        #80#12#20#(self.CH-11)*(self.CW-11)
        # self.num_cells_F = self.FH*self.FW

        if 'adj_mx' in extdata:
            self.adj_mat = extdata['adj_mx']
            print('using adj_mx in the prepdata')
        else:
            self.adj_mat = np.eye(self.num_nodes)
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D//2)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D//2),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        # self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        # self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZC', dtype=tf.float32)

        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D//2)])
        self.ZC_Conv = layers.Conv2D(D//2, self.ksize, padding='valid', activation=tf.nn.relu)

        self.X_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        

        # self.XC_trans_layer = [BipartiteAttention(self.K, self.d) for _ in range(self.L)]
        # self.XC_FC_layer = [layers.Dense(D, activation="relu") for _ in range(self.L)]

        
        
        # self.C_trans_layer = TransformAttention(self.K, self.d)
        # self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])
        

        
        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        
        # self.bns = [layers.BatchNormalization() for _ in range(self.L)]
        # self.bns2 = [layers.BatchNormalization() for _ in range(self.L-1)]



    def call(self, kwargs):
        X, ZC, TE = kwargs['X'], kwargs['ZC'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]


        # ZC = tf.expand_dims(ZC, -1)
        # ZC = self.FC_ZC_in(ZC)
        # ZC = tf.reshape(ZC, (-1, self.CH, self.CW, self.D))
        # ZC = self.FC_ZC_Conv(ZC)
        # ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[1]*ZC.shape[2], self.D))
        # STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # ZC = ZC + STEZC

        print(ZC.shape)
        ZC = tf.reshape(ZC, (-1, self.ksize, self.ksize, 2))
        ZC = self.ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, self.num_nodes, self.D//2))
        print('ZC.shape', ZC.shape)

    

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X) + STEX_P


        X = tf.concat((X, ZC), -1)

        X = self.X_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        print(Y.shape)

        return Y