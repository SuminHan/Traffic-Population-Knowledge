import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules import *
from submodules.deconvolutional_recurrent import DeConvLSTM2D


class MyOURS_GMANCF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_GMANCF, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        self.num_cells_C = self.CH*self.CW
        self.num_cells_F = self.FH*self.FW
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
                                        initializer='glorot_uniform',
                                        name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])


        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = keras.Sequential([
                                layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', return_sequences=True),
                                layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', return_sequences=True),
                                layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', return_sequences=True),
                            ])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = keras.Sequential([
                                layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', return_sequences=True),
                                layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', return_sequences=True),
                                layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', return_sequences=True),
                            ])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.concat_fusion = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])


    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX_P
        for i in range(self.L):
            X = self.GSTA_enc[i](X, STEX_P)


        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = tf.concat((X, ZF, ZC), -1)
        X = self.concat_fusion(X)
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y



class MyOURS_GMANMulti(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_GMANMulti, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        #[(W−K+2P)/S]+1
        def cnn_dim(W):
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            return W
        self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)
        #80#12#20#(self.CH-11)*(self.CW-11)
        # self.num_cells_F = self.FH*self.FW

        if 'adj_mx' in extdata:
            self.adj_mat = extdata['adj_mx']
            print('using adj_mx in the prepdata')
        else:
            self.adj_mat = np.eye(self.num_nodes)
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])

        
        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                            ])
        self.RSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.RSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_ZC_ConvTrans = keras.Sequential([
                                layers.Conv2DTranspose(D, 3, strides=(2, 2), padding='valid'),
                                layers.LeakyReLU(),
                                layers.Conv2DTranspose(D, 3, strides=(2, 2), padding='valid'),
                                layers.LeakyReLU(),
                                layers.Conv2DTranspose(D, 3, strides=(2, 2), padding='valid'),
                            ])

        
        self.XC_trans_layer = [BipartiteAttention(self.K, self.d) for _ in range(self.L)]
        
        
        self.XPQ_trans_layer = TransformAttention(self.K, self.d)
        self.ZPQ_trans_layer = TransformAttention(self.K, self.d)


        
        # self.ZC_trans_layer = [BipartiteAttention(self.K, self.d) for _ in range(self.L-1)]

        # self.XC_Bipartite = BipartiteAttention(self.K, self.d)
        # self.FC_ZC_trans = layers.Dense(self.num_nodes)

        # self.FC_ZF_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        # self.FC_ZF_Conv = keras.Sequential([
        #                         # layers.Reshape((-1, self.FH, self.FW, self.D)),
        #                         layers.Conv2D(D, 5, strides=(1, 1), padding='valid'),
        #                         layers.Conv2D(D, 5, strides=(1, 1), padding='valid'),
        #                         layers.Conv2D(D, 5, strides=(1, 1), padding='valid'),
        #                         # layers.Reshape((-1, self.P, self.FH, self.FW, self.D))
        #                     ])
        # self.FC_ZF_trans = layers.Dense(self.num_nodes)

        # self.concat_fusion = keras.Sequential([
        #                     layers.Dense(D, activation="relu")])
        self.bns = [layers.BatchNormalization() for _ in range(self.L)]
        self.bns2 = [layers.BatchNormalization() for _ in range(self.L-1)]



    def call(self, kwargs):
        X, Z, _, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]

        STEZC = self.STEZC_layer(self.SEZ, TE)
        STEZC_P, STEZC_Q = STEZC[:, :self.P, :], STEZC[:, self.P:, :]


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = tf.reshape(ZC, (-1, self.CH, self.CW, self.D))
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[1]*ZC.shape[2], self.D))
        
        # ZC = ZC + STEZC


    

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        #X = X + STEX_P
        for i in range(self.L):
            # X = X + self.GSTA_enc[i](X, ZC, ZC)
            ZC = self.RSTA_enc[i](ZC, STEZC)
            X = self.GSTA_enc[i](X, STEX_P)

            ZX = self.XC_trans_layer[i](X, STEZC_P, ZC)
            CX = self.ZC_trans_layer[i](ZC, STEX_P, X)

            ZC = self.bns2[i](ZC + CX)
            X = self.bns[i](X + ZX)


        
        # STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        # STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        # STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))



        # print(X.shape, ZC.shape)
        # atX = self.XC_Bipartite(X, ZC, ZC) 
        # print('atX:', atX.shape)
        # X = X + atX

        X = self.XPQ_trans_layer(X, STEX_P, STEX_Q)
        ZC = self.ZPQ_trans_layer(ZC, STEZC_P, STEZC_Q)
        
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)
        

        return Y




class MyOURS_GMAN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_GMAN, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        #[(W−K+2P)/S]+1
        def cnn_dim(W):
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            return W
        self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)
        #80#12#20#(self.CH-11)*(self.CW-11)
        # self.num_cells_F = self.FH*self.FW

        if 'adj_mx' in extdata:
            self.adj_mat = extdata['adj_mx']
            print('using adj_mx in the prepdata')
        else:
            self.adj_mat = np.eye(self.num_nodes)
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        # self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        # self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        # self.GSTA_enc = [BipartiteAttention(self.K, self.d) for _ in range(self.L)]
        self.XC_trans_layer = [BipartiteAttention(self.K, self.d) for _ in range(self.L)]
        self.XC_FC_layer = [layers.Dense(D, activation="relu") for _ in range(self.L)]

        
        self.RSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.RSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]

        
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])
        

        
        # self.ZC_trans_layer = [BipartiteAttention(self.K, self.d) for _ in range(self.L-1)]

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                # layers.Dropout(0.1),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                # layers.Dropout(0.1),
                                # layers.MaxPooling2D(pool_size=(2, 2), padding="valid"),
                                # layers.Conv2D(D, 3, strides=(1, 1), padding='valid'),
                            ])
        # self.XC_Bipartite = BipartiteAttention(self.K, self.d)
        # self.FC_ZC_trans = layers.Dense(self.num_nodes)

        # self.FC_ZF_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        # self.FC_ZF_Conv = keras.Sequential([ 
        #                         # layers.Reshape((-1, self.FH, self.FW, self.D)),
        #                         layers.Conv2D(D, 5, strides=(1, 1), padding='valid'),
        #                         layers.Conv2D(D, 5, strides=(1, 1), padding='valid'),
        #                         layers.Conv2D(D, 5, strides=(1, 1), padding='valid'),
        #                         # layers.Reshape((-1, self.P, self.FH, self.FW, self.D))
        #                     ])
        # self.FC_ZF_trans = layers.Dense(self.num_nodes)

        # self.concat_fusion = keras.Sequential([
        #                     layers.Dense(D, activation="relu")])
        self.bns = [layers.BatchNormalization() for _ in range(self.L)]
        # self.bns2 = [layers.BatchNormalization() for _ in range(self.L-1)]



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = tf.reshape(ZC, (-1, self.CH, self.CW, self.D))
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[1]*ZC.shape[2], self.D))
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # ZC = ZC + STEZC


    

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        #X = X + STEX_P
        for i in range(self.L):
            # X = X + self.GSTA_enc[i](X, ZC, ZC)
            ZC = self.RSTA_enc[i](ZC, STEZC)

            X = self.GSTA_enc[i](X, STEX_P)
            
            ZX = self.XC_trans_layer[i](STEX_P, STEZC, ZC)
            # if i < self.L-1:
            #     CX = self.ZC_trans_layer[i](ZC, STEX_P, X)
            #     ZC = self.bns2[i](ZC + CX)
            X = tf.concat((X, ZX), -1)
            X = self.XC_FC_layer[i](X)
            


        
        # STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        # STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        # STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))



        # print(X.shape, ZC.shape)
        # atX = self.XC_Bipartite(X, ZC, ZC) 
        # print('atX:', atX.shape)
        # X = X + atX

        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y






class MyOURS_empty(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_empty, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        self.num_cells_C = self.CH*self.CW
        self.num_cells_F = self.FH*self.FW
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        # self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        # self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZC', dtype=tf.float32)
        # self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        # self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        # self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        # self.FC_ZF_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        # self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        # STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        # STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        # STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1)
        # ZC = self.FC_ZC_in(ZC)
        # ZC = ZC + STEZC
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        # ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))
        # ZC = self.FC_ZC_trans(ZC)
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        # ZF = self.FC_ZF_in(ZF)
        # ZF = ZF + STEZF
        # ZF = self.FC_ZF_ConvLSTM(ZF)
        # ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))
        # ZF = self.FC_ZF_trans(ZF)
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X #+ ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyOURS_v1(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_v1, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        self.num_cells_C = self.CH*self.CW
        self.num_cells_F = self.FH*self.FW
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
                                        initializer='glorot_uniform',
                                        name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=True, return_sequences=True, go_backwards=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=True)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=True, return_sequences=True, go_backwards=True)
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=True)


 
    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


##########################################
# MyOURS2: MMD, Global Attention
##########################################


class MyOURS_v2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_v2, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        self.num_cells_C = self.CH*self.CW
        self.num_cells_F = self.FH*self.FW
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
                                        initializer='glorot_uniform',
                                        name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        EX = X
        EF = ZF
        EC = ZC
        X = X + ZF + ZC

        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return [Y, EX, EF, EC]
