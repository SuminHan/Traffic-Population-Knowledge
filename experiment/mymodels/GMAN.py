import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules import *



class MyGMAN0(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN0, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        # self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X) 
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)
        return Y



def custom_mae_loss_inside(label, pred):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss

class MyGMAN0IM(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN0IM, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        # self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

                        
        self.FC_XC_in_IM = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XM = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])
        
    def call(self, kwargs):
        X0, TE = kwargs['X'], kwargs['TE']
        mask = tf.not_equal(X0, 0)
        mask = tf.cast(mask, tf.float32)
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        
        ZX = tf.expand_dims(X0, -1)
        ZX = self.FC_XC_in_IM(ZX)
        ZX = ZX + STEX
        X_gen = self.FC_XM(ZX)[..., 0]
        X = mask*X0 + (1-mask)*X_gen

        custom_loss = 0.15*custom_mae_loss_inside(X0, X_gen)
        self.add_loss(custom_loss)

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X) 
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)
        return Y


class MyGM0ZCFC(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZCFC, self).__init__()
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
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
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


class MyGM0ZCFW(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZCFW, self).__init__()
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
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.weight_fusion = WeightFusion(D)


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

        X = tf.stack((X, ZF, ZC), -1)
        X = self.weight_fusion(X)
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y


class MyGM0ZCFB(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZCFB, self).__init__()
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
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



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
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        # ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y



class MyGM0ZCF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZCF, self).__init__()
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

        X = X + ZF + ZC
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y

class MyGM0ZC(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZC, self).__init__()
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
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)




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



        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        X = X +  ZC
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y

        
class MyGM0ZF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZF, self).__init__()
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

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



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

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))

        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y