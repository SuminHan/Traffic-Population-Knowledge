import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from submodules import *



class MyOURS_AE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AE, self).__init__()
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
        # self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)
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
                                        
        # self.FC_X_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        # self.GSTA_enc = [BipartiteAttention(self.K, self.d) for _ in range(self.L)]
        self.XC_trans_layer = [BipartiteAttention(self.K, self.d) for _ in range(self.L)]
        # self.XC_FC_layer = [layers.Dense(D, activation="relu") for _ in range(self.L)]

        
        self.RSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        # self.RSTA_enc = [TemporalAttention(self.K, self.d) for _ in range(self.L)]
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
        _, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]


        # ZC = tf.expand_dims(ZC, -1)
        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = tf.reshape(ZC, (-1, self.CH, self.CW, self.D))
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[1]*ZC.shape[2], self.D))
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # ZC = ZC + STEZC


    
        ZC = self.RSTA_enc[0](ZC, STEZC)           
        X = self.XC_trans_layer[0](STEX_P, STEZC, ZC)

        # X = tf.expand_dims(X, -1)
        # X = self.FC_X_in(X)
        #X = X + STEX_P
        for i in range(1, self.L):
            # X = X + self.GSTA_enc[i](X, ZC, ZC)
            ZC = self.RSTA_enc[i](ZC, STEZC)
            ZX = self.XC_trans_layer[i](STEX_P, STEZC, ZC)
            X = X + ZX
            

        for i in range(self.L):
            X = self.GSTA_enc[i](X, STEX_P)


        
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




class MyOURS_AEConv(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AEConv, self).__init__()
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

        self.num_cells_C = (self.CH) * (self.CW)

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

        self.conv1 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', return_sequences=True)
        self.conv2 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', go_backwards=True, return_sequences=True)
        self.ZC_FC_layer = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.XC_trans_layer = BipartiteAttention(self.K, self.d)
        self.FC_X_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)

        
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])
        
        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])


        ###
        
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])


        




    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]


        # ZC = tf.expand_dims(ZC, -1)
        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, self.CH * self.CW, self.D))
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        ZC = ZC + STEZC
        ZC = tf.reshape(ZC, (-1, self.P, self.CH, self.CW, self.D))

        ZC1 = self.conv1(ZC)
        ZC2 = self.conv2(ZC)[::-1]
        ZC = tf.concat((ZC1, ZC2), -1)
        ZC = self.ZC_FC_layer(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        
        X = self.XC_trans_layer(STEX_P, STEZC, ZC)
        X = self.FC_X_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y1 = tf.transpose(Y, (0, 2, 1))


        
        X = tf.expand_dims(X0, -1)
        X = self.FC_XC_in(X)
        X = X + STEX_P
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y2 = tf.transpose(Y, (0, 2, 1))

        

        return Y1+Y2



class MyOURS_AE0(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AE0, self).__init__()
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

        self.num_cells_C = (self.CH) * (self.CW)

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

        self.conv1 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', return_sequences=True)
        self.conv2 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', go_backwards=True, return_sequences=True)
        self.ZC_FC_layer = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.XC_trans_layer = BipartiteAttention(self.K, self.d)
        self.FC_X_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)

        
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])
        
        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])


    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]


        # ZC = tf.expand_dims(ZC, -1)
        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, self.CH * self.CW, self.D))
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        ZC = ZC + STEZC
        ZC = tf.reshape(ZC, (-1, self.P, self.CH, self.CW, self.D))

        ZC1 = self.conv1(ZC)
        ZC2 = self.conv2(ZC)[::-1]
        ZC = tf.concat((ZC1, ZC2), -1)
        ZC = self.ZC_FC_layer(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        
        X = self.XC_trans_layer(STEX_P, STEZC, ZC)
        X = self.FC_X_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

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

    
class MyOURS_AEGen3(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AEGen3, self).__init__()
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

        # self.num_cells_C = (self.CH) * (self.CW)
        def cnn_dim(W):
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            return W
        self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)

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

        # self.conv1 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', return_sequences=True)
        # self.conv2 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', go_backwards=True, return_sequences=True)
        
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                # layers.LeakyReLU(),
                                # layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                            ], name='FC_ZC_Conv')
        # self.ZC_FC_layer = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.XC_trans_layer = BipartiteAttention(self.K, self.d)

        
        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.FC_Xgen = keras.Sequential([
                            layers.Dense(1)])

        ###
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)], name='FC_XC_in')
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        # self.FC_XC_out = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(self.Q)], name='FC_XC_out')

        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        # self.FC_XC_out2 = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(self.Q)], name='FC_XC_out2')

        self.RSTA_enc = GSTAttBlock(self.K, self.d)
        # self.MASK_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)], name='MASK_in')
        # self.MASK_enc = GSTAttBlock(self.K, self.d)
        self.FC_final = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)], name='FC_final')
        
        # self.bn = layers.BatchNormalization()
        self.gated_fusion = GatedFusion(self.D)


    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        mask = tf.not_equal(X0, 0)
        mask = tf.cast(mask, tf.float32)

        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)

        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]
        # mmask = tf.expand_dims(mask, -1)
        # mmask = self.MASK_in(mmask)
        # STEX_P = self.MASK_enc(mmask, STEX_P)

        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))

        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]

        # ZC = self.RSTA_enc(ZC, STEZC)
        ZX = self.XC_trans_layer(STEX_P, STEZC, ZC)

        X_gen = self.FC_Xgen(ZX)[..., 0]
        XN = mask*X0 + (1-mask)*X_gen



        # import tensorflow_model_remediation.min_diff as tfmrmd
        # min_diff_loss = tfmrmd.losses.MMDLoss()
        # min_diff_weight = 1.0

        # mX_gen = ((1-mask)*X_gen)
        # custom_loss = 0
        # for i in range(self.num_nodes):
        #     custom_loss += min_diff_loss(X0[:, :, i], (mX_gen)[:, :, i], min_diff_weight)
        # self.add_loss(custom_loss/6*0.2)

        custom_loss = 0.7*custom_mae_loss_inside(X0, X_gen)
        self.add_loss(custom_loss)


        X = tf.expand_dims(X0, -1)
        X = self.FC_XC_in(X)
        EX = X

        X = X + STEX_P
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        YY1 = X
        # Y1 = self.FC_XC_out(X)


        EZX = ZX
        ZX = ZX + STEX_P
        ZX = self.FC_XC_DCGRU2(ZX)
        ZX = tf.reshape(ZX, (-1, self.num_nodes, self.D))
        YY2 = ZX
        # Y2 = self.FC_XC_out2(ZX)

        # import tensorflow_model_remediation.min_diff as tfmrmd
        # min_diff_loss = tfmrmd.losses.MMDLoss()
        # min_diff_weight = 1.0

        # print(Y1.shape, Y2.shape)
        # custom_loss = min_diff_loss(Y1[..., 0], Y2[..., 0], min_diff_weight)
        # self.add_loss(custom_loss*0.1)

        Y = self.FC_final(self.gated_fusion(YY1, YY2))
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyOURS_AEGen3GMAN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AEGen3GMAN, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        # self.SE = extdata['SE']
        self.CH = extdata['CH']
        self.CW = extdata['CW']
        def cnn_dim(W):
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            return W
        self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_XC_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])


        ##################################
        
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                            ])
        self.RSTA_enc = GSTAttBlock(self.K, self.d)
        self.XC_trans_layer = BipartiteAttention(self.K, self.d)

        self.GSTAC_enc2 = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer2 = TransformAttention(self.K, self.d)
        self.GSTAC_dec2 = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        # self.FC_XC_out2 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        self.FC_XM = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])
        self.gated_fusion = GatedFusion(self.D)
        self.FC_final = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)], name='FC_final')

        
    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        mask = tf.not_equal(X0, 0)
        mask = tf.cast(mask, tf.float32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STE[:, :self.P, :], STE[:, self.P:, :]

        X = tf.expand_dims(X0, -1)
        X = self.FC_XC_in(X) 
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX_P)
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEX_Q)
        YY1 = X
        # X = self.FC_XC_out(X)
        # Y1 = tf.squeeze(X, -1)

        ###############################
        STEZ = self.STEZC_layer(self.SEZC, TE)
        STEZ_P, STEZ_Q = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))

        ZC = self.RSTA_enc(ZC, STEZ_P)
        
        ZX = self.XC_trans_layer(STEX_P, STEZ_P, ZC)

        X_gen = self.FC_XM(ZX)[..., 0]
        X = mask*X0 + (1-mask)*X_gen
        custom_loss = 0*custom_mae_loss_inside(X0, X_gen)
        self.add_loss(custom_loss)

        for i in range(self.L):
            ZX = self.GSTAC_enc2[i](ZX, STEX_P)
        ZX = self.C_trans_layer2(ZX, STEX_P, STEX_Q)
        for i in range(self.L):
            ZX = self.GSTAC_dec2[i](ZX, STEX_Q)
        YY2 = ZX
        # ZX = self.FC_XC_out2(ZX)
        # Y2 = tf.squeeze(ZX, -1)

        # Y = Y1 + Y2

        
        Y = self.FC_final(self.gated_fusion(YY1, YY2))
        Y = tf.squeeze(Y, -1)


        return Y




class MyOURS_AEGen2trust(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AEGen2trust, self).__init__()
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

        # self.num_cells_C = (self.CH) * (self.CW)
        def cnn_dim(W):
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            return W
        self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)

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

        # self.conv1 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', return_sequences=True)
        # self.conv2 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', go_backwards=True, return_sequences=True)
        
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                # layers.LeakyReLU(),
                                # layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                            ])
        # self.ZC_FC_layer = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.XC_trans_layer = BipartiteAttention(self.K, self.d)

        
        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.FC_XM = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])

        ###
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        # self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_XC_out2 = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.RSTA_enc = GSTAttBlock(self.K, self.d)
        self.MASK_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.MASK_enc = GSTAttBlock(self.K, self.d)
        self.MASK_dec = GSTAttBlock(self.K, self.d)
        
        # self.bn = layers.BatchNormalization()
        # self.gated_fusion = GatedFusion(self.D)


    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        mask = tf.not_equal(X0, 0)
        mask = tf.cast(mask, tf.float32)

        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)



        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]

        mmask = tf.expand_dims(mask, -1)
        mmask = self.MASK_in(mmask)
        STEX_P = self.MASK_enc(mmask, STEX_P)

        STEX_Q = self.MASK_enc(mmask, STEX_Q)

        # ZC = tf.expand_dims(ZC, -1)
        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        # ZC = self.FC_ZC_in(ZC)
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))

        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]

        
        print(ZC.shape, STEZC.shape)

        ZC = self.RSTA_enc(ZC, STEZC)
        
        ZX = self.XC_trans_layer(STEX_P, STEZC, ZC)

        X_gen = self.FC_XM(ZX)[..., 0]
        X = mask*X0 + (1-mask)*X_gen

        custom_loss = 0.15*custom_mae_loss_inside(X0, X_gen)
        self.add_loss(custom_loss)


        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        X = X + STEX_P
        # X = self.gated_fusion(X, ZX)
        # X = self.bn(X)
        EX = X
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y1 = self.FC_XC_out(X)
        Y1 = tf.transpose(Y1, (0, 2, 1))

        
        ZX = ZX + STEX_P
        EZX = ZX 
        ZX = self.FC_XC_DCGRU(ZX)
        ZX = tf.reshape(ZX, (-1, self.num_nodes, self.D))
        Y2 = self.FC_XC_out2(ZX)
        Y2 = tf.transpose(Y2, (0, 2, 1))

        
        custom_loss = 0
        for i in range(self.P):
            custom_loss += mmd_loss(EX[:, i, ...], EZX[:, i, ...], 0.05)
        
            
        # custom_loss = 0.05*custom_mae_loss_inside(EX, EZX)

        # trust = tf.reduce_sum(mask, 1)
        # trust /= self.P
        # trust = tf.expand_dims(trust, 1)

        # return trust*Y1 + (1-trust)*Y2
        return Y1 + Y2

        # return (Y1+Y2)/2
        # return Y1*0.85(Y1+Y2)/2





class MyOURS_AEGen2GMAN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AEGen2GMAN, self).__init__()
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

        # self.num_cells_C = (self.CH) * (self.CW)
        def cnn_dim(W):
            W = (W-3)//2 + 1
            W = (W-3)//2 + 1
            return W
        self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)

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

        # self.conv1 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', return_sequences=True)
        # self.conv2 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', go_backwards=True, return_sequences=True)
        
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                                layers.BatchNormalization(),
                                # layers.LeakyReLU(),
                                # layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                            ])
        # self.ZC_FC_layer = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.XC_trans_layer = BipartiteAttention(self.K, self.d)

        
        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.FC_XM = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])

        ###
        self.RSTA_enc = GSTAttBlock(self.K, self.d)

        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        # self.GSTAC_enc = GSTAttBlock(self.K, self.d)
        # self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        # self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_XC_out2 = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])
        self.GSTAC_enc2 = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer2 = TransformAttention(self.K, self.d)
        self.GSTAC_dec2 = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]

        


        self.MASK_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.MASK_enc = GSTAttBlock(self.K, self.d)
        
        # self.bn = layers.BatchNormalization()
        # self.gated_fusion = GatedFusion(self.D)


    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        mask = tf.not_equal(X0, 0)
        mask = tf.cast(mask, tf.float32)

        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)



        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]

        mmask = tf.expand_dims(mask, -1)
        mmask = self.MASK_in(mmask)
        STEX_P = self.MASK_enc(mmask, STEX_P)

        # ZC = tf.expand_dims(ZC, -1)
        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        # ZC = self.FC_ZC_in(ZC)
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))

        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]

        
        print(ZC.shape, STEZC.shape)

        ZC = self.RSTA_enc(ZC, STEZC)
        
        ZX = self.XC_trans_layer(STEX_P, STEZC, ZC)

        X_gen = self.FC_XM(ZX)[..., 0]
        X = mask*X0 + (1-mask)*X_gen

        custom_loss = 0.15*custom_mae_loss_inside(X0, X_gen)
        self.add_loss(custom_loss)



        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X) 
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX_P)
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEX_Q)
        X = self.FC_XC_out(X)
        Y1 = tf.squeeze(X, -1)

        

        for i in range(self.L):
            ZX = self.GSTAC_enc2[i](ZX, STEX_P)
        ZX = self.C_trans_layer2(ZX, STEX_P, STEX_Q)
        for i in range(self.L):
            ZX = self.GSTAC_dec2[i](ZX, STEX_Q)
        ZX = self.FC_XC_out2(ZX)
        Y2 = tf.squeeze(ZX, -1)

        return Y1 + Y2

        # return (Y1+Y2)/2
        # return Y1*0.85(Y1+Y2)/2



class MyOURS_AEGen2multi(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyOURS_AEGen2multi, self).__init__()
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
        self.maxvalZC = extdata['maxvalZC']

        self.num_cells_C = (self.CH) * (self.CW)
        # def cnn_dim(W):
        #     W = (W-3)//2 + 1
        #     W = (W-3)//2 + 1
        #     return W
        # self.num_cells_C = cnn_dim(self.CH) * cnn_dim(self.CW)

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

        # self.conv1 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', return_sequences=True)
        # self.conv2 = tf.keras.layers.ConvLSTM2D(filters=self.D, kernel_size=(5,5), strides=(1, 1), padding='same', go_backwards=True, return_sequences=True)
        
        self.FC_ZC_Conv = keras.Sequential([
                                layers.Conv2D(D, 3, strides=(1, 1), padding='same'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2D(D, 3, strides=(1, 1), padding='same'),
                                layers.BatchNormalization(),
                                # layers.LeakyReLU(),
                                # layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                            ])
                            
        self.FC_ZC_ConvTranspose = keras.Sequential([
                                layers.Conv2DTranspose(D, 3, strides=(1, 1), padding='same'),
                                layers.BatchNormalization(),
                                layers.LeakyReLU(),
                                layers.Conv2DTranspose(2, 3, strides=(1, 1), padding='same'),
                                layers.BatchNormalization(),
                                # layers.LeakyReLU(),
                                # layers.Conv2D(D, 3, strides=(2, 2), padding='valid'),
                            ])
        # self.ZC_FC_layer = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.XC_trans_layer = BipartiteAttention(self.K, self.d)

        self.ZC_trans_layer = TransformAttention(self.K, self.d)
        
        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])

        self.FC_XM = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])

        ###
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.RSTA_enc = GSTAttBlock(self.K, self.d)
        self.RSTA_dec = GSTAttBlock(self.K, self.d)
        
        # self.bn = layers.BatchNormalization()


    def call(self, kwargs):
        X0, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        ZCY0 = kwargs['ZCY']
        mask = tf.not_equal(X0, 0)
        mask = tf.cast(mask, tf.float32)

        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]


        # ZC = tf.expand_dims(ZC, -1)
        print(ZC.shape)
        if len(ZC.shape) == 4:
            ZC = tf.expand_dims(ZC, -1)
        # ZC = self.FC_ZC_in(ZC)
        ZC = self.FC_ZC_Conv(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))

        STEZC = self.STEZC_layer(self.SEZC, TE)
        STEZC_P, STEZC_Q = STEZC[:, :self.P, :], STEZC[:, self.P:, :]

        
        print(ZC.shape, STEZC.shape)

        ZC = self.RSTA_enc(ZC, STEZC_P)
        ZCY = self.ZC_trans_layer(ZC, STEZC_P, STEZC_Q)
        ZCY = self.RSTA_dec(ZCY, STEZC_Q)
        ZCY = self.FC_ZC_ConvTranspose(ZCY)

        print('ZCY0.shape, ZCY.shape', ZCY0.shape, ZCY.shape)
        ZCY = tf.reshape(ZCY, (-1, self.CH, self.CW, 1))
        custom_loss1 = 0.0015*custom_mae_loss_inside(ZCY0, ZCY)
        self.add_loss(custom_loss1)
        
        ZX = self.XC_trans_layer(STEX_P, STEZC_P, ZC)

        X_gen = self.FC_XM(ZX)[..., 0]
        X = mask*X0 + (1-mask)*X_gen



        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        X = X + STEX_P
        # X = self.bn(X)
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        custom_loss = 0.15*custom_mae_loss_inside(X0, X_gen)
        self.add_loss(custom_loss)


        return Y


