import os, datetime, argparse, tqdm, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import utils
import mymodels
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os.path import join as pjoin
# import sys


parser = argparse.ArgumentParser()
parser.add_argument('--file_traf', type = str, default = '../prepdata/traffic-volume-A-20180101-20190101.df')
parser.add_argument('--file_coarse', type = str, default = '../prepdata/coarse_grained_lte.h5')
parser.add_argument('--file_fine', type = str, default = '../prepdata/fine_grained_lte.h5')
parser.add_argument('--file_coarse_idx', type = str, default = '../prepdata/coarse_idx.csv')
parser.add_argument('--model_name', type = str, default = 'LastRepeat')

# parser.add_argument('--patch_size', type = int, default = 4)
parser.add_argument('--memo', type = str, default = '')
parser.add_argument('--save_dir', default = 'test', help = 'save_dir')

parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
# parser.add_argument('--cnn_size', type = int, default = 3)
parser.add_argument('--P', type = int, default = 6)
parser.add_argument('--Q', type = int, default = 1)
parser.add_argument('--time_slot', type = int, default = 60, help = 'a time step is 60 mins')

parser.add_argument('--eta', type = float, default = 0.01,
                    help = 'weight of mmd')
parser.add_argument('--L', type = int, default = 1,
                    help = 'number of STAtt Blocks')
# parser.add_argument('--LZ', type = int, default = 2,
#                     help = 'number of Regional STAtt Blocks')
parser.add_argument('--K', type = int, default = 4,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 16,
                    help = 'dims of each head attention outputs')
parser.add_argument('--D', type = int, default = 64)
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 1000,
                    help = 'epoch to run')
parser.add_argument('--patience', type = int, default = 10,
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--decay_epoch', type=int, default = 5,
                    help = 'decay epoch')


args = parser.parse_args()

    
args.test_name = args.memo + args.model_name

args.model_checkpoint_dir = f'checkpoint/'
args.model_checkpoint = os.path.join(args.model_checkpoint_dir, args.test_name)
args.test_dir = f'test_exp/'

if not os.path.isdir(args.model_checkpoint_dir):
    os.makedirs(args.model_checkpoint_dir)
if not os.path.isdir(args.test_dir):
    os.makedirs(args.test_dir)


(trainX, trainZC, trainTE, trainY, 
    valX, valZC, valTE, valY, 
    testX, testZC, testTE, testY, extdata) = utils.loadVolumeCrazy(args)

# (trainX, trainZC, trainZF, trainTE, trainY, 
#             valX, valZC, valZF, valTE, valY, 
#             testX, testZC, testZF, testTE, testY, extdata) = utils.loadData(args)

print(trainX.shape, trainZC.shape, trainTE.shape, trainY.shape)
print(valX.shape, valZC.shape, valTE.shape, valY.shape)
print(testX.shape, testZC.shape, testTE.shape, testY.shape)

# import sys; sys.exit(0)

#####################################

pred = np.tile(np.expand_dims(testX.mean(axis=1) * extdata['maxval'], 1), [1, args.Q, 1])
label = testY
print('Closeness Mean', utils.metric(pred, label), sep='\t')

pred = np.tile(testX[:, -1:, :]  * extdata['maxval'], [1, args.Q, 1])
label = testY
print('Closeness Last', utils.metric(pred, label), sep='\t')

pred = trainY.mean(0)[np.newaxis, :, :]
label = testY
print('Historical Avg', utils.metric(pred, label), sep='\t')

#####################################


def model_define():
    X = layers.Input(shape=trainX.shape[1:], dtype=tf.float32)
    ZC = layers.Input(shape=trainZC.shape[1:], dtype=tf.float32)
    TE = layers.Input(shape=trainTE.shape[1:], dtype=tf.int32) # int32

    Y = mymodels.ModelSetOurs(model_name=args.model_name, extdata=extdata, args=args, X=X, ZC=ZC, TE=TE)
    
    Y = Y * extdata['maxval']
    model = keras.models.Model((X, ZC, TE), Y)


    return model

model = model_define()

# model.load_weights(args.model_checkpoint).expect_partial()
model.summary()
optimizer = keras.optimizers.Adam(learning_rate=args.learning_rate)

# optimizer = tf.keras.optimizers.experimental.Adagrad(learning_rate = args.learning_rate)



def custom_mape_loss(label, pred):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    # loss = tf.abs(tf.subtract(pred, label))
    loss = 100 * tf.abs((pred - label) / (label+1e-3))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss

def custom_mae_loss(label, pred):
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


def custom_rmse_loss(label, pred):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.square(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.sqrt(tf.reduce_mean(loss))
    loss = tf.reduce_mean(loss)
    return loss


model.compile(loss=custom_rmse_loss, optimizer=optimizer)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=args.decay_epoch)
model_ckpt = tf.keras.callbacks.ModelCheckpoint(
    filepath = args.model_checkpoint,
    monitor = 'val_loss',
    verbose = 0,
    save_best_only = True,
    save_weights_only = True,
    cooldown=3,
    mode = 'auto',
    save_freq='epoch',
)


model.fit(
    (trainX, trainZC, trainTE), trainY,
    batch_size=args.batch_size,
    epochs=args.max_epoch,
    validation_data=((valX, valZC, valTE), valY),
    callbacks=[early_stopping, reduce_lr, model_ckpt],
    # callbacks=[early_stopping, model_ckpt],
)

model = model_define()
model.load_weights(args.model_checkpoint).expect_partial()

pred = model.predict((testX, testZC, testTE))
label = testY
print(f'{args.model_name}', utils.metric(pred, label), sep='\t')


np.save(f'{args.test_dir}/label.npy', label)
np.save(f'{args.test_dir}/pred_{args.test_name}.npy', pred)
