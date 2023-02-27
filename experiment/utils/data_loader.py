import numpy as np
import pandas as pd
import os, h5py
import scipy.sparse as sp
import pickle
from utils.extra import *

def load_h5(filename, keywords):
	f = h5py.File(filename, 'r')
	data = []
	for name in keywords:
		data.append(np.array(f[name]))
	f.close()
	if len(data) == 1:
		return data[0]
	return data


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def seq2instance(data, P, Q):
    num_step = data.shape[0]
    data_type = data.dtype
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(num_sample):
        x[i] = data[i : i + P].astype(data_type)
        y[i] = data[i + P : i + P + Q].astype(data_type)
    return x, y

def seq2instance_fill(data, data_fill, P, Q):
    num_step = data.shape[0]
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(num_sample):
        x[i] = data_fill[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def seq2instance2(data, P, Q):
    print(data.shape)
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def fill_missing(data):
	data = data.copy()
	data[data < 1e-5] = float('nan')
	data = data.fillna(method='pad')
	data = data.fillna(method='bfill')
	return data

    
def pearson_corr(X, Y):
    return stats.pearsonr(X, Y)[0]


    
def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

# def convert_to_adj_mx(dist_mx, threshold=3000):
#     if threshold > 0:
#         dist_mx[dist_mx > threshold] = np.inf
#     distances = dist_mx[~np.isinf(dist_mx)].flatten()
#     std = distances.std()
#     adj_mx = np.exp(-np.square(dist_mx / std))
#     return row_normalize(adj_mx)






# def loadVolumeData(args):
#     #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
#     traf_df = pd.read_hdf(args.file_traf)

#     Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
#     #Traffic_fill = fill_missing(df).values.astype(np.float32)
#     num_step, num_sensors = traf_df.shape
    
#     # train/val/test 
#     train_steps = round(args.train_ratio * num_step)
#     test_steps = round(args.test_ratio * num_step)
#     val_steps = num_step - train_steps - test_steps

#     train = Traffic[: train_steps]
#     val = Traffic[train_steps : train_steps + val_steps]
#     test = Traffic[-test_steps :]

#     # X, Y 
#     trainX, trainY = seq2instance(train, args.P, args.Q)
#     valX, valY = seq2instance(val, args.P, args.Q)
#     testX, testY = seq2instance(test, args.P, args.Q)
#     # normalization
#     maxval = np.max(train)
#     trainX = trainX / maxval
#     valX = valX / maxval
#     testX = testX / maxval
    
    
#     # temporal embedding 
#     Time = traf_df.index
#     dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
#     timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#                 // (args.time_slot*60) #// Time.freq.delta.total_seconds()
#     timeofday = np.reshape(timeofday, newshape = (-1, 1))    
#     Time = np.concatenate((dayofweek, timeofday), axis = -1)
#     # train/val/test
#     train = Time[: train_steps]
#     val = Time[train_steps : train_steps + val_steps]
#     test = Time[-test_steps :]
#     # shape = (num_sample, P + Q, 2)
#     trainTE = seq2instance(train, args.P, args.Q)
#     trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
#     valTE = seq2instance(val, args.P, args.Q)
#     valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
#     testTE = seq2instance(test, args.P, args.Q)
#     testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

#     extdata = dict()
#     extdata['maxval'] = maxval 
#     extdata['num_nodes'] = num_sensors 

#     return (trainX, trainTE, trainY, 
#             valX, valTE, valY, 
#             testX, testTE, testY, extdata)


def loadVolumeData2(args):
    #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
    traf_df = pd.read_hdf(args.file_traf)
    coarse_df = pd.read_hdf(args.file_coarse)
    # coarse_zero = np.zeros((1, len(coarse_df.columns)), dtype=np.float32)
    # coarse_diff = np.concatenate((coarse_zero, np.diff(coarse_df.values, axis=0)), 0)
    # coarse_diff_df = pd.DataFrame(coarse_diff, columns=coarse_df.columns)
    # coarse_diff_df.index = coarse_df.index
    # coarse_df = coarse_diff_df

    fine_df = pd.read_hdf(args.file_fine)
    # coarse_df = fine_df

    # fine_zero = np.zeros((1, len(fine_df.columns)), dtype=np.float32)
    # fine_diff = np.concatenate((fine_zero, np.diff(fine_df.values, axis=0)), 0)
    # fine_diff_df = pd.DataFrame(fine_diff, columns=fine_df.columns)
    # fine_diff_df.index = fine_df.index
    # fine_df = fine_diff_df
    
    traf_df = traf_df.iloc[:24*7*30]
    coarse_df = coarse_df.iloc[:24*7*30]
    fine_df = fine_df.iloc[:24*7*30]

    sid_list = []
    traf_vals = np.nan_to_num(traf_df.values.astype(np.float32))
    for sid, val in enumerate(np.sum(traf_vals == 0, 0)):
        if val < traf_vals.shape[0]*0.3:
            sid_list.append(traf_df.columns[sid])
            
    
    traf_df = traf_df[sid_list]
    print('traf_df', len(sid_list), traf_df.shape)

    extdata = dict()

    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    # Traffic_fill = fill_missing(traf_df).values.astype(np.float32)
    num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
    # train/val/test 
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    
    # train_fill = train_traf = Traffic_fill[: train_steps]
    # val_fill = Traffic_fill[train_steps : train_steps + val_steps]
    # test_fill = Traffic_fill[-test_steps :]


    np.random.seed(0)
    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # test[np.random.rand(*test.shape) < 0.5] = 0

    # X, Y 
    # trainX, trainY = seq2instance_fill(train, train_fill, args.P, args.Q)
    # valX, valY = seq2instance_fill(val, val_fill, args.P, args.Q)
    # testX, testY = seq2instance_fill(test, test_fill, args.P, args.Q)
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    maxval = np.max(train); extdata['maxval'] = maxval 
    trainX = trainX / maxval
    valX = valX / maxval
    testX = testX / maxval
    np.save(f'{args.test_dir}/label_zeroed.npy', testY)




    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    _, trainY = seq2instance(train, args.P, args.Q)
    _, valY = seq2instance(val, args.P, args.Q)
    _, testY = seq2instance(test, args.P, args.Q)
    # np.random.seed(0)
    # trainX[np.random.rand(*trainX.shape) < 0.3] = 0
    # valX[np.random.rand(*valX.shape) < 0.3] = 0

    # testX[np.random.rand(*testX.shape) < 0.15] = 0
    np.save(f'{args.test_dir}/label_original.npy', testY)


    

    
    # print(train_traf.shape)
    # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    # # cr = int(args.cnn_size//2)
    # for i in range(num_sensors):
    #     adj_prcc[i, i] = 1
    #     for j in range(i+1, num_sensors):
    #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
    # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

    # coarse train/val/test
    # CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    # CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    # extdata['CH'] = CH; extdata['CW'] = CW
    # CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

    CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    extdata['CH'] = CH; extdata['CW'] = CW
    CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

    oLTE = CoarseLTE
    lte_zeros =  np.zeros(CoarseLTE[0].shape)
    lte_diff = np.diff(CoarseLTE, axis=0)
    lte_diff0 = np.concatenate(([lte_zeros], lte_diff), 0)
    CoarseLTE = lte_diff0
    CoarseLTE = np.stack((oLTE, lte_diff0), -1)

    
    train = CoarseLTE[: train_steps]
    val = CoarseLTE[train_steps : train_steps + val_steps]
    test = CoarseLTE[-test_steps :]


    # X, Y 
    trainZC, _ = seq2instance(train, args.P, args.Q)
    valZC, _ = seq2instance(val, args.P, args.Q)
    testZC, _ = seq2instance(test, args.P, args.Q)
    # normalization
    # maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
    # trainZC = trainZC / maxvalZC
    # valZC = valZC / maxvalZC
    # testZC = testZC / maxvalZC
    meanZC = np.mean(train); extdata['meanZC'] = meanZC = 0
    stdZC = np.std(train); extdata['stdZC'] = stdZC
    trainZC = (trainZC - meanZC) / stdZC
    valZC = (valZC - meanZC) / stdZC
    testZC = (testZC - meanZC) / stdZC

    
    # fine train/val/test
    FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))

    FineLTE = np.diff(FineLTE, axis=0)
    FineLTE = np.sign(FineLTE) * np.log10(np.abs(FineLTE)+1)

    # FineLTE = np.concatenate((np.zeros((0, FineLTE.shape[1])), np.diff(FineLTE, axis=0)), 0)
    FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
    extdata['FH'] = FH; extdata['FW'] = FW
    FineLTE = FineLTE.reshape(-1, FH, FW)
    
    train = FineLTE[: train_steps]
    val = FineLTE[train_steps : train_steps + val_steps]
    test = FineLTE[-test_steps :]

    # X, Y 
    trainZF, _ = seq2instance(train, args.P, args.Q)
    valZF, _ = seq2instance(val, args.P, args.Q)
    testZF, _ = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZF = np.max(train); extdata['maxvalZF'] = maxvalZF = 0
    trainZF = trainZF / maxvalZF
    valZF = valZF / maxvalZF
    testZF = testZF / maxvalZF

    
    # temporal embedding 
    Time = traf_df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (args.time_slot*60) #// Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


    return (trainX, trainZC, trainZF, trainTE, trainY, 
            valX, valZC, valZF, valTE, valY, 
            testX, testZC, testZF, testTE, testY, extdata)


def loadVolumeData2multi(args):
    #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
    traf_df = pd.read_hdf(args.file_traf)
    coarse_df = pd.read_hdf(args.file_coarse)
    # coarse_zero = np.zeros((1, len(coarse_df.columns)), dtype=np.float32)
    # coarse_diff = np.concatenate((coarse_zero, np.diff(coarse_df.values, axis=0)), 0)
    # coarse_diff_df = pd.DataFrame(coarse_diff, columns=coarse_df.columns)
    # coarse_diff_df.index = coarse_df.index
    # coarse_df = coarse_diff_df

    fine_df = pd.read_hdf(args.file_fine)
    # fine_zero = np.zeros((1, len(fine_df.columns)), dtype=np.float32)
    # fine_diff = np.concatenate((fine_zero, np.diff(fine_df.values, axis=0)), 0)
    # fine_diff_df = pd.DataFrame(fine_diff, columns=fine_df.columns)
    # fine_diff_df.index = fine_df.index
    # fine_df = fine_diff_df
    
    traf_df = traf_df.iloc[:24*7*50]
    coarse_df = coarse_df.iloc[:24*7*50]
    fine_df = fine_df.iloc[:24*7*50]

    sid_list = []
    traf_vals = np.nan_to_num(traf_df.values.astype(np.float32))
    for sid, val in enumerate(np.sum(traf_vals == 0, 0)):
        if val < 1000:
            sid_list.append(traf_df.columns[sid])
            
    
    traf_df = traf_df[sid_list]
    print('traf_df', len(sid_list), traf_df.shape)

    extdata = dict()

    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    # Traffic_fill = fill_missing(traf_df).values.astype(np.float32)
    num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
    # train/val/test 
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    # train_fill = train_traf = Traffic_fill[: train_steps]
    # val_fill = Traffic_fill[train_steps : train_steps + val_steps]
    # test_fill = Traffic_fill[-test_steps :]

    # X, Y 
    # trainX, trainY = seq2instance_fill(train, train_fill, args.P, args.Q)
    # valX, valY = seq2instance_fill(val, val_fill, args.P, args.Q)
    # testX, testY = seq2instance_fill(test, test_fill, args.P, args.Q)
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    maxval = np.max(train); extdata['maxval'] = maxval 
    trainX = trainX / maxval
    valX = valX / maxval
    testX = testX / maxval
    
    
    # print(train_traf.shape)
    # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    # # cr = int(args.cnn_size//2)
    # for i in range(num_sensors):
    #     adj_prcc[i, i] = 1
    #     for j in range(i+1, num_sensors):
    #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
    # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

    # coarse train/val/test
    # CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    # CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    # extdata['CH'] = CH; extdata['CW'] = CW
    # CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

    CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    extdata['CH'] = CH; extdata['CW'] = CW
    CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

    oLTE = CoarseLTE
    lte_zeros =  np.zeros(CoarseLTE[0].shape)
    lte_diff = np.diff(CoarseLTE, axis=0)
    lte_diff0 = np.concatenate(([lte_zeros], lte_diff), 0)
    CoarseLTE = lte_diff0
    CoarseLTE = np.stack((oLTE, lte_diff0), -1)

    
    train = CoarseLTE[: train_steps]
    val = CoarseLTE[train_steps : train_steps + val_steps]
    test = CoarseLTE[-test_steps :]

    # X, Y 
    trainZC, trainZCY = seq2instance(train, args.P, args.Q)
    valZC, valZCY = seq2instance(val, args.P, args.Q)
    testZC, testZCY = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
    trainZC = trainZC / maxvalZC
    valZC = valZC / maxvalZC
    testZC = testZC / maxvalZC
    # meanZC = np.mean(train); extdata['meanZC'] = meanZC = 0
    # stdZC = np.std(train); extdata['stdZC'] = stdZC
    # trainZC = (trainZC - meanZC) / stdZC
    # valZC = (valZC - meanZC) / stdZC
    # testZC = (testZC - meanZC) / stdZC

    
    # fine train/val/test
    FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))

    FineLTE = np.diff(FineLTE, axis=0)
    FineLTE = np.sign(FineLTE) * np.log10(np.abs(FineLTE)+1)

    # FineLTE = np.concatenate((np.zeros((0, FineLTE.shape[1])), np.diff(FineLTE, axis=0)), 0)
    FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
    extdata['FH'] = FH; extdata['FW'] = FW
    FineLTE = FineLTE.reshape(-1, FH, FW)
    
    train = FineLTE[: train_steps]
    val = FineLTE[train_steps : train_steps + val_steps]
    test = FineLTE[-test_steps :]

    # X, Y 
    trainZF, _ = seq2instance(train, args.P, args.Q)
    valZF, _ = seq2instance(val, args.P, args.Q)
    testZF, _ = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZF = np.max(train); extdata['maxvalZF'] = maxvalZF = 0
    trainZF = trainZF / maxvalZF
    valZF = valZF / maxvalZF
    testZF = testZF / maxvalZF

    
    # temporal embedding 
    Time = traf_df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (args.time_slot*60) #// Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


    return (trainX, trainZC, trainZF, trainTE, trainY, trainZCY, 
            valX, valZC, valZF, valTE, valY, valZCY, 
            testX, testZC, testZF, testTE, testY, testZCY, extdata)




# def loadVolumeCrazy(args):
#     #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
#     traf_df = pd.read_hdf(args.file_traf)
#     coarse_df = pd.read_hdf(args.file_coarse)
#     fine_df = pd.read_hdf(args.file_fine)
    
#     dstart = 0
#     dend = 24*7*10
#     traf_df = traf_df.iloc[dstart:dend]
#     coarse_df = coarse_df.iloc[dstart:dend]
#     fine_df = fine_df.iloc[dstart:dend]

#     sid_list = []
#     traf_vals = np.nan_to_num(traf_df.values.astype(np.float32))
#     for sid, val in enumerate(np.sum(traf_vals == 0, 0)):
#         if val < 1000:
#             sid_list.append(traf_df.columns[sid])
            
    
#     traf_df = traf_df[sid_list]
#     print('traf_df', len(sid_list), traf_df.shape)


#     CH, CW = coarse_df.columns[-1].split(',');  CH = int(CH)+1; CW = int(CW)+1; CN = CH*CW
#     FH, FW = fine_df.columns[-1].split(',');    FH = int(FH)+1; FW = int(FW)+1; FN = FH*FW

#     sensor_coarse_df = pd.read_csv('../prepdata/coarse_idx.csv')
#     sensor2cidx = dict()
#     for _, item in sensor_coarse_df.iterrows():
#         sensor2cidx[item.sensor_id] = (item.y, item.x)


#     extdata = dict()

#     Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
#     num_step, num_nodes = traf_df.shape; extdata['num_nodes'] = num_nodes
    
#     # train/val/test 
#     train_steps = round(args.train_ratio * num_step)
#     test_steps = round(args.test_ratio * num_step)
#     val_steps = num_step - train_steps - test_steps

#     train = Traffic[1: train_steps] # !skipping first step
#     val = Traffic[train_steps : train_steps + val_steps]
#     test = Traffic[train_steps + val_steps:]

#     trainX, trainY = seq2instance(train, args.P, args.Q)
#     valX, valY = seq2instance(val, args.P, args.Q)
#     testX, testY = seq2instance(test, args.P, args.Q)
#     # normalization
#     maxval = np.max(train); extdata['maxval'] = maxval 
#     # print('maxval.shape', maxval.shape)
#     trainX = trainX / maxval
#     valX = valX / maxval
#     testX = testX / maxval
    


#     # temporal embedding 
#     Time = traf_df.index
#     dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
#     timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#                 // (args.time_slot*60) #// Time.freq.delta.total_seconds()
#     timeofday = np.reshape(timeofday, newshape = (-1, 1))    
#     Time = np.concatenate((dayofweek, timeofday), axis = -1)
#     # train/val/test
#     train = Time[1: train_steps] # !skipping first step
#     val = Time[train_steps : train_steps + val_steps]
#     test = Time[train_steps + val_steps:]
#     # shape = (num_sample, P + Q, 2)
#     trainTE = seq2instance(train, args.P, args.Q)
#     trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
#     valTE = seq2instance(val, args.P, args.Q)
#     valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
#     testTE = seq2instance(test, args.P, args.Q)
#     testTE = np.concatenate(testTE, axis = 1).astype(np.int32)






    
#     # print(train_traf.shape)
#     # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
#     # # cr = int(args.cnn_size//2)
#     # for i in range(num_sensors):
#     #     adj_prcc[i, i] = 1
#     #     for j in range(i+1, num_sensors):
#     #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
#     # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

#     # coarse train/val/test
#     # CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
#     # CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
#     # extdata['CH'] = CH; extdata['CW'] = CW
#     # CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

#     CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))

#     ksize = 3
#     extdata['ksize'] = ksize
#     if os.path.isfile(f'temp/AC_flow_{ksize}_{dstart}_{dend}.npy'):
#         with open(f'temp/AC_flow_{ksize}_{dstart}_{dend}.npy', 'rb') as f:
#             AC_flow = np.load(f)
#     else:
#         AC_list = []
#         import tqdm
#         for i in tqdm.tqdm(range(CoarseLTE.shape[0]-1)):
#             Z1 = CoarseLTE[i]
#             Z2 = CoarseLTE[i+1]
#             Z2 = Z2.flatten().reshape(CN, 1)
#             Z1 = Z1.flatten().reshape(CN, 1)

#             A = (Z2 @ np.linalg.pinv(Z1))
#             AC_list.append(A)
        
#         AC_list = np.array(AC_list)

#         AC_flow1 = []
#         AC_flow2 = []
#         ker_list = []
#         for i in range(-ksize, ksize+1):
#             for j in range(-ksize, ksize+1):
#                 ker_list.append((i, j))


#         for sensor in tqdm.tqdm(range(num_nodes)):
            
#             flow1 = []
#             for k, (ki, kj) in enumerate(ker_list):
#                 sy, sx = sensor2cidx[sensor]
#                 wy, wx = sy+ki, sx+kj
#                 flow1.append(AC_list[:, sy*CW+sx, wy*CW+wx])
#             AC_flow1.append(flow1)
                
#             flow2 = []
#             for k, (ki, kj) in enumerate(ker_list):
#                 sy, sx = sensor2cidx[sensor]
#                 wy, wx = sy+ki, sx+kj
#                 flow2.append(AC_list[:, wy*CW+wx, sy*CW+sx])
#             AC_flow2.append(flow2)
        

#         AC_flow1 = np.array(AC_flow1)
#         AC_flow2 = np.array(AC_flow2)

#         AC_flow1 = np.transpose(AC_flow1, (2, 0, 1))
#         AC_flow2 = np.transpose(AC_flow2, (2, 0, 1))

#         AC_flow = np.stack([AC_flow1, AC_flow2], -1)
 
#         print('AC_flow', AC_flow.shape)

#         with open(f'temp/AC_flow_{ksize}_{dstart}_{dend}.npy', 'wb') as fp:
#             np.save(fp, AC_flow)

        





#     extdata['CH'] = CH; extdata['CW'] = CW; extdata['CN'] = CN;

#     train = AC_flow[: train_steps-1]
#     val = AC_flow[train_steps-1 : train_steps + val_steps-1]
#     test = AC_flow[train_steps + val_steps-1:]

#     # X, Y 
#     trainZC, _ = seq2instance(train, args.P, args.Q)
#     valZC, _ = seq2instance(val, args.P, args.Q)
#     testZC, _ = seq2instance(test, args.P, args.Q)
#     # normalization
#     # maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
#     # trainZC = trainZC / maxvalZC
#     # valZC = valZC / maxvalZC
#     # testZC = testZC / maxvalZC
#     stdZC = np.std(train); extdata['stdZC'] = stdZC
#     trainZC = trainZC / stdZC
#     valZC = valZC  / stdZC
#     testZC = testZC / stdZC

    
#     # # fine train/val/test
#     # FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))

#     # FineLTE = np.diff(FineLTE, axis=0)
#     # FineLTE = np.sign(FineLTE) * np.log10(np.abs(FineLTE)+1)

#     # # FineLTE = np.concatenate((np.zeros((0, FineLTE.shape[1])), np.diff(FineLTE, axis=0)), 0)
#     # FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
#     # extdata['FH'] = FH; extdata['FW'] = FW
#     # FineLTE = FineLTE.reshape(-1, FH, FW)
    
#     # train = FineLTE[: train_steps]
#     # val = FineLTE[train_steps : train_steps + val_steps]
#     # test = FineLTE[-test_steps :]

#     # # X, Y 
#     # trainZF, _ = seq2instance(train, args.P, args.Q)
#     # valZF, _ = seq2instance(val, args.P, args.Q)
#     # testZF, _ = seq2instance(test, args.P, args.Q)
#     # # normalization
#     # maxvalZF = np.max(train); extdata['maxvalZF'] = maxvalZF = 0
#     # trainZF = trainZF / maxvalZF
#     # valZF = valZF / maxvalZF
#     # testZF = testZF / maxvalZF

    


#     return (trainX, trainZC, trainTE, trainY, 
#             valX, valZC, valTE, valY, 
#             testX, testZC, testTE, testY, extdata)


# def loadVolumeData3(args):
#     #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
#     traf_df = pd.read_hdf(args.file_traf); traf_df = traf_df.iloc[24*59:24*151]
#     coarse_df = pd.read_hdf(args.file_coarse); coarse_df = coarse_df.iloc[24*59:24*151]
#     fine_df = pd.read_hdf(args.file_fine); fine_df = fine_df.iloc[24*59:24*151]
    
#     extdata = dict()

#     Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
#     #Traffic_fill = fill_missing(df).values.astype(np.float32)
#     num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
#     # train/val/test 
#     train_steps = round(args.train_ratio * num_step)
#     test_steps = round(args.test_ratio * num_step)
#     val_steps = num_step - train_steps - test_steps

#     train = train_traf = Traffic[: train_steps]
#     val = Traffic[train_steps : train_steps + val_steps]
#     test = Traffic[-test_steps :]

#     # X, Y 
#     trainX, trainY = seq2instance(train, args.P, args.Q)
#     valX, valY = seq2instance(val, args.P, args.Q)
#     testX, testY = seq2instance(test, args.P, args.Q)
#     # normalization
#     maxval = np.max(train); extdata['maxval'] = maxval 
#     trainX = trainX / maxval
#     valX = valX / maxval
#     testX = testX / maxval
    
    
    
#     # print(train_traf.shape)
#     # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
#     # # cr = int(args.cnn_size//2)
#     # for i in range(num_sensors):
#     #     adj_prcc[i, i] = 1
#     #     for j in range(i+1, num_sensors):
#     #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
#     # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

#     # coarse train/val/test
#     CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
#     prev_CoarseLTE = np.concatenate((np.zeros((1, CoarseLTE.shape[1])), CoarseLTE[:-1, ...]), 0)
#     print(CoarseLTE.shape, prev_CoarseLTE.shape)
#     CoarseLTE = np.concatenate((prev_CoarseLTE, CoarseLTE), -1)


#     CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
#     extdata['CH'] = CH; extdata['CW'] = CW
#     CoarseLTE = CoarseLTE.reshape(-1, CH, CW, 2)
    
#     train = CoarseLTE[: train_steps]
#     val = CoarseLTE[train_steps : train_steps + val_steps]
#     test = CoarseLTE[train_steps + val_steps :]

#     # X, Y 
#     trainZC, _ = seq2instance(train, args.P, args.Q)
#     valZC, _ = seq2instance(val, args.P, args.Q)
#     testZC, _ = seq2instance(test, args.P, args.Q)
#     # normalization
#     maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
#     trainZC = trainZC / maxvalZC
#     valZC = valZC / maxvalZC
#     testZC = testZC / maxvalZC

    
#     # fine train/val/test
#     FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))
#     prev_FineLTE = np.concatenate((np.zeros((1, FineLTE.shape[1])), FineLTE[:-1, ...]), 0)
#     FineLTE = np.concatenate((prev_FineLTE, FineLTE), -1)

#     FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
#     extdata['FH'] = FH; extdata['FW'] = FW
#     FineLTE = FineLTE.reshape(-1, FH, FW, 2)
    
#     train = FineLTE[: train_steps]
#     val = FineLTE[train_steps : train_steps + val_steps]
#     test = FineLTE[train_steps + val_steps :]

#     # X, Y 
#     trainZF, _ = seq2instance(train, args.P, args.Q)
#     valZF, _ = seq2instance(val, args.P, args.Q)
#     testZF, _ = seq2instance(test, args.P, args.Q)
#     # normalization 
#     maxvalZF = np.max(train, 0); extdata['maxvalZF'] = maxvalZF
#     trainZF = trainZF / maxvalZF
#     valZF = valZF / maxvalZF
#     testZF = testZF / maxvalZF

    
#     # temporal embedding 
#     Time = traf_df.index
#     dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
#     timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#                 // (args.time_slot*60) #// Time.freq.delta.total_seconds()
#     timeofday = np.reshape(timeofday, newshape = (-1, 1))    
#     Time = np.concatenate((dayofweek, timeofday), axis = -1)
#     # train/val/test 
#     train = Time[: train_steps]
#     val = Time[train_steps : train_steps + val_steps]
#     test = Time[-test_steps :]
#     # shape = (num_sample, P + Q, 2) 
#     trainTE = seq2instance(train, args.P, args.Q)
#     trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
#     valTE = seq2instance(val, args.P, args.Q)
#     valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
#     testTE = seq2instance(test, args.P, args.Q)
#     testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


#     extdata['adj_mat'] = row_normalize(np.eye(num_sensors) + 0.01)
#     # extdata['adj_mat'] = row_normalize(np.exp(np.random.rand(num_sensors, num_sensors)))


#     return (trainX, trainZC, trainZF, trainTE, trainY, 
#             valX, valZC, valZF, valTE, valY, 
#             testX, testZC, testZF, testTE, testY, extdata)



# def loadVolumeDataB(args):
#     traf_df = pd.read_hdf('../benchdata/pems-bay.h5')
    
#     extdata = dict()

#     Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
#     #Traffic_fill = fill_missing(df).values.astype(np.float32)
#     num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
#     # train/val/test 
#     train_steps = round(args.train_ratio * num_step)
#     test_steps = round(args.test_ratio * num_step)
#     val_steps = num_step - train_steps - test_steps

#     train = train_traf = Traffic[: train_steps]
#     val = Traffic[train_steps : train_steps + val_steps]
#     test = Traffic[-test_steps :]

#     # X, Y 
#     trainX, trainY = seq2instance(train, args.P, args.Q)
#     valX, valY = seq2instance(val, args.P, args.Q)
#     testX, testY = seq2instance(test, args.P, args.Q)
#     # normalization
#     # maxval = np.max(train); extdata['maxval'] = maxval 
#     # trainX = trainX / maxval
#     # valX = valX / maxval
#     # testX = testX / maxval
#     stdval = np.std(train, axis=0, keepdims=True)+1; extdata['stdval'] = stdval 
#     meanval = np.mean(train, axis=0, keepdims=True); extdata['meanval'] = meanval 
#     trainX = (trainX - meanval) / stdval
#     valX = (valX - meanval) / stdval
#     testX = (testX - meanval) / stdval
    
    
#     # temporal embedding 
#     Time = traf_df.index
#     dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
#     timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#                 // (args.time_slot*5) #// Time.freq.delta.total_seconds()
#     timeofday = np.reshape(timeofday, newshape = (-1, 1))    
#     Time = np.concatenate((dayofweek, timeofday), axis = -1)
#     # train/val/test 
#     train = Time[: train_steps]
#     val = Time[train_steps : train_steps + val_steps]
#     test = Time[-test_steps :]
#     # shape = (num_sample, P + Q, 2) 
#     trainTE = seq2instance(train, args.P, args.Q)
#     trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
#     valTE = seq2instance(val, args.P, args.Q)
#     valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
#     testTE = seq2instance(test, args.P, args.Q)
#     testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


#     _, _, adj_mx = load_graph_data('../benchdata/adj_mx_bay.pkl')
#     extdata['adj_mat'] = adj_mx


#     return (trainX, trainX, trainX, trainTE, trainY, 
#             valX, valX, valX, valTE, valY, 
#             testX, testX, testX, testTE, testY, extdata)


import geopandas as gpd
# def loadData(args):
#     # Traffic
#     print("Help")
#     extdata = dict()
#     args.data_dir = '../AAAI22-ID4313-supplementary-final-0912-1745/dataprocess/prepdata'
#     args.region = 'hongik' #'jamsil' #'gangnam'
#     args.cell_size = 150
#     traffic_file = f'{args.data_dir}/{args.region}/traffic.h5'
#     SE_file = f'{args.data_dir}/{args.region}/SE.txt'
#     SEZ_npy_file = f'{args.data_dir}/{args.region}/lte_cell_poi_features_{args.cell_size}.npy'
#     SAT_npy_file = f'{args.data_dir}/{args.region}/cell_satellite_features_{args.cell_size}.npy'

#     lte_adj_mx_file = f'{args.data_dir}/{args.region}/road_lte_adj_mx_distance_{args.cell_size}.npy'
#     coarse_df = pd.read_hdf(args.file_coarse)
#     lte_file = f'{args.data_dir}/{args.region}/lte_cell_{args.cell_size}.h5'
#     # lte_cell_geofile = f'{args.data_dir}/{args.region}/lte_cell_{args.cell_size}.geojson'
#     graph_pkl_filename = f'{args.data_dir}/{args.region}/adj_mat.pkl'
    
#     df = pd.read_hdf(traffic_file)

#     Traffic = df.values.astype(np.float32)
#     Traffic_fill = fill_missing(df).values.astype(np.float32)
#     num_step, num_sensors = df.shape 



#     # DIST_GR = np.load(lte_adj_mx_file).astype(np.float32)

#     # POISEZ = np.load(SEZ_npy_file).astype(np.float32)
#     # SATSEZ = np.load(SAT_npy_file).astype(np.float32)
    
#     # LH, LW, _ = POISEZ.shape
#     # POISEZ = POISEZ.reshape(LH*LW, -1)
#     # SATSEZ = SATSEZ.reshape(LH*LW, -1)


#     # lte_gdf = gpd.read_file(lte_cell_geofile)
#     # loc_val = np.array([lte_gdf['geometry'].centroid.x.values, 
#     #                     lte_gdf['geometry'].centroid.y.values], dtype=np.float32).T
#     # LOCSEZ = np.array(loc_val)

#     # SEZ = [LOCSEZ]
#     # if not args.NOPOI:
#     #     SEZ.append(POISEZ)
#     # if not args.NOSAT:
#     #     SEZ.append(SATSEZ)
    
#     # SEZ = np.concatenate(SEZ, axis=-1) # last two is lng and lat
#     # print("SEZ:", SEZ.shape)

    
#     coarse_df = coarse_df[(coarse_df.index >= df.index.min()) & (coarse_df.index < df.index.max())]
#     # lte_df = pd.read_hdf(lte_file)
#     num_cells = lte_df.shape[1]
#     LTE = lte_df.values.astype(np.float32)
#     LTE = LTE.reshape(-1, LH, LW)

    
#     CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
#     CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
#     extdata['CH'] = CH; extdata['CW'] = CW
#     CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

#     oLTE = CoarseLTE
#     lte_zeros =  np.zeros(CoarseLTE[0].shape)
#     lte_diff = np.diff(CoarseLTE, axis=0)
#     lte_diff0 = np.concatenate(([lte_zeros], lte_diff), 0)
#     CoarseLTE = lte_diff0
#     CoarseLTE = np.stack((oLTE, lte_diff0), -1)


#     SEZ = None

    


#     # train/val/test 
#     train_steps = round(args.train_ratio * num_step)
#     test_steps = round(args.test_ratio * num_step)
#     val_steps = num_step - train_steps - test_steps

#     train = Traffic[: train_steps]
#     val = Traffic[train_steps : train_steps + val_steps]
#     test = Traffic[-test_steps :]

#     train_fill = Traffic_fill[: train_steps]
#     val_fill = Traffic_fill[train_steps : train_steps + val_steps]
#     test_fill = Traffic_fill[-test_steps :]

#     # X, Y 
#     trainX, trainY = seq2instance_fill(train, train_fill, args.P, args.Q)
#     valX, valY = seq2instance_fill(val, val_fill, args.P, args.Q)
#     testX, testY = seq2instance_fill(test, test_fill, args.P, args.Q)
#     # normalization
#     # mean, std = np.mean(trainX), np.std(trainX)
#     # trainX = (trainX - mean) / std
#     # valX = (valX - mean) / std
#     # testX = (testX - mean) / std
#     maxval = np.max(train); extdata['maxval'] = maxval 
#     trainX = trainX / maxval
#     valX = valX / maxval
#     testX = testX / maxval

    
#     train = lte_train = LTE[: train_steps]
#     val = LTE[train_steps : train_steps + val_steps]
#     test = LTE[-test_steps :]
#     # X, Y 
#     trainZ, trainZY = seq2instance(train, args.P, args.Q)
#     valZ, valZY = seq2instance(val, args.P, args.Q)
#     testZ, testZY = seq2instance(test, args.P, args.Q)
#     # normalization
#     meanL, stdL = np.mean(trainZ), np.std(trainZ)
#     trainZ = (trainZ - meanL) / stdL
#     valZ = (valZ - meanL) / stdL
#     testZ = (testZ - meanL) / stdL

        
#     # temporal embedding 
#     Time = df.index
#     dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
#     timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
#                 // (args.time_slot*60) #// Time.freq.delta.total_seconds()
#     timeofday = np.reshape(timeofday, newshape = (-1, 1))    
#     Time = np.concatenate((dayofweek, timeofday), axis = -1)
#     # train/val/test
#     train = Time[: train_steps]
#     val = Time[train_steps : train_steps + val_steps]
#     test = Time[-test_steps :]
#     # shape = (num_sample, P + Q, 2)
#     trainTE = seq2instance(train, args.P, args.Q)
#     trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
#     valTE = seq2instance(val, args.P, args.Q)
#     valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
#     testTE = seq2instance(test, args.P, args.Q)
#     testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

#     # N2V spatial embedding 
#     f = open(SE_file, mode = 'r')
#     lines = f.readlines()
#     temp = lines[0].split(' ')
#     N, dims = int(temp[0]), int(temp[1])
#     SE = np.zeros(shape = (N, dims), dtype = np.float32)
#     for line in lines[1 :]:
#         temp = line.split(' ')
#         index = int(temp[0])
#         SE[index] = temp[1 :]

#     # return (trainX, trainZ, trainTE, trainY, trainZY, 
#     #         valX, valZ, valTE, valY, valZY, 
#     #         testX, testZ, testTE, testY, testZY,
#     #         SE, SEZ, DIST_GR, ADJ_DY, mean, std, LH, LW)

#     extdata['num_nodes'] = N
#     extdata['SE'] = SE
#     extdata['SEZ'] = SEZ
#     # extdata['mean'] = mean
#     # extdata['std'] = std
#     extdata['LH'] = LH
#     extdata['LW'] = LW
#     extdata['CH'] = LH; extdata['CW'] = LW
#     extdata['FH'] = LH; extdata['FW'] = LW

#     graph_pkl_filename = os.path.join(f'{args.data_dir}/{args.region}', 'adj_mat.pkl')
#     sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)
#     extdata['adj_mx'] = adj_mx
    
#     # return (trainX, trainX, trainX, trainTE, trainY, 
#     #         valX, valX, valX, valTE, valY, 
#     #         testX, testX, testX, testTE, testY, extdata)
#     return (trainX, trainZ, trainZ, trainTE, trainY, 
#             valX, valZ, valZ, valTE, valY, 
#             testX, testZ, testZ, testTE, testY, extdata)




def loadData(args):
    # Traffic
    extdata = dict()
    args.data_dir = '../AAAI22-ID4313-supplementary-final-0912-1745/dataprocess/prepdata'
    args.region = 'hongik' #'jamsil' #'gangnam'
    args.cell_size = 150
    traffic_file = f'{args.data_dir}/{args.region}/traffic.h5'
    SE_file = f'{args.data_dir}/{args.region}/SE.txt'
    SEZ_npy_file = f'{args.data_dir}/{args.region}/lte_cell_poi_features_{args.cell_size}.npy'
    SAT_npy_file = f'{args.data_dir}/{args.region}/cell_satellite_features_{args.cell_size}.npy'

    lte_adj_mx_file = f'{args.data_dir}/{args.region}/road_lte_adj_mx_distance_{args.cell_size}.npy'
    coarse_df = pd.read_hdf(args.file_coarse)
    lte_file = f'{args.data_dir}/{args.region}/lte_cell_{args.cell_size}.h5'
    # lte_cell_geofile = f'{args.data_dir}/{args.region}/lte_cell_{args.cell_size}.geojson'
    graph_pkl_filename = f'{args.data_dir}/{args.region}/adj_mat.pkl'
    
    traf_df = pd.read_hdf(traffic_file)
    coarse_df = coarse_df[(coarse_df.index >= traf_df.index.min()) & (coarse_df.index < traf_df.index.max())]
    fine_df = coarse_df

    # fine_df = pd.read_hdf(args.file_fine)
    # coarse_df = fine_df

    # fine_zero = np.zeros((1, len(fine_df.columns)), dtype=np.float32)
    # fine_diff = np.concatenate((fine_zero, np.diff(fine_df.values, axis=0)), 0)
    # fine_diff_df = pd.DataFrame(fine_diff, columns=fine_df.columns)
    # fine_diff_df.index = fine_df.index
    # fine_df = fine_diff_df
    
    # traf_df = traf_df.iloc[:24*7*30]
    # coarse_df = coarse_df.iloc[:24*7*30]
    # fine_df = fine_df.iloc[:24*7*30]

    sid_list = []
    traf_vals = np.nan_to_num(traf_df.values.astype(np.float32))
    for sid, val in enumerate(np.sum(traf_vals == 0, 0)):
        if val < traf_vals.shape[0]*0.3:
            sid_list.append(traf_df.columns[sid])
            
    
    traf_df = traf_df[sid_list]
    print('traf_df', len(sid_list), traf_df.shape)

    extdata = dict()

    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    # Traffic_fill = fill_missing(traf_df).values.astype(np.float32)
    num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
    # train/val/test 
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    
    # train_fill = train_traf = Traffic_fill[: train_steps]
    # val_fill = Traffic_fill[train_steps : train_steps + val_steps]
    # test_fill = Traffic_fill[-test_steps :]


    np.random.seed(0)
    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    # test[np.random.rand(*test.shape) < 0.5] = 0

    # X, Y 
    # trainX, trainY = seq2instance_fill(train, train_fill, args.P, args.Q)
    # valX, valY = seq2instance_fill(val, val_fill, args.P, args.Q)
    # testX, testY = seq2instance_fill(test, test_fill, args.P, args.Q)
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    maxval = np.max(train); extdata['maxval'] = maxval 
    trainX = trainX / maxval
    valX = valX / maxval
    testX = testX / maxval
    np.save(f'{args.test_dir}/label_zeroed.npy', testY)




    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]
    _, trainY = seq2instance(train, args.P, args.Q)
    _, valY = seq2instance(val, args.P, args.Q)
    _, testY = seq2instance(test, args.P, args.Q)
    # np.random.seed(0)
    # trainX[np.random.rand(*trainX.shape) < 0.3] = 0
    # valX[np.random.rand(*valX.shape) < 0.3] = 0

    # testX[np.random.rand(*testX.shape) < 0.15] = 0
    np.save(f'{args.test_dir}/label_original.npy', testY)


    

    
    # print(train_traf.shape)
    # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    # # cr = int(args.cnn_size//2)
    # for i in range(num_sensors):
    #     adj_prcc[i, i] = 1
    #     for j in range(i+1, num_sensors):
    #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
    # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

    # coarse train/val/test
    # CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    # CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    # extdata['CH'] = CH; extdata['CW'] = CW
    # CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

    CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    extdata['CH'] = CH; extdata['CW'] = CW
    CoarseLTE = CoarseLTE.reshape(-1, CH, CW)

    # oLTE = CoarseLTE
    # lte_zeros =  np.zeros(CoarseLTE[0].shape)
    # lte_diff = np.diff(CoarseLTE, axis=0)
    # lte_diff0 = np.concatenate(([lte_zeros], lte_diff), 0)
    # CoarseLTE = lte_diff0
    # CoarseLTE = np.stack((oLTE, lte_diff0), -1)

    
    train = CoarseLTE[: train_steps]
    val = CoarseLTE[train_steps : train_steps + val_steps]
    test = CoarseLTE[-test_steps :]


    # X, Y 
    trainZC, _ = seq2instance(train, args.P, args.Q)
    valZC, _ = seq2instance(val, args.P, args.Q)
    testZC, _ = seq2instance(test, args.P, args.Q)
    # normalization
    # maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
    # trainZC = trainZC / maxvalZC
    # valZC = valZC / maxvalZC
    # testZC = testZC / maxvalZC
    meanZC = np.mean(train); extdata['meanZC'] = meanZC = 0
    stdZC = np.std(train); extdata['stdZC'] = stdZC
    trainZC = (trainZC - meanZC) / stdZC
    valZC = (valZC - meanZC) / stdZC
    testZC = (testZC - meanZC) / stdZC

    
    # fine train/val/test
    FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))

    FineLTE = np.diff(FineLTE, axis=0)
    FineLTE = np.sign(FineLTE) * np.log10(np.abs(FineLTE)+1)

    # FineLTE = np.concatenate((np.zeros((0, FineLTE.shape[1])), np.diff(FineLTE, axis=0)), 0)
    FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
    extdata['FH'] = FH; extdata['FW'] = FW
    FineLTE = FineLTE.reshape(-1, FH, FW)
    
    train = FineLTE[: train_steps]
    val = FineLTE[train_steps : train_steps + val_steps]
    test = FineLTE[-test_steps :]

    # X, Y 
    trainZF, _ = seq2instance(train, args.P, args.Q)
    valZF, _ = seq2instance(val, args.P, args.Q)
    testZF, _ = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZF = np.max(train); extdata['maxvalZF'] = maxvalZF = 0
    trainZF = trainZF / maxvalZF
    valZF = valZF / maxvalZF
    testZF = testZF / maxvalZF

    
    # temporal embedding 
    Time = traf_df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (args.time_slot*60) #// Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


    return (trainX, trainZC, trainZF, trainTE, trainY, 
            valX, valZC, valZF, valTE, valY, 
            testX, testZC, testZF, testTE, testY, extdata)
