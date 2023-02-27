import argparse
import numpy as np
from scipy.special import softmax
from scipy import stats

def row_normalize(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix

def convert_to_adj_mx(dist_mx, threshold=3000):
    adj_mx = np.zeros(dist_mx.shape)
    if threshold > 0:
        adj_mx[dist_mx > threshold] = 0
        adj_mx[dist_mx <= threshold] = 1
    
    return row_normalize(adj_mx)

def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: raise argparse.ArgumentTypeError('Boolean value expected.')
