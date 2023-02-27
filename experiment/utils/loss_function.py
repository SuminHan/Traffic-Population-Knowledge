import numpy as np
import tensorflow as tf



# def custom_mae_loss(label, pred):
#     mask = tf.not_equal(label, 0)
#     mask = tf.cast(mask, tf.float32)
#     mask /= tf.reduce_mean(mask)
#     mask = tf.compat.v2.where(
#         condition = tf.math.is_nan(mask), x = 0., y = mask)
#     loss = tf.abs(tf.subtract(pred, label))
#     loss *= mask
#     loss = tf.compat.v2.where(
#         condition = tf.math.is_nan(loss), x = 0., y = loss)
#     loss = tf.reduce_mean(loss)
#     return loss


# def custom_rmse_loss(label, pred):
#     mask = tf.not_equal(label, 0)
#     mask = tf.cast(mask, tf.float32)
#     mask /= tf.reduce_mean(mask)
#     mask = tf.compat.v2.where(
#         condition = tf.math.is_nan(mask), x = 0., y = mask)
#     loss = tf.square(tf.subtract(pred, label))
#     loss *= mask
#     loss = tf.compat.v2.where(
#         condition = tf.math.is_nan(loss), x = 0., y = loss)
#     loss = tf.sqrt(tf.reduce_mean(loss))
#     return loss


# def custom_mape_loss(label, pred):
#     mask = tf.not_equal(label, 0)
#     mask = tf.cast(mask, tf.float32)
#     mask /= tf.reduce_mean(mask)
#     mask = tf.compat.v2.where(
#         condition = tf.math.is_nan(mask), x = 0., y = mask)
#     # loss = tf.abs(tf.subtract(pred, label))
#     loss = 100 * tf.abs((pred - label) / (label+1e-3))
#     loss *= mask
#     loss = tf.compat.v2.where(
#         condition = tf.math.is_nan(loss), x = 0., y = loss)
#     loss = tf.reduce_mean(loss)
#     return loss



def compute_pairwise_distances(x, y):
  if not len(x.get_shape()) == len(y.get_shape()) == 2:
    raise ValueError('Both inputs should be matrices.')

  if x.get_shape().as_list()[1] != y.get_shape().as_list()[1]:
    raise ValueError('The number of features should be the same.')

  norm = lambda x: tf.reduce_sum(tf.square(x), 1)
  return tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))

def gaussian_kernel_matrix(x, y, sigmas):
    beta = 1. / (2. * (tf.expand_dims(sigmas, 1)))
    dist = compute_pairwise_distances(x, y)
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    return tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))

def maximum_mean_discrepancy(x, y, kernel=gaussian_kernel_matrix):
    # \E{ K(x, x) } + \E{ K(y, y) } - 2 \E{ K(x, y) }
    cost = tf.reduce_mean(kernel(x, x))
    cost += tf.reduce_mean(kernel(y, y))
    cost -= 2 * tf.reduce_mean(kernel(x, y))

    # We do not allow the loss to become negative.
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def mmd_loss(source_samples, target_samples, weight, scope=None):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight
    # assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
    # with tf.control_dependencies([assert_op]):
        # tag = 'MMD Loss'
        # if scope:
            # tag = scope + tag
    # tf.summary.scalar(tag, loss_value)
    # tf.losses.add_loss(loss_value)n 

    return loss_value


def my_mmd_loss(source_samples, target_samples, weight, scope=None):
    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    gaussian_kernel = partial(
        gaussian_kernel_matrix, sigmas=tf.constant(sigmas))

    
    loss_value = maximum_mean_discrepancy(
        source_samples, target_samples, kernel=gaussian_kernel)
    loss_value = tf.maximum(1e-4, loss_value) * weight
    # assert_op = tf.Assert(tf.is_finite(loss_value), [loss_value])
    # with tf.control_dependencies([assert_op]):
        # tag = 'MMD Loss'
        # if scope:
            # tag = scope + tag
    # tf.summary.scalar(tag, loss_value)
    # tf.losses.add_loss(loss_value)n 

    return loss_value
