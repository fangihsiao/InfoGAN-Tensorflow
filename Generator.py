import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.slim as slim


def generator(inputs, params):
    with slim.arg_scope([layers.fully_connected, layers.conv2d_transpose],
                        activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
                        normalizer_params={'updates_collections': None, 'is_training': params.is_training, 'decay': 0.9}):
        with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
            net = layers.fully_connected(inputs, 1024)
            net = layers.fully_connected(net, 7*7*128)
            net = tf.reshape(net, [-1, 7, 7, 128])
            net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
            net = layers.conv2d_transpose(net, 1, [4, 4], stride=2, normalizer_fn=None, activation_fn=tf.nn.sigmoid)

    return net
