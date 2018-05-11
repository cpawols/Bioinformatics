import tensorflow as tf
import numpy as np


def get_melanoma_cnn(features: np.array):
    # TODO Add assertion about dimension!

    conv_1 = tf.layers.conv2d(inputs=features,
                              filters=32,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              activation=tf.nn.relu)

    pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(1, 1))

    conv_2 = tf.layers.conv2d(inputs=pool_1,
                              filters=64,
                              kernel_size=[3, 3],
                              strides=(1, 1),
                              activation=tf.nn.relu)

    pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(1, 1))
    flatten_input = tf.contrib.layers.flatten(pool_2)

    fc_1 = tf.layers.dense(flatten_input, 512, activation=tf.nn.relu)

    fc_2 = tf.layers.dense(fc_1, 2, activation=tf.nn.relu)

    return fc_2


epoch_number = 100

input_x = tf.placeholder(tf.int32, shape=[None, 32, 32], name='input_features')
labels = tf.placeholder(tf.int32, shape=[None, 2], name='true_labels')

logits_predictions = get_melanoma_cnn(None)

loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_predictions, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)

with tf.Session() as sess:
    for i in range(epoch_number):
        current_batch = None
        current_batch_labels = None
        _, loss_value = sess.run([optimizer, loss_function],
                                 feed_dict={input_x: current_batch, labels: current_batch_labels})
