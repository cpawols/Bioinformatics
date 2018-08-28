"""
Module contains CNN based autoencoder for images.
"""

import os
import sys

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm

from melanoma.constants.constants import (HEIGHT, WIDTH, CHANNELS, LEARNING_RATE, MODEL_DIR, EPOCH_NUMBER, BATCH_SIZE,
                                          CHECKPOINTS, MODEL_NAME)
from melanoma.data_preparator.read_data import iterator
from melanoma.nets.BaseNetwork import BaseNetwork
from melanoma.nets.config.CnnAutoencoderConfig import cnn_autoencoder_config


class CnnAutoencoder(BaseNetwork):
    """
    Autoencoder for color images.

    NOTE: Current architecture is prepared for images 200 x 200 pixels.
    In case changing image size you have to change number of filters in conv layers.
    """

    def __init__(self, configuration: dict):
        super().__init__(configuration)

        # Input to network.
        self.input_x = None
        self.target = None
        self.droput_keep = None

        self.encoded = None
        self.decoded = None

    def build_graph(self):
        """
        Builds network graph. Network contains 3 convolutional layers for encoder and 3 convolutional layers
        for decoder.
        """
        with tf.name_scope('inputs'):
            # (batch, height, width, channels)
            self.input_x = tf.placeholder(tf.float32,
                                          shape=[None, self._conf[HEIGHT], self._conf[WIDTH], self._conf[CHANNELS]],
                                          name='features')
            # (batch, height, width, channels)
            self.target = tf.placeholder(tf.float32,
                                         shape=[None, self._conf[HEIGHT], self._conf[WIDTH], self._conf[CHANNELS]])
            # droput keep probability
            self.droput_keep = tf.placeholder(dtype=tf.float32)

        # Number of filters for each of convolution layer.
        layer_1 = 32
        layer_2 = 16
        layer_3 = 8

        with tf.name_scope("encoder"):
            # (batch_size, width, hight, 32)
            conv_1 = tf.layers.conv2d(self.input_x, filters=layer_1, kernel_size=(5, 5), padding='same',
                                      activation=tf.nn.relu)
            pool_1 = tf.layers.max_pooling2d(conv_1, (2, 2), 2, padding='same')

            # (batch_size, width/2, hight/2, 8)
            conv_2 = tf.layers.conv2d(pool_1, filters=layer_2, kernel_size=(3, 3), padding='same',
                                      activation=tf.nn.relu)
            pool_2 = tf.layers.max_pooling2d(conv_2, (2, 2), 2, padding='same')

            # (batch_size, width/4, hight/4, 8)
            conv_3 = tf.layers.conv2d(pool_2, filters=layer_3, kernel_size=(3, 3), padding='same',
                                      activation=tf.nn.relu)
            self.encoded = tf.layers.max_pooling2d(conv_3, (2, 2), 2, padding='same', name='a')

        with tf.name_scope("decoder"):
            decoder = tf.layers.conv2d(self.encoded, filters=layer_3, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
            decoded = self.unsaple_2d(decoder, 2)

            decoder = tf.layers.conv2d(decoded, filters=layer_2, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
            decoder = self.unsaple_2d(decoder, 2)

            decoder = tf.layers.conv2d(decoder, filters=layer_1, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
            decoder = self.unsaple_2d(decoder, 2)

            self.decoded = tf.layers.conv2d(decoder, filters=self._conf[CHANNELS], kernel_size=(5, 5), padding='same',
                                            activation=tf.nn.sigmoid, name='decoded')

        with tf.name_scope('loss'):
            # Loss function - square error between original picture and reconstructed
            self.loss = tf.reduce_sum(tf.square(self.input_x - self.decoded))
            tf.summary.scalar('loss', self.loss)
        self.saver = tf.train.Saver(tf.global_variables())

    def build_optimizer(self):
        with tf.name_scope("optimizer"):
            self.global_step = tf.Variable(0, trainable=False)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self._conf[LEARNING_RATE]) \
                .minimize(self.loss, global_step=self.global_step)
        self.saver = tf.train.Saver(tf.global_variables())

    @staticmethod
    def unsaple_2d(image: tf.Variable, size: int):
        """
        Operation which produced image `size` times bigger.

        If input image has size (10,10) then output image wil have (10*size, 10*size) shape.
        """
        width = int(image.get_shape()[1] * size)
        height = int(image.get_shape()[2] * size)
        return tf.image.resize_nearest_neighbor(image, (height, width))

    def train(self, session: tf.Session, images: np.array):
        feed_dict = {
            self.input_x: images,

            self.target: images
        }

        _, summary, step, loss = session.run([self.optimizer, self.summary, self.global_step, self.loss],
                                             feed_dict=feed_dict)

        return summary, step, loss

    def predict(self, session: tf.Session, images: np.array):
        feed_dict = {
            self.input_x: images
        }

        return session.run(self.decoded, feed_dict=feed_dict)

    def load(self, session: tf.Session):
        super().load(session)
        self.input_x = session.graph.get_operation_by_name('inputs/features').outputs[0]
        self.decoded = session.graph.get_tensor_by_name('decoder/decoded/Sigmoid:0')


def train_model(configuration: dict):
    with tf.Graph().as_default():

        if not os.path.exists(configuration[MODEL_DIR]):
            os.mkdir(configuration[MODEL_DIR])

        session = tf.Session()

        with session.as_default():
            model = CnnAutoencoder(configuration)
            model.build_graph()
            model.build_optimizer()
            model.merge_all()

            checkpoint = tf.train.get_checkpoint_state(configuration[MODEL_DIR])
            if (checkpoint and tf.train.checkpoint_exists(checkpoint.model_checkpoint_path) and
                    configuration[CHECKPOINTS]):
                print("Reading model parameters from {}".format(checkpoint.model_checkpoint_path))
                model.saver.restore(session, checkpoint.model_checkpoint_path)
            else:
                print("Creating model with new parameters")
                tf.global_variables_initializer().run()

            summary_folder = os.path.join(configuration[MODEL_DIR], 'autoencoder')
            summary_train = os.path.join(summary_folder, 'training')
            writer_train = tf.summary.FileWriter(summary_train, session.graph)

            print('Start training')
            for epoch in tqdm(range(configuration[EPOCH_NUMBER])):
                batch_x, batch_y = iterator.get_next_batch()
                batch_x = batch_x[:2]

                summary, step, loss = model.train(session, batch_x)

                if epoch % 1000 == 0:
                    checkpoint_path = os.path.join(configuration[MODEL_DIR],
                                                   "{}.ckpt".format(configuration[MODEL_NAME]))
                    model.saver.save(session, checkpoint_path, global_step=step)
                    writer_train.add_summary(summary, step)

                if epoch % 500 == 0:
                    print(loss)

            model.save(session)


def predict(configuration: dict):
    with tf.Graph().as_default():
        if not os.path.exists(configuration[MODEL_DIR]):
            sys.exit('{} model does not exist'.format(configuration[MODEL_DIR]))

        session = tf.Session()

        with session.as_default():
            model = CnnAutoencoder(configuration)
            model.load(session)

            batch_xs, batch_ys = iterator.get_next_batch()
            batch_xs = batch_xs.reshape(-1, configuration[HEIGHT], configuration[WIDTH], configuration[CHANNELS])

            reconstructed_images = model.predict(session, batch_xs)
            return reconstructed_images


def show_image(image: np.array):
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    image = predict(cnn_autoencoder_config)
    batch_xs, batch_ys = iterator.get_next_batch()
    show_image(image[0])
