import os
import sys

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from melanoma.constants.Constants import HEIGHT, WIDTH, CHANNELS, LEARNING_RATE, MODEL_DIR, EPOCH_NUMBER, BATCH_SIZE, \
    CHECKPOINTS
from melanoma.data_preparator.read_data import iterator
from melanoma.nets.BaseNetwork import BaseNetwork
from melanoma.nets.classify.CnnAutoencoderConfig import cnn_autoencoder_config
from melanoma.nets.classify.CnnConfig import model_config


class CnnAutoencoder(BaseNetwork):
    def __init__(self, configuration):
        super().__init__(configuration)

        self.input_x = None
        self.target = None
        self.droput_keep = None

        self.encoded = None
        self.decoded = None

    def build_graph(self):
        with tf.name_scope('inputs'):
            self.input_x = tf.placeholder(tf.float32,
                                          shape=[None, self._conf[HEIGHT], self._conf[WIDTH], self._conf[CHANNELS]],
                                          name='features')
            self.target = tf.placeholder(tf.float32,
                                         shape=[None, self._conf[HEIGHT], self._conf[WIDTH], self._conf[CHANNELS]])

            self.droput_keep = tf.placeholder(dtype=tf.float32)

        layer_1 = 32
        layer_2 = 16
        layer_3 = 8

        with tf.name_scope("encoder"):
            # (batch_size, width, hight, 16)
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
            self.encoded = tf.layers.max_pooling2d(conv_3, (2, 2), 2, padding='same')
            print(self.encoded.shape)

        with tf.name_scope("decoder"):
            decoder = tf.layers.conv2d(self.encoded, filters=layer_3, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
            decoded = self.unsaple_2d(decoder, 2)

            decoder = tf.layers.conv2d(decoded, filters=layer_2, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
            decoder = self.unsaple_2d(decoder, 2)
            print(decoder.get_shape())

            decoder = tf.layers.conv2d(decoder, filters=layer_1, kernel_size=(3, 3), padding='same',
                                       activation=tf.nn.relu)
            decoder = self.unsaple_2d(decoder, 2)
            print(decoder.get_shape())

            self.decoded = tf.layers.conv2d(decoder, filters=self._conf[CHANNELS], kernel_size=(5, 5), padding='same',
                                            activation=tf.nn.sigmoid, name='decoded_image')
            print(decoder.get_shape())

        with tf.name_scope('loss'):
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
    def unsaple_2d(image, size):
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

        _, summary, step = session.run([self.optimizer, self.summary, self.global_step], feed_dict=feed_dict)

        return summary, step

    def predict(self, session: tf.Session, images: np.array):
        feed_dict = {
            self.input_x: images
        }

        return session.run(self.decoded, feed_dict=feed_dict)

    def load(self, session: tf.Session):
        super().load()
        self.input_x = session.graph.get_operation_by_name('inputs/features').outputs[0]
        self.decoded = session.graph.get_operation_by_name('decoder/decoded_image').outputs[0]


def train_model(configuration):
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

                
                summary, step = model.train(session, batch_x)

                if epoch % 1000 == 0:
                    checkpoint_path = os.path.join(configuration[MODEL_DIR], "autoencoder.ckpt")
                    model.saver.save(session, checkpoint_path, global_step=step)
                    writer_train.add_summary(summary, step)

            model.save(session)


def predict(configuration):
    with tf.Graph().as_default():
        if not os.path.exists(configuration[MODEL_DIR]):
            sys.exit('{} model does not exist'.format(configuration[MODEL_DIR]))

        session = tf.Session()

        with session.as_default():
            model = CnnAutoencoder(configuration)
            model.load(session)

            batch_xs, batch_ys = iterator(configuration[BATCH_SIZE])
            batch_xs = batch_xs.reshape(-1, configuration[HEIGHT], configuration[WIDTH], configuration[CHANNELS])

            reconstructed_images = model.predict(session, batch_xs)
            return reconstructed_images


if __name__ == "__main__":
    train_model(cnn_autoencoder_config)
