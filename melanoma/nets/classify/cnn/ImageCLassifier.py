import os
import sys

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.examples.tutorials.mnist import input_data
from tqdm import tqdm

from melanoma.constants.Constants import HEIGHT, WIDTH, CHANNELS, CLASS_NUMBER, DROPOUT_KEEP_PROB, MODEL_NAME, \
    EPOCH_NUMBER, BATCH_SIZE, MODEL_DIR, CHECKPOINTS, MODEL_SUMMARY_DIR, LEARNING_RATE
from melanoma.nets.BaseNetwork import BaseNetwork
from melanoma.nets.classify.CnnAutoencoderConfig import cnn_autoencoder_config

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class MnistCnnClassifier(BaseNetwork):
    def __init__(self, configuration):
        # Configuration for classifier
        super().__init__(configuration)

        self.input_x = None
        self.labels = None
        self.keep_prob = None
        self.accuracy = None
        self.learning_rate = self._conf[LEARNING_RATE]

        self.logits = None

        self.predictions = None

    def build_graph(self):
        with tf.name_scope('inputs'):
            self.input_x = tf.placeholder(tf.float32,
                                          shape=[None, self._conf[HEIGHT], self._conf[WIDTH], self._conf[CHANNELS]],
                                          name='input_features')
            self.labels = tf.placeholder(tf.int32, shape=[None, self._conf[CLASS_NUMBER]], name='true_labels')
            self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        with tf.name_scope("conv_1"):
            conv_1 = tf.layers.conv2d(inputs=self.input_x,
                                      filters=32,
                                      kernel_size=[5, 5],
                                      strides=(1, 1),
                                      activation=tf.nn.relu)
            pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(1, 1))

        with tf.name_scope("conv_2"):
            conv_2 = tf.layers.conv2d(inputs=pool_1,
                                      filters=64,
                                      kernel_size=[3, 3],
                                      strides=(1, 1),
                                      activation=tf.nn.relu)
            pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(1, 1))

        with tf.name_scope("flatten"):
            flatten_size = np.prod(pool_2.shape[1:])
            flatten_input = tf.reshape(pool_2, [-1, flatten_size])

        with tf.name_scope("output"):
            fc_1 = tf.layers.dense(flatten_input, 1024, activation=tf.nn.relu)
            fc_1 = tf.nn.dropout(fc_1, self.keep_prob)
            self.logits = tf.layers.dense(fc_1, self._conf[CLASS_NUMBER], name='logits')
            self.predictions = tf.nn.softmax(self.logits, axis=1, name='predictions')

        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
            tf.summary.scalar('training_accuracy', self.accuracy)

        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
            tf.summary.scalar('loss', self.loss)

        self.saver = tf.train.Saver(tf.global_variables())

    def build_optimizer(self):
        with tf.name_scope('training_operations'):
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            optimizer = tf.train.AdamOptimizer(self.learning_rate, name='Adam')
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, session: tf.Session, input_images: np.array, labels: np.array):
        feed_dict = {
            # (batch_size, height, width, nr_channels)
            self.input_x: input_images,

            # One hot vector (batch_size, class_number)
            self.labels: labels,

            # Probability of keep neurons
            self.keep_prob: self._conf[DROPOUT_KEEP_PROB]
        }

        _, step, loss, merged = session.run([self.train_op, self.global_step, self.loss, self.summary], feed_dict)

        return step, loss, merged

    def validate(self, session: tf.Session, input_images: np.array):
        feed_dict = {
            # (batch_size, height, width, nr_channels)
            self.input_x: input_images,

            # No dropout in validate mode.
            self.keep_prob: 1.0
        }

        step, logits = session.run([self.global_step, self.logits], feed_dict)
        return step, logits

    def predict(self, session: tf.Session, input_images: np.array):
        feed_dict = {
            # (batch_size, height, width, nr_channels)
            self.input_x: input_images,

            # No dropout in predict mode.
            self.keep_prob: 1.0
        }

        return session.run([self.predictions], feed_dict=feed_dict)[0]

    def load(self, session: tf.Session):
        super().load(session)
        self.input_x = session.graph.get_operation_by_name('inputs/input_features').outputs[0]
        self.keep_prob = session.graph.get_operation_by_name('inputs/keep_prob').outputs[0]
        self.predictions = session.graph.get_operation_by_name('output/predictions').outputs[0]


def train_model(configuration):
    with tf.Graph().as_default():

        if not os.path.exists(configuration[MODEL_DIR]):
            os.mkdir(configuration[MODEL_DIR])

        session = tf.Session()

        with session.as_default():
            model = MnistCnnClassifier(configuration)
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

            summary_folder = os.path.join(configuration[MODEL_SUMMARY_DIR], configuration[MODEL_NAME])
            summary_train = os.path.join(summary_folder, 'training')
            writer_train = tf.summary.FileWriter(summary_train, session.graph)

            print('Start training')
            for epoch in tqdm(range(configuration[EPOCH_NUMBER])):
                batch_xs, batch_ys = mnist.train.next_batch(configuration[BATCH_SIZE])
                batch_xs = batch_xs.reshape(-1, configuration[HEIGHT], configuration[WIDTH], configuration[CHANNELS])

                step, loss, summary = model.train(session, batch_xs, batch_ys)

                if epoch % 1000 == 0:
                    checkpoint_path = os.path.join(configuration[MODEL_DIR],
                                                   "{}.ckpt".format(configuration[MODEL_NAME]))
                    model.saver.save(session, checkpoint_path, global_step=step)
                    writer_train.add_summary(summary, step)

            model.save(session)


def predict(configuration):
    with tf.Graph().as_default():
        if not os.path.exists(configuration[MODEL_DIR]):
            sys.exit('{} model does not exist'.format(configuration[MODEL_DIR]))

        session = tf.Session()

        with session.as_default():
            model = MnistCnnClassifier(configuration)
            model.load(session)

            batch_xs, batch_ys = mnist.test.next_batch(configuration[BATCH_SIZE])
            batch_xs = batch_xs.reshape(-1, configuration[HEIGHT], configuration[WIDTH], configuration[CHANNELS])

            predictions = model.predict(session, batch_xs)
            predictions = np.argmax(predictions, axis=1)
            golden = np.argmax(batch_ys, axis=1)

            print(accuracy_score(predictions, golden))


if __name__ == "__main__":
    predict(cnn_autoencoder_config)
