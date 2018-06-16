import tensorflow as tf

from melanoma.data_preparator.read_data import iterator


class MelanomaClassifier:
    def __init__(self, config=None):
        # TODO config will be necessary later.
        self._config = config

        self.input_x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='input_features')
        self.labels = tf.placeholder(tf.int32, shape=[None, 10], name='true_labels')
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        self._get_net()

    def _get_net(self):
        conv_1 = tf.layers.conv2d(inputs=self.input_x,
                                  filters=64,
                                  kernel_size=[5, 5],
                                  strides=(1, 1),
                                  activation=tf.nn.relu)

        pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(1, 1))

        conv_2 = tf.layers.conv2d(inputs=pool_1,
                                  filters=32,
                                  kernel_size=[5, 5],
                                  strides=(1, 1),
                                  activation=tf.nn.relu)

        pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(1, 1))

        flatten_input = tf.contrib.layers.flatten(pool_2)

        fc_1 = tf.layers.dense(flatten_input, 64, activation=tf.nn.relu)
        fc_1 = tf.nn.dropout(fc_1, self.keep_prob)

        self.logits = tf.layers.dense(fc_1, 10)
        self.predictions = tf.nn.softmax(self.logits)

        return self.logits

    def _initialize_metrics(self):
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_accuracy(self):
        return self._accuracy

    def train(self):

        loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)

        self._initialize_metrics()

        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(2000):
                # XTrain, yTrain = iterator.get_next_batch()

                batch_xs, batch_ys = mnist.train.next_batch(100)
                batch_xs = batch_xs.reshape(-1, 28, 28, 1)

                _, loss_value = sess.run([optimizer, loss_function],
                                         feed_dict={self.input_x: batch_xs, self.labels: batch_ys,
                                                    self.keep_prob: 0.5})

                if i % 10 == 0:
                    test_xs, test_ys = mnist.test.next_batch(3000)
                    test_xs = test_xs.reshape(-1,28,28,1)
                    print('Current Accuracy = {}'.format(
                        sess.run(self._accuracy,
                                 feed_dict={self.input_x: test_xs, self.labels: test_ys, self.keep_prob: 1.0})))




if __name__ == "__main__":
    net = MelanomaClassifier()
    net.train()
