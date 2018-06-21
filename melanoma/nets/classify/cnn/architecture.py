import tensorflow as tf

from melanoma.data_preparator.read_data import iterator


class MelanomaClassifier:
    EPOCH_NUMBER = 50
    BATCH_SIZE = 100

    IMAGE_WIDTH = 28
    IMAGE_HIGHT = 28
    CHANEL_NUMBER = 1
    CLASS_NUMBER = 10

    def __init__(self, model_dir, config=None):
        # TODO config will be necessary later.
        self._config = config
        self._model_dir = model_dir

        self.input_x = tf.placeholder(tf.float32, shape=[None, self.IMAGE_HIGHT, self.IMAGE_WIDTH, self.CHANEL_NUMBER],
                                      name='input_features')
        self.labels = tf.placeholder(tf.int32, shape=[None, self.CLASS_NUMBER], name='true_labels')
        self.keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

        self._get_net()

    def _get_net(self):

        with tf.name_scope("conv_layers"):
            conv_1 = tf.layers.conv2d(inputs=self.input_x,
                                      filters=32,
                                      kernel_size=[5, 5],
                                      strides=(1, 1),
                                      activation=tf.nn.relu)

            pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(1, 1))

            conv_2 = tf.layers.conv2d(inputs=pool_1,
                                      filters=64,
                                      kernel_size=[3, 3],
                                      strides=(1, 1),
                                      activation=tf.nn.relu)

            pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(1, 1))

        with tf.name_scope("flatten_layer"):
            flatten_input = tf.contrib.layers.flatten(pool_2)

        with tf.name_scope("prediction_layer"):
            fc_1 = tf.layers.dense(flatten_input, 1024, activation=tf.nn.relu)
            fc_1 = tf.nn.dropout(fc_1, self.keep_prob)

            self.logits = tf.layers.dense(fc_1, self.CLASS_NUMBER, name='predictions')
            print(self.logits)
            self.predictions = tf.nn.softmax(self.logits)

        return self.logits

    def _initialize_metrics(self):
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.labels, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def get_accuracy(self):
        return self._accuracy

    def train(self, auto_eval=True):
        loss_function = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss_function)

        self._initialize_metrics()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.EPOCH_NUMBER):
                batch_xs, batch_ys = mnist.train.next_batch(self.BATCH_SIZE)
                batch_xs = batch_xs.reshape(-1, 28, 28, 1)

                _, loss_value = sess.run([optimizer, loss_function],
                                         feed_dict={self.input_x: batch_xs, self.labels: batch_ys, self.keep_prob: 0.5})

            self.save(sess)

            if auto_eval:
                self.evaluate(sess)

    def evaluate(self, sess):
        test_xs, test_ys = mnist.test.next_batch(300)
        test_xs = test_xs.reshape(-1, 28, 28, 1)

        accuracy = sess.run(self._accuracy,
                            feed_dict={self.input_x: test_xs, self.labels: test_ys, self.keep_prob: 1.0})
        print("Test score = {}".format(accuracy * 100))

    def predict(self, inputs):
        self.load_model(self._model_dir)

    def save(self, session):
        saver = tf.train.Saver()
        saver.save(sess=session, save_path=self._model_dir)

    def load_model(self, model_dir):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            self.logits = graph.get_tensor_by_name('prediction_layer/predictions/BiasAdd:0')
            saver.restore(sess, tf.train.latest_checkpoint('/home/pawols/tmp_model/'))
            predictions = sess.run(self.logits, feed_dict={self.input_x:test_xs, self.keep_prob:1.0})
            print(predictions)


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

if __name__ == "__main__":
    path = '/home/pawols/tmp_model/mymodel'
    net = MelanomaClassifier(model_dir=path)
    test_xs, test_ys = mnist.test.next_batch(2)
    test_xs = test_xs.reshape(-1, 28, 28, 1)
    # net.train()
    net.load_model('/home/pawols/tmp_model/')
