import tensorflow as tf
import numpy as np

from melanoma.data_preparator.read_data import iterator

batch_x, batch_y = iterator.get_next_batch()
batch_x = batch_x[:2]

class CnnAutoencoder:
    EPOCH_NUMBER = 1000
    BATCH_SIZE = 2
    LEARNING_RATE = 0.0001

    IMAGE_WIDTH = 200
    IMAGE_HIGHT = 200
    CHANEL_NUMBER = 3
    CLASS_NUMBER = 10

    def __init__(self, model_path):
        self._model_path = model_path
        self._initialize_placeholders()

        self._create_encoder()

    def _initialize_placeholders(self):
        self.input_x = tf.placeholder(tf.float32, shape=[None, self.IMAGE_WIDTH, self.IMAGE_HIGHT, self.CHANEL_NUMBER])
        self.target = tf.placeholder(tf.float32, shape=[None, self.IMAGE_WIDTH, self.IMAGE_HIGHT, self.CHANEL_NUMBER])

        self.droput_keep = tf.placeholder(dtype=tf.float32)

    def _create_encoder(self):
        layer_1 = 16
        layer_2 = 8
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
            print(self.encoded.get_shape())

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

            self.decoded = tf.layers.conv2d(decoder, filters=self.CHANEL_NUMBER, kernel_size=(5, 5), padding='same',
                                            activation=tf.nn.sigmoid, name='image')
            print(decoder.get_shape())


    def unsaple_2d(self, image, size):
        """
        Operation which produced image `size` times bigger.

        If input image has size (10,10) then output image wil have (10*size, 10*size) shape.
        """
        width = int(image.get_shape()[1] * size)
        hight = int(image.get_shape()[2] * size)
        print(image.get_shape())
        return tf.image.resize_nearest_neighbor(image, (hight, width))

    def train(self):
        loss_function = tf.reduce_sum(tf.square(self.input_x - self.decoded))
        tf.summary.scalar('loss', loss_function)
        merged_summary_op = tf.summary.merge_all()

        summary_writer = tf.summary.FileWriter('/home/pawols/tmp_model/autoencoder/logs', graph=tf.get_default_graph())

        optimizer = tf.train.AdamOptimizer(learning_rate=self.LEARNING_RATE).minimize(loss_function)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(self.EPOCH_NUMBER):
                _, loss_value, summary = sess.run([optimizer, loss_function, merged_summary_op],
                                         feed_dict={self.input_x: batch_x, self.target: batch_x})
                summary_writer.add_summary(summary, i)
                if i%10 == 0:
                    print(loss_value)


            self.save(sess)
            # images = sess.run(self.decoded, {self.input_x: batch_xs[:1]})
            #
            # import matplotlib.pyplot as plt
            # first_image = np.array(images, dtype='float')
            # pixels = first_image.reshape((28, 28))
            # plt.imshow(pixels, cmap='gray')
            # plt.show()

            # first_image = np.array(batch_xs[:1], dtype='float')
            # pixels = first_image.reshape((28, 28))
            # plt.imshow(pixels, cmap='gray')
            # plt.show()


    def save(self, session):
        saver = tf.train.Saver()
        saver.save(sess=session, save_path=self._model_path)

    def predict(self, images, model_dir):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, tf.train.latest_checkpoint(model_dir))
            graph = tf.get_default_graph()
            decoded = graph.get_tensor_by_name('decoder/image/Sigmoid:0')


            reconstructed_images = sess.run(decoded, {self.input_x:images})

            return reconstructed_images



if __name__ == "__main__":
    model_name = '/home/pawols/tmp_model/autoencoder/my_autoencoder'
    model_dir = '/home/pawols/tmp_model/autoencoder/'
    clf = CnnAutoencoder(model_name)
    clf.train()
    # reconstructed = clf.predict(batch_x, model_dir)
    # import matplotlib.pyplot as plt
    # #
    # # print(batch_x[0].shape)
    # plt.imshow(reconstructed[0])
    # plt.show()


