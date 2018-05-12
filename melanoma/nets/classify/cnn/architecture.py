import tensorflow as tf
# from datastore import iterator


from tensorflow.examples.tutorials.mnist import input_data

from melanoma.data_preparator.read_data import iterator


def get_melanoma_cnn(features, keep_drop):
    conv_1 = tf.layers.conv2d(inputs=features,
                              filters=32,
                              kernel_size=[5, 5],
                              strides=(1, 1),
                              activation=tf.nn.relu)

    pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(1, 1))

    conv_2 = tf.layers.conv2d(inputs=pool_1,
                              filters=64,
                              kernel_size=[5, 5],
                              strides=(1, 1),
                              activation=tf.nn.relu)

    pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(1, 1))

    flatten_input = tf.contrib.layers.flatten(pool_2)

    fc_1 = tf.layers.dense(flatten_input, 64, activation=tf.nn.relu)
    fc_1 = tf.nn.dropout(fc_1, keep_drop)


    fc_2 = tf.layers.dense(fc_1, 2)

    return fc_2


input_x = tf.placeholder(tf.float32, shape=[None, 64, 64, 3], name='input_features')
labels = tf.placeholder(tf.int32, shape=[None, 2], name='true_labels')
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)

logits_predictions = get_melanoma_cnn(input_x, keep_prob)
predictions = tf.nn.softmax(logits_predictions)
correct_pred = tf.equal(tf.argmax(logits_predictions, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


loss_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_predictions, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss_function)


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    for i in range(600):
        XTrain, yTrain = iterator.get_next_batch()

        _, loss_value = sess.run([optimizer, loss_function],
                                 feed_dict={input_x: XTrain, labels: yTrain, keep_prob:0.75})

        if i % 30 == 0:
            print(loss_value)
            print('Current Accuracy = {}'.format(sess.run(accuracy, feed_dict={input_x:XTrain, labels:yTrain, keep_prob:1.0})))
        
    # ac = []
    # for i in range(5):
    #     q = mnist.test.next_batch(2000)
    #     imgs = q[0].reshape((-1,28,28,1))
    #     l = q[1]
    #
    #
    #     curr_acc  = sess.run(accuracy, feed_dict={input_x: imgs, labels: l, keep_prob:1.0})
    #     ac.append(curr_acc)
    #     print('Current Accuracy',curr_acc)
    #
    # print(sum(ac)/len(ac))

            # predictions = sess.run(pred, feed_dict={input_x: imgs})
            #
            # # AUC = roc_auc_score(ba[:,1], predictions[:,1])
            # #
            # # # print(predictions.shape)
            # predictions2 = np.argmax(predictions, axis=1)
            # ACC= np.mean(predictions2 == np.argmax(batch[1], axis=1)[:,np.newaxis])
            # if i % 10 == 0:
            #     print(loss_value)
            #     # print(np.unique(predictions2))
            #     print("ACC = {}, AUC = {}".format(ACC,0))
