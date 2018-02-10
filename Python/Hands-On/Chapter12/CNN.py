import numpy as np
import tensorflow as tf


X = tf.placeholder(tf.float32, shape = (None, 784), name = 'X')
y = tf.placeholder(tf.int64, shape = (None), name = 'y')

X_ = tf.reshape(X, shape = [-1, 28, 28, 1])

with tf.name_scope('CNN'):
    C1 = tf.layers.conv2d(X_, filters = 6, kernel_size = 5, strides = [1, 1], padding = 'SAME')
    C1_ = tf.tanh(C1)
    S2 = tf.nn.avg_pool(C1_, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    S2_ = tf.tanh(S2)
    C3 = tf.layers.conv2d(S2_, filters = 16, kernel_size = 5, strides = 1, padding = 'VALID')
    C3_ = tf.tanh(C3)
    S4 = tf.nn.avg_pool(C3_, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'VALID')
    S4_ = tf.tanh(S4)
    C5 = tf.layers.conv2d(S4_, filters = 120, kernel_size = 5, strides = [1, 1], padding = 'VALID')
    C5_ = tf.tanh(C5)

with tf.name_scope('FC'):
    F6 = tf.layers.dense(C5_, 84, activation = tf.tanh)
    logits = tf.layers.dense(F6, 10)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    loss = tf.reduce_mean(xentropy, name = 'loss')

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    logits_ = logits[:, 0, 0, :]
    correct = tf.nn.in_top_k(logits_, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/")

X_test = mnist.test.images
y_test = mnist.test.labels

n_epochs = 150
batch_size = 50

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict = {X: X_batch, y: y_batch})
        acc_test = accuracy.eval(feed_dict = {X: X_test, y: y_test})
        print(epoch, 'Train accuracy:', acc_train, 'Test accuracy:', acc_test)
    save_path = saver.save(sess, './my_model_final.ckpt')
