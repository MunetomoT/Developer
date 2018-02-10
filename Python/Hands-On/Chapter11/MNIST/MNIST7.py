import tensorflow as tf
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data
import pdb

mnist = input_data.read_data_sets("/tmp/data/")
X_train = mnist.train.images
X_test = mnist.test.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")

def reset_graph():
    tf.reset_default_graph()

n_inputs = 28 * 28
n_hidden1 = 1000
n_hidden2 = 1000
n_hidden3 = 100
n_outputs = 10

reset_graph()

X = tf.placeholder(tf.float32, shape = (None, n_inputs), name = "X")
y = tf.placeholder(tf.int64, shape = (None), name = 'y')

my_dense_layer = partial(tf.layers.dense, activation = tf.nn.relu, kenel_regularozer = tf.contrib.layers.l1_regularizer(scale))

with tf.name_scope('dnn'):
    hidden1 = my_dense_layer(X, n_hidden1, name = 'hidden1')
    hidden2 = my_dense_layer(hidden1, n_hidden2, name = 'hidden2')
#   hidden3 = tf.layers.dense(hidden2, n_hidden3, name = 'hidden3', activation = tf.nn.selu)
    logits = my_dense_layer(hidden2, n_outputs, name = 'outputs', activation = None)

with tf.name_scope('loss'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits)
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    xentropy = tf.add_n(reg_losses + xentropy)
    loss = tf.reduce_mean(xentropy, name = 'loss')
    loss_summary = tf.summary.scalar('log_loss', loss)
    loss_summary_train = tf.summary.scalar('log_loss_train', loss)

learning_rate = 0.01

with tf.name_scope('train'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope('eval'):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32)
                             )
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    accuracy_summary_train = tf.summary.scalar('accuracy_train', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

from datetime import datetime

def log_dir(prefix=''):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

logdir = log_dir("mnist_dnn")

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

X_valid = mnist.validation.images
y_valid = mnist.validation.labels

m, n = X_train.shape

n_epochs = 10001
batch_size = 100
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "/tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, 'rb') as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for iteration in range(mnist.train.num_examples // batch_size):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            sess.run(training_op, feed_dict = {X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict = {X: X_valid, y: y_valid})
        accuracy_training_str, loss_training_str = sess.run([accuracy_summary_train, loss_summary_train], feed_dict = {X: X_train, y: y_train})
        file_writer.add_summary(accuracy_training_str, epoch)
        file_writer.add_summary(loss_training_str, epoch)
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy: {:.3f}%".format(accuracy_val * 100), 
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, 'wb') as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break

os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict = {X: X_test, y: y_test})
    accuracy_test = accuracy.eval(feed_dict = {X:X_train, y:y_train})
print ('Test score {}'.format(accuracy_val))
print ('Train score {}'.format(accuracy_test))

