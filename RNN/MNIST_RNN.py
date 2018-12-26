import numpy as np
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow.contrib.slim as slim
from utils.layers import lstm
tf.set_random_seed(2018)

# get Mnist data
mnist = input_data.read_data_sets('/data/langjunwei/translation/tensorflowSample/NeuralNetwork/MNIST_data', one_hot=True, reshape=False)
train_set = mnist.train
test_set = mnist.test
train_images, train_labels = train_set.next_batch(64)
print(train_images.shape)
print(train_labels.shape)

# define the placeholder
input_ph = tf.placeholder(dtype=tf.float32, shape=(None, 28, 28, 1))
label_ph = tf.placeholder(dtype=tf.int64, shape=(None, 10))
batch_size_ph = tf.placeholder(dtype=tf.int32, shape=[])
keep_prob_ph = tf.placeholder(dtype=tf.float32, shape=[])

# transfer image data into what that fits rnn_input (to be time major)
input_op = tf.transpose(tf.squeeze(input_ph, axis=[-1]), (1, 0, 2))
print(input_op.shape)


def rnn_classify(inputs, rnn_units=100, rnn_layers=2, batch_size=64, keep_prob=1, num_classes=10):
    # build a multi-layer rnn
    rnn_out, rnn_state = lstm(inputs, rnn_units, rnn_layers, batch_size, keep_prob=keep_prob)

    # fetch the final output of multi-lstm
    net = rnn_out[-1]  # (batch_size, cell.output_size)

    # layer to classify
    net = slim.flatten(net)  # flatten the tensor into (batch_size, ?) for full-connected layer
    net = slim.fully_connected(net, num_classes, activation_fn=None, scope='classification')

    return net


out = rnn_classify(input_op, batch_size=batch_size_ph, keep_prob=keep_prob_ph)

loss = tf.losses.softmax_cross_entropy(logits=out, onehot_labels=label_ph)

acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, axis=-1), tf.argmax(label_ph, axis=-1)), dtype=tf.float32))

lr = 0.01
optimizer = tf.train.MomentumOptimizer(lr, 0.9)
train_op = optimizer.minimize(loss)

# start to train
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
for e in range(10000):
    images, labels = train_set.next_batch(64)
    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels, batch_size_ph: 64, keep_prob_ph: 0.5})
    if e % 1000 == 999:
        test_imgs, test_labels = test_set.next_batch(128)
        loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: images, label_ph: labels, batch_size_ph: 64, keep_prob_ph: 1.0})
        loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: test_imgs, label_ph: test_labels, batch_size_ph: 128, keep_prob_ph: 1.0})
        print('STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f} test_acc: {:.6f}'.format(e + 1, loss_train, acc_train, loss_test, acc_test))

print('Train Done!')
print('-'*30)

train_loss = []
train_acc = []
for _ in range(train_set.num_examples // 100):
    image, label = train_set.next_batch(100)
    loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label, batch_size_ph: 100, keep_prob_ph: 1.0})
    train_loss.append(loss_train)
    train_acc.append(acc_train)

print('Train loss: {:.6f}'.format(np.array(train_loss).mean()))
print('Train accuracy: {:.6f}'.format(np.array(train_acc).mean()))

test_loss = []
test_acc = []
for _ in range(test_set.num_examples // 100):
    image, label = test_set.next_batch(100)
    loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label, batch_size_ph: 100, keep_prob_ph: 1.0})
    test_loss.append(loss_test)
    test_acc.append(acc_test)

print('Test loss: {:.6f}'.format(np.array(test_loss).mean()))
print('Test accuracy: {:.6f}'.format(np.array(test_acc).mean()))