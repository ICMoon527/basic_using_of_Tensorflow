import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.examples.tutorials.mnist.input_data as input_data
import os
# os.environ['CUDA_VISIBLE_DIVICES'] = '6'

tf.set_random_seed(2018)
# reset the graph
tf.reset_default_graph()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
train_set = mnist.train
test_set = mnist.test
# see some pics from MNIST
images, labels = train_set.next_batch(12, shuffle=False)

for ind, (image, label) in enumerate(zip(images, labels)):
    # image is a vector of 784 dimensions, so we reshape it to 28 * 28
    image = image.reshape((28, 28))

    # label is a vector of 10 dimensions, the location of '1' represents the actual number
    label = label.argmax()

    row = ind // 6
    col = ind % 6
    plt.subplot(6, 2, ind+1)
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(label)
plt.show()


def variable_summaries(var):
    with tf.name_scope('summaries'):
        # calculate mean of var
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        # calculate the stddev
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        # add the max and the min scalar to summaries
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # add the var to summaries which is a vector
        tf.summary.histogram('var_histogram', var)


def hidden_layer(layer_input, output_depth, scope='hidden_layer', reuse=None):
    input_depth = layer_input.get_shape()[-1]
    print('layer_input = ', layer_input.get_shape())
    with tf.variable_scope(scope, reuse=reuse):
        # use truncated_normal_initializer to make distribution more concentrated
        w = tf.get_variable(name='weight',
                            shape=(input_depth, output_depth),
                            dtype=tf.float32,
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable(name='bias',
                            shape=(output_depth),
                            dtype=tf.float32,
                            initializer=tf.constant_initializer(0.1))
        net = tf.matmul(layer_input, w) + b
        return net


def DNN(input, output_depths, scope='DNN', reuse=None):
    net = input
    for i, output_depth in enumerate(output_depths):
        net = hidden_layer(layer_input=net,
                           output_depth=output_depth,
                           scope='layer{}'.format(i),
                           reuse=reuse)
        tf.summary.histogram('pre_activation', net)
        net = tf.nn.relu(net)
        tf.summary.histogram('output', net)

    # the last layer should be 10 dimensions to classify
    net = hidden_layer(layer_input=net,
                       output_depth=10,
                       scope='classification',
                       reuse=reuse)
    return net


# define the placeholder rather than variable for changing input_data during training and testing
input_ph = tf.placeholder(dtype=tf.float32,
                          shape=(None, 784))
label_ph = tf.placeholder(dtype=tf.int8,
                          shape=(None, 10))
# create a 4-layer neural networks whose hidden sizes are [400, 200, 100, 10]
dnn = DNN(input=input_ph,
          output_depths=[400, 200, 100])

with tf.name_scope('cross_entropy'):
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label_ph,
                                           logits=dnn)
    tf.summary.scalar('cross_entropy', loss)

with tf.name_scope('accuracy'):
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(dnn, axis=-1), tf.argmax(label_ph, axis=-1)), dtype=tf.float32))
    tf.summary.scalar('accuracy', acc)

with tf.name_scope('train'):
    learning_rate = 0.01
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss)

# start to train
merged = tf.summary.merge_all()
sess = tf.InteractiveSession()
# open file writer
train_writer = tf.summary.FileWriter('MNIST_data/train', sess.graph)
test_writer = tf.summary.FileWriter('MNIST_data/test', sess.graph)
batch_size=64
sess.run(tf.global_variables_initializer())

for e in range(20000):
    images, labels = train_set.next_batch(batch_size=batch_size, shuffle=True)
    # print('image_shape = ', images.shape)
    sess.run(train_op, feed_dict={input_ph: images, label_ph: labels})
    if (e + 1) % 1000 == 0:
        test_images, test_labels = test_set.next_batch(batch_size=batch_size, shuffle=True)
        # calculate the loss and the accuracy of train_set and test_set
        sum_train, loss_train, acc_train = sess.run([merged, loss, acc], feed_dict={input_ph: images, label_ph: labels})
        train_writer.add_summary(sum_train, e)
        sum_test, loss_test, acc_test = sess.run([merged, loss, acc], feed_dict={input_ph: test_images, label_ph: test_labels})
        test_writer.add_summary(sum_test, e)
        print('STEP {}: train_loss: {:.6f} train_acc: {:.6f} test_loss: {:.6f} test_acc: {:.6f}'.format(
            e + 1, loss_train, acc_train, loss_test, acc_test))
        # print(sess.run(dnn, feed_dict={input_ph: test_images[0: 10, :]}))

# close file writer
train_writer.close()
test_writer.close()
print('Training is done')
print('-' * 30)

# calculate the loss and the accuracy of all the train_set
train_loss = []
train_acc = []

for e in range(train_set.num_examples // 100):
    image, label = train_set.next_batch(100)
    loss_train, acc_train = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    train_loss.append(loss_train)
    train_acc.append(acc_train)

print('Train loss: {:.6f}'.format(np.array(train_loss).mean()))
print('Train accuracy: {:.6f}'.format(np.array(train_acc).mean()))

# calculate the loss and the accuracy of all the test_set
test_loss = []
test_acc = []

for e in range(test_set.num_examples // 100):
    image, label = test_set.next_batch(100)
    loss_test, acc_test = sess.run([loss, acc], feed_dict={input_ph: image, label_ph: label})
    test_loss.append(loss_test)
    test_acc.append(acc_test)

print('Test loss: {:.6f}'.format(np.array(test_loss).mean()))
print('Test accuracy: {:.6f}'.format(np.array(test_acc).mean()))

sess.close()