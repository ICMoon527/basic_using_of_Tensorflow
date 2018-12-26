import sys
sys.path.append('..')
import tensorflow as tf
from utils import cifar10_pruning as cifar10
from utils.learning import train

batch_size = 128

# the name of directory is in cifar10_pruning.py
cifar10.maybe_download_and_extract()
# fetch the training data which will be shuffled and default image size is 24
train_imgs, train_labels = cifar10.distorted_inputs()
# fetch the testing data which will not be shuffled
test_imgs, test_labels = cifar10.inputs(True)
print('So we have {} to train per epoch and {} to test per epoch'.format(cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN,
                                                                         cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL))


# construct function to generate weight and bias
def construct_weight(shape, stddev=0.05):
    return tf.get_variable(name='weight', shape=shape, initializer=tf.truncated_normal_initializer(stddev=stddev))


def construct_bias(shape):
    return tf.get_variable(name='bias', shape=shape, initializer=tf.constant_initializer(0.1))


def convolution(input, kernel_size, output_depth, strides, padding='SAME', activation=tf.nn.relu, scope='conv_layer', reuse=None):
    """
    :param input: input_data
    :param kernel_size: a list with length of two
    :param output_depth:
    :param strides:
    :param padding: the padding strategy
    :param activation: which activate function we want to use
    :param scope:
    :param reuse: whether to reuse
    :return: the result after convolution
    """
    input_depth = input.shape.as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        shape = kernel_size + [input_depth, output_depth]
        with tf.variable_scope('kernel'):
            kernel = construct_weight(shape)
        # convolution will not be done to the first and fourth dims of inputs, so these dims of strides will be one
        strides = [1, strides[0], strides[1], 1]
        # convolution is ready
        conv = tf.nn.conv2d(input, kernel, strides, padding, name='conv')
        # construct the bias
        with tf.variable_scope('bias'):
            bias = construct_bias(output_depth)
        preact = tf.nn.bias_add(conv, bias)
        out = activation(preact, name='conv_output')
        return out


def max_pool(input, kernel_size, strides, padding='SAME', name='pooling'):
    """
    :param input:
    :param kernel_size:
    :param strides:
    :param padding:
    :param name:
    :return:
    """
    return tf.nn.max_pool(input, [1, kernel_size[0], kernel_size[1], 1], [1, strides[0], strides[1], 1], padding, name=name)


def fc(input, output_depth, activation=tf.nn.relu, scope='full_connect', reuse=None):
    """
    :param input:
    :param output_depth:
    :param act:
    :param scope:
    :param reuse:
    :return:
    """
    in_depth = input.shape.as_list()[-1]
    with tf.variable_scope(scope, reuse=reuse):
        with tf.variable_scope('weight'):
            weight = construct_weight([in_depth, output_depth])
        with tf.variable_scope('bias'):
            bias = construct_bias(output_depth)
        fc = tf.nn.bias_add(tf.matmul(input, weight), bias, name='fc')
        out = activation(fc, name='fc_output')
        return out


def alexnet(input, reuse=None):
    """
    :param input:
    :param reuse:
    :return:
    """
    print('input_shape: ', input.shape.as_list())
    with tf.variable_scope('AlexNet', reuse=reuse):
        conv1 = convolution(input, [5, 5], 64, [1, 1], 'VALID', scope='conv1')
        pool1 = max_pool(conv1, [3, 3], [2, 2], 'VALID', name='pool1')
        conv2 = convolution(pool1, [5, 5], 64, [1, 1], 'SAME', scope='conv2')
        pool2 = max_pool(conv2, [3, 3], [2, 2], 'VALID', name='pool2')
        if reuse is True:
            print('pool1_input: {}\nconv2_input: {}\npool2_input: {}\npool2_output: {}'.format(conv1.shape.as_list(), pool1.shape.as_list(), conv2.shape.as_list(), pool2.shape.as_list()))
        pool2 = tf.reshape(pool2, [-1, 4*4*64])
        fc1 = fc(pool2, 384, scope='fc1')
        fc2 = fc(fc1, 192, scope='fc2')
        # we don't use activation funtion here, why tf.identity
        out = fc(fc2, 10, scope='fc3', activation=tf.identity)
        return out


train_out = alexnet(train_imgs)
# if we forget to reuse, we will create a new set of variables because we didn't define any placeholder
test_out = alexnet(test_imgs, reuse=True)
# if labels are one-hot vector, we use softmax_cross_entropy() instead
with tf.variable_scope('loss'):
    train_loss = tf.losses.sparse_softmax_cross_entropy(train_labels, train_out, scope='train')
    test_loss = tf.losses.sparse_softmax_cross_entropy(test_labels, test_out, scope='test')
with tf.name_scope('accuracy'):
    with tf.name_scope('train'):
        train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(train_out, axis=-1, output_type=tf.int32), train_labels), tf.float32))
    with tf.name_scope('test'):
        test_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(test_out, axis=-1, output_type=tf.int32), test_labels), tf.float32))

lr = 0.01

opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
train_op = opt.minimize(train_loss)

train(train_op, train_loss, train_acc, test_loss, test_acc, 20000, batch_size)
