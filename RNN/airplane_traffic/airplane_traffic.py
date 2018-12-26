import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils.layers import lstm


data_csv = pd.read_csv('airplane_traffic.csv', usecols=[1])
print(data_csv)
plt.plot(data_csv)
plt.show()

# data preprocessing
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))


def create_dataset(dataset, look_back=2):
    """
    :param dataset:
    :param look_back: the number of months we look back when we predict this month
    :return: data for input, data for output
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


data_input, data_output = create_dataset(dataset)
train_size = (int(len(data_input) * 0.35)) * 2  # train_size is 98 here which should be an even number
test_size = len(data_input) - train_size
train_input = data_input[:train_size]
train_output = data_output[:train_size]
test_input = data_input[train_size:]
test_output = data_output[train_size:]

train_input = train_input.reshape(-1, 1, 2)
train_output = train_output.reshape(-1, 1, 1)
test_input = test_input.reshape(-1, 1, 2)
test_output = test_output.reshape(-1, 1, 1)

input_ph = tf.placeholder(shape=[None, 1, 2], dtype=tf.float32, name='input')
target_ph = tf.placeholder(shape=[None, 1, 1], dtype=tf.float32, name='target')


def lstm_reg(inputs, num_units, output_size=1, keep_prob=1, num_layers=2, scope='lstm_reg', reuse=None):
    """
    :param inputs: data
    :param num_units: the hidden size of RNN
    :param output_size: at last we should concentrate the output of hidden neurons into one output (prediction)
    :param keep_prob: 1 - drop out ratio
    :param num_layers: multilayers
    :param scope: namescope
    :param reuse: whether to reuse
    :return:
    """
    with tf.variable_scope(scope, reuse=reuse):
        net, state = lstm(inputs, num_units, num_layers, 1, keep_prob=keep_prob)  # (time_steps, batch_size, fixed_length)

        s, b, n = net.get_shape().as_list()  # none 1 4 = time_steps batch_size num_units
        # print(s, b, n)
        net = tf.reshape(net, (-1, num_units))
        net = slim.fully_connected(net, output_size, activation_fn=None, scope='regression')  # the last layer
        net = tf.reshape(net, (-1, b, output_size))

        return net


out = lstm_reg(input_ph, 4)
loss = tf.losses.mean_squared_error(target_ph, out)
opt = tf.train.AdamOptimizer(1e-2)
train_op = opt.minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for e in range(1000):
    feed_dict = {input_ph: train_input, target_ph: train_output}
    sess.run(train_op, feed_dict=feed_dict)
    if (e + 1) % 100 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, sess.run(loss, feed_dict=feed_dict)))