from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import math

np.random.seed(2017)

x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168],
                    [9.779], [6.182], [7.59], [2.167], [7.042],
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573],
                    [3.366], [2.596], [2.53], [1.221], [2.827],
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

x = tf.constant(x_train, name='x')
y = tf.constant(y_train, name='y')
w = tf.Variable(initial_value=tf.random_normal(shape=(), seed=2017), dtype=tf.float32, name='weight')
b = tf.Variable(initial_value=0, dtype=tf.float32, name='bias')

with tf.variable_scope('linearModel'):
    y_predict = w * x + b

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
lr = 6e-2/5

for i in range(10):
    y_predict_numpy = y_predict.eval(session=sess)
    plt.plot(x_train, y_train, 'bo', label='real')
    plt.plot(x_train, y_predict_numpy, 'ro', label='predict')
    plt.show()

    loss = tf.reduce_mean(tf.square(y - y_predict))
    print('loss =', loss.eval(session=sess))
    w_grad, b_grad = tf.gradients(loss, [w, b])
    print('w_grad =', w_grad)  # aim to get tensors' name, shape, type rather than value
    print('b_grad =', b_grad)
    w_update = w.assign_sub(lr * w_grad / math.exp(i/2))
    b_update = b.assign_sub(lr * b_grad / math.exp(i))
    sess.run([w_update, b_update])