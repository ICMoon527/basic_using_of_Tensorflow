import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(2018)


def plot_decision_boundary(model, x, y):
    # 找到x, y的最大值和最小值, 并在周围填充一个像素
    x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
    y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
    h = 0.01
    # 构建一个宽度为`h`的网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # 计算模型在网格上所有点的输出值
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # 画图显示
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), cmap=plt.cm.Spectral)


np.random.seed(1)
m = 400  # 样本数量
N = int(m/2)  # 每一类的点的个数
D = 2 # 维度
x = np.zeros((m, D))
y = np.zeros((m, 1), dtype='uint8') # label 向量，0 表示红色，1 表示蓝色
a = 4

for j in range(2):
    ix = range(N*j,N*(j+1))
    t = np.linspace(j*3.12,(j+1)*3.12,N) + np.random.randn(N)*0.2 # theta
    r = a*np.sin(4*t) + np.random.randn(N)*0.2 # radius
    x[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
    y[ix] = j

plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), s=40, cmap=plt.cm.Spectral)
plt.show()

with tf.variable_scope('layer1'):
    w1 = tf.get_variable(name='weight1',
                         shape=(2, 4),
                         dtype=tf.float32,
                         initializer=tf.random_normal_initializer(stddev=0.01))
    b1 = tf.get_variable(name='bias1',
                         shape=(4),
                         dtype=tf.float32,
                         initializer=tf.zeros_initializer())

with tf.variable_scope('layer2'):
    w2 = tf.get_variable(name='weight2',
                         shape=(4, 1),
                         dtype=tf.float32,
                         initializer=tf.random_normal_initializer(stddev=0.01))
    b2 = tf.get_variable(name='bias2',
                         shape=(1),
                         dtype=tf.float32,
                         initializer=tf.zeros_initializer())


def two_network(input):
    with tf.variable_scope('two_network'):
        net = tf.matmul(input, w1) + b1
        net = tf.tanh(net)
        net = tf.matmul(net, w2) + b2
        return tf.sigmoid(net)


x = tf.constant(x, dtype=tf.float32, name='x')
y = tf.constant(y, dtype=tf.float32, name='y')
net = two_network(x)
loss = tf.losses.log_loss(labels=y, predictions=net, scope='loss')
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1, name='optimizer')
train_op = optimizer.minimize(loss=loss, var_list=[w1, b1, w2, b2])

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for e in range(10000):  # 0.24857676029205322
    sess.run(train_op)
    if (e + 1) % 1000 == 0:
        loss_numpy = loss.eval(session=sess)
        print('Epoch: {}, loss: {}'.format((e + 1), loss_numpy))
    if (e + 1) % 5000 == 0:
        saver.save(sess=sess, save_path='multilayerNeuralNetwork/model.ckpt', global_step=(e + 1))


sess.close()

# 恢复模型结构
# saver = tf.train.import_meta_graph('First_Save/model.ckpt-10000.meta')

# 恢复模型参数
# saver.restore(sess, 'First_Save/model.ckpt-10000')