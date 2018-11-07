import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
tf.set_random_seed(2018)


def logistic_regression(x, w, b):
    with tf.variable_scope("LogisticRegression"):
        return tf.sigmoid(tf.matmul(x, w) + b)   # remember to make them to 1 or 0


def logistic_loss(yPredict, y):
    with tf.variable_scope("LogisticLoss"):
        return -tf.reduce_mean(y * tf.log(yPredict) + (1 - y) * tf.log(1 - yPredict))


def correct_num(yPredict, y):
    count = 0
    for i, item in enumerate(yPredict):
        if yPredict[i] > 0.5:
            yPredict[i] = 1
        else:
            yPredict[i] = 0
        print('yPredict[{}] = {}, y[{}] = {}'.format(i, yPredict[i], i, y[i]))
        if yPredict[i] == y[i]:
            count += 1
            # print('yPredict{} = {}, y{} = {}'.format(i, yPredict[i], i, y[i]))
    return count


data_csv = pd.read_csv('LogisticRegression/data.csv', sep=',')
data_csv.columns = ["grade1", "grade2", "label"]
# print(data_csv.head())
label = data_csv['label'].map(lambda x: float(x.rstrip(';')))
label = np.array(label)
data = np.array(data_csv[['grade1', 'grade2']])
x = data_csv['grade1']
y = data_csv['grade2']
x0 = []   # the target divided into zero
y0 = []
x1 = []   # the target divided into one
y1 = []

# scale or normalize the data is extremely important
scale_data = preprocessing.StandardScaler().fit(data)
data_scaled = scale_data.transform(data)
xNorm = data_scaled[:, 0]
yNorm = data_scaled[:, 1]

for i, item in enumerate(label):   # prepare for plotting the pic
    if item == 1:
        x1.append(xNorm[i])
        y1.append(yNorm[i])
    else:
        x0.append(xNorm[i])
        y0.append(yNorm[i])

xNorm = np.array(xNorm)
yNorm = np.array(yNorm)
dataNorm = np.column_stack((xNorm, yNorm))   # put inputs together and prepare to be trained
print("dataNorm.shape =", dataNorm.shape)
print("label.shape =", label.shape)

x_data = tf.constant(dataNorm, name='input', dtype=tf.float32)
y_data = tf.constant(label, name='label', dtype=tf.float32)
w = tf.get_variable(name='weight',
                    shape=(2, 1),
                    dtype=tf.float32,
                    initializer=tf.random_normal_initializer(seed=2017)
                    )
b = tf.get_variable(name='bias',
                    shape=(1),
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer())

sess = tf.InteractiveSession()

learning_rate = 0.02
sess.run(tf.global_variables_initializer())
wNumpy = w.eval(session=sess)
bNumpy = b.eval(session=sess)
wx = wNumpy[0]
wy = wNumpy[1]
bias = bNumpy[0]
xPlot = np.arange(-2, 2, 0.01)
yPlot = - (wx * xPlot + bias) / wy
plt.plot(xPlot, yPlot, 'g', label='line')
plt.plot(x0, y0, 'ro', label='0')
plt.plot(x1, y1, 'bo', label='1')
plt.show()

# start training
# yPredict = logistic_regression(x_data, w, b)
# loss = logistic_loss(yPredict, y_data)
yPredict = tf.sigmoid(tf.matmul(x_data, w) + b)
loss = tf.losses.log_loss(labels=tf.reshape(y_data, (-1, 1)), predictions=yPredict)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate, name="optimizer")
train_op = optimizer.minimize(loss=loss)

# print('x_data =\n', x_data.eval(session=sess))
for i in range(1000):
    sess.run(train_op)
    if (i + 1) % 200 == 0:
        # plot the picture
        # print('yPridict =\n', yPredict.eval(session=sess))
        # print('correct number :', correct_num(yPredict.eval(session=sess), label))
        wNumpy = w.eval(session=sess)
        bNumpy = b.eval(session=sess)
        print('bNumpy = ', bNumpy)
        wx = wNumpy[0]
        wy = wNumpy[1]
        bias = bNumpy[0]
        xPlot = np.arange(-2, 2, 0.01)
        yPlot = (-wx * xPlot - bias) / wy
        plt.plot(xPlot, yPlot, 'g', label='line')   # why it seems like a line rather than points putting together
        plt.plot(x0, y0, 'ro', label='0')
        plt.plot(x1, y1, 'bo', label='1')
        plt.show()

        y_true_label = y_data.eval(session=sess)
        y_pred_numpy = yPredict.eval(session=sess)
        y_pred_label = np.greater_equal(y_pred_numpy, 0.5).astype(np.float32)
        accuracy = np.mean(y_pred_label == y_true_label)
        loss_numpy = loss.eval(session=sess)
        print('Epoch %d, Loss: %.4f, Acc: %.4f' % (i + 1, loss_numpy, accuracy))

sess.close()