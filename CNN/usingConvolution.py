import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

image = Image.open('kitty.png')
image_grey = image.convert('L')
image_grey.show()
image_grey = np.array(image_grey, dtype=np.float32)
print(image_grey.shape)
# transform picture matrix into tensor
image = tf.constant(image_grey.reshape(1, image_grey.shape[0], image_grey.shape[1], 1), name='input')
# define a kernel
sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
sobel_kernel = tf.constant(sobel_kernel, shape=[3, 3, 1, 1])

# convolution and test the params of padding
edge_same = tf.nn.conv2d(image, sobel_kernel, strides=[1, 1, 1, 1], padding='SAME', name='same_conv')
edge_valid = tf.nn.conv2d(image, sobel_kernel, strides=[1, 1, 1, 1], padding='VALID', name='valid_conv')
sess = tf.InteractiveSession()
edge_same_np = sess.run(edge_same)
edge_valid_np = sess.run(edge_valid)

# show the difference between same_conv and valid_conv
plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(edge_same_np), cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(edge_valid_np), cmap='gray')
plt.show()

# max_pooling and test the params of padding
pool_same = tf.nn.max_pool(image, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='same_pool')