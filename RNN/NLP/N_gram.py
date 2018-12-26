from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

CONTEXT_SIZE = 2 # 依据的单词数
EMBEDDING_DIM = 10 # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

# the size of window is three
trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]
vocabulary = set(test_sentence)
word_to_idx = {word: i for i, word in enumerate(vocabulary)}  # dictionary from word to index, this index isn't one-hot
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}  # dictionary from index to word


def n_gram(inputs, vocab_size, context_size=CONTEXT_SIZE, n_dim=EMBEDDING_DIM, scope='n-gram', reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        with tf.device('/cpu:0'):
            embeddings = tf.get_variable('embeddings', shape=[vocab_size, n_dim],
                                         initializer=tf.random_uniform_initializer)
        embed = tf.nn.embedding_lookup(embeddings, inputs)  # through this, we got a bridge between input and embedding

        net = tf.reshape(embed, (1, -1))
        net = slim.fully_connected(net, vocab_size, activation_fn=None, scope='classification')

        return net


input_ph = tf.placeholder(dtype=tf.int64, shape=[2, ], name='input')
label_ph = tf.placeholder(dtype=tf.int64, shape=[1, ], name='label')

net = n_gram(input_ph, len(word_to_idx))
loss = tf.losses.sparse_softmax_cross_entropy(label_ph, net, scope='loss')
opt = tf.train.MomentumOptimizer(1e-2, 0.9)
train_op = opt.minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for e in range(100):
    train_loss = 0
    for word, label in trigram[:100]:
        word = [word_to_idx[i] for i in word]  # get the index of input words
        label = [word_to_idx[label]]  # get the index of output words

        _, curr_loss, output = sess.run([train_op, loss, net], feed_dict={input_ph: word, label_ph: label})
        train_loss += curr_loss

    if (e + 1) % 20 == 0:
        print('Epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / 100))

# testing part
word, label = trigram[19]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = [word_to_idx[i] for i in word]
out = sess.run(net, feed_dict={input_ph: word})
pred_label_idx = out[0].argmax()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}\n'.format(label, predict_word))

word, label = trigram[75]
print('input: {}'.format(word))
print('label: {}'.format(label))
print()
word = [word_to_idx[i] for i in word]
out = sess.run(net, feed_dict={input_ph: word})
pred_label_idx = out[0].argmax()
predict_word = idx_to_word[pred_label_idx]
print('real word is {}, predicted word is {}'.format(label, predict_word))