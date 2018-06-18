import pandas as pd
import numpy as np
import tensorflow as tf
import random
import math
from lib.data_utils import Dataset

DIRECTORY = ('data/Digit_Recognizer/')
data = pd.read_csv(DIRECTORY + 'train.csv')
tdata = pd.read_csv(DIRECTORY + 'test.csv')

cols = set(data.columns) - set(['label'])

answer_set = Dataset(data, discreteColumns=set(['label']))
answers = np.asarray(answer_set.data).T

train = Dataset(data, continuousColumns=cols)
test = Dataset(tdata, continuousColumns=cols)

dmat = np.concatenate(train.data, 0).T
tdmat = np.concatenate(test.data, 0).T

CONV1_FILTERS = 8
CONV1_KERNAL_SIZE = 5
POOL1_SIZE = 2
HEIGHT = 28
WIDTH = 28
NHID = 32

VALIDATION_SIZE = 50
v_dmat = dmat[:VALIDATION_SIZE]
dmat = dmat[VALIDATION_SIZE:]
v_answers = answers[:VALIDATION_SIZE]
answers = answers[VALIDATION_SIZE:]

x = tf.placeholder(tf.float32, [None, train.columnCount])

# Input Layer
input_layer = tf.reshape(x, [-1, HEIGHT, WIDTH, 1])

# Convolutional Layer #1
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=CONV1_FILTERS,
    kernel_size=[CONV1_KERNAL_SIZE, CONV1_KERNAL_SIZE],
    padding="same",
    activation=tf.nn.relu)

# Pooling Layer #1
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[POOL1_SIZE, POOL1_SIZE], strides=2)

# Dense Layer
pool1_flat = tf.reshape(pool1, [-1, CONV1_FILTERS*HEIGHT*WIDTH//(POOL1_SIZE*POOL1_SIZE)])
dense = tf.layers.dense(inputs=pool1_flat, units=NHID, activation=tf.nn.relu)

# Logits Layer
logits = tf.layers.dense(inputs=dense, units=10)

y = tf.nn.softmax(logits)
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
prediction = tf.round(tf.sigmoid(y))

batch_size = 300
end_batch_size = 800
switched = True
correct_prediction = tf.equal([tf.round(tf.sigmoid(y))], [y_])
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(1000):
    batch_xs, batch_ys = Dataset.get_batch(dmat, answers,batch_size, 10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i % 10 == 0:
        print('iteration:', i)
        print('Train:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))
        batch_xs, batch_ys = Dataset.get_batch(v_dmat,v_answers,VALIDATION_SIZE, 10)
        v_acc = accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys})
        print('Validation:', v_acc)
        if not switched and v_acc > 0.8:
            print('SWITCH')
            train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
            batch_size = end_batch_size
            switched = True

batch_xs, batch_ys = Dataset.get_batch(v_dmat, v_answers, VALIDATION_SIZE, 10)
print('Validation Accuracy:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))

test_answers = [int(k) if not math.isnan(k) else 0 for k in prediction.eval(session=sess, feed_dict = {x: tdmat})]

f = open(DIRECTORY + 'submission.txt', 'w')
f.write('PassengerId,Survived\n')
for i in range(len(data), len(data) + len(tdata)):
    f.write(str(i+1) +','+ str(test_answers[i-len(data)]) +'\n')
f.close()