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

CONV1_FILTERS = 32
CONV1_KERNEL_SIZE = 5
POOL1_SIZE = 2
HEIGHT = 28
WIDTH = 28
NHID = 32
LOCAL_MIN_THRESH = 1e-4
NOISE_TO_ADD = 0.2
TIME_STEP = 10
ITERATIONS = 500
BATCH_SIZE = 3000
INIT_STEPS = 50
LEARNING_RATE = 1e-4


VALIDATION_SIZE = 50
v_dmat = dmat[:VALIDATION_SIZE]
dmat = dmat[VALIDATION_SIZE:]
v_answers = answers[:VALIDATION_SIZE]
answers = answers[VALIDATION_SIZE:]
INIT_DENOM = 100
x = tf.placeholder(tf.float32, [None, train.columnCount])

# Input Layer
input_layer = tf.reshape(x, [-1, HEIGHT, WIDTH, 1])

# Convolutional Layer #1
kernel = tf.Variable(tf.random_normal([CONV1_KERNEL_SIZE, CONV1_KERNEL_SIZE, 1, CONV1_FILTERS])/INIT_DENOM)

conv1_in = tf.nn.conv2d(
    input=input_layer,
    filter=kernel,
    strides=[1, 2, 2, 1],
    padding="SAME"
    )
    
conv1 = tf.nn.relu(conv1_in)

# Pooling Layer #1
pool1 = tf.nn.max_pool(
    value=conv1,
    ksize=[1, 2, 2, 1],
    strides=[1,1,1,1],
    padding="SAME"
    )
    

# Fully Connected Layer
pool1_flat = tf.reshape(pool1, [-1, CONV1_FILTERS*HEIGHT*WIDTH//(POOL1_SIZE*POOL1_SIZE)])
W = tf.Variable(tf.random_normal([CONV1_FILTERS*HEIGHT*WIDTH//(POOL1_SIZE*POOL1_SIZE), 10])/INIT_DENOM)
b = tf.Variable(tf.random_normal([1, 10])/INIT_DENOM)
fully_connected = tf.nn.relu(tf.matmul(pool1_flat, W)+b)

# Softmax Layer
y = tf.nn.softmax(fully_connected)
y_ = tf.placeholder(tf.float32, [None, 10])
    
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
prediction = tf.argmax(y,1)


a0 = [0]*5
b0 = [0]*5
a1 = [0]*5
b1 = [0]*5
for i in range(ITERATIONS):
    batch_xs, batch_ys = Dataset.get_batch(dmat, answers,BATCH_SIZE, 10)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i % TIME_STEP == 0:
        print('iteration:', i)
        print('Train:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))
        batch_xs, batch_ys = Dataset.get_batch(v_dmat,v_answers,VALIDATION_SIZE, 10)
        v_acc = accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys})
        print('Validation:', v_acc)
        
        a0 = sess.run(W)[0][:5]
        b0 = sess.run(kernel)[0][0][:5][0][:5]
        diff1 = a0-a1
        diff2 = b0-b1
        print(diff1)
        print(diff2)
        if i > INIT_STEPS:
            if np.sum(np.abs(diff1)) < LOCAL_MIN_THRESH:
                print("Add noise to W")
                W += tf.random_normal(W.get_shape())*((tf.reduce_mean(W).eval(session=sess))*NOISE_TO_ADD)
            if np.sum(np.abs(diff2)) < LOCAL_MIN_THRESH:
                print("Add noise to kernel")
                kernel += tf.random_normal(kernel.get_shape())*((tf.reduce_mean(kernel).eval(session=sess))*NOISE_TO_ADD)
        a1 = a0
        b1 = b0

batch_xs, batch_ys = Dataset.get_batch(v_dmat, v_answers, VALIDATION_SIZE, 10)
print('Validation Accuracy:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))

test_answers = prediction.eval(session=sess, feed_dict = {x: tdmat})
f = open(DIRECTORY + 'submission.txt', 'w')
f.write('ImageId,Label\n')
for i in range(len(tdata)):
    f.write(str(i+1) +','+ str(test_answers[i]) +'\n')
f.close()