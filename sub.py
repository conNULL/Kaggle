import pandas as pd
import numpy as np
import tensorflow as tf
import random
import math
from lib.data_utils import Dataset
    

DIRECTORY = ('data/Titanic/')
data = pd.read_csv(DIRECTORY + 'train.csv')
tdata = pd.read_csv(DIRECTORY + 'test.csv')

age = data['Age'].mean()
data['Age'] =data['Age'].fillna(age)
tdata['Age'] =tdata['Age'].fillna(age)
data['Embarked'] =data['Embarked'].fillna('nan')
tdata['Embarked'] =tdata['Embarked'].fillna('nan')

binary = set(['Sex'])
cont = set(['Pclass','Age', 'Parch', 'Fare', 'SibSp'])
multiVal = set(['Embarked'])
answers = data['Survived'].values

train = Dataset(data, binary, cont, multiVal, set(['nan']))
test = Dataset(tdata, binary, cont, multiVal, set(['nan']))

dmat = np.concatenate(train.data, 0).T
tdmat = np.concatenate(test.data, 0).T

NHID = 4
VALIDATION_SIZE = 50
v_dmat = dmat[:VALIDATION_SIZE]
dmat = dmat[VALIDATION_SIZE:]
v_answers = answers[:VALIDATION_SIZE]
answers = answers[VALIDATION_SIZE:]

x = tf.placeholder(tf.float32, [None, train.columnCount])
W0 = tf.Variable(tf.random_normal([train.columnCount, NHID]))
b0 = tf.Variable(tf.random_normal([1, NHID]))
W1 = tf.Variable(tf.random_normal([NHID, 1]))
b1 = tf.Variable(tf.random_normal([1, 1]))

layer1 = tf.nn.tanh(tf.matmul(x, W0) + b0)
y = tf.matmul(layer1, W1) + b1
y_ = tf.placeholder(tf.float32, [None, 1])

cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)
train_step = tf.train.AdamOptimizer(0.05).minimize(cross_entropy)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
prediction = tf.round(tf.sigmoid(y))

batch_size = 300
end_batch_size = 800
switched = False
correct_prediction = tf.equal([tf.round(tf.sigmoid(y))], [y_])
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(30000):
    batch_xs, batch_ys = Dataset.get_batch(dmat, answers,batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    
    if i % 300 == 0:
        print('iteration:', i)
        print('Train:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))
        batch_xs, batch_ys = Dataset.get_batch(v_dmat,v_answers,VALIDATION_SIZE)
        v_acc = accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys})
        print('Validation:', v_acc)
        if not switched and v_acc > 0.8:
            print('SWITCH')
            train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
            batch_size = end_batch_size
            switched = True

batch_xs, batch_ys = Dataset.get_batch(v_dmat, v_answers, VALIDATION_SIZE)
print('Validation Accuracy:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))

test_answers = [int(k) if not math.isnan(k) else 0 for k in prediction.eval(session=sess, feed_dict = {x: tdmat})]

f = open(DIRECTORY + 'submission.txt', 'w')
f.write('PassengerId,Survived\n')
for i in range(len(data), len(data) + len(tdata)):
    f.write(str(i+1) +','+ str(test_answers[i-len(data)]) +'\n')
f.close()





