import pandas as pd
import numpy as np
import tensorflow as tf
import random
import math


def get_batch(x, y, size):
    
    ind = random.sample(range(len(x)), size)
    batch_x = np.asarray([x[k] for k in ind])
    batch_y = np.asarray([y[k] for k in ind])
    
    return batch_x, batch_y.reshape(size, 1)
    

DIRECTORY = ('data/Titanic/')
data = pd.read_csv(DIRECTORY + 'train.csv')
tdata = pd.read_csv(DIRECTORY + 'test.csv')

age = data['Age'].mean()
data['Age'] =data['Age'].fillna(age)
tdata['Age'] =tdata['Age'].fillna(age)
cols = []
tcols = []
binary = set(['Sex'])
cont = set(['Pclass','Age', 'Parch', 'Fare', 'SibSp'])
multiVal = set(['Embarked'])
answers = data['Survived'].values
EXTRA_COLS = 3
EMBARKED = {'S':[1,0,0],'C':[0,1,0],'Q':[0,0,1], 'nan':[0,0,0]}
data['Embarked'] = data['Embarked'].fillna('nan')
tdata['Embarked'] = tdata['Embarked'].fillna('nan')
data['Embarked'] = data['Embarked'].map(lambda x: EMBARKED[x])
tdata['Embarked'] = tdata['Embarked'].map(lambda x: EMBARKED[x])
for i in range(len(data.columns)):
    col = data.columns[i]
    typ = data.dtypes[i]
    
    if col in binary:
        cols.append((data[col] == data[col][0]).values.astype(np.float32).reshape(1, len(data[col])))
        tcols.append((tdata[col] == tdata[col][0]).values.astype(np.float32).reshape(1, len(tdata[col])))
    elif col in cont:
        cols.append(data[col].values.astype(np.float32).reshape(1, len(data[col])))
        tcols.append(tdata[col].values.astype(np.float32).reshape(1, len(tdata[col])))
    elif col in multiVal:
        for j in range(len(data[col][0])):
            cols.append(np.array([k[j] for k in data[col].values]).astype(np.float32).reshape(1, len(data[col])))
            tcols.append(np.array([k[j] for k in tdata[col].values]).astype(np.float32).reshape(1, len(tdata[col])))
    


dmat = np.concatenate(cols, 0).T#.reshape([len(cols), len(cols[0])])
tdmat = np.concatenate(tcols, 0).T#.reshape([len(cols), len(cols[0])])

NHID = 5
VALIDATION_SIZE = 50
v_dmat = dmat[:VALIDATION_SIZE]
dmat = dmat[VALIDATION_SIZE:]
v_answers = answers[:VALIDATION_SIZE]
answers = answers[VALIDATION_SIZE:]
cols = len(cont) + len(binary) + EXTRA_COLS


x = tf.placeholder(tf.float32, [None, cols])
W0 = tf.Variable(tf.random_normal([cols, NHID]))
b0 = tf.Variable(tf.random_normal([1, NHID]))
W1 = tf.Variable(tf.random_normal([NHID, 1]))
b1 = tf.Variable(tf.random_normal([1, 1]))

layer1 = tf.nn.tanh(tf.matmul(x, W0) + b0)
y = tf.matmul(layer1, W1) + b1
y_ = tf.placeholder(tf.float32, [None, 1])

cross_entropy = tf.losses.sigmoid_cross_entropy(multi_class_labels=y_, logits=y)
train_step = tf.train.AdadeltaOptimizer(5).minimize(cross_entropy)
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
prediction = tf.round(tf.sigmoid(y))
# print(W.eval(session = sess))
# Test trained model
batch_size = 300
end_batch_size = 800
switched = False
correct_prediction = tf.equal([tf.round(tf.sigmoid(y))], [y_])
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(20000):
    batch_xs, batch_ys = get_batch(dmat, answers,batch_size)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    # sess.run(train_step, feed_dict={x: dmat, y_: answers.reshape(len(answers), 1)})
    if i % 300 == 0:
        print('iteration:', i)
        # print(W0.eval(session = sess))
        print('Train:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))
        batch_xs, batch_ys = get_batch(v_dmat,v_answers,VALIDATION_SIZE)
        v_acc = accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys})
        print('Validation:', v_acc)
        if not switched and v_acc > 0.8:
            print('SWITCH')
            train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
            batch_size = end_batch_size
            switched = True
        # print(cross_entropy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))

batch_xs, batch_ys = get_batch(v_dmat, v_answers, VALIDATION_SIZE)
print('Validation Accuracy:', accuracy.eval(session = sess, feed_dict={x: batch_xs, y_: batch_ys}))
# print(prediction.eval(session=sess, feed_dict = {x: dmat}))

test_answers = [int(k) if not math.isnan(k) else 0 for k in prediction.eval(session=sess, feed_dict = {x: tdmat})]

f = open(DIRECTORY + 'submission.txt', 'w')
f.write('PassengerId,Survived\n')
for i in range(len(data), len(data) + len(tdata)):
    f.write(str(i+1) +','+ str(test_answers[i-len(data)]) +'\n')
f.close()





