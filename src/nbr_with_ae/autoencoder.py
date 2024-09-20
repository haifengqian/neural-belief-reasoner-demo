#!/usr/bin/env python3
import tensorflow as tf
import pickle
import sys
import random

width     = 11
hwidth    = [100,8]
batchSize = 64
outFile   = 'ae.pickle'
tf.set_random_seed(1)
random.seed(1)

def getBatch( data, index ):
    begin = index
    end = index + batchSize
    size = len(data)
    index = end % size
    if end <= size: return data[begin:end],index,False
    batch = data[begin:size]
    batch.extend(data[0:index])
    return batch,index,True

data= [[int(bit) for bit in format(i,'0'+str(width)+'b')] for i in range(pow(2,width))]
random.shuffle(data)

W1 = tf.Variable(tf.truncated_normal([width,hwidth[0]],0.,1./float(width)))
b1 = tf.Variable(tf.constant(0.1,shape=[hwidth[0]]))
W2 = tf.Variable(tf.truncated_normal(hwidth,0.,1./float(hwidth[0])))
b2 = tf.Variable(tf.constant(0.1, shape=[hwidth[1]]))
W3 = tf.Variable(tf.truncated_normal([hwidth[1],hwidth[0]],0.,1./float(hwidth[1])))
b3 = tf.Variable(tf.constant(0.1, shape=[hwidth[0]]))
W4 = tf.Variable(tf.truncated_normal([hwidth[0],width],0.,1./float(hwidth[0])))
b4 = tf.Variable(tf.constant(0.1,shape=[width]))
x  = tf.placeholder(tf.float32,shape=[None,width])
h  = tf.nn.relu(tf.matmul(x,W1)+b1)
h  = tf.nn.relu(tf.matmul(h,W2)+b2)
h  = tf.nn.relu(tf.matmul(h,W3)+b3)
o  = tf.matmul(h,W4)+b4
wd    = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)+tf.nn.l2_loss(W3)+tf.nn.l2_loss(W4)
loss  = tf.reduce_mean(tf.square(o-x))
train1= tf.train.AdamOptimizer(1e-3).minimize(loss+0.0001*wd)
train2= tf.train.AdamOptimizer(1e-4).minimize(loss+0.00001*wd)
sess  = tf.Session()
sess.run(tf.global_variables_initializer())

batchBase = 0
for i in range(100000):
    batch,batchBase,shuffle = getBatch(data,batchBase)
    if shuffle: random.shuffle(data)
    sess.run(train1 if i<50000 else train2,{x:batch})
    if i%5000 == 0:
        print('iter',i,'loss',sess.run(loss,{x:data}))
        sys.stdout.flush()
finaloss = sess.run(loss,{x:data})
print('final loss',finaloss)
pickle.dump([sess.run(W1),sess.run(b1),sess.run(W2),sess.run(b2),
             sess.run(W3),sess.run(b3),sess.run(W4),sess.run(b4)],open(outFile,'wb'))

