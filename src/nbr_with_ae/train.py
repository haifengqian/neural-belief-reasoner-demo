#!/usr/bin/env python3
import tensorflow as tf
import pickle
import random
import sys

dataFile1 = 'samples1.txt'
dataFile2 = 'samples2.txt'
aeFile    = 'ae.pickle'
outFile   = 'nbr.pickle'
batchSize = 64
widths    = [200]*2
tf.set_random_seed(1)
random.seed(1)

def readFile( filename ):
    data = []
    file = open( filename, 'r' )
    while(1):
        line = file.readline()
        words = line.split()
        if len(words) < 1: break
        data.append( [int(word) for word in words] )
    file.close()
    return data

def genData(samples1,samples2):
    data10 = []
    data11 = []
    data20 = []
    data21 = []
    for sample in samples1:
        buffer = sample[:]
        buffer.append(0)
        data10.append(buffer[:])
        buffer[width-1] = 1
        data11.append(buffer[:])
    for sample in samples2:
        buffer = sample[:]
        buffer.insert(0,0)
        data20.append(buffer[:])
        buffer[0] = 1
        data21.append(buffer[:])
    return data10,data11,data20,data21

def getBatch( data1, data2, data3, data4, index ):
    begin = index
    end = index + batchSize
    size = len(data1)
    index = end % size
    if end <= size: return data1[begin:end],data2[begin:end],data3[begin:end],data4[begin:end],index,False
    batch1 = data1[begin:size]
    batch1.extend(data1[0:index])
    batch2 = data2[begin:size]
    batch2.extend(data2[0:index])
    batch3 = data3[begin:size]
    batch3.extend(data3[0:index])
    batch4 = data4[begin:size]
    batch4.extend(data4[0:index])
    return batch1,batch2,batch3,batch4,index,True

def getBatch1( data, index ):
    begin = index
    end = index + batchSize
    size = len(data)
    index = end % size
    if end <= size: return data[begin:end],index,False
    batch = data[begin:size]
    batch.extend(data[0:index])
    return batch,index,True

def shuffleData( data1, data2 ):
    comb = list(zip(data1,data2))
    random.shuffle(comb)
    d1,d2 = zip(*comb)
    return list(d1),list(d2)

def inference( x ):
    layer = tf.concat([tf.concat([tf.minimum(x,[1.]*(width-1)+[0.]),
                                  tf.maximum(x,[0.]*(width-1)+[1.])],0),
                       tf.concat([tf.minimum(x,[0.]+[1.]*(width-1)),
                                  tf.maximum(x,[1.]+[0.]*(width-1))],0)],0)
    layer = tf.nn.relu(tf.matmul(layer,aeW1)+aeb1)
    layer = tf.nn.relu(tf.matmul(layer,aeW2)+aeb2)
    layer = tf.reshape(layer,[2,-1,hwidth])
    layer = tf.nn.relu([tf.matmul(layer[i],rW1[i])+rb1[i] for i in range(2)])
    layer = tf.nn.relu([tf.matmul(layer[i],rW2[i])+rb2[i] for i in range(2)])
    layer = [tf.tensordot(layer[i],rW3[i],1)+rb3[i] for i in range(2)]
    layer = tf.reduce_min(tf.reshape(layer,[2,2,-1]),1)
    return tf.sigmoid(layer)

def inference2( x ):
    layer = tf.nn.relu(tf.matmul(x,aeW1)+aeb1)
    layer = tf.nn.relu(tf.matmul(layer,aeW2)+aeb2)
    layer = tf.nn.relu([tf.matmul(layer,rW1[i])+rb1[i] for i in range(2)])
    layer = tf.nn.relu([tf.matmul(layer[i],rW2[i])+rb2[i] for i in range(2)])
    layer = [tf.tensordot(layer[i],rW3[i],1)+rb3[i] for i in range(2)]
    return tf.sigmoid(layer)

def mylog( t ):
    return tf.log(tf.maximum(t,1e-6))

def getLoss( x0, x1 ):
    o0 = inference(x0)
    o1 = inference(x1)
    return -tf.reduce_mean(mylog((1-b[0])*(1-b[1])+b[0]*(1-b[1])*.5*(o0[0]+o1[0])+b[1]*(1-b[0])*.5*(o0[1]+o1[1])+b[0]*b[1]*.5*(tf.minimum(o0[0],o0[1])+tf.minimum(o1[0],o1[1]))))

def getLossP( x ):
    o = inference2(x)
    return (1-b[0])*(1-b[1])+b[0]*(1-b[1])*tf.reduce_mean(o[0])+b[1]*(1-b[0])*tf.reduce_mean(o[1])+b[0]*b[1]*tf.reduce_mean(tf.minimum(o[0],o[1]))

def getNormalVar( shape ):
    return [tf.Variable(tf.truncated_normal(shape,0.,1./float(shape[0]))) for i in range(2)]

def load( sess ):
    sess.run([[tf.assign(rW1[ii],ct[0][ii]),tf.assign(rb1[ii],ct[1][ii]),
               tf.assign(rW2[ii],ct[2][ii]),tf.assign(rb2[ii],ct[3][ii]),
               tf.assign(rW3[ii],ct[4][ii]),tf.assign(rb3[ii],ct[5][ii]),
               bv[ii]] for ii in range(2)])

samples1 = readFile( dataFile1 )
samples2 = readFile( dataFile2 )
width    = len(samples1[0])+1
data10,data11,data20,data21 = genData(samples1,samples2)
data10,data11 = shuffleData(data10,data11)
data20,data21 = shuffleData(data20,data21)
prio     = [[int(bit) for bit in format(i,'0'+str(width)+'b')] for i in range(pow(2,width))]
random.shuffle(prio)
aeChckpt = pickle.load(open(aeFile,"rb"))
aeW1     = tf.constant(aeChckpt[0])
aeb1     = tf.constant(aeChckpt[1])
aeW2     = tf.constant(aeChckpt[2])
aeb2     = tf.constant(aeChckpt[3])
hwidth   = len(aeChckpt[3])

rW1 = getNormalVar([hwidth,widths[0]])
rb1 = [tf.Variable(tf.constant(0.1,shape=[widths[0]])) for i in range(2)]
rW2 = getNormalVar([widths[0],widths[1]])
rb2 = [tf.Variable(tf.constant(0.1,shape=[widths[1]])) for i in range(2)]
rW3 = getNormalVar([widths[1]])
rb3 = [tf.Variable(tf.constant(0.0)) for i in range(2)]
bv  = [tf.Variable(tf.constant(0.0)) for i in range(2)]
x10 = tf.placeholder(tf.float32, shape=[None, width])
x11 = tf.placeholder(tf.float32, shape=[None, width])
x20 = tf.placeholder(tf.float32, shape=[None, width])
x21 = tf.placeholder(tf.float32, shape=[None, width])
p   = tf.placeholder(tf.float32, shape=[None, width])
epc = tf.placeholder(tf.float32)
b   = tf.sigmoid(bv)
lossk = (getLoss(x10,x11)+getLoss(x20,x21))*.5
ep    = getLossP(p)
loss  = lossk+mylog(ep)
losst = lossk+ep/epc
train = [tf.train.AdamOptimizer(1e-3).minimize(losst),
         tf.train.AdamOptimizer(5e-4).minimize(losst),
         tf.train.AdamOptimizer(2e-4).minimize(losst),
         tf.train.AdamOptimizer(1e-4).minimize(losst),
         tf.train.AdamOptimizer(5e-5).minimize(losst)]
sess  = tf.Session()
sess.run(tf.global_variables_initializer())

batchBase  = 0
batchBaseP = 0
newepc     = sess.run(ep,{p:prio})
lastloss   = 0
ti         = 0
for i in range(100000):
    batch10,batch11,batch20,batch21,batchBase,shuffle=getBatch(data10,data11,data20,data21,batchBase)
    batchP,batchBaseP,shuffleP = getBatch1(prio,batchBaseP)
    sess.run(train[ti],{x10:batch10,x11:batch11,x20:batch20,x21:batch21,p:batchP,epc:newepc})
    if shuffle:
        data10,data11 = shuffleData(data10,data11)
        data20,data21 = shuffleData(data20,data21)
    if shuffleP: random.shuffle(prio)
    if i%20 == 0: newepc = sess.run(ep,{p:prio})
    if i%500 == 0:
        newloss = sess.run(loss,{x10:data10,x11:data11,x20:data20,x21:data21,p:prio})
        print('iter',i,'loss',newloss)
        sys.stdout.flush()
        if newloss < lastloss:
            lastloss = newloss
            ct = sess.run([rW1,rb1,rW2,rb2,rW3,rb3,bv])
        if i in {30000,60000,80000,90000}:
            ti += 1
            load(sess)
load(sess)
finaloss = sess.run(loss,{x10:data10,x11:data11,x20:data20,x21:data21,p:prio})
print('final loss',finaloss)
pickle.dump([sess.run(rW1),sess.run(rb1),sess.run(rW2),sess.run(rb2),
             sess.run(rW3),sess.run(rb3),sess.run(b)],
            open(outFile,'wb'))

