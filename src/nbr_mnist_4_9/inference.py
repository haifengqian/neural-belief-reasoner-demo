#!/usr/bin/env python3
import sys
import os
import time
import pickle
import numpy as np
import tensorflow as tf

TEST_IMAGE_FILE  = 't10k-images-idx3-ubyte'
TEST_LABEL_FILE  = 't10k-labels-idx1-ubyte'
MODEL_FILE       = 'nbr_mnist_4_9.pickle'

def readImages( filename ):
    if filename == '' or os.path.isfile(filename) == False:
        print('unable to open input file:',filename)
        sys.exit()
    file = open( filename, "rb" )
    u32 = np.dtype(np.uint32).newbyteorder('>')
    magic = np.frombuffer(file.read(4), dtype=u32)[0]
    if magic != 2051:
        print('invalid input file:',filename)
        sys.exit()
    count = np.frombuffer(file.read(4), dtype=u32)[0]
    dim1 = np.frombuffer(file.read(4), dtype=u32)[0]
    if dim1 != 28:
        print('invalid input file:',filename)
        sys.exit()
    dim2 = np.frombuffer(file.read(4), dtype=u32)[0]
    if dim2 != 28:
        print('invalid input file:',filename)
        sys.exit()
    data = np.frombuffer(file.read(count*dim1*dim2),dtype=np.uint8).reshape(count,dim1,dim2,1)
    file.close()
    return data

def readLabels( filename ):
    if filename == '' or os.path.isfile(filename) == False:
        print('unable to open input file:',filename)
        sys.exit()
    file = open( filename, "rb" )
    u32 = np.dtype(np.uint32).newbyteorder('>')
    magic = np.frombuffer(file.read(4), dtype=u32)[0]
    if magic != 2049:
        print('invalid input file:',filename)
        sys.exit()
    count = np.frombuffer(file.read(4), dtype=u32)[0]
    data = np.frombuffer( file.read(count), dtype=np.uint8)
    file.close()
    return data

def pooling( inlayer ):
    return 2.0*tf.sqrt(tf.nn.avg_pool(inlayer*inlayer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME'))

def inference( x, filename ):
    checkpt   = pickle.load(open(filename,"rb"))
    lgt       = []
    for j in range(len(checkpt[1])):
        layer = x
        for i in range(len(checkpt[1][j])):
            layer = tf.nn.conv2d(layer/float(len(checkpt[1][j][i])),tf.constant(checkpt[1][j][i]),strides=[1,1,1,1],padding='SAME')+tf.constant(checkpt[2][j][i])
            layerf= tf.nn.relu(-layer)
            layer = tf.nn.relu( layer)
            layer = tf.concat([layer,layerf],3)
        layer = pooling(layer)
        for i in range(len(checkpt[3][j])):
            layer = tf.nn.conv2d(layer/float(len(checkpt[3][j][i])),tf.constant(checkpt[3][j][i]),strides=[1,1,1,1],padding='SAME')+tf.constant(checkpt[4][j][i])
            layerf= tf.nn.relu(-layer)
            layer = tf.nn.relu( layer)
            layer = tf.concat([layer,layerf],3)
        layer = pooling(layer)
        layer = tf.reshape(layer,[-1,len(checkpt[5][j][0])])
        for i in range(len(checkpt[5][j])):
            layer = tf.matmul(layer,tf.constant(checkpt[5][j][i]))+tf.constant(checkpt[6][j][i])
            layerf= tf.nn.relu(-layer)
            layer = tf.nn.relu( layer)
            layer = tf.concat([layer,layerf],1)
        lgt.append(tf.reduce_sum(layer*tf.constant(checkpt[7][j]),1)+tf.constant(checkpt[8][j]))
    lgt  = tf.transpose(lgt)
    imgx = tf.reshape(x,[-1,1,784])
    for i in range(len(checkpt[15])):
        buf = [i.tolist() for i in checkpt[15][i][0]]
        buf = tf.constant(checkpt[15][i][1])-tf.norm(imgx-tf.constant(buf),axis=-1)
        lgt = tf.concat([lgt,buf],1)
    o    = tf.sigmoid(lgt*tf.constant(checkpt[9],tf.float32))-0.5
    o1   = 0.5+tf.nn.relu(o)-tf.nn.relu(-o)*tf.constant(checkpt[12])
    o2   = 1.-o1
    od   = o2-o1
    sign = tf.nn.relu(tf.sign(od))
    b    = tf.sigmoid(tf.constant(checkpt[10]))
    w1   = tf.log(tf.maximum(1.-b*od*sign/(1.-b+b*o2),1e-6))
    w2   = tf.log(tf.maximum(1.+b*od*(1.-sign)/(1.-b+b*o1),1e-6))
    rmat1= tf.constant([[[float(ii in checkpt[13][i][0]) for ii in range(len(checkpt[9]))]] for i in range(10)])
    rmat2= tf.constant([[[float(ii in checkpt[13][i][1]) for ii in range(len(checkpt[9]))]] for i in range(10)])
    return tf.transpose(tf.reduce_sum(w1*rmat1,-1)+tf.reduce_sum(w2*rmat2,-1))


x        = tf.placeholder(tf.float32,[None,28,28,1])
y        = tf.placeholder(tf.int64,  [None])
o        = tf.minimum(inference(x,MODEL_FILE),[-1e6]*4+[1e6]+[-1e6]*4+[1e6])
accu     = tf.reduce_sum(tf.cast(tf.equal(y,tf.argmax(o,1)),tf.float32))
testData = readImages(TEST_IMAGE_FILE)
testLabl = readLabels(TEST_LABEL_FILE)
testData = [testData[i] for i in range(len(testData)) if testLabl[i] in {4,9}]
testLabl = [i for i in testLabl if i in {4,9}]
testData = np.divide(testData,255.)
size     = len(testData)
sess     = tf.Session()
bsize    = 500
accuO    = 0.
timestart= time.time()
for i in range(0,size,bsize):
    limit = min(size,i+bsize)
    accuO += sess.run(accu,{x:testData[i:limit],y:testLabl[i:limit]})
timeend  = time.time()
print("Took",timeend-timestart,"seconds")
print("Natural accuracy:",accuO/float(size))

