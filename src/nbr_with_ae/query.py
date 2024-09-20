#!/usr/bin/env python3
import tensorflow as tf
import pickle
import numpy as np

class NBR:
    def __init__( self, aeFile, nbrFile ):
        checkpt = pickle.load(open(aeFile,"rb" ))
        aeW1 = tf.constant(checkpt[0])
        aeb1 = tf.constant(checkpt[1])
        aeW2 = tf.constant(checkpt[2])
        aeb2 = tf.constant(checkpt[3])
        checkpt = pickle.load(open(nbrFile,"rb" ))
        rW1 = [tf.constant(checkpt[0][i]) for i in range(2)]
        rb1 = [tf.constant(checkpt[1][i]) for i in range(2)]
        rW2 = [tf.constant(checkpt[2][i]) for i in range(2)]
        rb2 = [tf.constant(checkpt[3][i]) for i in range(2)]
        rW3 = [tf.constant(checkpt[4][i]) for i in range(2)]
        rb3 = [tf.constant(checkpt[5][i]) for i in range(2)]
        self.b0= checkpt[6][0]
        self.b1= checkpt[6][1]
        self.x = tf.placeholder(tf.float32,shape=[None,11])
        layer  = tf.nn.relu(tf.matmul(self.x,aeW1)+aeb1)
        layer  = tf.nn.relu(tf.matmul(layer,aeW2)+aeb2)
        layer  = tf.nn.relu([tf.matmul(layer,rW1[i])+rb1[i] for i in range(2)])
        layer  = tf.nn.relu([tf.matmul(layer[i],rW2[i])+rb2[i] for i in range(2)])
        self.o = tf.sigmoid([tf.tensordot(layer[i],rW3[i],1)+rb3[i] for i in range(2)])
        self.s = tf.Session()

    def query( self, index, condition ):
        data0,data1 = NBR.generateData(index, condition)
        o0  = self.s.run(self.o,{self.x:data0})
        r00 = max(o0[0])
        r01 = max(o0[1])
        r02 = max(np.minimum(o0[0],o0[1]))
        o1  = self.s.run(self.o,{self.x:data1})
        r10 = max(o1[0])
        r11 = max(o1[1])
        r12 = max(np.minimum(o1[0],o1[1]))
        c0  = (1.-self.b0)*(1.-self.b1)
        c   = self.b0*self.b1*max(r02,r12)+self.b0*(1.0-self.b1)*max(r00,r10)+self.b1*(1.0-self.b0)*max(r01,r11)+c0
        pl  = self.b0*self.b1*r12+self.b0*(1.0-self.b1)*r10+self.b1*(1.0-self.b0)*r11+c0
        pl0 = self.b0*self.b1*r02+self.b0*(1.0-self.b1)*r00+self.b1*(1.0-self.b0)*r01+c0
        return 1.-pl0/c,pl/c

    @staticmethod
    def generateData( index, condition ):
        indices = [i for i in range(11) if i not in condition and i != index]
        newwidth = len(indices)
        partials = [[int(bit) for bit in format(sample,'0'+ str(newwidth)+'b')] for sample in [i for i in range(pow(2,newwidth))]]
        size = len(partials)
        buffer = [0]*11
        for i in condition: buffer[i] = condition[i]
        buffer[index] = 0
        data0 = [buffer[:] for i in range(size)]
        buffer[index] = 1
        data1 = [buffer[:] for i in range(size)]
        for i in range(size):
            for ii in range(newwidth):
                data0[i][indices[ii]] = partials[i][ii]
                data1[i][indices[ii]] = partials[i][ii]
        return data0,data1

def display( cmd ):
    print(cmd,':',eval(cmd))

nbr = NBR('ae.pickle','nbr.pickle')
display('nbr.query(10,{0:1})')
display('nbr.query(10,{0:0})')
display('nbr.query(5,{0:1,1:0,2:0,3:0,4:0})')
display('nbr.query(5,{0:1,1:0,2:0,3:0,4:0,10:1})')
display('nbr.query(5,{0:1,1:0,2:0,3:0,4:0,6:1,7:1,8:1,9:1,10:1})')

