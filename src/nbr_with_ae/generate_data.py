#!/usr/bin/env python3
import random

width = 9
samplesize  = 10
noiseratio1 = 10
noiseratio2 = 5
outfile1 = 'samples1.txt'
outfile2 = 'samples2.txt'
random.seed(1)

def writeFile( filename, data ):
    dataFile = open( filename, 'w' )
    for sample in data:
        for bit in sample: dataFile.write( repr(bit)+' ' )
        dataFile.write( '\n' )
    dataFile.close()

centerall = [[int(bit) for bit in format(sample, '0' + str(width) + 'b')] for sample in [i for i in range(pow(2,width))] ]
samples = []
for sample in centerall:
    buffer = sample[:]
    buffer.insert( 0, int(sum(sample)*2 > width) )
    for i in range(samplesize-int(samplesize/noiseratio1)): samples.append(buffer[:])
    buffer[0] = 1-buffer[0]
    for i in range(int(samplesize/noiseratio1)): samples.append(buffer[:])
random.shuffle(samples)
writeFile(outfile1,samples)
samples = []
for sample in centerall:
    buffer = sample[:]
    buffer.append( int(sum(sample)*2 < width) )
    for i in range(samplesize-int(samplesize/noiseratio2)): samples.append(buffer[:])
    buffer[width] = 1-buffer[width]
    for i in range(int(samplesize/noiseratio2)): samples.append(buffer[:])
random.shuffle(samples)
writeFile(outfile2,samples)



