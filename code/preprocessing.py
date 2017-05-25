#!/usr/bin/python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import one_hot

import math
import os
import random
import sys
import time

import numpy as np
import h5py

#reload : 
"""
txt="seq_t.txt"
h5file="test.h5"
chars="0ACGU"
"""
def read_h5_data(h5file): #matfile read  or h5
    h5f=h5py.File(h5file)
    return h5f
    
def txttobinary(txt,h5file,chars,maxlength=10000): # arbitrary variable length ints
    output=h5py.File(h5file,'w')
    seqs=[]
    f=open(txt)
    maxlen=0
    ctable = CharacterTable(chars)
    for l in f:
        l=l.strip()
        if len(l)<maxlength:
            l=ctable.encode(l)
            seqs.append(l)
            maxlen+=1
    
    seqs=np.asarray(seqs)
    dt = h5py.special_dtype(vlen=np.dtype('int32'))
    binaries=output.create_dataset('seqs',(maxlen,),dtype=dt)
    for i in range(seqs.shape[0]):
        binaries[i]=seqs[i]
    output.close() 

def one_hot(seqs,chars): # seq(1d-> 2d one hot)
    ints_len = seqs.shape[1]
    dictionary_k = len(chars)+1 #np.int(np.max(ints)+1)
    ints_enc = np.zeros((ints_len, dictionary_k))
    ints_enc[np.arange(ints_len), [k for k in seqs]] = 1
    ints_enc=np.delete(ints_enc,0,axis=1) # delete padded sequence
    return ints_enc.tolist()

def one_hot_seq2seq(seqs,chars): # seq(1d-> 2d one hot)
    ints_len = seqs.shape[1]
    dictionary_k = len(chars) #np.int(np.max(ints)+1)
    ints_enc = np.zeros((ints_len, dictionary_k))
    ints_enc[np.arange(ints_len), [k for k in seqs]] = 1
    return ints_enc.tolist()

def tensorshape(seq,chars): #whole seq-> whole one hot return 3d shaping 
    codedseq=[]
    for i in range(seq.shape[0]):
        hot=one_hot(seq[i:i+1,:],chars)
        codedseq.append(hot)
    return np.array(codedseq)

def tensorshape_seq2seq(seq,chars): #whole seq-> whole one hot return 3d shaping 
    codedseq=[]
    for i in range(seq.shape[0]):
        hot=one_hot_seq2seq(seq[i:i+1,:],chars)
        codedseq.append(hot)
    return np.array(codedseq)

class CharacterTable(object): #make encoding table
    '''
    Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    #chars : 0 (padding ) + other characters
    '''
    def __init__(self, chars):
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
    
    def encode(self, l):
        X = np.zeros((len(l)),dtype=int)
        for i, c in enumerate(l):
            X[i]= self.char_indices[c]
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)
        
def dataset(seq,maxlen):
    sset=[]
    
    for i in range(seq.shape[0]):
        s=seq[i:i+1,:]
        #ns=np.nonzero(s)
        #s=s[ns]
        s=[int(x)+1 for x in s]
        sset.append(s)
        
    sset=pad_sequences(sset,timestep,padding='pre')
    return sset

"""
def edsequence(seq,struct):
    modelinput=np.concatenate([seq,struct],axis=1)
    modeloutput=np.concatenate([np.zeros((struct.shape[0],timestep)),struct[:,1:],np.zeros((struct.shape[0],1))],axis=1)
    modeloutput=np.array(modeloutput,dtype='int32')
    return modelinput,modeloutput
"""
