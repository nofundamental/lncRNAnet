#!/usr/bin/python3
import math, os, random, sys, time
from random import shuffle
from math import ceil

#keras
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model, load_model
from keras.layers import LSTM, Dropout, Dense, Input, Activation, Masking, merge, Lambda, TimeDistributed, Flatten
#etc
from Bio import SeqIO
from Bio.Seq import Seq
import numpy as np

#usr
import preprocessing
#####################################################################################
#preference
sys.setrecursionlimit(100000)
toolname='lncRNAnet'
#variable
chars="0AGCT"
inD=len(chars)-1
outD=2
#hyperparameters
dropout=0.5
height=2
hidden=100
orfref=10000
orfD=2
batch_size=128
clen=10000
_buckets=[500*i for i in range(1,201)]

def findORF(seq):
    orflen=0
    orf=""
    o_s=0
    o_e=0
    length=len(seq)
    seq=[seq]
    seq=preprocessing.pad_sequences(seq,maxlen=length,padding='post')
    seq=(np.arange(seq.max()+1) == seq[:,:,None]).astype(dtype='float32')
    seq=np.delete(seq,0,axis=-1)
    if seq.shape[2]==3:
        zeros_col = np.zeros((seq.shape[0],seq.shape[1],1))
        seq = np.concatenate((seq,zeros_col),axis=2)
    for frame in range(3):
        tseq=stopmodel.predict(seq[:,frame:])[:,:(length-frame)//3]
        tseq=np.argmax(tseq,axis=-1)-1
        sseq=np.append(-1,np.where(tseq==1)[1])
        sseq=np.append(sseq,tseq.shape[1])
        lseq=np.diff(sseq)-1
        flenp=np.argmax(lseq)
        flen=lseq[flenp]
        n_s=frame+3*sseq[flenp]+3
        n_e=frame+3*sseq[flenp+1]
        
        if flen>orflen or ((orflen==flen) and n_s<o_s):
            orflen=flen
            o_s=n_s
            o_e=n_e
                
    return o_s,o_e

def batchfindORF(seqs):
    batchsize=len(seqs)
    batches=[]
    for seq in seqs:
        length=len(seq)
        o_s,o_e=findORF(seq)
        orf=np.ones(o_e-o_s)
        orf=np.concatenate([2*np.ones(o_s),orf,2*np.ones(length-o_e)],axis=0)
        batches.append(orf)
    
    batches=np.array(batches)
    return batches

def bucket_generator_ORF(X,Y,batchsize):
    data_set = [[] for _ in _buckets]
    for i,x in enumerate(X):
        for b_id, _ in enumerate(_buckets):
            if len(x)<=_:
                data_set[b_id].append(i)
                break
    k=0
    len_set=[int(len(_)/batchsize)+ceil((len(_)%batchsize)/batchsize) for _ in data_set]
    tot_batch=sum(len_set)
    while(1):
        if k%tot_batch==0:
            k=0
            shuffled_batch=np.arange(tot_batch)#batches shuffle
            shuffled_data_set=[]
            for data in data_set:
                b_data=data
                shuffle(b_data)
                shuffled_data_set.append(b_data)#bucket shuffle
            
            np.random.shuffle(shuffled_batch)
        
        cur_batch=shuffled_batch[k]
        for s_i,l_b in enumerate(len_set):
            if cur_batch<l_b:
                batch_index=np.array(shuffled_data_set[s_i][cur_batch*batchsize:(cur_batch+1)*batchsize])
                X_batch=X[batch_index]
                orf_batch=batchfindORF(X_batch)
                orf_batch=preprocessing.pad_sequences(orf_batch,maxlen=_buckets[s_i])
                orf_batch=(np.arange(orf_batch.max()+1) == orf_batch[:,:,None]).astype(dtype='float32')
                orf_batch=np.delete(orf_batch,0,axis=-1)
                X_batch=preprocessing.pad_sequences(X_batch, maxlen=_buckets[s_i])
                X_batch=(np.arange(X_batch.max()+1) == X_batch[:,:,None]).astype(dtype='float32')
                X_batch=np.delete(X_batch,0,axis=-1)
                Y_batch=Y[batch_index]
                #Y_batch=(np.arange(2)==Y_batch).astype(dtype='float32')
                
                yield [X_batch,orf_batch],Y_batch
                break
            else:
                cur_batch-=l_b   
        k+=1

def count_batches(X,Y,batchsize):
    data_set = [[] for _ in _buckets]
    for i,x in enumerate(X):
        for b_id, _ in enumerate(_buckets):
            if len(x)<=_:
                data_set[b_id].append(i)
                break
    
    return tot_batch

def predict_splitter(X,batchsize):
    X_bucket=[]
    data_set = [[] for _ in _buckets]
    tot_index=np.zeros(0)
    for i,x in enumerate(X):
        for b_id, _ in enumerate(_buckets):
            if len(x)<=_:
                data_set[b_id].append(i)
                break
    
    for b_id, _ in enumerate(data_set):
        if len(data_set[b_id])==0:
            continue
        batches=len(_)
        x_index=np.array(data_set[b_id])
        tot_index=np.concatenate([tot_index,x_index],axis=0)
        
        index=np.arange(batches)
        X_prime=X[x_index[index]]
        orf_prime=batchfindORF(X_prime)
        orf_prime=preprocessing.pad_sequences(orf_prime,maxlen=_buckets[b_id])
        orf_prime=(np.arange(orf_prime.max()+1) == orf_prime[:,:,None]).astype(dtype='float32')
        orf_prime=np.delete(orf_prime,0,axis=-1)
        X_prime=preprocessing.pad_sequences(X_prime,maxlen=_buckets[b_id])
        X_prime=(np.arange(X_prime.max()+1) == X_prime[:,:,None]).astype(dtype='float32') #one_hot
        X_prime=np.delete(X_prime,0,axis=-1)
        X_bucket.append([X_prime,orf_prime])
    
    
    return X_bucket,tot_index

def readfile(filename):
    f=open(filename)
    sl=list(SeqIO.parse(f,'fasta'))
    ids=[]
    seqs=[]
    
    for s in sl:
        id=s.id
        seq=str(s.seq).upper()
        ids.append(id)
        seqs.append(seq)
    
    return ids, seqs

def seqtodata(seqs):
    X=[]
    for seq in seqs:
        X.append(ctable.encode(seq))
    
    X=np.array(X)
    return X

ctable = preprocessing.CharacterTable(chars)

def predict(infile,outfile):
    ids,seqs=readfile(infile)
    numseq=len(ids)
    X=seqtodata(seqs)
    Y_predicted=np.empty([0,2])
    X_b,tot_index=predict_splitter(X,batch_size)
    if X_b[0][0].shape[2] == 3:
        zeros_col = np.zeros((X_b[0][0].shape[0],X_b[0][0].shape[1],1))
        X_b[0][0] = np.concatenate((X_b[0][0],zeros_col),axis=2)
    if X_b[0][1].shape[2] == 1:
        col = np.abs(X_b[0][1]-1)
        X_b[0][1] = np.concatenate((X_b[0][1],col),axis=2)
    for i in range(len(X_b)):
        Y_predicted=np.concatenate([Y_predicted,model.predict(X_b[i],batch_size=1024)],axis=0)
    
    Y_predicted=Y_predicted[np.argsort(tot_index)]
    
    g=open(outfile,'w')
    
    for i in range(numseq):
        g.write(ids[i]+'\t'+str(len(seqs[i]))+'\t'+str(Y_predicted[i,1]-Y_predicted[i,0])+'\n')
    
    g.close()
    print("Prediction complete: total", str(len(Y_predicted)), " sequences")
    return Y_predicted

def nn():
    
    RNN=LSTM
    
    rnn_input=Input(shape=(None,inD))
    orf_input=Input(shape=(None,orfD))
    
    #orf
    orf_size=Lambda(lambda x: K.expand_dims(K.sum(x,axis=-2)[:,0]/orfref,axis=-1), output_shape=lambda  s: (s[0],1))(orf_input)#.repeat(1)
    orf_ratio=Lambda(lambda x: K.sum(x,axis=-1),output_shape=lambda s: (s[0],s[1]))(rnn_input)
    orf_ratio=Lambda(lambda x: orfref/(K.sum(x,axis=-1,keepdims=True)+1),output_shape=lambda s: (s[0],1))(orf_ratio)
    orf_ratio=merge([orf_size,orf_ratio],mode='dot')
        
    orf_in=Masking()(orf_input)
    rnn_in=Masking()(rnn_input)
    
    orf_in=RNN(hidden,return_sequences=True,consume_less='gpu')(orf_in)
    rnn_in=RNN(hidden,return_sequences=True,consume_less='gpu')(rnn_in)
        
    rnn_in=merge([orf_in,rnn_in],mode='concat')
    rnn_in=RNN(hidden,return_sequences=False,consume_less='gpu') (rnn_in)
    rnn_in=Dropout(dropout)(rnn_in)
        
    rnn_in=merge([rnn_in,orf_size,orf_ratio],mode='concat')
    rnn_out=Dense(outD)(rnn_in)
    rnn_act=Activation('softmax')(rnn_out)
    
    model=Model(input=[rnn_input,orf_input],output=rnn_act)
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.summary()
    
    return model

stopmodel=load_model('./data/model/stopfinder_singleframe.h5')
model=nn()
model.load_weights('./data/model/lncRNAnet.h5')

def main():
        options=sys.argv
        inputf=options[-2]
        outputf=options[-1]
        Y_p=predict(inputf,outputf)
        


if __name__=="__main__":
    main()

