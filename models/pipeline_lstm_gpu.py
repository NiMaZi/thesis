import os
import json
import boto3
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM,Bidirectional,Masking,BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K
from gensim.models import word2vec as w2v

inner_batch_size=256
outer_batch_size=2048
dim=128
maxlen=512
volume=1000
homedir=os.environ['HOME']

s3 = boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')

def load_models():
    path=homedir+"/results/models/e2v_sg_10000_e100_d64.model"
    e2v_model=w2v.Word2Vec.load(path)
    f=open(homedir+"/results/ontology/KG_n2v_d64.json",'r')
    n2v_model=json.load(f)
    f.close()
    return e2v_model,n2v_model

e2v_model,n2v_model=load_models()

def load_sups():
    f=open(homedir+"/results/ontology/c2id.json",'r')
    c2id=json.load(f)
    f.close()
    f=open(homedir+"/results/ontology/full_word_list.json",'r')
    word_list=json.load(f)
    f.close()
    prefix='http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'
    return c2id,prefix,word_list[1:]

c2id,prefix,word_list=load_sups()

def get_emb(_code):
    e_vec=list(e2v_model.wv[_code])
    n_vec=n2v_model[str(c2id[prefix+_code])]
    return e_vec+n_vec

def load_corpus(_path):
    f=open(_path,'r')
    pre_corpus=f.read()
    f.close()
    pre_list=pre_corpus.split("\n")[:-1]
    corpus=[]
    for i,p in enumerate(pre_list):
        _p=p.split(" ")[:-1]
        corpus.append(_p)
    return corpus[:volume]

path=homedir+"/thesiswork/corpus/fullcorpusall.txt"
corpus=load_corpus(path)

def upload_model(_model_path,_model_name):
    f=open(_model_path,'rb')
    d=f.read()
    f.close
    myBucket.put_object(Body=d,Key="yalun/results/models/"+_model_name)


def my_cos_loss(y_true,y_pred):
    ny_true=K.l2_normalize(y_true,axis=-1)
    ny_pred=K.l2_normalize(y_pred,axis=-1)


def build_model(_input_dim,_input_length):
    model=Sequential()
    model.add(Masking(mask_value=0.0,input_shape=(_input_length,_input_dim)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(_input_dim,return_sequences=False,dropout=0.4,activation="tanh"),merge_mode='ave'))
    # model.add(BatchNormalization())
    model.compile(optimizer='nadam',loss='cosine_proximity')
    return model

# model=load_model(homedir+"/results/models/BiLSTM40.h5")
model=build_model(dim,maxlen)

def train_on_data(_corpus,_maxlen,_model,_epochs):
    early_stopping=EarlyStopping(monitor='loss',patience=5)
    early_stopping_val=EarlyStopping(monitor='val_loss',patience=5)
    count=167
    i=619
    comp_vec=[0.0 for i in range(0,128)]
    ndata=[]
    while(i<len(_corpus)-1):
        print("training on doc "+str(i))
        _body=_corpus[i]
        b_emb=[]
        if len(_body)<_maxlen:
            print("short body.")
            for j in range(0,len(_body)-1):
                b_emb.append(get_emb(_body[j]))
            for j in range(len(b_emb)-1,_maxlen-1):
                b_emb.append(comp_vec)
            b_emb.append(get_emb(_body[len(_body)-1]))
            ndata.append(b_emb)
            if len(ndata)>=outer_batch_size:
                print("training on batch "+str(count))
                # print(len(ndata),len(ndata[0]),len(ndata[0][0]))
                N_all=np.array(ndata)
                # N_all=N_all.reshape(1024,_maxlen,128)
                # print(N_all.shape)
                X_train=N_all[:,:-1,:]
                y_train=N_all[:,-1,:]
                _model.fit(X_train,y_train,batch_size=inner_batch_size,epochs=_epochs,validation_split=1.0/16.0,verbose=0,shuffle=True,callbacks=[early_stopping,early_stopping_val])
                if count%10==0:
                    _model.save(homedir+"/results/models/BiLSTM"+str(count)+".h5")
                    upload_model(homedir+"/results/models/BiLSTM"+str(count)+".h5","BiLSTMGPU"+str(count)+".h5")
                count+=1
                ndata=[]
        else:
            print("long body.")
            for j in range(0,len(_body)-_maxlen+1):
                b_emb=[]
                for wj in range(0,_maxlen):
                    w=_body[j+wj]
                    b_emb.append(get_emb(w))
                ndata.append(b_emb)
                if len(ndata)>=outer_batch_size:
                    print("training on batch "+str(count))
                    N_all=np.array(ndata)
                    # N_all=N_all.reshape(1024,_maxlen,128)
                    # print(N_all.shape)
                    X_train=N_all[:,:-1,:]
                    y_train=N_all[:,-1,:]
                    _model.fit(X_train,y_train,batch_size=inner_batch_size,epochs=_epochs,validation_split=1.0/16.0,verbose=1,shuffle=True,callbacks=[early_stopping,early_stopping_val])
                    if count%10==0:
                        _model.save(homedir+"/results/models/BiLSTM"+str(count)+".h5")
                        upload_model(homedir+"/results/models/BiLSTM"+str(count)+".h5","BiLSTMGPU"+str(count)+".h5")
                    count+=1
                    ndata=[]
        i+=2
    if ndata:
        print("training on batch "+str(count))
        N_all=np.array(ndata)
        X_train=N_all[:,:-1,:]
        y_train=N_all[:,-1,:]
        _model.fit(X_train,y_train,batch_size=inner_batch_size,epochs=_epochs,validation_split=1.0/16.0,verbose=0,shuffle=True,callbacks=[early_stopping,early_stopping_val])
        if count%10==0:
            _model.save(homedir+"/results/models/BiLSTM"+str(count)+".h5")
            upload_model(homedir+"/results/models/BiLSTM"+str(count)+".h5","BiLSTMGPU"+str(count)+".h5")
        count+=1
        ndata=[]

train_on_data(corpus,maxlen+1,model,10)