import json
import random
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Bidirectional, Masking, BatchNormalization
from keras.callbacks import EarlyStopping
from gensim.models import word2vec as w2v

dim=128
maxlen=300

def load_models():
    path="/home/ubuntu/results/models/e2v_sg_10000_e100_d64.model"
    e2v_model=w2v.Word2Vec.load(path)
    f=open("/home/ubuntu/results/ontology/KG_n2v_d64.json",'r')
    n2v_model=json.load(f)
    f.close()
    return e2v_model,n2v_model

e2v_model,n2v_model=load_models()

def load_sups():
    f=open("/home/ubuntu/results/ontology/c2id.json",'r')
    c2id=json.load(f)
    f.close()
    f=open("/home/ubuntu/results/ontology/full_word_list.json",'r')
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
    return corpus

path="/home/ubuntu/thesiswork/source/corpus/fullcorpus5000.txt"
corpus=load_corpus(path)

def build_model(_input_dim,_input_length):
    model=Sequential()
    model.add(Masking(mask_value=0.0,input_shape=(maxlen,dim)))
    model.add(BatchNormalization())
    model.add(Bidirectional(LSTM(1,return_sequences=False,dropout=0.5,activation="relu"),merge_mode='ave'))
    model.compile(optimizer='nadam',loss='binary_crossentropy')
    return model

# model=load_model("/home/ubuntu/results/models/LSTM1001.h5")
model=build_model(dim,maxlen)

def train_on_data(_corpus,_maxlen,_model,_epochs):
    early_stopping=EarlyStopping(monitor='loss',patience=10)
    i=0
    comp_vec=[0.0 for i in range(0,128)]
    while(i<len(_corpus)-1):
        print("training on doc "+str(i))
        ndata=[]
        nlabel=[]
        _abs=_corpus[i]
        a_emb=[]
        for w in _abs:
            a_emb.append(get_emb(w))
        for j in range(len(a_emb),_maxlen-1):
            a_emb.append(comp_vec)
        i+=1
        _body=_corpus[i]
        for w in _body:
            ndata.append(a_emb+[get_emb(w)])
            nlabel.append(1.0)
            ct=3
            while ct>0:
                negw=random.choice(word_list).split('#')[1]
                if negw in _body:
                    continue
                ndata.append(a_emb+[get_emb(negw)])
                nlabel.append(0.0)
                ct-=1
        i+=1
        X_train=np.array(ndata)
        y_train=np.array(nlabel)
        _model.fit(X_train,y_train,batch_size=1024,epochs=_epochs,verbose=0,shuffle=True,callbacks=[early_stopping])
        if i%10==0:
            _model.save("/home/ubuntu/results/models/BiLSTM_cls_doc"+str(i)+".h5")

train_on_data(corpus,maxlen,model,5)