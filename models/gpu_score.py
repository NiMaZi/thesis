import os
import json
import numpy as np
from scipy.spatial.distance import cosine
from keras.models import Sequential, load_model
from keras.layers import LSTM,Bidirectional,Masking,BatchNormalization
from keras.callbacks import EarlyStopping
from gensim.models import word2vec as w2v

dim=128
maxlen=512
volume=1000
homedir=os.environ['HOME']

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
    return corpus[volume:volume+200]

path=homedir+"/thesiswork/source/corpus/fullcorpusall.txt"
corpus=load_corpus(path)

def find_match(vec,num):
    min_dis=np.inf
    min_word=None
    for w in word_list:
        dis=cosine(vec,get_emb(w.split('#')[1]))
        if dis<min_dis:
            min_dis=dis
            min_word=w.split('#')[1]
    return min_word,min_dis

def test_on_data(_corpus,_maxlen,_model):
    i=0
    comp_vec=[0.0 for i in range(0,128)]
    ndata=[]
    hit=0.0
    while(i<len(_corpus)-1):
        _body=_corpus[i]
        i+=1
        _rbody=set(_corpus[i])
        b_emb=[]
        if len(_body)<_maxlen:
            for w in _body:
                b_emb.append(get_emb(w))
            for j in range(len(b_emb),_maxlen):
                b_emb.append(comp_vec)
            ndata=np.array([b_emb])
            y_out=model.predict(ndata)
            match,dis=find_match(y_out[0],1)
            if match in _rbody:
                hit+=1.0
        else:
            all_match=[]
            for j in range(0,len(_body)-_maxlen+1):
                b_emb=[]
                for wj in range(0,_maxlen):
                    w=_body[j+wj]
                    b_emb.append(get_emb(w))
                ndata=[b_emb]
                y_out=model.predict(ndata)
                match,dis=find_match(y_out[0],1)
                all_match.append((match,dis))
            best_match=min(all_match,key=lambda x:x[1])[0]
            if best_match in _rbody:
                hit+=1.0
        i+=1
    hit/=len(_corpus)/2.0
    return hit

mod_no=140
logf=open(homedir+"/results/logs/BiLSTMGPU_log.txt",'a')
while mod_no<=170:
    model=load_model(homedir+"/results/models/BiLSTMGPU"+str(mod_no)+".h5")
    hit=test_on_data(corpus,maxlen,model)
    logf.write("%d,%.3f\n"%(mod_no,hit))
    mod_no+=10
logf.close()