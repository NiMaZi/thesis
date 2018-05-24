import json
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
    return corpus[10000:10200]

path="/home/ubuntu/thesiswork/source/corpus/fullcorpus10000.txt"
corpus=load_corpus(path)

def topn(l,n):
    r=[]
    for i in l:
        r.append(i)
        if len(r)>n:
            r.remove(min(r,key=lambda x:x[1]))
    return sorted(r,key=lambda x:-x[1])

def test_on_data(_corpus,_maxlen,_model):
    i=0
    P1=0.0
    P5=0.0
    P1k=0.0
    R1=0.0
    R5=0.0
    R1k=0.0
    comp_vec=[0.0 for i in range(0,128)]
    while(i<len(_corpus)-1):
        ndata=[]
        nlabel=[]
        _abs=_corpus[i]
        a_emb=[]
        for w in _abs:
            a_emb.append(get_emb(w))
        for j in range(len(a_emb),_maxlen-1):
            a_emb.append(comp_vec)
        i+=1
        _body=set(_corpus[i])
        pred=[]
        for j,w in enumerate(word_list):
            ndata=[a_emb+[get_emb(w.split('#')[1])]]
            y_out=_model.predict(np.array(ndata))
            if y_out[0][0]:
                pred.append((w.split('#')[1],y_out[0][0]))
        result=topn(pred,1000)
        i+=1
        H1=0.0
        H5=0.0
        H1k=0.0
        for j,wt in enumerate(result):
            if wt[0] in _body:
                if j<=100:
                    H1+=1
                if j<=500:
                    H5+=1
                if j<=1000:
                    H1k+=1
        print("hit @ 100: %.3f, hit @ 500: %.3f, hit @ 1k: %.3f."%(H1/100,H5/500,H1k/1000))
        P1+=H1/100
        P5+=H5/500
        P1k+=H1k/1000
        R1+=H1/len(_body)
        R5+=H5/len(_body)
        R1k+=H1k/len(_body)
    P1/=len(_corpus)/2
    P5/=len(_corpus)/2
    P1k/=len(_corpus)/2
    R1/=len(_corpus)/2
    R5/=len(_corpus)/2
    R1k/=len(_corpus)/2
    return P1,R1,P5,R5,P1k,R1k
            
doc=100
logf=open("/home/ubuntu/results/logs/BiLSTM_cls_log.txt",'a')
while(doc<=2500):
    model=load_model("/home/ubuntu/results/models/BiLSTM_cls_doc"+str(doc)+".h5")
    P1,R1,P5,R5,P1k,R1k=test_on_data(corpus,maxlen,model)
    logf.write("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n"%(doc,P1,R1,P5,R5,P1k,R1k))
    doc+=100
logf.close()