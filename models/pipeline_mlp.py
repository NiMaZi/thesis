import sys
import csv
import json
import numpy as np
from gensim.models import word2vec
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Activation
from keras.layers import Dropout

batch_size=int(sys.argv[1])
epochs=int(sys.argv[2])
hidden_expansion=float(sys.argv[3])
dropout_rate=float(sys.argv[4])
es_patience=float(sys.argv[5])

input_dimension=542

MLP_model=Sequential()
MLP_model.add(Dense(int(hidden_expansion*input_dimension),input_dim=input_dimension,activation='linear'))
MLP_model.add(Dropout(dropout_rate))
MLP_model.add(Dense(input_dimension,activation='sigmoid'))
MLP_model.add(Dropout(dropout_rate))
MLP_model.add(Dense(1,activation='sigmoid'))
MLP_model.compile(optimizer='sgd',loss='binary_crossentropy')

early_stopping=EarlyStopping(monitor='loss',patience=es_patience)

f=open("/home/ubuntu/results/saliency/featured_list_com.json",'r')
featured_list_com=json.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/word_list.json",'r')
word_list=json.load(f)
f.close()

jaccard=np.load("/home/ubuntu/results/ontology/jaccard.npy")
milne_witten=np.load("/home/ubuntu/results/ontology/milne_witten.npy")
adamic_adar=np.load("/home/ubuntu/results/ontology/adamic_adar.npy")
dice=np.load("/home/ubuntu/results/ontology/dice.npy")

f=open("/home/ubuntu/results/ontology/n2id.json",'r')
n2id=json.load(f)
f.close()
f=open("/home/ubuntu/results/statistics/idf.json",'r')
idf=json.load(f)
f.close()
f=open("/home/ubuntu/results/statistics/tf_all.json",'r')
tf_all=json.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/KG_n2v.json",'r')
KG_n2v=json.load(f)
f.close()

KG_e2v=word2vec.Word2Vec.load("/home/ubuntu/results/e2v_sg_e100.model")

cooc_simple=np.load("/home/ubuntu/results/statistics/cooc_simple.npy")

batched_list=[]
for i,record in enumerate(featured_list_com):
    if i>=4000:
        break
    if len(batched_list)>batch_size:
        X_train=np.array(batched_list)[:,1:]
        y_train=np.array(batched_list)[:,0]
        MLP_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,callbacks=[early_stopping])
        batched_list=[]
    _abs=record['abs']
    _body=record['body']
    for a in _abs.keys():
        if not a in word_list:
            continue
        for b in _body.keys():
            if not b in word_list or a==b:
                continue
            jacc=jaccard[n2id[a]][n2id[b]]
            mlwt=milne_witten[n2id[a]][n2id[b]]
            aa=adamic_adar[n2id[a]][n2id[b]]
            di=dice[n2id[a]][n2id[b]]
            cocs=cooc_simple[n2id[a]][n2id[b]]
            idf_a=idf[a]
            idf_b=idf[b]
            tf_all_a=tf_all[a]
            tf_all_b=tf_all[b]
            tfidf_a=tf_all_a*idf_a
            tfidf_b=tf_all_b*idf_b
            nodevec_a=KG_n2v[str(n2id[a])]
            nodevec_b=KG_n2v[str(n2id[b])]
            wordvec_a=list(KG_e2v.wv[a])
            wordvec_b=list(KG_e2v.wv[b])
            term_feature=[1.0,_abs[a][0],_abs[a][1],_abs[a][2],jacc,mlwt,aa,di,cocs,idf_a,idf_b,tf_all_a,tf_all_b,tfidf_a,tfidf_b]
            term_embedding=nodevec_a+nodevec_b+wordvec_a+wordvec_b
            term_all=term_feature+term_embedding
            batched_list.append(term_all)
            ct=4
            while(ct):
                w=word_list[np.random.randint(len(word_list))]
                if not w in _body.keys():
                    jacc=jaccard[n2id[a]][n2id[w]]
                    mlwt=milne_witten[n2id[a]][n2id[w]]
                    aa=adamic_adar[n2id[a]][n2id[w]]
                    di=dice[n2id[a]][n2id[w]]
                    cocs=cooc_simple[n2id[a]][n2id[w]]
                    idf_a=idf[a]
                    idf_b=idf[w]
                    tf_all_a=tf_all[a]
                    tf_all_b=tf_all[w]
                    tfidf_a=tf_all_a*idf_a
                    tfidf_b=tf_all_b*idf_b
                    nodevec_a=KG_n2v[str(n2id[a])]
                    nodevec_b=KG_n2v[str(n2id[w])]
                    wordvec_a=list(KG_e2v.wv[a])
                    wordvec_b=list(KG_e2v.wv[w])
                    term_feature=[0.0,_abs[a][0],_abs[a][1],_abs[a][2],jacc,mlwt,aa,di,cocs,idf_a,idf_b,tf_all_a,tf_all_b,tfidf_a,tfidf_b]
                    term_embedding=nodevec_a+nodevec_b+wordvec_a+wordvec_b
                    term_all=term_feature+term_embedding
                    batched_list.append(term_all)
                    ct-=1

if batched_list:
    X_train=np.array(batched_list)[:,1:]
    y_train=np.array(batched_list)[:,0]
    MLP_model.fit(X_train,y_train,batch_size=batch_size,epochs=epochs,callbacks=[early_stopping])

path="/home/ubuntu/results_new/models/MLP_r.h5"
MLP_model.save(path)