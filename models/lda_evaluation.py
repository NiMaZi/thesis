import os
import sys
import csv
import json
import boto3
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping

from gensim.corpora import Dictionary
from gensim.models import *
homedir=os.environ['HOME']

def get_bucket():
	s3 = boto3.resource("s3")
	myBucket=s3.Bucket('workspace.scitodate.com')
	return myBucket

def load_sups():
	f=open(homedir+"/results/ontology/ConCode2Vid.json",'r')
	cc2vid=json.load(f)
	f.close()
	return cc2vid

def get_model_S3(_name):
	bucket=get_bucket()
	homedir=os.environ['HOME']
	try:
		os.remove(homedir+"/temp/tmp_model2.h5")
	except:
		pass
	bucket.download_file("yalun/results/models/"+_name+".h5",homedir+"/temp/tmp_model2.h5")
	model=load_model(homedir+"/temp/tmp_model2.h5")
	return model

def get_model_local(path):
	return load_model(path)

def test_on_doc_S3_atmodel(_lda,_model,_volume,_alpha=0.5,_threshold=0.0):
    bucket=get_bucket()
    cc2vid=load_sups()
    P_all=0.0
    R_all=0.0
    all_count=0.0
    sample_list=[]
    for i in range(0,_volume*100,100):
        try:
            bucket.download_file("yalun/annotated_papers_authors/abs"+str(i)+".csv",homedir+"/temp/tmp.csv")
            bucket.download_file("yalun/annotated_papers_authors/authors"+str(i)+".json",homedir+"/temp/tmp.json")
        except:
            continue
        abs_vec=[0.0 for ii in range(0,len(cc2vid))]
        abs_count=0.0
        with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
            rd=csv.reader(cf)
            for item in rd:
                if item[0]=="Mention":
                    continue
                try:
                    abs_vec[cc2vid[item[1]]]+=1.0
                    abs_count+=1.0
                except:
                    pass
        if not abs_count:
            continue
        abs_vec=np.array(abs_vec)/abs_count
        try:
            bucket.download_file("yalun/annotated_papers_authors/body"+str(i)+".csv",homedir+"/temp/tmp.csv")
        except:
            continue
        body_vec=[0.0 for ii in range(0,len(cc2vid))]
        body_count=0.0
        with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
            rd=csv.reader(cf)
            for item in rd:
                if item[0]=="Mention":
                    continue
                try:
                    body_vec[cc2vid[item[1]]]+=1.0
                    body_count+=1.0
                except:
                    pass
        if not body_count:
            continue
        body_vec=list(np.array(body_vec)/body_count)
        author_vec=[0.0 for ii in range(0,len(cc2vid))]
        f=open(homedir+"/temp/tmp.json")
        authors=json.load(f)
        f.close()
        author_count=0
        for author in authors:
            try:
                b_topic=max(_lda.get_author_topics(author),key=lambda x:x[1])[0]
                terms=_lda.get_topic_terms(b_topic,20)
                for term in terms:
                    author_vec[cc2vid[dictionary.id2token[term[0]]]]+=term[1]
                author_count+=1
            except:
                pass
        if author_count:
            author_vec=np.array(author_vec)/author_count
        refined_vec=list(_alpha*author_vec*(abs_vec.max()/author_vec.max())+(1-_alpha)*abs_vec)
        sample_list=[refined_vec+body_vec]
        N_test=np.array(sample_list)
        X_test=N_test[:,:len(cc2vid)]
        Y_test=np.clip(np.ceil(N_test[:,len(cc2vid):])-np.ceil(X_test),0.0,1.0)[0].astype(int)
        Y_pred=_model.predict(X_test)[0]
        Y_pred/=Y_pred.max()
        Y_pred[Y_pred<_threshold]=0.0
        Y_pred[Y_pred>=_threshold]=1.0
        Y_pred=np.ceil(Y_pred).astype(int)
        tp=float(np.sum(Y_pred&Y_test))
        fp=float(np.sum(Y_pred-(Y_pred&Y_test)))
        fn=float(np.sum(Y_test-(Y_pred&Y_test)))
        try:
            P=tp/(tp+fp)
        except:
            P=0.0
        try:
            R=tp/(tp+fn)
        except:
            R=0.0
        P_all+=P
        R_all+=R
        all_count+=1
    P_all/=all_count
    R_all/=all_count
    try:
        F1=2*P_all*R_all/(P_all+R_all)
    except:
        F1=0.0
    return P_all,R_all,F1

f=open(homedir+"/results/ontology/c2n.json",'r')
c2n=json.load(f)
f.close()
prefix='http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'
ncit_dict=[k.split('#')[1] for k in c2n.keys()]
dictionary=Dictionary([ncit_dict]);dictionary[0]

model_name="MLPsparse_1hidden"
model=get_model_S3(model_name)

topic_num=[5,10,20,25,40,50,100,200,250]

for tn in topic_num:
	lda=AuthorTopicModel.load(homedir+"/results/models/lda2000_topic"+str(tn))
	threshold=0.0
	volume=100
	while threshold<1.0:
		alpha=0.0
		while alpha<1.0:
			P,R,F=test_on_doc_S3_atmodel(lda,model,volume,alpha,threshold)
			f=open(homedir+"/results/logs/lda_eval_topic"+str(tn),'a')
			f.write("%.3f,%.3f,%.3f,%.3f,%.3f\n"%(threshold,alpha,P,R,F))
			f.close
			alpha+=0.1
		threshold+=0.1