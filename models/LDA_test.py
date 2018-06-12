import os
import csv
import json
import numpy as np
import boto3
s3 = boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')
homedir=os.environ['HOME']
from gensim.corpora import Dictionary
from gensim.models import *
f=open(homedir+"/results/ontology/c2n.json",'r')
c2n=json.load(f)
f.close()
prefix='http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'
ncit_dict=[k.split('#')[1] for k in c2n.keys()]
dictionary=Dictionary([ncit_dict]);dictionary[0]
dictionary.save(homedir+"/results/models/lda_dict")
tfidf=TfidfModel.load(homedir+"/results/models/tfidf_model")

for i in [2,3,4]:
    lda=AuthorTopicModel(id2word=dictionary.id2token,num_topics=i,eval_every=False)
    f=open(homedir+"/thesiswork/source/corpus/lda_doc_2k.json",'r')
    _corpus=json.load(f)
    f.close()
    bow_corpus=[dictionary.doc2bow(doc) for doc in _corpus]
    tfidf_corpus=[tfidf[doc] for doc in bow_corpus]
    f=open(homedir+"/thesiswork/source/corpus/lda_d2a_2k.json",'r')
    _d2a=json.load(f)
    f.close()
    d2a={}
    for k in _d2a.keys():
        d2a[int(k)]=_d2a[k]
    lda.update(corpus=tfidf_corpus,doc2author=d2a)
    lf=open(homedir+"/results/logs/lda_training_log.txt",'a')
    lf.write("finish training "+str(i)+" topics.\n")
    lf.close()
    lda.save(homedir+"/results/models/lda2000_body_topic"+str(i))