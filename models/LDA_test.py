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
tfidf=TfidfModel.load(homedir+"/models/tfidf_model")

for i in [5,10,20,25,40,50,100,200,250,400,500,1000]:
    lda=AuthorTopicModel(id2word=dictionary.id2token,num_topics=i,eval_every=False)
    for j in range(0,19):
        f=open(homedir+"/thesiswork/source/corpus/lda_doc_part"+str(j),'r')
        _corpus=json.load(f)
        f.close()
        bow_corpus=[dictionary.doc2bow(doc) for doc in _corpus]
        tfidf_corpus=[tfidf[doc] for doc in bow_corpus]
        f=open(homedir+"/thesiswork/source/corpus/lda_d2a_part"+str(j),'r')
        d2a=json.load(f)
        f.close()
        lda=AuthorTopicModel(corpus=tfidf_corpus,doc2author=d2a)
        f.open(homedir+"/results/logs/lda_training_log.txt",'a')
        f.write("finish training %d-%d\n"%(i,j))
        f.close()
    lda.save(homedir+"/results/models/lda20000_topic"+str(i))