import os
import csv
import json
import numpy as np
import boto3
s3 = boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')
homedir=os.environ['HOME']

source="annotated_papers_authors"
corpus=[]
d2a={}
c=0
for i in range(0,200000,100):
    tmpc=[]
    try:
        myBucket.download_file("yalun/"+source+"/body"+str(i)+".csv",homedir+"/temp/ldactmp.csv")
        with open(homedir+"/temp/ldactmp.csv",'r',encoding='utf-8') as csvf:
            rd=csv.reader(csvf)
            for item in rd:
                if item[0]=='Mention':
                    continue
                tmpc.append(item[1])
        myBucket.download_file("yalun/"+source+"/authors"+str(i)+".json",homedir+"/temp/ldajtmp.json")
        f=open(homedir+"/temp/ldajtmp.json",'r',encoding='utf-8')
        authors=json.load(f)
        f.close()
        if not authors:
            continue
        corpus.append(tmpc)
        d2a[c]=authors
        c+=1
    except:
        pass

author_list=[]
for d in d2a.keys():
    author_list+=d2a[d]
author_list=list(set(author_list))
from gensim.corpora import Dictionary
from gensim.models import *
f=open(homedir+"/results/ontology/c2n.json",'r')
c2n=json.load(f)
f.close()
prefix='http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'
ncit_dict=[k.split('#')[1] for k in c2n.keys()]
dictionary=Dictionary([corpus])
print(dictionary[0])
bow_corpus=[dictionary.doc2bow(doc) for doc in corpus]
tfidf=TfidfModel(bow_corpus)
tfidf_corpus=[tfidf[doc] for doc in bow_corpus]

res=[]
for i in [5,10,20,25,40,50,100,200,250,400,500,1000]:
    lda=AuthorTopicModel(tfidf_corpus,num_topics=i,doc2author=d2a,eval_every=False)
    lda.save(homedir+"/results/models/lda2000rdc_topic"+str(i))
    try:
        cm=CoherenceModel(model=lda,corpus=tfidf_corpus,texts=corpus,dictionary=dictionary,coherence='u_mass',processes=4)
        u_mass=cm.get_coherence()
    except:
        u_mass=''
    try:
        cm=CoherenceModel(model=lda,corpus=tfidf_corpus,texts=corpus,dictionary=dictionary,coherence='c_v',processes=4)
        c_v=cm.get_coherence()
    except:
        c_v=''
    try:
        cm=CoherenceModel(model=lda,corpus=tfidf_corpus,texts=corpus,dictionary=dictionary,coherence='c_uci',processes=4)
        c_uci=cm.get_coherence()
    except:
        c_uci=''
    try:
        cm=CoherenceModel(model=lda,corpus=tfidf_corpus,texts=corpus,dictionary=dictionary,coherence='c_npmi',processes=4)
        c_npmi=cm.get_coherence()
    except:
        c_npmi=''
    res.append((i,u_mass,c_v,c_uci,c_npmi))

f=open(homedir+"/results/logs/lda2000rdc.csv",'w')
wt=csv.writer(f)
for r in res:
    wt.writerow(r)
f.close()