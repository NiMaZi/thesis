import boto3
s3=boto3.resource("s3")
sourceBucket=s3.Bucket('papers.scitodate.com')

import os
homedir=os.environ['HOME']
import json
import csv
import time
import numpy as np
myBucket=s3.Bucket('workspace.scitodate.com')

author_hist={}

for i in range(0,10000):#26294
    try:
        myBucket.download_file("yalun/Dependence/body"+str(i)+".csv",homedir+"/temp/tmpcsv.csv")
    except:
        continue
    try:
        myBucket.download_file("yalun/Dependence/authors"+str(i)+".json",homedir+"/temp/tmpjson.json")
    except:
        continue
    body_vec=np.array([0.0 for i in range(0,5)])
    body_count=0.0
    with open(homedir+"/temp/tmpcsv.csv",'r',encoding='utf-8') as cf:
        rd=csv.reader(cf)
        for item in rd:
            if item[0]=='Mention':
                continue
            if item[1]=='C93040':	#Alcohol
                body_vec[0]=1.0
            if item[1]=='C35386' or item[1]=='C35387' or item[1]=='C34445':	#Cannabis
                body_vec[1]=1.0
            if item[1]=='C34492' or item[1]=='C35389' or item[1]=='C35388':	#Cocaine
                body_vec[2]=1.0
            if item[1]=='C34694':	#Heroin
                body_vec[3]=1.0
            if item[1]=='C70989' or item[1]=='C54203' or item[1]=='C15985':	#Nicotine
                body_vec[4]=1.0
    f=open(homedir+"/temp/tmpjson.json",'r',encoding='utf-8')
    authors=json.load(f)
    f.close()
    for author in authors:
        if author in author_hist.keys():
            author_hist[author][0]+=body_vec
            author_hist[author][1]+=1.0
        else:
            author_hist[author]=[body_vec,1.0]

for author in author_hist.keys():
    if not author_hist[author][1]==0.0:
        author_hist[author][0]/=author_hist[author][1]
for author in author_hist.keys():
    author_hist[author][0]=list(author_hist[author][0])

f=open(homedir+"/results/statistics/authorhist_dependence.json",'w')
json.dump(author_hist,f)
f.close()