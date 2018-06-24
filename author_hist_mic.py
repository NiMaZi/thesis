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
        myBucket.download_file("yalun/Microscopy/body"+str(i)+".csv",homedir+"/temp/tmpcsvmic.csv")
    except:
        continue
    try:
        myBucket.download_file("yalun/Microscopy/authors"+str(i)+".json",homedir+"/temp/tmpjsonmic.json")
    except:
        continue
    body_vec=np.array([0.0 for i in range(0,5)])
    body_count=0.0
    with open(homedir+"/temp/tmpcsvmic.csv",'r',encoding='utf-8') as cf:
        rd=csv.reader(cf)
        for item in rd:
            if item[0]=='Mention':
                continue
            if item[1]=='C78814' or item[1]=='C78815': #SEM
                body_vec[0]=1.0
            if item[1]=='C78860' or item[1]=='C78813':  #TEM
                body_vec[1]=1.0
            if item[1]=='C17374':   #STM
                body_vec[2]=1.0
            if item[1]=='C78804':   #AFM
                body_vec[3]=1.0
            if item[1]=='C17753' or item[1]=='C122390' or item[1]=='C116477' or item[1]=='C116481': #Confocal
                body_vec[4]=1.0
    f=open(homedir+"/temp/tmpjsonmic.json",'r',encoding='utf-8')
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

f=open(homedir+"/results/statistics/authorhist_microscopy.json",'w')
json.dump(author_hist,f)
f.close()