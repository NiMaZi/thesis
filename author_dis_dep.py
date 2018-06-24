import boto3
s3 = boto3.resource("s3")
sourceBucket=s3.Bucket('papers.scitodate.com')
import os
homedir=os.environ['HOME']
import json
import csv
import time
import numpy as np
import scipy.stats as scis
myBucket=s3.Bucket('workspace.scitodate.com')

f=open(homedir+"/results/statistics/authorhist_dependence.json",'r',encoding='utf-8')
author_hist=json.load(f)
f.close()

res=[]
for i in range(0,10000):
    try:
        myBucket.download_file("yalun/Dependence/body"+str(i)+".csv",homedir+"/temp/tmpcsv1.csv")
    except:
        continue
    try:
        myBucket.download_file("yalun/Dependence/authors"+str(7*i)+".json",homedir+"/temp/tmpjson1.json")
    except:
        continue
    body_vec=np.array([0.0 for i in range(0,5)])
    body_count=0.0
    with open(homedir+"/temp/tmpcsv1.csv",'r',encoding='utf-8') as cf:
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
    f=open(homedir+"/temp/tmpjson1.json",'r',encoding='utf-8')
    authors=json.load(f)
    f.close()
    ahs=[]
    for author in authors:
        if author in author_hist.keys():
            ahs.append(author_hist[author][0])
    if not all([ah==ahs[0] for ah in ahs]):
        for ah in ahs:
            dis=np.linalg.norm(body_vec-np.array(ah))
            top=max(ah)
            cut=max(ah)-min(ah)
            var=np.var(np.array(ah))
            res.append((dis,top,cut,var))

res=list(set(res))
f=open(homedir+"/results/logs/authordis_dep.csv",'w')
wt=csv.writer(f)
for r in res:
	wt.writerow(r)
f.close()

dis=[r[0] for r in res]
top=[r[1] for r in res]
cut=[r[2] for r in res]
var=[r[3] for r in res]
f=open(homedir+"/results/logs/authordis_dep.txt",'w')
f.write("%f,%f,%f,%f,%f,%f\n"%(scis.pearsonr(dis,top)[0],scis.pearsonr(dis,top)[1],scis.spearmanr(dis,top)[0],scis.spearmanr(dis,top)[1],scis.kendalltau(dis,top)[0],scis.kendalltau(dis,top)[1]))
f.write("%f,%f,%f,%f,%f,%f\n"%(scis.pearsonr(dis,cut)[0],scis.pearsonr(dis,cut)[1],scis.spearmanr(dis,cut)[0],scis.spearmanr(dis,cut)[1],scis.kendalltau(dis,cut)[0],scis.kendalltau(dis,cut)[1]))
f.write("%f,%f,%f,%f,%f,%f\n"%(scis.pearsonr(dis,var)[0],scis.pearsonr(dis,var)[1],scis.spearmanr(dis,var)[0],scis.spearmanr(dis,var)[1],scis.kendalltau(dis,var)[0],scis.kendalltau(dis,var)[1]))
f.close()