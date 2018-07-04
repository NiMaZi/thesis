import os
import sys
import csv
import json
import boto3
import numpy as np

homedir=os.environ['HOME']
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
es=Elasticsearch(['localhost:9200'])

s3 = boto3.resource("s3")
mybucket=s3.Bucket("workspace.scitodate.com")

id2date={}
for i in range(0,100):
    try:
        mybucket.download_file("yalun/Dependence/abs"+str(i)+".txt",homedir+"/temp/tmptxt0.txt")
    except:
        continue
    f=open(homedir+"/temp/tmptxt0.txt",'r',encoding='utf-8')
    abstract=f.read().split(",")[0]
    f.close()
    results=scan(es,
        query={
            "query": {
                "bool": {
                    "must": [{"match_phrase": {"abstract": abstract}}]
                }
            }
        },
        size=1
    )
    for n,result in enumerate(results):
        if n>10:
            break
        if abstract in result['_source']['abstract']:
            try:
                id2date[i]=result['_source']['date']
            except:
                break

f=open(homedir+"/temp/id2date0.json",'w')
json.dump(id2date,f)
f.close()
f=open(homedir+"/temp/id2date0.json",'rb')
d=f.read()
f.close()
mybucket.put_object(Body=d,Key="yalun/Dependence/id2date.json")