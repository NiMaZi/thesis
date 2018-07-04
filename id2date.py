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

s3key=sys.argv[1]

id2date={}
for i in range(0,15000):
    try:
        mybucket.download_file("yalun/"+s3key+"/abs"+str(i)+".txt",homedir+"/temp/tmptxt"+s3key+".txt")
    except:
        continue
    f=open(homedir+"/temp/tmptxt"+s3key+".txt",'r',encoding='utf-8')
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

f=open(homedir+"/temp/id2date"+s3key+".json",'w')
json.dump(id2date,f)
f.close()
f=open(homedir+"/temp/id2date"+s3key+".json",'rb')
d=f.read()
f.close()
mybucket.put_object(Body=d,Key="yalun/"+s3key+"/id2date.json")