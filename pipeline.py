import subprocess
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import boto3
import random
s3 = boto3.resource("s3")
myBucket=s3.Bucket('workspace.scitodate.com')
import os
import sys
homedir=os.environ['HOME']
import csv
import json
elastic_ip_address=sys.argv[1]
term=sys.argv[2]
S3_key="yalun/experiment_generic/"
folder_term=term.replace(" ","_")+"/"
print(term,folder_term)
subprocess.call(command='./tunnel_elastic.bash '+elastic_ip_address)
es=Elasticsearch(['localhost:9200'])
results=scan(es,
    query={
        "query": {
            "bool": {
                "must": [
                    {"exists": {"field":"abstract"}},
                    {"exists": {"field":"body"}},
                    {"match_phrase": {"body":{"query":term,"slop":5}}}
                ],
            }
        }
    },
    size=10
)
paper_id_list=[]
for i,result in enumerate(results):
    paper_id_list.append(result['_id'])
paper_id_list=list(set(paper_id_list))
for t in range(0,10):
    random.shuffle(paper_id_list)
f=open(homedir+"/temp/index.json",'w')
json.dump(paper_id_list,f)
f.close()
f=open(homedir+"/temp/index.json",'r')
d=f.read()
f.close()
myBucket.put_object(Body=d,Key=S3_key+folder_term+"index.json")
for i,a in enumerate(paper_id_list):
    results=scan(es,
        query={
            "query": {
                "bool": {
                    "must": [{"match": {"ids": a}}]
                }
            }
        },
        size=1
    )

    for result in results:
        try:
            abstract=result['_source']['abstract']

            f=open("tmp.txt",'w',encoding='utf-8')
            f.write(str(abstract))
            f.close()
            f=open("tmp.txt",'r',encoding='utf-8')
            data=f.read()
            f.close()
            myBucket.put_object(Body=data,Key=S3_key+folder_term+a+"_abs.txt")

            body=result['_source']['body']

            f=open("tmp.txt",'w',encoding='utf-8')
            f.write(str(body))
            f.close()
            f=open("tmp.txt",'r',encoding='utf-8')
            data=f.read()
            f.close()
            myBucket.put_object(Body=data,Key=S3_key+folder_term+a+"_body.txt")

            title=result['_source']['title']

            f=open("tmp.txt",'w',encoding='utf-8')
            f.write(str(title))
            f.close()
            f=open("tmp.txt",'r',encoding='utf-8')
            data=f.read()
            f.close()
            myBucket.put_object(Body=data,Key=S3_key+folder_term+a+"_title.txt")

            authors=result['_source']['authors']

            f=open("tmp.json",'w',encoding='utf-8')
            json.dump(authors,f)
            f.close()
            f=open("tmp.json",'r',encoding='utf-8')
            data=f.read()
            f.close()
            myBucket.put_object(Body=data,Key=S3_key+folder_term+a+"_authors.json")

        except:
            pass