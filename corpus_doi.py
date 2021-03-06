import os
import sys
import boto3
import json
import jsonlines
import csv
import subprocess

pid=int(sys.argv[1])
term=sys.argv[2]
# start=int(sys.argv[2])
# end=int(sys.argv[3])

homedir=os.environ['HOME']
s3 = boto3.resource("s3")
sourceBucket=s3.Bucket('workspace.scitodate.com')
targetBucket=s3.Bucket('workspace.scitodate.com')
obj_prefix="yalun/experiment_generic/"+term+"/"

def get_annotation(_inpath):
    subprocess.call(['java','-jar',homedir+'/ner/NobleJar/NobleCoder-1.0.jar','-terminology','NCI_Thesaurus','-input',_inpath,'-output',homedir+'/thesiswork/disambiguation'+str(pid),'-search','best-match','-selectBestCandidates'])
    f=open(homedir+'/thesiswork/disambiguation'+str(pid)+'/RESULTS.tsv','r',encoding='utf-8')
    unamb=f.read()
    f.close()
    tmp_list=unamb.split('\n')
    unamb_list=[]
    for item in tmp_list:
        unamb_list.append(item.split('\t'))
    result_list=[]
    result_list.append(['Mention','ConceptCode','ConceptName','SemanticType','Start'])
    unamb_list=unamb_list[1:len(unamb_list)-1]
    for item in unamb_list:
        result_list.append([item[1],item[2],item[3],item[4],item[5].split(',')[0].split('/')[1]])
    f=open(homedir+'/thesiswork/annotation'+str(pid)+'.csv','w',encoding='utf-8')
    wr=csv.writer(f)
    for row in result_list:
        wr.writerow(row)
    f.close()
    _outpath=homedir+'/thesiswork/annotation'+str(pid)+'.csv'
    return _outpath

def upload_to_S3(_inpath,_key):
    f=open(_inpath,"r",encoding='utf-8')
    data=f.read()
    f.close()
    targetBucket.put_object(Body=data,Key=_key)

def get_index(_pid):
    sourceBucket.download_file(obj_prefix+"index.json",homedir+"/temp/doi_index"+str(_pid)+".json")
    f=open(homedir+"/temp/doi_index"+str(_pid)+".json",'r',encoding='utf-8')
    doi_index=json.load(f)
    f.close()
    return doi_index

doi_index=get_index(pid)
start=0
end=len(doi_index)
logf=open(homedir+"/results/logs/annotator_log_experiment_generic.txt",'a')
for i in range (start,end):
    try:
        sourceBucket.download_file(obj_prefix+doi_index[i]+"_abs.txt",homedir+"/thesiswork/source/papers/tmp"+str(pid)+".txt")
        txt_path=homedir+"/thesiswork/source/papers/tmp"+str(pid)+".txt"
        path=get_annotation(txt_path)
        key=obj_prefix+doi_index[i]+"_abs.csv"
        upload_to_S3(path,key)

        sourceBucket.download_file(obj_prefix+doi_index[i]+"_body.txt",homedir+"/thesiswork/source/papers/tmp"+str(pid)+".txt")
        txt_path=homedir+"/thesiswork/source/papers/tmp"+str(pid)+".txt"
        path=get_annotation(txt_path)
        key=obj_prefix+doi_index[i]+"_body.csv"
        upload_to_S3(path,key)

        sourceBucket.download_file(obj_prefix+doi_index[i]+"_title.txt",homedir+"/thesiswork/source/papers/tmp"+str(pid)+".txt")
        txt_path=homedir+"/thesiswork/source/papers/tmp"+str(pid)+".txt"
        path=get_annotation(txt_path)
        key=obj_prefix+doi_index[i]+"_title.csv"
        upload_to_S3(path,key)
        logf.write("process "+str(pid)+", source file "+str(i)+"\n")
    except:
        pass
logf.close()