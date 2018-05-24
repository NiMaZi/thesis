import os
import sys
import csv
import json
import boto3
import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping

def get_bucket():
	s3 = boto3.resource("s3")
	myBucket=s3.Bucket('workspace.scitodate.com')
	return myBucket

def load_sups():
	homedir=os.environ['HOME']
	f=open(homedir+"/results/ontology/ConCode2Vid.json",'r')
	cc2vid=json.load(f)
	f.close()
	f=open(homedir+"/results/ontology/c2n.json",'r')
	c2n=json.load(f)
	f.close()
	prefix='http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#'
	return cc2vid,c2n,prefix

def get_model_S3(_name):
	bucket=get_bucket()
	homedir=os.environ['HOME']
	try:
		os.remove(homedir+"/temp/tmp_model.h5")
	except:
		pass
	bucket.download_file("yalun/results/models/"+_name+".h5",homedir+"/temp/tmp_model.h5")
	model=load_model(homedir+"/temp/tmp_model.h5")
	return model

def get_weight(_model):
	res=[]
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid,c2n,prefix=load_sups()
	for i,k in enumerate(list(cc2vid.keys())):
		in_vec=[0.0 for i in range(0,len(cc2vid))]
		in_vec[cc2vid[k]]=1.0
		in_vec=np.array([in_vec])
		res.append((i,c2n[prefix+k],model.predict(in_vec)[0][0]))
	l=sorted(res,key=lambda x:-x[2])
	return l

if __name__=="__main__":
	homedir=os.environ['HOME']
	model_name="MLPsparse_1hidden_eeg_gpuopt"
	model=get_model_S3(model_name)
	sorted_weight=get_weight(model)
	with open(homedir+"/results/statistics/eeg_weight_gpuopt.csv",'w',encoding='utf-8') as cf:
		wt=csv.writer(cf)
		for d in sorted_weight:
			wt.writerow(d)