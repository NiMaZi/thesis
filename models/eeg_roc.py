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
# 	f=open(homedir+"/results/ontology/cc2vid_eegrdc_enriched_2nd.json",'r')
# 	f=open(homedir+"/results/ontology/cc2vid_leaf.json",'r')
	f=open(homedir+"/results/ontology/ConCode2Vid.json",'r')
	cc2vid=json.load(f)
	f.close()
	return cc2vid

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

def get_model_local(path):
	return load_model(path)

def test_on_doc_S3_all(_model,_volume,_threshold=0.0):
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid=load_sups()
	sample_list=[]
	hits=0.0
	tp=0.0
	tn=0.0
	fp=0.0
	fn=0.0
	error_count=0.0
	all_count=0.0   
	for i in range(20000,20000+_volume):
		abs_vec=[0.0 for i in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/EEG_raw/abs"+str(i+10000)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
# 					abs_vec[cc2vid[item[1]]]=1.0                    
					abs_vec[cc2vid[item[1]]]+=1.0
					abs_count+=1.0
				except:
					pass
		if not abs_count==0.0:
			abs_vec=list(np.array(abs_vec)/abs_count)
		body_res=0.0
		try:
			bucket.download_file("yalun/EEG_raw/body"+str(i+10000)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=="C38054":
					body_res=1.0
					break
		X_test=np.array([abs_vec])
		Y_pred=_model.predict(X_test)[0][0]
# 		print(Y_pred)
		if Y_pred>=_threshold:
			Y_res=1.0
		else:
			Y_res=0.0
		if Y_res==body_res:
			hits+=1.0
			if Y_res==1.0:
				tp+=1.0
			else:
				tn+=1.0
		else:
			error_count+=1.0
			if Y_res==1.0:
				fp+=1.0
			else:
				fn+=1.0
		all_count+=1.0
		abs_vec=[0.0 for i in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/EEG_filter/abs"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
# 					abs_vec[cc2vid[item[1]]]=1.0                    
					abs_vec[cc2vid[item[1]]]+=1.0
					abs_count+=1.0
				except:
					pass
		if not abs_count==0.0:
			abs_vec=list(np.array(abs_vec)/abs_count)
		body_res=0.0
		try:
			bucket.download_file("yalun/EEG_filter/body"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=="C38054":
					body_res=1.0
					break
		X_test=np.array([abs_vec])
		Y_pred=_model.predict(X_test)[0][0]
# 		print(Y_pred)
		if Y_pred>=_threshold:
			Y_res=1.0
		else:
			Y_res=0.0
		if Y_res==body_res:
			hits+=1.0
			if Y_res==1.0:
				tp+=1.0
			else:
				tn+=1.0
		else:
			error_count+=1.0
			if Y_res==1.0:
				fp+=1.0
			else:
				fn+=1.0
		all_count+=1.0
		abs_vec=[0.0 for i in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/annotated_papers_meta/abs"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
# 					abs_vec[cc2vid[item[1]]]=1.0                    
					abs_vec[cc2vid[item[1]]]+=1.0
					abs_count+=1.0
				except:
					pass
		if not abs_count==0.0:
			abs_vec=list(np.array(abs_vec)/abs_count)
		body_res=0.0
		try:
			bucket.download_file("yalun/annotated_papers_meta/body"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=="C38054":
					body_res=1.0
					break
		X_test=np.array([abs_vec])
		Y_pred=_model.predict(X_test)[0][0]
		if Y_pred>=_threshold:
			Y_res=1.0
		else:
			Y_res=0.0
		if Y_res==body_res:
			hits+=1.0
			if Y_res==1.0:
				tp+=1.0
			else:
				tn+=1.0
		else:
			error_count+=1.0
			if Y_res==1.0:
				fp+=1.0
			else:
				fn+=1.0
		all_count+=1.0
	acc=hits/all_count
	tpr=tp/(tp+fn)
	fpr=fp/(fp+tn)
	return acc,tpr,fpr

if __name__=="__main__":
	model_name="MLPsparse_1hidden_eeg_gpuopt"
	model=get_model_S3(model_name)
	res=[]
	threshold=0.0
	volume=1000
	while threshold<4.2:
		acc,tpr,fpr=test_on_doc_S3_all(model,volume,threshold)
		res.append((fpr,tpr))
		threshold+=0.05
	res=list(set(res))
	homedir=os.environ['HOME']
	f=open(homedir+"/results/eeg_gpuopt_roc.json",'w')
	json.dump(res,f)
	f.close()
	l=sorted(res,key=lambda x:x[0])
	res=[]
	m=0.0
	for r in l:
		if r[1]>=m:
			m=r[1]
			res.append(r)
	fpr=[0.0]
	tpr=[0.0]
	for rec in res:
		fpr.append(rec[0])
		tpr.append(rec[1])
	AUC=0.0
	for i in range(0,len(tpr)-1):
		AUC+=(tpr[i]+tpr[i+1])*(fpr[i+1]-fpr[i])*0.5
	f=open(homedir+"/results/AUC.txt",'a')
	f.write("%s,%f"%(model_name,AUC))
	f.close()