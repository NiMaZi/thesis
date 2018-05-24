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

def test_on_doc_S3_all(_model,_volume,_threshold=0.0,_idx=0):
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
	for i in range(5000,5000+_volume):
		abs_vec=[0.0 for i in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/port/spectroscopy/abs"+str(i)+".csv",homedir+"/temp/tmp.csv")
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
		body_vec=[0.0 for k in range(0,41)]
		try:
			bucket.download_file("yalun/port/spectroscopy/body"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=='C38054':	#EEG
					body_vec[0]=1.0
				if item[1]=='C16809':	#MRI
					body_vec[1]=1.0
				if item[1]=='C116454':	#fMRI
					body_vec[2]=1.0
				if item[1]=='C116655':	#TMS
					body_vec[3]=1.0
				if item[1]=='C129862':	#tDCS
					body_vec[4]=1.0
				if item[1]=='C16811':	#MEG
					body_vec[5]=1.0
				if item[1]=='C78814' or item[1]=='C78815':	#SEM
					body_vec[6]=1.0
				if item[1]=='C78860' or item[1]=='C78813':	#TEM
					body_vec[7]=1.0
				if item[1]=='C17374':	#STM
					body_vec[8]=1.0
				if item[1]=='C78804':	#AFM
					body_vec[9]=1.0
				if item[1]=='C17753' or item[1]=='C122390' or item[1]=='C116477' or item[1]=='C116481':	#Confocal
					body_vec[10]=1.0
				if item[1]=='C93040':	#Alcohol
					body_vec[11]=1.0
				if item[1]=='C35386' or item[1]=='C35387' or item[1]=='C34445':	#Cannabis
					body_vec[12]=1.0
				if item[1]=='C34492' or item[1]=='C35389' or item[1]=='C35388':	#Cocaine
					body_vec[13]=1.0
				if item[1]=='C34694':	#Heroin
					body_vec[14]=1.0
				if item[1]=='C70989' or item[1]=='C54203' or item[1]=='C15985':	#Nicotine
					body_vec[15]=1.0
				if item[1]=='C73850':	#duck
					body_vec[16]=1.0
				if item[1]=='C91813':	#quail
					body_vec[17]=1.0
				if item[1]=='C91812':	#pigeon
					body_vec[18]=1.0
				if item[1]=='C77098':	#leghorn chicken
					body_vec[19]=1.0
				if item[1]=='C76362':	#broiler chicken
					body_vec[20]=1.0
				if item[1]=='C17611':	#Bronchoscope
					body_vec[21]=1.0
				if item[1]=='C17613':	#Colonoscope
					body_vec[22]=1.0
				if item[1]=='C17614':	#Colposcope
					body_vec[23]=1.0
				if item[1]=='C17616':	#Cystoscope
					body_vec[24]=1.0
				if item[1]=='C17620':	#Gastroscope
					body_vec[25]=1.0
				if item[1]=='C28167':	#Laparoscope
					body_vec[26]=1.0
				if item[1]=='C17618':	#Laryngoscope
					body_vec[27]=1.0
				if item[1]=='C85574':	#Atomic Absorption Spectroscopy
					body_vec[28]=1.0
				if item[1]=='C94374':	#Optical Spectroscopy
					body_vec[29]=1.0
				if item[1]=='C17157':	#Raman Spectroscopy
					body_vec[30]=1.0
				if item[1]=='C78869':	#Energy Dispersive Spectroscopy
					body_vec[31]=1.0
				if item[1]=='C16810':	#Magnetic Resonance Spectroscopy
					body_vec[32]=1.0
				if item[1]=='C62329':	#Photon Correlation Spectroscopy
					body_vec[33]=1.0
				if item[1]=='C78871':	#X-Ray Photoelectron Spectroscopy
					body_vec[34]=1.0
				if item[1]=='C142750':	#web server
					body_vec[35]=1.0
				if item[1]=='C47907':	#compiler
					body_vec[36]=1.0
				if item[1]=='C142380' or item[1]=='C80012':	#HTML
					body_vec[37]=1.0
				if item[1]=='C69302':	#R programming
					body_vec[38]=1.0
				if item[1]=='C75940' or item[1]=='C81018':	#Hidden Markov
					body_vec[39]=1.0
				if item[1]=='C142403' or item[1]=='C142404':	#Bayesian
					body_vec[40]=1.0
		sample_list=[abs_vec+body_vec]
		N_test=np.array(sample_list)
		X_test=N_test[:,:len(cc2vid)]
		body_res=N_test[:,len(cc2vid):][0][_idx]
		Y_pred=_model.predict(X_test)[0][_idx]
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
	term=sys.argv[1]
	idx=int(sys.argv[2])
	p0=float(sys.argv[3])
	p1=float(sys.argv[4])
	model_name="MLPsparse_1hidden_port"
	# model=get_model_S3(model_name)
	model=get_model_local("/home/ubuntu/temp/tmp_model2.h5")
	res=[]
	threshold=p0
	volume=100
	m_acc=0.0
	while threshold<=p1:
		acc,tpr,fpr=test_on_doc_S3_all(model,volume,threshold,idx)
		if acc>m_acc:
			m_acc=acc
		res.append((fpr,tpr))
		threshold+=0.001
	res=list(set(res))
	homedir=os.environ['HOME']
	f=open(homedir+"/results/spec_"+term+"_roc.json",'w')
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
	f.write("%s,%s,%f,%f\n"%(model_name,term,AUC,m_acc))
	f.close()