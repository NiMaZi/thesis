import os
import sys
import csv
import json
import time
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
	f=open(homedir+"/results/ontology/code2svec.json",'r')
	c2sv=json.load(f)
	f.close()
	return cc2vid,c2sv

def build_model(_input_dim=133609,_hidden_dim=512,_drate=0.5):
	model=Sequential()
	model.add(Dense(_hidden_dim,input_shape=(_input_dim,),activation='relu'))
	model.add(Dropout(_drate))
	model.add(BatchNormalization())
	model.add(Dense(129,activation='relu'))
	model.compile(optimizer='nadam',loss='binary_crossentropy')
	return model

def train_on_batch_S3(_model,_source,_volume,_bcount,_batch,_mbatch,_epochs=5):
	early_stopping=EarlyStopping(monitor='loss',patience=2)
	early_stopping_val=EarlyStopping(monitor='val_loss',patience=2)
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid,c2sv=load_sups()
	sample_list=[]
	batch_count=_bcount
	for i in range(0,_volume):
		abs_vec=np.array([0.0 for i in range(0,len(cc2vid))])
		abs_count=0.0
		try:
			bucket.download_file("yalun/"+_source+"/abs"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
					abs_vec+=np.array(c2sv[item[1]])
					abs_count+=1.0
				except:
					pass
		if not abs_count:
			continue
		abs_vec=list(abs_vec/np.sum(abs_vec))
		body_vec=np.array([0.0 for i in range(0,len(c2sv))])
		body_count=0.0
		try:
			bucket.download_file("yalun/"+_source+"/body"+str(i)+".csv",homedir+"/temp/tmp.csv")
		except:
			continue
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
					body_vec+=np.array(c2sv[item[1]])
					body_count+=1.0
				except:
					pass
		if not body_count:
			continue
		body_vec=list(body_vec/np.sum(body_vec))
		sample_list.append(abs_vec+body_vec)
		if len(sample_list)>=_batch:
			N_all=np.array(sample_list)
			X_train=N_all[:,:len(cc2vid)]
			Y_train=np.ceil(N_all[:,len(cc2vid):])
			_model.fit(X_train,Y_train,batch_size=_mbatch,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
			try:
				os.remove(homedir+"/temp/tmp_model.h5")
			except:
				pass
			_model.save(homedir+"/temp/tmp_model.h5")
			s3f=open(homedir+"/temp/tmp_model.h5",'rb')
			updata=s3f.read()
			bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_sem"+str(batch_count)+".h5")
			s3f.close()
			logf=open(homedir+"/results/logs/bow_training_log_sem.txt",'a')
			logf.write("%s,%d\n"%(_source,batch_count))
			logf.close()
			batch_count+=1
			sample_list=[]
	if len(sample_list):
		N_all=np.array(sample_list)
		X_train=N_all[:,:len(cc2vid)]
		Y_train=np.ceil(N_all[:,len(cc2vid):])
		_model.fit(X_train,Y_train,batch_size=_mbatch,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
		try:
			os.remove(homedir+"/temp/tmp_model.h5")
		except:
			pass
		_model.save(homedir+"/temp/tmp_model.h5")
		s3f=open(homedir+"/temp/tmp_model.h5",'rb')
		updata=s3f.read()
		bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_sem"+str(batch_count)+".h5")
		s3f.close()
		logf=open(homedir+"/results/logs/bow_training_log_sem.txt",'a')
		logf.write("%s,%d\n"%(_source,batch_count))
		logf.close()
		batch_count+=1
	return _model,batch_count

if __name__=="__main__":
	model=build_model()
	source_key="kdata"
	model,bcount=train_on_batch_S3(model,source_key,12000,0,1088,1024)
	source_key="annotated_papers"
	model,bcount=train_on_batch_S3(model,source_key,14000,bcount,1088,1024)
	source_key="annotated_papers_with_txt"
	model,bcount=train_on_batch_S3(model,source_key,13000,bcount,1088,1024)
	source_key="annotated_papers_with_txt_new"
	model,bcount=train_on_batch_S3(model,source_key,15000,bcount,1088,1024)
	source_key="annotated_papers_with_txt_new2"
	model,bcount=train_on_batch_S3(model,source_key,95000,bcount,1088,1024)