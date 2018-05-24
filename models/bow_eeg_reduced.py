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
	bucket=get_bucket()
	bucket.download_file("yalun/results/ontology/cc2vid_leaf.json",homedir+"/results/ontology/cc2vid_leaf.json")
	f=open(homedir+"/results/ontology/cc2vid_leaf.json",'r')
	cc2vid=json.load(f)
	f.close()
	return cc2vid

def build_model(_input_dim=110184,_hidden_dim=512,_drate=0.5):
	model=Sequential()
	model.add(Dense(_hidden_dim,input_shape=(_input_dim,),activation='relu'))
	model.add(Dropout(_drate))
	model.add(BatchNormalization())
	model.add(Dense(1,activation='relu'))
	model.compile(optimizer='nadam',loss='binary_crossentropy')
	return model

def train_on_batch_S3(_model,_source,_volume,_bcount,_batch,_mbatch,_epochs=5):
	early_stopping=EarlyStopping(monitor='loss',patience=2)
	early_stopping_val=EarlyStopping(monitor='val_loss',patience=2)
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid=load_sups()
	sample_list=[]
	batch_count=_bcount
	for i in range(0,_volume):
		abs_vec=[0.0 for k in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/"+_source[0]+"/abs"+str(i)+".csv",homedir+"/temp/tmp0.csv")
		except:
			continue
		with open(homedir+"/temp/tmp0.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
					abs_vec[cc2vid[item[1]]]+=1.0
					abs_count+=1.0
				except:
					pass
		if not abs_count==0.0:
			abs_vec=list(np.array(abs_vec)/abs_count)
		body_vec=[0.0]
		try:
			bucket.download_file("yalun/"+_source[0]+"/body"+str(i)+".csv",homedir+"/temp/tmp0.csv")
		except:
			continue
		with open(homedir+"/temp/tmp0.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=='C38054':	#EEG
					body_vec[0]=1.0
					break
		sample_list.append(abs_vec+body_vec)
		abs_vec=[0.0 for k in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/"+_source[1]+"/abs"+str(i)+".csv",homedir+"/temp/tmp0.csv")
		except:
			continue
		with open(homedir+"/temp/tmp0.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
					abs_vec[cc2vid[item[1]]]+=1.0
					abs_count+=1.0
				except:
					pass
		if not abs_count==0.0:
			abs_vec=list(np.array(abs_vec)/abs_count)
		body_vec=[0.0]
		try:
			bucket.download_file("yalun/"+_source[1]+"/body"+str(i)+".csv",homedir+"/temp/tmp0.csv")
		except:
			continue
		with open(homedir+"/temp/tmp0.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=='C38054':	#EEG
					body_vec[0]=1.0
					break
		sample_list.append(abs_vec+body_vec)
		if len(sample_list)>=_batch:
			N_all=np.array(sample_list)
			X_train=N_all[:,:len(cc2vid)]
			Y_train=N_all[:,len(cc2vid):]
			_model.fit(X_train,Y_train,batch_size=_mbatch,shuffle=True,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
			try:
				os.remove(homedir+"/temp/tmp_model0.h5")
			except:
				pass
			_model.save(homedir+"/temp/tmp_model0.h5")
			s3f=open(homedir+"/temp/tmp_model0.h5",'rb')
			updata=s3f.read()
			bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_eegrdc_leaf.h5")
			s3f.close()
			logf=open(homedir+"/results/logs/bow_training_log_eegrdc_leaf.txt",'a')
			logf.write("%s,%d\n"%(str(_source),batch_count))
			logf.close()
			batch_count+=1
			sample_list=[]
	if len(sample_list):
		N_all=np.array(sample_list)
		X_train=N_all[:,:len(cc2vid)]
		Y_train=N_all[:,len(cc2vid):]
		_model.fit(X_train,Y_train,batch_size=_mbatch,shuffle=True,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
		try:
			os.remove(homedir+"/temp/tmp_model0.h5")
		except:
			pass
		_model.save(homedir+"/temp/tmp_model0.h5")
		s3f=open(homedir+"/temp/tmp_model0.h5",'rb')
		updata=s3f.read()
		bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_eegrdc_leaf.h5")
		s3f.close()
		logf=open(homedir+"/results/logs/bow_training_log_eegrdc_leaf.txt",'a')
		logf.write("%s,%d\n"%(str(_source),batch_count))
		logf.close()
		batch_count+=1
	return _model,batch_count

if __name__=="__main__":
	model=build_model()
	source_key=["EEG_raw","annotated_papers_with_txt_new2"]
	model,bcount=train_on_batch_S3(model,source_key,30000,0,1088,1024)