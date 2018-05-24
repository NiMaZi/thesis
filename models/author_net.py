import os
import sys
import csv
import json
import time
import boto3
import numpy as np
from keras.models import Model,load_model
from keras.layers import Dense,Concatenate,Input,Dropout,BatchNormalization
from keras.callbacks import EarlyStopping
from keras import backend as K

def get_bucket():
	s3 = boto3.resource("s3")
	myBucket=s3.Bucket('workspace.scitodate.com')
	return myBucket

def load_sups():
	homedir=os.environ['HOME']
	f=open(homedir+"/results/ontology/ConCode2Vid.json",'r')
	cc2vid=json.load(f)
	f.close()
	f=open(homedir+"/results/statistics/fa2vid.json",'r')
	fa2vid=json.load(f)
	f.close()
	return cc2vid,fa2vid

def build_model(_input_dim_entity=133609,_input_dim_author=7858,_hidden_dim=512,_drate=0.5):
	in_1=Input(shape=(_input_dim_entity,))
	in_2=Input(shape=(_input_dim_author,))
	b1=BatchNormalization()(in_1)
	b2=BatchNormalization()(in_2)
	d1=Dropout(_drate)(b1)
	d2=Dropout(_drate)(b2)
	x1=Dense(_hidden_dim,activation='relu')(d1)
	x2=Dense(_hidden_dim,activation='relu')(d2)
	concat=Concatenate()([x1,x2])
	hidden=Dense(_hidden_dim,activation='relu')(concat)
	out1=Dense(_input_dim_entity,activation='relu')(hidden)
	model=Model(inputs=[in_1,in_2], outputs=[out1])
	model.compile(optimizer='Nadam',loss='binary_crossentropy')
	return model

def train_on_batch_S3(_model,_source,_volume,_bcount,_batch,_mbatch,_epochs=5):
	early_stopping=EarlyStopping(monitor='loss',patience=2)
	early_stopping_val=EarlyStopping(monitor='val_loss',patience=2)
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid,fa2vid=load_sups()
	sample_list=[]
	batch_count=_bcount
	for i in range(0,_volume):
		abs_vec=[0.0 for i in range(0,len(cc2vid))]
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
					abs_count+=1.0
					abs_vec[cc2vid[item[1]]]+=1.0
				except:
					pass
		if not abs_count:
			continue
		abs_vec=list(np.array(abs_vec)/abs_count)
		body_vec=[0.0 for i in range(0,len(cc2vid))]
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
					body_count+=1.0
					body_vec[cc2vid[item[1]]]+=1.0
				except:
					pass
		if not body_count:
			continue
		body_vec=list(np.array(body_vec)/body_count)
		author_vec=[0.0 for i in range(0,len(fa2vid))]
		try:
			bucket.download_file("yalun/"+_source+"/authors"+str(i)+".json",homedir+"/temp/tmpjson.json")
		except:
			continue
		tmpf=open(homedir+"/temp/tmpjson.json",'r',encoding='utf-8')
		authors=json.load(tmpf)
		tmpf.close()
		for author in authors:
			try:
				author_vec[fa2vid[author]]=1.0
			except:
				pass
		sample_list.append(abs_vec+author_vec+body_vec)
		if len(sample_list)>=_batch:
			N_all=np.array(sample_list)
			X_train_1=N_all[:,:len(cc2vid)]
			X_train_2=N_all[:,len(cc2vid):len(cc2vid)+len(fa2vid)]
			Y_train=np.clip(np.ceil(N_all[:,len(cc2vid)+len(fa2vid):])-np.ceil(X_train_1),0.0,1.0)
			_model.fit([X_train_1,X_train_2],[Y_train],batch_size=_mbatch,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
			try:
				os.remove(homedir+"/temp/tmp_model.h5")
			except:
				pass
			_model.save(homedir+"/temp/tmp_model.h5")
			s3f=open(homedir+"/temp/tmp_model.h5",'rb')
			updata=s3f.read()
			bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_authornet2.h5")
			s3f.close()
			logf=open(homedir+"/results/logs/bow_training_log_authornet.txt",'a')
			logf.write("%s,%d\n"%(_source,batch_count))
			logf.close()
			batch_count+=1
			sample_list=[]
	if len(sample_list):
		N_all=np.array(sample_list)
		X_train_1=N_all[:,:len(cc2vid)]
		X_train_2=N_all[:,len(cc2vid):len(cc2vid)+len(fa2vid)]
		Y_train=np.clip(np.ceil(N_all[:,len(cc2vid)+len(fa2vid):])-np.ceil(X_train_1),0.0,1.0)
		_model.fit([X_train_1,X_train_2],[Y_train],batch_size=_mbatch,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
		try:
			os.remove(homedir+"/temp/tmp_model.h5")
		except:
			pass
		_model.save(homedir+"/temp/tmp_model.h5")
		s3f=open(homedir+"/temp/tmp_model.h5",'rb')
		updata=s3f.read()
		bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_authornet2.h5")
		s3f.close()
		logf=open(homedir+"/results/logs/bow_training_log_authornet.txt",'a')
		logf.write("%s,%d\n"%(_source,batch_count))
		logf.close()
		batch_count+=1
	return _model,batch_count

if __name__=="__main__":
	model=build_model()
	source_key="annotated_papers_authors"
	model,bcount=train_on_batch_S3(model,source_key,100000,0,1088,1024)