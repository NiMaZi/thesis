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
	f=open(homedir+"/results/ontology/cc2vid_eprdc.json",'r')
	cc2vid=json.load(f)
	f.close()
	f=open(homedir+"/results/statistics/author_emb.json",'r')
	fa2v=json.load(f)
	f.close()
	return cc2vid,fa2v

def build_model(_input_dim_entity=18639,_input_dim_author=512,_hidden_dim=512,_drate=0.5):
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
	out1=Dense(6,activation='relu')(hidden)
	model=Model(inputs=[in_1,in_2], outputs=[out1])
	model.compile(optimizer='Nadam',loss='binary_crossentropy')
	return model

def train_on_batch_S3(_model,_source,_volume,_bcount,_batch,_mbatch,_epochs=5):
	early_stopping=EarlyStopping(monitor='loss',patience=2)
	early_stopping_val=EarlyStopping(monitor='val_loss',patience=2)
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid,fa2v=load_sups()
	sample_list=[]
	batch_count=_bcount
	for i in range(0,_volume):
		abs_vec=[0.0 for k in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/"+_source[0]+"/abs"+str(7*i)+".csv",homedir+"/temp/tmp1.csv")
		except:
			continue
		with open(homedir+"/temp/tmp1.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				try:
					abs_vec[cc2vid[item[1]]]+=1.0
					abs_count+=1.0
				except:
					pass
		if not abs_count:
			continue
		abs_vec=list(np.array(abs_vec)/abs_count)
		author_vec=np.array([0.0 for i in range(0,512)])
		try:
			bucket.download_file("yalun/"+_source[0]+"/authors"+str(7*i)+".json",homedir+"/temp/tmpjson1.json")
		except:
			continue
		tmpf=open(homedir+"/temp/tmpjson1.json",'r',encoding='utf-8')
		authors=json.load(tmpf)
		tmpf.close()
		for author in authors:
			try:
				author_vec+=fa2v[author]
			except:
				pass
		if len(authors):
			author_vec=list(author_vec/len(authors))
		else:
			author_vec=list(author_vec)
		body_vec=[0.0 for k in range(0,6)]
		try:
			bucket.download_file("yalun/"+_source[0]+"/body"+str(7*i)+".csv",homedir+"/temp/tmp1.csv")
		except:
			continue
		with open(homedir+"/temp/tmp1.csv",'r',encoding='utf-8') as cf:
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
		sample_list.append(abs_vec+author_vec+body_vec)
		abs_vec=[0.0 for k in range(0,len(cc2vid))]
		abs_count=0.0
		try:
			bucket.download_file("yalun/"+_source[1]+"/abs"+str(i)+".csv",homedir+"/temp/tmp1.csv")
		except:
			continue
		with open(homedir+"/temp/tmp1.csv",'r',encoding='utf-8') as cf:
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
		author_vec=np.array([0.0 for i in range(0,512)])
		try:
			bucket.download_file("yalun/"+_source[1]+"/authors"+str(i)+".json",homedir+"/temp/tmpjson1.json")
		except:
			continue
		tmpf=open(homedir+"/temp/tmpjson1.json",'r',encoding='utf-8')
		authors=json.load(tmpf)
		tmpf.close()
		for author in authors:
			try:
				author_vec+=fa2v[author]
			except:
				pass
		if len(authors):
			author_vec=list(author_vec/len(authors))
		else:
			author_vec=list(author_vec)
		body_vec=[0.0 for k in range(0,6)]
		try:
			bucket.download_file("yalun/"+_source[1]+"/body"+str(i)+".csv",homedir+"/temp/tmp1.csv")
		except:
			continue
		with open(homedir+"/temp/tmp1.csv",'r',encoding='utf-8') as cf:
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
		sample_list.append(abs_vec+author_vec+body_vec)
		if len(sample_list)>=_batch:
			N_all=np.array(sample_list)
			X_train_1=N_all[:,:len(cc2vid)]
			X_train_2=N_all[:,len(cc2vid):len(cc2vid)+512]
			Y_train=N_all[:,len(cc2vid)+512:]
			_model.fit([X_train_1,X_train_2],[Y_train],batch_size=_mbatch,shuffle=True,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
			try:
				os.remove(homedir+"/temp/tmp_model1.h5")
			except:
				pass
			_model.save(homedir+"/temp/tmp_model1.h5")
			s3f=open(homedir+"/temp/tmp_model1.h5",'rb')
			updata=s3f.read()
			bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_eprdc_authornet2.h5")
			s3f.close()
			logf=open(homedir+"/results/logs/bow_training_log_eprdc_authornet.txt",'a')
			logf.write("%s,%d\n"%(str(_source),batch_count))
			logf.close()
			batch_count+=1
			sample_list=[]
	if len(sample_list):
		X_train_1=N_all[:,:len(cc2vid)]
		X_train_2=N_all[:,len(cc2vid):len(cc2vid)+512]
		Y_train=N_all[:,len(cc2vid)+512:]
		_model.fit([X_train_1,X_train_2],[Y_train],batch_size=_mbatch,shuffle=True,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
		try:
			os.remove(homedir+"/temp/tmp_model1.h5")
		except:
			pass
		_model.save(homedir+"/temp/tmp_model1.h5")
		s3f=open(homedir+"/temp/tmp_model1.h5",'rb')
		updata=s3f.read()
		bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_eprdc_authornet2.h5")
		s3f.close()
		logf=open(homedir+"/results/logs/bow_training_log_eprdc_authornet.txt",'a')
		logf.write("%s,%d\n"%(str(_source),batch_count))
		logf.close()
		batch_count+=1
	return _model,batch_count

if __name__=="__main__":
	model=build_model()
	source_key=["EEG_expansion","annotated_papers_meta"]
	model,bcount=train_on_batch_S3(model,source_key,20000,0,1088,1024)