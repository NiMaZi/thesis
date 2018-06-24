import os
import sys
import csv
import json
import boto3
import numpy as np
from time import time
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
	return cc2vid

def build_model(_input_dim=133609,_hidden_dim=512,_drate=0.5):
	model=Sequential()
	model.add(Dense(_hidden_dim,input_shape=(_input_dim,),activation='relu'))
	model.add(Dropout(_drate))
	model.add(BatchNormalization())
	model.add(Dense(6,activation='relu'))
	model.compile(optimizer='nadam',loss='binary_crossentropy')
	return model

def get_model_local(path):
	return load_model(path)

def train_on_batch_S3(_model,_source,_volume,_bcount,_batch,_mbatch,_epochs=5):
	early_stopping=EarlyStopping(monitor='loss',patience=2)
	early_stopping_val=EarlyStopping(monitor='val_loss',patience=2)
	homedir=os.environ['HOME']
	bucket=get_bucket()
	cc2vid=load_sups()
	doi_indices=[]
	for s in _source[:-1]:
		bucket.download_file("yalun/"+s+"index.json",homedir+"/temp/expgen_index.json")
		f=open(homedir+'/temp/expgen_index.json','r',encoding='utf-8')
		index=json.load(f)
		f.close()
		doi_indices.append(index)
	doi_ratio=[1,3,1,3,3,3]
	sample_list=[]
	batch_count=_bcount
	for i in range(0,_volume):
		for j in range(0,len(_source)-1):
			for n in range(0,doi_ratio[j]):
				try:
					abs_vec=[0.0 for k in range(0,len(cc2vid))]
					abs_count=0.0
					bucket.download_file("yalun/"+_source[j]+doi_indices[j][i*doi_ratio[j]+n]+"_abs.csv",homedir+"/temp/tmp1.csv")
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
					body_vec=[0.0 for k in range(0,6)]
					body_vec[j]=1.0
				except:
					continue
				sample_list.append(abs_vec+body_vec)
		for n in range(0,3):
			try:
				abs_vec=[0.0 for k in range(0,len(cc2vid))]
				abs_count=0.0
				bucket.download_file("yalun/"+_source[6]+"/abs"+str(i*3+n)+".csv",homedir+"/temp/tmp1.csv")
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
				body_vec=[0.0 for k in range(0,6)]
			except:
				continue
			sample_list.append(abs_vec+body_vec)
			if len(sample_list)>=_batch:
				N_all=np.array(sample_list)
				X_train=N_all[:,:len(cc2vid)]
				Y_train=N_all[:,len(cc2vid):]
				_model.fit(X_train,Y_train,batch_size=_mbatch,shuffle=True,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
				try:
					os.remove(homedir+"/temp/tmp_model1.h5")
				except:
					pass
				_model.save(homedir+"/temp/tmp_model1.h5")
				s3f=open(homedir+"/temp/tmp_model1.h5",'rb')
				updata=s3f.read()
				bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_expgen.h5")
				s3f.close()
				logf=open(homedir+"/results/logs/bow_training_log_expgen.txt",'a')
				logf.write("ep_port,%s,%d,%d,%d\n"%(str(_source),_epochs,_mbatch,batch_count))
				logf.close()
				batch_count+=1
				sample_list=[]
	if len(sample_list):
		N_all=np.array(sample_list)
		X_train=N_all[:,:len(cc2vid)]
		Y_train=N_all[:,len(cc2vid):]
		_model.fit(X_train,Y_train,batch_size=_mbatch,shuffle=True,verbose=0,epochs=_epochs,validation_split=1.0/17.0,callbacks=[early_stopping,early_stopping_val])
		try:
			os.remove(homedir+"/temp/tmp_model1.h5")
		except:
			pass
		_model.save(homedir+"/temp/tmp_model1.h5")
		s3f=open(homedir+"/temp/tmp_model1.h5",'rb')
		updata=s3f.read()
		bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_expgen.h5")
		s3f.close()
		logf=open(homedir+"/results/logs/bow_training_log_expgen.txt",'a')
		logf.write("ep_port,%s,%d,%d,%d\n"%(str(_source),_epochs,_mbatch,batch_count))
		logf.close()
		batch_count+=1
	return _model,batch_count

if __name__=="__main__":
	start=time()
	model=build_model()
	# model=get_model_local("/home/yzg550/temp/tmp_model1.h5")
	source_key=["experiment_generic/safety_cabinet/","experiment_generic/freeze_dryer/","experiment_generic/cold_trap/","experiment_generic/vacuum_concentrator/","experiment_generic/cooling_bath/","experiment_generic/centrifuge/","annotated_papers_with_txt_new2"]
	model,bcount=train_on_batch_S3(model,source_key,3,0,272,256)
	model,bcount=train_on_batch_S3(model,source_key,3,0,1088,256)
	model,bcount=train_on_batch_S3(model,source_key,3,0,1088,1024)
	end=time()
	f=open(homedir+"/temp/expgen_time.txt",'w')
	f.write(str(end-start))
	f.close()