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
	return cc2vid

def build_model(_input_dim=133609,_hidden_dim=512,_drate=0.5):
	model=Sequential()
	model.add(Dense(_hidden_dim,input_shape=(_input_dim,),activation='relu'))
	model.add(Dropout(_drate))
	model.add(BatchNormalization())
	model.add(Dense(41,activation='relu'))
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
	sample_list=[]
	batch_count=_bcount
	for i in range(0,_volume):
		for j in range(0,len(_source)):
			try:
				abs_vec=[0.0 for k in range(0,len(cc2vid))]
				abs_count=0.0
				bucket.download_file("yalun/"+_source[j]+"/abs"+str(i)+".csv",homedir+"/temp/tmp1.csv")
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
				body_vec=[0.0 for k in range(0,41)]
				bucket.download_file("yalun/"+_source[j]+"/body"+str(i)+".csv",homedir+"/temp/tmp1.csv")
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
				bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_port.h5")
				s3f.close()
				logf=open(homedir+"/results/logs/bow_training_log_port.txt",'a')
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
		bucket.put_object(Body=updata,Key="yalun/results/models/MLPsparse_1hidden_port.h5")
		s3f.close()
		logf=open(homedir+"/results/logs/bow_training_log_port.txt",'a')
		logf.write("ep_port,%s,%d,%d,%d\n"%(str(_source),_epochs,_mbatch,batch_count))
		logf.close()
		batch_count+=1
	return _model,batch_count

if __name__=="__main__":
	model=build_model()
	# model=get_model_local("/home/yzg550/temp/tmp_model1.h5")
	source_key=["Dependence","Microscopy","port/bird","port/computer","port/endoscope","port/spectroscopy","annotated_papers_with_txt_new2"]
	model,bcount=train_on_batch_S3(model,source_key,5000,0,272,256)
	model,bcount=train_on_batch_S3(model,source_key,5000,0,1088,256)
	model,bcount=train_on_batch_S3(model,source_key,5000,0,1088,1024)