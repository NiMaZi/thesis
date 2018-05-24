import csv
import sys
import json
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Reshape
from keras.callbacks import EarlyStopping

def load_trained_model(_path):
	model=load_model(_path)
	return model

def build_model(_input_dim,_input_length):
	model=Sequential()
	model.add(LSTM(_input_dim,input_dim=_input_dim,input_length=_input_length,return_sequences=True,activation="relu"))
	model.add(Reshape((_input_length*_input_dim,), input_shape=(_input_length,_input_dim)))
	model.add(Dense(int(_input_dim*_input_length*1.5),input_dim=_input_dim*_input_length,activation="relu"))
	model.add(Dense(_input_dim*_input_length,input_dim=int(_input_dim*_input_length*1.5),activation="relu"))
	model.add(Reshape((_input_length,_input_dim), input_shape=(_input_dim*_input_length,)))
	model.add(LSTM(_input_dim,input_dim=_input_dim,input_length=_input_length,return_sequences=False,activation="relu"))
	model.compile(optimizer='rmsprop',loss='binary_crossentropy')
	return model

def build_data(_volume,_chunk,_split):
	fp=open("/home/ubuntu/results_new/ontology/word2tvec.json",'r',encoding='utf-8')
	word2tvec=json.load(fp)
	fp.close()
	seq_list=[]
	for i in range(0,_volume):
		time_steps=[]
		f=open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[2]=="ConceptName":
				continue
			try:
				time_steps.append(word2tvec[item[1]])
			except:
				pass
			if len(time_steps)>=_chunk:
				seq_list.append(time_steps)
				time_steps=[]
		if time_steps:
			seq_list.append(time_steps)
	for seq in seq_list:
		if len(seq)<_chunk:
			for i in range(0,_chunk-len(seq)):
				seq.append([-1.0 for i in range(0,len(seq[0]))])
	N_all=np.array(seq_list)
	N_train=N_all[:int(_split*len(seq_list)),:,:]
	N_test=N_all[int(_split*len(seq_list)):,:,:]
	X_train=N_train[:,:_chunk-1,:]
	y_train=N_train[:,_chunk-1,:]
	X_test=N_test[:,:_chunk-1,:]
	y_test=N_test[:,_chunk-1,:]
	input_dim=X_train.shape[2]
	input_length=X_train.shape[1]
	return X_train,y_train,X_test,y_test,input_dim,input_length

def save_model(_model,_path):
	_model.save(_path)

if __name__=="__main__":
	if len(sys.argv)<7:
		print("Usage: -mode -volume -chunk-size -split-rate -batch-size -epoch.\n")
		sys.exit(0)
	mode=int(sys.argv[1])
	volume=int(sys.argv[2])
	chunk=int(sys.argv[3])
	split=float(sys.argv[4])
	batch=int(sys.argv[5])
	epoch=int(sys.argv[6])
	path="/home/ubuntu/results_new/models/LSTM.h5"
	X_train,y_train,X_test,y_test,input_dim,input_length=build_data(volume,chunk,split)
	if mode==0:
		model=build_model(input_dim,input_length)
	else:
		model=load_trained_model(path)
	early_stopping=EarlyStopping(monitor='loss',patience=10)
	model.fit(X_train,y_train,batch_size=batch,epochs=epoch,callbacks=[early_stopping])
	score=model.evaluate(X_test,y_test,batch_size=batch)
	print(score)
	save_model(model,path)