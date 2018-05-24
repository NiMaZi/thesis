import csv
import sys
import json
import numpy as np
from keras.models import load_model

ratio_body2abs=496.27/59.5

def load_trained_model(_path):
	model=load_model(_path)
	return model

def decode(_vec,_dict):
	min_dist=np.inf
	target_word=''
	for w in _dict.keys():
		dist=np.linalg.norm(_vec-np.array(_dict[w]))
		if dist<min_dist:
			min_dist=dist
			target_word=w
	return target_word,min_dist

def sliding_prediction(_array,_size,_model,_dict):
	res_list={}
	vec_dim=len(_array[0])
	incre=int(len(_array)*ratio_body2abs/(len(_array)-_size))
	for i in range(0,len(_array)-_size):
		seq=_array[i:i+_size]
		for j in range(0,incre):
			X_in=np.array(seq).reshape(1,_size,vec_dim)
			y_out=_model.predict(X_in)
			w,d=decode(y_out[0],_dict)
			res_list[w]=d
			v_inc=_dict[w]
			seq=seq[1:]+[v_inc]
	return res_list

def test_corpus(_offset,_volume,_chunk,_model,_verbose):
	fp=open("/home/ubuntu/results_new/ontology/word2tvec.json",'r',encoding='utf-8')
	word2tvec=json.load(fp)
	fp.close()
	fp=open("/home/ubuntu/results_new/ontology/word_dict.json",'r',encoding='utf-8')
	word_dict=json.load(fp)
	fp.close()
	P_all=0.0
	R_all=0.0
	for i in range(_offset,_offset+_volume):
		seq_list=[]
		abs_set=set()
		f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[2]=="ConceptName":
				continue
			try:
				abs_set.add(item[1])
				seq_list.append(word2tvec[item[1]])
			except:
				pass
		f.close()
		res_list=sliding_prediction(seq_list,_chunk,_model,word2tvec)
		# N_all=np.array(seq_list)
		# X_in=N_all[:,:_chunk,:]
		# y_out=_model.predict(X_in)
		# res_list={}
		# for p in y_out:
		# 	word,dist=decode(p,word2tvec)
		# 	res_list[word]=dist
		body_set=set()
		f=open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[2]=="ConceptName":
				continue
			try:
				body_set.add(item[1])
			except:
				pass
		real_set=body_set-abs_set
		# tp=0.0
		# fp=0.0
		# fn=0.0
		pred_set=set(list(res_list.keys()))
		tp=len(real_set&pred_set)
		fp=len(pred_set-(real_set&pred_set))
		fn=len(real_set-(real_set&pred_set))
		# for key in res_list.keys():
		# 	if key in real_set:
		# 		tp+=1.0
		# 	else:
		# 		fp+=1.0
		P=tp/(tp+fp)
		R=tp/(tp+fn)
		P_all+=P
		R_all+=R
		if _verbose>=1:
			print(res_list)
			print(real_set)
		if _verbose>=0:
			print(P,R)
	P_all/=_volume
	R_all/=_volume
	F1=2*P_all*R_all/(P_all+R_all)
	return P_all,R_all,F1

if __name__ == '__main__':
	if len(sys.argv)<4:
		print("Usage: -offset -volume -verbose.\n")
		sys.exit(0)
	offset=int(sys.argv[1])
	volume=int(sys.argv[2])
	verbose=int(sys.argv[3])
	path="/home/ubuntu/results_new/models/LSTM.h5"
	model=load_trained_model(path)
	chunk=model.layers[0].get_config()['batch_input_shape'][1]
	P,R,F1=test_corpus(offset,volume,chunk,model,verbose)
	print(P,R,F1)