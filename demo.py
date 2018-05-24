import sys
import math
import pickle
import random
import numpy as np
from sklearn import svm
from sklearn import linear_model as lm

test_mode=int(sys.argv[1])
article_id=int(sys.argv[2])
confidence=float(sys.argv[3])
taxonomy_distance=float(sys.argv[4])

f=open("/home/ubuntu/results/saliency/featured.pkl","rb")
featured_list=pickle.load(f)
f.close()

abs_dict=featured_list[article_id]['abs']
body_dict=featured_list[article_id]['body']

if not abs_dict:
	print("cannot find entity mentions in abstract.\n")
	sys.exit(0)

print("entities mentioned in the abstract:\n")

for key in abs_dict.keys():
	print(key)

print("\nentities mentioned in the body:\n")

for key in body_dict.keys():
	print(key)

print("\nentities predicted:\n")

f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
b_word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_word2taxonomy.pkl","rb")
b_word2tvec=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/idf.pkl","rb")
idf=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/centrality.pkl","rb")
centrality=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/keyphrase.pkl","rb")
key_phrase=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/svmclf.pkl","rb")
s_clf=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/simplemat.pkl","rb")
dev_mat=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_sgd.pkl","rb")
clf_sgd=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/b_clf_sgd_test.pkl","rb")
b_clf_sgd=pickle.load(f)
f.close()

# f=open("/home/ubuntu/results/coclf/clf_sgd_correction.pkl","rb")
# clf_sgd_correction=pickle.load(f)
# f.close()

# f=open("/home/ubuntu/results/coclf/clf_sgd_filter.pkl","rb")
# clf_sgd_filter=pickle.load(f)
# f.close()

def tree_distance(vec_1,vec_2):
	distance=0.0
	for i in range(0,len(vec_1)):
		if not vec_1[i]==vec_2[i]:
			for j in range(i,len(vec_1)):
				if vec_1[j]==-1.0:
					break
				distance+=1.0
			for j in range(i,len(vec_2)):
				if vec_2[j]==-1.0:
					break
				distance+=1.0
	return distance


if test_mode==1:
	pred_dict_rbf={}
	max_conf_rbf=0
	for a_key in abs_dict.keys():
		if a_key in word_list:
			a_key_phrase=key_phrase[word_list.index(a_key)]
			a_idf=word_list.index(a_key)
		else:
			a_key_phrase=0.0
			a_idf=0.0
		if a_key in centrality.keys():
			a_centrality=centrality[a_key]
		else:
			a_centrality=0.0
		pred_input=np.array([[a_key_phrase,abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],a_idf,a_centrality]])
		pred_saliency=list(s_clf.predict(pred_input))[0]
		for b_key in word_list:
			if a_key==b_key:
				continue
			if b_key in centrality.keys():
				b_centrality=centrality[b_key]
			else:
				b_centrality=0.0
			if a_key in word_list and b_key in word_list:
				dev_cor=dev_mat[word_list.index(a_key)][word_list.index(b_key)]
			else:
				dev_cor=0.0
			sample_input=np.array([[a_centrality,b_centrality,dev_cor,pred_saliency]])
			pred_label_rbf=list(clf_sgd.predict(sample_input))[0]
			# if pred_label_rbf==1:
			# 	pred_label_correction=list(clf_sgd_correction.predict(sample_input))[0]
			# 	pred_label_filter=list(clf_sgd_filter.predict(sample_input))[0]
			# 	if cor_rate==-1:
			# 		pred_label_rbf=1
			# 	else:
			# 		pred_label_rbf=cor_rate*pred_label_correction+(1.0-cor_rate)*pred_label_filter
			# 		if pred_label_rbf>0.5:
			# 			pred_label_rbf=1
			# 		else:
			# 			pred_label_rbf=0
			if pred_label_rbf==1:
				if b_key in pred_dict_rbf.keys():
					pred_dict_rbf[b_key]+=1.0
					if pred_dict_rbf[b_key]>max_conf_rbf:
						max_conf_rbf=pred_dict_rbf[b_key]
				else:
					pred_dict_rbf[b_key]=1.0
					if pred_dict_rbf[b_key]>max_conf_rbf:
						max_conf_rbf=pred_dict_rbf[b_key]
	for key in pred_dict_rbf.keys():
		pred_dict_rbf[key]/=max_conf_rbf
	pred_set_rbf=set(abs_dict.keys())
	for key in pred_dict_rbf.keys():
		if pred_dict_rbf[key]>confidence:
			pred_set_rbf.add(key)
	real_set=set(body_dict.keys())
	tp=len(pred_set_rbf&real_set)
	fp=len(pred_set_rbf-(pred_set_rbf&real_set))
	fn=len(real_set-(real_set&pred_set_rbf))
	P=tp/(tp+fp)
	R=tp/(tp+fn)

	for entity in pred_set_rbf:
		if entity in body_dict.keys():
			print("\033[1;31m"+entity)
		else:
			print("\033[0m"+entity)

	print("\033[0m\nprecision=%.2f,recall=%.2f."%(P,R))
elif test_mode==2:
	max_conf_rbf=0
	pred_dict_rbf={}
	abs_key_list=list(abs_dict.keys())
	for i1 in range(0,len(abs_key_list)):
		for i2 in range(i1,len(abs_key_list)):
			a_key_1=abs_key_list[i1]
			a_key_2=abs_key_list[i2]
			if a_key_1==a_key_2:
				continue
			if not a_key_1 in b_word_list or not a_key_2 in b_word_list:
				continue
			for b_key in word_list:
				if a_key_1==b_key or a_key_2==b_key:
					continue
				if not b_key in b_word_list:
					continue
				a1_tvec=b_word2tvec[a_key_1]
				a2_tvec=b_word2tvec[a_key_2]
				b_tvec=b_word2tvec[b_key]
				dis_sum=tree_distance(a1_tvec,b_tvec)+tree_distance(a2_tvec,b_tvec)
				if dis_sum>taxonomy_distance:
					continue
				_list=[dev_mat[b_word_list.index(a_key_1)][b_word_list.index(b_key)],dev_mat[b_word_list.index(a_key_2)][b_word_list.index(b_key)]]
				_list.extend(b_word2tvec[a_key_1])
				_list.extend(b_word2tvec[a_key_2])
				sample_input=np.array([_list])
				pred_label_rbf=list(b_clf_sgd.predict(sample_input))[0]
				if pred_label_rbf==1:
					if b_key in pred_dict_rbf.keys():
						pred_dict_rbf[b_key]+=1.0
						if pred_dict_rbf[b_key]>max_conf_rbf:
							max_conf_rbf=pred_dict_rbf[b_key]
					else:
						pred_dict_rbf[b_key]=1.0
						if pred_dict_rbf[b_key]>max_conf_rbf:
							max_conf_rbf=pred_dict_rbf[b_key]
	for key in pred_dict_rbf.keys():
		pred_dict_rbf[key]/=max_conf_rbf
	pred_set_rbf=set(abs_dict.keys())
	for key in pred_dict_rbf.keys():
		if pred_dict_rbf[key]>confidence:
			pred_set_rbf.add(key)
	real_set=set(body_dict.keys())
	tp=len(pred_set_rbf&real_set)
	fp=len(pred_set_rbf-(pred_set_rbf&real_set))
	fn=len(real_set-(real_set&pred_set_rbf))
	try:
		P=tp/(tp+fp)
	except:
		P=0.0
	try:
		R=tp/(tp+fn)
	except:
		R=0.0

	for entity in pred_set_rbf:
		if entity in body_dict.keys():
			print("\033[1;31m"+entity)
		else:
			print("\033[0m"+entity)

	print("\033[0m\nprecision=%.2f,recall=%.2f."%(P,R))
elif test_mode==3:
	f=open("/home/ubuntu/results/coclf/b_clf_sgd_plusdis_minusvec.pkl","rb")
	b_clf_sgd=pickle.load(f)
	f.close()
	max_conf_rbf=0
	pred_dict_rbf={}
	abs_key_list=list(abs_dict.keys())
	for i1 in range(0,len(abs_key_list)):
		for i2 in range(i1,len(abs_key_list)):
			a_key_1=abs_key_list[i1]
			a_key_2=abs_key_list[i2]
			if a_key_1==a_key_2:
				continue
			if not a_key_1 in b_word_list or not a_key_2 in b_word_list:
				continue
			for b_key in word_list:
				if a_key_1==b_key or a_key_2==b_key:
					continue
				if not b_key in b_word_list:
					continue
				a1_tvec=b_word2tvec[a_key_1]
				a2_tvec=b_word2tvec[a_key_2]
				b_tvec=b_word2tvec[b_key]
				disa1a2=tree_distance(a1_tvec,a2_tvec)
				disa1b=tree_distance(a1_tvec,b_tvec)
				disa2b=tree_distance(a2_tvec,b_tvec)
				# dis_sum=tree_distance(a1_tvec,b_tvec)+tree_distance(a2_tvec,b_tvec)
				# if dis_sum>taxonomy_distance:
				# 	continue
				try:
					_list=[dev_mat[b_word_list.index(a_key_1)][b_word_list.index(b_key)],dev_mat[b_word_list.index(a_key_2)][b_word_list.index(b_key)]]
				except:
					_list=[0.0,0.0]
				_list.extend([disa1a2,disa1b,disa2b])
				# _list.extend(a1_tvec)
				# _list.extend(a2_tvec)
				# _list.extend(b_tvec)
				sample_input=np.array([_list])
				pred_label_rbf=list(b_clf_sgd.predict(sample_input))[0]
				if pred_label_rbf==1:
					if b_key in pred_dict_rbf.keys():
						pred_dict_rbf[b_key]+=1.0
						if pred_dict_rbf[b_key]>max_conf_rbf:
							max_conf_rbf=pred_dict_rbf[b_key]
					else:
						pred_dict_rbf[b_key]=1.0
						if pred_dict_rbf[b_key]>max_conf_rbf:
							max_conf_rbf=pred_dict_rbf[b_key]
	for key in pred_dict_rbf.keys():
		pred_dict_rbf[key]/=max_conf_rbf
	pred_set_rbf=set(abs_dict.keys())
	for key in pred_dict_rbf.keys():
		if pred_dict_rbf[key]>confidence:
			pred_set_rbf.add(key)
	real_set=set(body_dict.keys())
	tp=len(pred_set_rbf&real_set)
	fp=len(pred_set_rbf-(pred_set_rbf&real_set))
	fn=len(real_set-(real_set&pred_set_rbf))
	try:
		P=tp/(tp+fp)
	except:
		P=0.0
	try:
		R=tp/(tp+fn)
	except:
		R=0.0

	for entity in pred_set_rbf:
		if entity in body_dict.keys():
			print("\033[1;31m"+entity)
		else:
			print("\033[0m"+entity)

	print("\033[0m\nprecision=%.2f,recall=%.2f."%(P,R))
else:
	f=open("/home/ubuntu/results/coclf/b_clf_sgd_without_onto.pkl","rb")
	b_clf_sgd=pickle.load(f)
	f.close()
	max_conf_rbf=0
	pred_dict_rbf={}
	abs_key_list=list(abs_dict.keys())
	for i1 in range(0,len(abs_key_list)):
		for i2 in range(i1,len(abs_key_list)):
			a_key_1=abs_key_list[i1]
			a_key_2=abs_key_list[i2]
			if a_key_1==a_key_2:
				continue
			if not a_key_1 in b_word_list or not a_key_2 in b_word_list:
				continue
			for b_key in word_list:
				if a_key_1==b_key or a_key_2==b_key:
					continue
				if not b_key in b_word_list:
					continue
				a1_tvec=b_word2tvec[a_key_1]
				a2_tvec=b_word2tvec[a_key_2]
				b_tvec=b_word2tvec[b_key]
				disa1a2=tree_distance(a1_tvec,a2_tvec)
				disa1b=tree_distance(a1_tvec,b_tvec)
				disa2b=tree_distance(a2_tvec,b_tvec)
				dis_sum=tree_distance(a1_tvec,b_tvec)+tree_distance(a2_tvec,b_tvec)
				if dis_sum>taxonomy_distance:
					continue
				try:
					_list=[dev_mat[b_word_list.index(a_key_1)][b_word_list.index(b_key)],dev_mat[b_word_list.index(a_key_2)][b_word_list.index(b_key)]]
				except:
					_list=[0.0,0.0]
				sample_input=np.array([_list])
				pred_label_rbf=list(b_clf_sgd.predict(sample_input))[0]
				if pred_label_rbf==1:
					if b_key in pred_dict_rbf.keys():
						pred_dict_rbf[b_key]+=1.0
						if pred_dict_rbf[b_key]>max_conf_rbf:
							max_conf_rbf=pred_dict_rbf[b_key]
					else:
						pred_dict_rbf[b_key]=1.0
						if pred_dict_rbf[b_key]>max_conf_rbf:
							max_conf_rbf=pred_dict_rbf[b_key]
	for key in pred_dict_rbf.keys():
		pred_dict_rbf[key]/=max_conf_rbf
	pred_set_rbf=set(abs_dict.keys())
	for key in pred_dict_rbf.keys():
		if pred_dict_rbf[key]>confidence:
			pred_set_rbf.add(key)
	real_set=set(body_dict.keys())
	tp=len(pred_set_rbf&real_set)
	fp=len(pred_set_rbf-(pred_set_rbf&real_set))
	fn=len(real_set-(real_set&pred_set_rbf))
	try:
		P=tp/(tp+fp)
	except:
		P=0.0
	try:
		R=tp/(tp+fn)
	except:
		R=0.0

	for entity in pred_set_rbf:
		if entity in body_dict.keys():
			print("\033[1;31m"+entity)
		else:
			print("\033[0m"+entity)

	print("\033[0m\nprecision=%.2f,recall=%.2f."%(P,R))