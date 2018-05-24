import sys
import math
import pickle
import numpy as np
from sklearn import svm

split_ratio=float(sys.argv[1])
volume=int(sys.argv[2])

f=open("/home/ubuntu/results/saliency/featured.pkl","rb")
featured_list=pickle.load(f)
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

f=open("/home/ubuntu/results/coclf/bsvd_linear.pkl","rb")
clf_linear=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/bsvd_rbf_default.pkl","rb")
clf_rbf=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/bsvd_rfc.pkl","rb")
clf_rfc=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/bsvd_abc.pkl","rb")
clf_abc=pickle.load(f)
f.close()

count=0

tp_linear=0.0
fp_linear=0.0
fn_linear=0.0
tp_rbf=0.0
fp_rbf=0.0
fn_rbf=0.0

pred_pos=0
pred_neg=0

for i in range(int(split_ratio*len(featured_list)),len(featured_list)):
	abs_dict=featured_list[i]['abs']
	body_dict=featured_list[i]['body']
	pred_set_linear=set()
	pred_set_rbf=set()
	if count>volume:
		break
	a_mat=[]
	for a_key in abs_dict.keys():
		pred_input=np.array([[key_phrase[word_list.index(a_key)],abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],idf[word_list.index(a_key)],centrality[a_key]]])
		pred_saliency=list(s_clf.predict(pred_input))[0]
		a_mat.append([abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],idf[word_list.index(a_key)],centrality[a_key],pred_saliency])
	if not a_mat:
		continue
	U,S,V=np.linalg.svd(np.array(a_mat),full_matrices=True)
	for b_key in word_list:
		if b_key in abs_dict.keys():
			continue
		flag=0
		for a_dec in abs_dict.keys():
			if dev_mat[word_list.index(a_dec)][word_list.index(b_key)]>0.0:
				flag=1
				break
		if not flag:
			continue
		count+=1
		# label=0
		# if b_key in body_dict.keys():
		# 	label=-1
		# else:
		# 	label=1
		print(count,abs_dict.keys(),b_key)
		_feature=[idf[word_list.index(b_key)],centrality[b_key]]
		_feature.extend(list(V.flatten()))
		sample_input=np.array([_feature])
		pred_label_linear=list(clf_linear.predict(sample_input))[0]
		pred_label_rbf=list(clf_rbf.predict(sample_input))[0]
		pred_label_rfc=list(clf_rfc.predict(sample_input))[0]
		pred_label_abc=list(clf_abc.predict(sample_input))[0]
		if pred_label_linear==1:
			pred_set_linear.add(b_key)
		if pred_label_rbf==1:
			pred_set_rbf.add(b_key)
			pred_pos+=1
		else:
			pred_neg+=1
	real_set=set(body_dict.keys())
	tp_linear+=len(pred_set_linear&real_set)
	fp_linear+=len(pred_set_linear-(pred_set_linear&real_set))
	fn_linear+=len(real_set-(real_set&pred_set_linear))
	tp_rbf+=len(pred_set_rbf&real_set)
	fp_rbf+=len(pred_set_rbf-(pred_set_rbf&real_set))
	fn_rbf+=len(real_set-(real_set&pred_set_rbf))
		# if pred_label_linear==label:
		# 	if pred_label_linear==1:
		# 		tp_linear+=1
		# else:
		# 	if pred_label_linear==1:
		# 		fp_linear+=1
		# 	else:
		# 		fn_linear+=1
		# if pred_label_rbf==label:
		# 	if pred_label_rbf==-1:
		# 		tp_rbf+=1
		# else:
		# 	if pred_label_rbf==-1:
		# 		fp_rbf+=1
		# 	else:
		# 		fn_rbf+=1
print(pred_pos,pred_neg)

P=tp_linear/(tp_linear+fp_linear)
R=tp_linear/(tp_linear+fn_linear)
F1=2*P*R/(P+R)

print(P,R,F1)

P=tp_rbf/(tp_rbf+fp_rbf)
R=tp_rbf/(tp_rbf+fn_rbf)
F1=2*P*R/(P+R)

print(P,R,F1)