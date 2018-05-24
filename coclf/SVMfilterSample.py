import sys
import math
import pickle
import numpy as np
from sklearn import svm
from sklearn import linear_model as lm

front_split_ratio=float(sys.argv[1])
end_split_ratio=float(sys.argv[2])
cor_rate=float(sys.argv[3])
confidence=0.0

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

f=open("/home/ubuntu/results/coclf/clf_lr.pkl","rb")
clf_lr=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_sgd.pkl","rb")
clf_sgd=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_sgd_correction.pkl","rb")
clf_sgd_correction=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_sgd_filter.pkl","rb")
clf_sgd_filter=pickle.load(f)
f.close()

count=0
p_count=0.0
pp_count=0.0
pc_count=0.0
pf_count=0.0

tp_linear=0.0
fp_linear=0.0
fn_linear=0.0
tp_rbf=0.0
fp_rbf=0.0
fn_rbf=0.0

for i in range(int(front_split_ratio*len(featured_list)),int(end_split_ratio*len(featured_list))):
	p_count+=1
	abs_dict=featured_list[i]['abs']
	body_dict=featured_list[i]['body']
	max_conf_linear=0
	max_conf_rbf=0
	pred_dict_linear={}
	pred_dict_rbf={}
	# if count>20000:
	# 	break
	for a_key in abs_dict.keys():
		pred_input=np.array([[key_phrase[word_list.index(a_key)],abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],idf[word_list.index(a_key)],centrality[a_key]]])
		pred_saliency=list(s_clf.predict(pred_input))[0]
		for b_key in word_list:
			if a_key==b_key:
				continue
			count+=1
			label=0
			if b_key in body_dict.keys():
				label=1
			else:
				label=0
			print(count,a_key,b_key,label)
			sample_input=np.array([[centrality[a_key],centrality[b_key],dev_mat[word_list.index(a_key)][word_list.index(b_key)],pred_saliency]])

			# pred_label_linear=list(clf_lr.predict(sample_input))[0]
			# if pred_label_linear==1:
			# 	if b_key in pred_dict_linear.keys():
			# 		pred_dict_linear[b_key]+=1.0
			# 		if pred_dict_linear[b_key]>max_conf_linear:
			# 			max_conf_linear=pred_dict_linear[b_key]
			# 	else:
			# 		pred_dict_linear[b_key]=1.0
			# 		if pred_dict_linear[b_key]>max_conf_linear:
			# 			max_conf_linear=pred_dict_linear[b_key]

			pred_label_rbf=list(clf_sgd.predict(sample_input))[0]
			if pred_label_rbf==1:
				pred_label_correction=list(clf_sgd_correction.predict(sample_input))[0]
				pred_label_filter=list(clf_sgd_filter.predict(sample_input))[0]
				pred_label_rbf=cor_rate*pred_label_correction+(1.0-cor_rate)*pred_label_filter
				if pred_label_rbf>0.5:
					pred_label_rbf=1
				else:
					pred_label_rbf=0

			if pred_label_rbf==1:
				if b_key in pred_dict_rbf.keys():
					pred_dict_rbf[b_key]+=1.0
					if pred_dict_rbf[b_key]>max_conf_rbf:
						max_conf_rbf=pred_dict_rbf[b_key]
				else:
					pred_dict_rbf[b_key]=1.0
					if pred_dict_rbf[b_key]>max_conf_rbf:
						max_conf_rbf=pred_dict_rbf[b_key]

	for key in pred_dict_linear.keys():
		pred_dict_linear[key]/=max_conf_linear

	for key in pred_dict_rbf.keys():
		pred_dict_rbf[key]/=max_conf_rbf

	pred_set_linear=set()
	pred_set_rbf=set()

	for key in pred_dict_linear.keys():
		if pred_dict_linear[key]>confidence:
			pred_set_linear.add(key)

	for key in pred_dict_rbf.keys():
		if pred_dict_rbf[key]>confidence:
			pred_set_rbf.add(key)

	# pred_set_combine=pred_set_rbf&pred_set_linear

	real_set=set(body_dict.keys())-set(abs_dict.keys())
	# tp_linear+=len(pred_set_linear&real_set)
	# fp_linear+=len(pred_set_linear-(pred_set_linear&real_set))
	# fn_linear+=len(real_set-(real_set&pred_set_linear))
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
			# 	if pred_label_rbf==1:
			# 		tp_rbf+=1
			# else:
			# 	if pred_label_rbf==1:
			# 		fp_rbf+=1
			# 	else:
			# 		fn_rbf+=1

# P=tp_linear/(tp_linear+fp_linear)
# R=tp_linear/(tp_linear+fn_linear)
# F1=2*P*R/(P+R)

# print(P,R,F1)

P=tp_rbf/(tp_rbf+fp_rbf)
R=tp_rbf/(tp_rbf+fn_rbf)
F1=2*P*R/(P+R)

hit_rate=(tp_rbf+fp_rbf)/p_count

print(P,R,F1,hit_rate)