import sys
import math
import pickle
import numpy as np
# from sklearn import svm
from sklearn import linear_model as lm

front_split_ratio=float(sys.argv[1])
end_split_ratio=float(sys.argv[2])
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

# f=open("/home/ubuntu/results/coclf/clf_lr.pkl","rb")
# clf_lr=pickle.load(f)
# f.close()

f=open("/home/ubuntu/results/coclf/clf_sgd.pkl","rb")
clf_sgd=pickle.load(f)
f.close()

count=0

while confidence<1.0:
	tp=0.0
	fp=0.0
	fn=0.0
	for i in range(int(front_split_ratio*len(featured_list)),int(end_split_ratio*len(featured_list))):
		abs_dict=featured_list[i]['abs']
		body_dict=featured_list[i]['body']
		max_conf_linear=0
		max_conf_rbf=0
		pred_dict_linear={}
		pred_dict_rbf={}
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
		tp+=len(pred_set_rbf&real_set)
		fp+=len(pred_set_rbf-(pred_set_rbf&real_set))
		fn+=len(real_set-(real_set&pred_set_rbf))
	try:
		P=tp/(tp+fp)
	except:
		P=0.0
	try:
		R=tp/(tp+fn)
	except:
		R=0.0
	try:
		F1=2*P*R/(P+R)
	except:
		F1=0.0
	f=open("/home/ubuntu/results/coclf/sgd_test_log_single.txt","a")
	f.write(str(confidence)+","+str(P)+","+str(R)+","+str(F1)+"\n")
	f.close()
	confidence+=0.1

# print(len(wrong_samples),len(filter_samples))

# clf_sgd_correction=lm.SGDClassifier()
# clf_sgd_filter=lm.SGDClassifier()

# nTrain=np.array(wrong_samples)
# nX=nTrain[:,0:4]
# ny=nTrain[:,4]

# clf_sgd_correction.fit(nX,ny)

# nTrain=np.array(filter_samples)
# nX=nTrain[:,0:4]
# ny=nTrain[:,4]

# clf_sgd_filter.fit(nX,ny)

# f=open("/home/ubuntu/results/coclf/clf_sgd_correction.pkl","wb")
# pickle.dump(clf_sgd_correction,f)
# f.close()

# f=open("/home/ubuntu/results/coclf/clf_sgd_filter.pkl","wb")
# pickle.dump(clf_sgd_filter,f)
# f.close()

# # P=tp_linear/(tp_linear+fp_linear)
# # R=tp_linear/(tp_linear+fn_linear)
# # F1=2*P*R/(P+R)

# # print(P,R,F1)

# # P=tp_rbf/(tp_rbf+fp_rbf)
# # R=tp_rbf/(tp_rbf+fn_rbf)
# # F1=2*P*R/(P+R)

# # print(P,R,F1)