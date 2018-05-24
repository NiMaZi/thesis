import sys
import math
import pickle
import numpy as np
from sklearn import svm

split_ratio=float(sys.argv[1])

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

sample_prelist=[]
count=0

for i in range(0,int(split_ratio*len(featured_list))):
	abs_dict=featured_list[i]['abs']
	body_dict=featured_list[i]['body']
	a_mat=[]
	for a_key in abs_dict.keys():
		pred_input=np.array([[key_phrase[word_list.index(a_key)],abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],idf[word_list.index(a_key)],centrality[a_key]]])
		pred_saliency=list(s_clf.predict(pred_input))[0]
		a_mat.append([abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],idf[word_list.index(a_key)],centrality[a_key],pred_saliency])
	if not a_mat:
		continue
	U,S,V=np.linalg.svd(np.array(a_mat),full_matrices=True)
	for b_key in word_list:
		if b_key in abs_dict.keys() or b_key in body_dict.keys():
			continue
		count+=1
		print(count,abs_dict.keys(),b_key)
		_feature=[idf[word_list.index(b_key)],centrality[b_key]]
		_feature.extend(list(V.flatten()))
		sample_prelist.append(_feature)

# print(sample_prelist)

f=open("/home/ubuntu/results/coclf/svd_inverse_trainlist.pkl","wb")
pickle.dump(sample_prelist,f)
f.close()