import sys
import math
import pickle
import random
import numpy as np
from sklearn import linear_model as lm

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

split_ratio=float(sys.argv[1])

partial_volume=10000

f=open("/home/ubuntu/results/saliency/featured.pkl","rb")
featured_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/wordlist.pkl","rb")
t_word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_word2taxonomy.pkl","rb")
word2tvec=pickle.load(f)
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

pos_list=[]
neg_prelist=[]
count=0
t_count=0
clf_sgd=lm.SGDClassifier()

for i in range(0,int(len(featured_list)*split_ratio)):
	abs_dict=featured_list[i]['abs']
	body_dict=featured_list[i]['body']
	abs_elist=list(abs_dict.keys())
	for i1 in range(0,len(abs_elist)):
		for i2 in range(i1,len(abs_elist)):
			a_key_1=abs_elist[i1]
			a_key_2=abs_elist[i2]
			if a_key_1==a_key_2:
				continue
			if not a_key_1 in word_list or not a_key_2 in word_list:
				continue
			# pred_input=np.array([[key_phrase[word_list.index(a_key_1)],abs_dict[a_key_1][0],abs_dict[a_key_1][1]-abs_dict[a_key_1][0],abs_dict[a_key_1][2],idf[word_list.index(a_key_1)],centrality[a_key_1]],[key_phrase[word_list.index(a_key_2)],abs_dict[a_key_2][0],abs_dict[a_key_2][1]-abs_dict[a_key_2][0],abs_dict[a_key_2][2],idf[word_list.index(a_key_2)],centrality[a_key_2]]])
			# pred_saliency_1=list(s_clf.predict(pred_input))[0]
			# pred_saliency_2=list(s_clf.predict(pred_input))[1]
			for b_key in t_word_list:
				if not b_key in word_list:
					continue
				if a_key_1==b_key or a_key_2==b_key:
					continue
				# disa1a2=tree_distance(word2tvec[a_key_1],word2tvec[a_key_2])
				# disa1b=tree_distance(word2tvec[a_key_1],word2tvec[b_key])
				# disa2b=tree_distance(word2tvec[a_key_2],word2tvec[b_key])
				try:
					_list=[dev_mat[word_list.index(a_key_1)][word_list.index(b_key)],dev_mat[word_list.index(a_key_2)][word_list.index(b_key)]]
				except:
					_list=[0.0,0.0]
				# _list.extend([disa1a2,disa1b,disa2b])
				if b_key in body_dict.keys():
					label=1
					_list.append(label)
					pos_list.append(_list)
				else:
					label=0
					_list.append(label)
					neg_prelist.append(_list)
					# neg_prelist.append([centrality[a_key_1],centrality[a_key_2],centrality[b_key],dev_mat[word_list.index(a_key_1)][word_list.index(b_key)],dev_mat[word_list.index(a_key_1)][word_list.index(b_key)],pred_saliency_1,pred_saliency_2,label])
				# print(count,a_key_1,a_key_2,b_key,label)
				count+=1
				t_count+=1
	if count>partial_volume:
		# sample_prelist=pos_list+neg_prelist
		sample_prelist=random.sample(neg_prelist,len(pos_list))
		sample_prelist.extend(pos_list)
		tlen=len(sample_prelist[0])
		nTrain=np.array(sample_prelist)
		nX=nTrain[:,0:tlen-1]
		ny=nTrain[:,tlen-1]
		clf_sgd.partial_fit(nX,ny,classes=[0,1])
		pos_list=[]
		neg_prelist=[]
		count=0

f=open("/home/ubuntu/results/coclf/b_clf_sgd_without_onto.pkl","wb")
pickle.dump(clf_sgd,f)
f.close()
print(sys.argv[0])
print(t_count)