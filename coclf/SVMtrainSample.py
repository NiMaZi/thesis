import sys
import math
import pickle
import random
import numpy as np
# from sklearn import svm
from sklearn import linear_model as lm

split_ratio=float(sys.argv[1])

f=open("/home/ubuntu/results/saliency/featured.pkl","rb")
featured_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
b_word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_word2taxonomy.pkl","rb")
b_word2tvec=pickle.load(f)
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

for i in range(0,int(len(featured_list)*split_ratio)):
	abs_dict=featured_list[i]['abs']
	body_dict=featured_list[i]['body']
	for a_key in abs_dict.keys():
		pred_input=np.array([[key_phrase[word_list.index(a_key)],abs_dict[a_key][0],abs_dict[a_key][1]-abs_dict[a_key][0],abs_dict[a_key][2],idf[word_list.index(a_key)],centrality[a_key]]])
		pred_saliency=list(s_clf.predict(pred_input))[0]
		for b_key in word_list:
			if a_key==b_key:
				continue
			if b_key in body_dict.keys():
				label=1
				pos_list.append([centrality[a_key],centrality[b_key],dev_mat[word_list.index(a_key)][word_list.index(b_key)],pred_saliency,label])
			else:
				label=0
				neg_prelist.append([centrality[a_key],centrality[b_key],dev_mat[word_list.index(a_key)][word_list.index(b_key)],pred_saliency,label])
			print(count,a_key,b_key,label)
			count+=1

sample_prelist=random.sample(neg_prelist,len(pos_list))
sample_prelist.extend(pos_list)
random.shuffle(sample_prelist)

print(len(sample_prelist))

# clf_linear=svm.LinearSVC()
# clf_rbf=svm.SVC(gamma=0.25)

clf_lr=lm.LogisticRegression()
clf_sgd=lm.SGDClassifier()

nTrain=np.array(sample_prelist)
nX=nTrain[:,0:4]
ny=nTrain[:,4]

# clf_linear.fit(nX,ny)
# clf_rbf.fit(nX,ny)
clf_lr.fit(nX,ny)
clf_sgd.fit(nX,ny)

f=open("/home/ubuntu/results/coclf/clf_linear.pkl","wb")
pickle.dump(clf_linear,f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_rbf.pkl","wb")
pickle.dump(clf_rbf,f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_lr.pkl","wb")
pickle.dump(clf_lr,f)
f.close()

f=open("/home/ubuntu/results/coclf/clf_sgd.pkl","wb")
pickle.dump(clf_sgd,f)
f.close()