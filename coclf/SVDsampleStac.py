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
pos_count=0
neg_count=0

for i in range(0,int(split_ratio*len(featured_list))):
	abs_dict=featured_list[i]['abs']
	body_dict=featured_list[i]['body']
	if not abs_dict:
		continue
	for b_key in word_list:
		if b_key in abs_dict.keys():
			continue
		count+=1
		if b_key in body_dict.keys():
			pos_count+=1
		else:
			neg_count+=1
		print(count,abs_dict.keys(),b_key)

print(pos_count,neg_count,count)