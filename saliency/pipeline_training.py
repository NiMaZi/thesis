import csv
import sys
import json
import math
import pickle
import numpy as np
from sklearn import svm
from difflib import SequenceMatcher as Sqm

from semantic_type import isolated_num
from semantic_type import isolated_list

volume=int(sys.argv[1])
secondary_match_rate=float(sys.argv[1])
samples=[]
key_phrase={}

f=open("/home/ubuntu/results/saliency/keywords_list","rb")
keywords_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results_new/ontology/word_list.json","r")
word_list=json.load(f)
f.close()

for w in word_list:
	key_phrase[w]=0.0

for i in range(0,volume):

	f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".txt",'r',encoding='utf-8')
	doc=f.read()
	abs_length=len(doc)
	f.close()

	salient_mentions=set()

	f=open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".csv",'r',encoding='utf-8')
	reader=csv.reader(f)
	for item in reader:
		if item[1]=='ConceptCode':
			continue
		salient_mentions.add(item[1])
	f.close()

	f=open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv",'r',encoding='utf-8')
	reader=csv.reader(f)
	for item in reader:
		if item[1]=='ConceptCode':
			continue
		salient_mentions.add(item[1])
	f.close()

	abs_dict={}

	f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv",'r',encoding='utf-8')
	reader=csv.reader(f)
	for item in reader:
		if item[1]=='ConceptCode':
			continue
		if item[1] in abs_dict.keys():
			abs_dict[item[1]][0]+=1.0
			new_dis=float(item[4])/float(abs_length)
			if new_dis>abs_dict[item[1]][2]:
				abs_dict[item[1]][2]=new_dis
		else:
			try:
				s_type=isolated_list.index(item[3].split('|')[0])
			except:
				s_type=0
			dis=float(item[4])/float(abs_length)
			label=0
			if item[1] in salient_mentions:
				label=1
			else:
				for kw in keywords_list[i]:
					if Sqm(None,kw,item[0].lower()).ratio()>=secondary_match_rate:
						label=1
						break
			abs_dict[item[1]]=[1.0,dis,dis,s_type,label]

	for key in abs_dict.keys():
		samples.append([key,abs_dict[key][0],abs_dict[key][1],abs_dict[key][2]-abs_dict[key][1],abs_dict[key][3],abs_dict[key][4]])
		key_phrase[key]+=abs_dict[key][4]

sample_set=[]
for sample in samples:
	sample_set.append([key_phrase[sample[0]]]+sample[1:])

training_set=sample_set[0:int(len(sample_set)*0.5)]
testing_set=sample_set[int(len(sample_set)*0.5):len(sample_set)]

n_training=np.array(training_set)
n_testing=np.array(testing_set)
n_training_X=n_training[:,0:5]
n_training_y=n_training[:,5]
n_testing_X=n_testing[:,0:5]
n_testing_y=list(n_testing[:,5])

clf_rbf=svm.SVC(kernel='rbf',gamma=(1.0/5.0),class_weight={1:(1/0.187),0:(1/0.813)})
clf_rbf.fit(n_training_X,n_training_y)
n_predicted_y=list(clf_rbf.predict(n_testing_X))

tp=0.0
fp=0.0
fn=0.0

for i in range(0,len(n_testing_y)):
	if n_testing_y[i]==n_predicted_y[i]:
		if n_predicted_y[i]==1:
			tp+=1
	else:
		if n_predicted_y[i]==1:
			fp+=1
		else:
			fn+=1

P=tp/(tp+fp)
R=tp/(tp+fn)
F1=2*P*R/(P+R)

f=open("/home/ubuntu/results_new/log.txt","w")
f.write(str(P))
f.write("\n")
f.write(str(R))
f.write("\n")
f.write(str(F1))
f.write("\n")
f.close()