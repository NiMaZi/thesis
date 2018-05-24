import sys
import pickle
import random
import numpy as np
from difflib import SequenceMatcher as Sqm

split_ratio=float(sys.argv[1])

f=open("/home/ubuntu/results/saliency/featured.pkl","rb")
featured_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/distanced.pkl","rb")
dis_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/saliency/keywords_list","rb")
keywords_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
word_list=pickle.load(f)
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

sample_prelist=[]
count=0
p_count=0

for i in range(0,len(dis_list)):
	if not featured_list[i]['abs']:
		continue
	if not dis_list[i]['title'] and not dis_list[i]['keywords']:
		continue
	label_set=set()
	body_set=set()
	for item in dis_list[i]['title']:
		label_set.add(item[1])
	for item in dis_list[i]['keywords']:
		label_set.add(item[1])
	for item in dis_list[i]['body']:
		body_set.add(item[1])
	for key in featured_list[i]['abs'].keys():
		# count+=1
		label=0
		if key in label_set:
			label=1
		for kw in keywords_list[i]:
			if Sqm(None,kw,key.lower()).ratio()>=0.5:
				label=1
				break
		# p_count+=label
		if key in word_list:
			k_idf=idf[word_list.index(key)]
		else:
			k_idf=0.0
		sample_prelist.append([key,featured_list[i]['abs'][key][0],featured_list[i]['abs'][key][1]-featured_list[i]['abs'][key][0],featured_list[i]['abs'][key][2],k_idf,centrality[key],label]) #(str)entity, (float)distance, (float)spread, (int)count, (float)idf, (float)centrality, label

split=int(split_ratio*len(sample_prelist))

key_phrase=[0 for i in range(0,len(word_list))]

for i in range(0,split):
	try:
		key_phrase[word_list.index(sample_prelist[i][0])]+=sample_prelist[i][6]
	except:
		pass

f=open("/home/ubuntu/results/saliency/keyphrase.pkl","wb")
pickle.dump(key_phrase,f)
f.close()

training_list=[]
for i in range(0,split):
	if sample_prelist[i][0] in word_list:
		kpns=key_phrase[word_list.index(sample_prelist[i][0])]
		_list=list(np.array(word2tvec[sample_prelist[i][0]])+1.0)
	else:
		kpns=0.0
		_list=[0.0 for i in range(0,16)]
	_list.extend([kpns,sample_prelist[i][1],sample_prelist[i][2],sample_prelist[i][3],sample_prelist[i][4],sample_prelist[i][5],sample_prelist[i][6]])
	training_list.append(_list)

f=open("/home/ubuntu/results/saliency/trainlist.pkl","wb")
pickle.dump(training_list,f)
f.close()

test_list=[]
for i in range(split,len(sample_prelist)):
	if sample_prelist[i][0] in word_list:
		kpns=key_phrase[word_list.index(sample_prelist[i][0])]
		_list=list(np.array(word2tvec[sample_prelist[i][0]])+1.0)
	else:
		kpns=0.0
		_list=[0.0 for i in range(0,16)]
	_list.extend([kpns,sample_prelist[i][1],sample_prelist[i][2],sample_prelist[i][3],sample_prelist[i][4],sample_prelist[i][5],sample_prelist[i][6]])
	test_list.append(_list)

f=open("/home/ubuntu/results/saliency/testlist.pkl","wb")
pickle.dump(test_list,f)
f.close()