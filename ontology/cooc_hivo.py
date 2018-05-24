import json
import numpy as np

f=open("/home/ubuntu/results/saliency/dis_list_com.json",'r')
dis_list_com=json.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/n2id.json",'r')
n2id=json.load(f)
f.close()

cooc_simple=np.zeros((1295,1295))

for record in dis_list_com:
	_mention=list(set([item[1] for item in record['abs']+record['body']]))
	for i in range(0,len(_mention)):
		for j in range(i,len(_mention)):
			if i==j:
				continue
			try:
				cooc_simple[n2id[_mention[i]]][n2id[_mention[j]]]+=1.0
				cooc_simple[n2id[_mention[j]]][n2id[_mention[i]]]+=1.0
			except:
				pass

cooc_simple/=len(dis_list_com)

path="/home/ubuntu/results/statistics/cooc_simple.npy"
np.save(path,cooc_simple)