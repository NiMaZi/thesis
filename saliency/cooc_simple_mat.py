import pickle
import numpy as np

# after distanced & wordlist

f=open("/home/ubuntu/results/saliency/distanced.pkl","rb")
dis_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

simple_mat=np.zeros((len(word_list),len(word_list)))

for entry in dis_list:

	_list=entry['abs']
	_set=set()
	for item in _list:
		_set.add(item[1])
	for a in list(_set):
		for b in list(_set):
			try:
				simple_mat[word_list.index(a)][word_list.index(b)]+=1.0
			except:
				pass

	_list=entry['body']
	_set=set()
	for item in _list:
		_set.add(item[1])
	for a in list(_set):
		for b in list(_set):
			try:
				simple_mat[word_list.index(a)][word_list.index(b)]+=1.0
			except:
				pass

for i in range(0,len(word_list)):
	for j in range(0,len(word_list)):
		simple_mat[i][j]/=(1.0*float(len(dis_list)))

for i in range(0,len(word_list)):
	simple_mat[i][i]=0.0

f=open("/home/ubuntu/results/saliency/simplemat.pkl","wb")
pickle.dump(simple_mat,f)
f.close()