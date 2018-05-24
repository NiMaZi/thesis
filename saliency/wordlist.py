# after distanced

import pickle
import numpy as np

f=open("/home/ubuntu/results/saliency/distanced.pkl","rb")
dis_list=pickle.load(f)
f.close()

# f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
# word_list=pickle.load(f)
# f.close()

word_count={}

# listed_word_set=[]

for entry in dis_list:

	# entry_set=set()

	_list=entry['abs']
	for item in _list:
		if item[1] in word_count.keys():
			word_count[item[1]]+=1.0
		else:
			word_count[item[1]]=1.0
		# entry_set.add(item[1])

	_list=entry['body']
	for item in _list:
		if item[1] in word_count.keys():
			word_count[item[1]]+=1.0
		else:
			word_count[item[1]]=1.0
		# entry_set.add(item[1])

	_list=entry['title']
	for item in _list:
		if item[1] in word_count.keys():
			word_count[item[1]]+=1.0
		else:
			word_count[item[1]]=1.0
		# entry_set.add(item[1])

	_list=entry['keywords']
	for item in _list:
		if item[1] in word_count.keys():
			word_count[item[1]]+=1.0
		else:
			word_count[item[1]]=1.0
		# entry_set.add(item[1])

	# listed_word_set.append(entry_set)



# print(len(word_list))

# idf=[]
# volume=len(dis_list)

# for word in word_list:
# 	d_count=0
# 	for doc in listed_word_set:
# 		if word in doc:
# 			d_count+=1
# 	idf.append(np.log(float(volume)/float(1.0+d_count)))

f=open("/home/ubuntu/results/saliency/w_count.pkl","wb")
pickle.dump(word_count,f)
f.close()

for key in word_count.keys():
	word_count[key]/=len(dis_list)

f=open("/home/ubuntu/results/saliency/wf_count.pkl","wb")
pickle.dump(word_count,f)
f.close()