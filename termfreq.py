import sys
import csv
import pickle
import numpy as np

volume=int(sys.argv[1])

f=open("/home/ubuntu/results/initDict.pickle","rb")
article_list=pickle.load(f)
f.close()

freq={}
word_count=0

for i in range(0,volume):
	print("counting term freq in article "+str(i)+".\n")
	for word in article_list[i]['abs'].keys():
		word_count+=article_list[i]['abs'][word]
		if word in freq:
			freq[word]+=article_list[i]['abs'][word]
		else:
			freq[word]=article_list[i]['abs'][word]
	for word in article_list[i]['body'].keys():
		word_count+=article_list[i]['body'][word]
		if word in freq:
			freq[word]+=article_list[i]['body'][word]
		else:
			freq[word]=article_list[i]['body'][word]

print("normalize.\n")
for word in freq:
	freq[word]=float(freq[word])/float(word_count)

f=open("/home/ubuntu/results/tfidf/tfall.pickle","wb")
pickle.dump(freq,f)
f.close()