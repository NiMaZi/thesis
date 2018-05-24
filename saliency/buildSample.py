import sys
import csv
import pickle
import random

volume=int(sys.argv[1])

# tf, tfall, idf, first-position, label.

f=open("/home/ubuntu/results/tfidf/tfidf3500.pickle","rb")
tfidf=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/tfidf/wordlist3500.pickle","rb")
word_list=pickle.load(f)
f.close()

f=open("/home/ubuntu/results/tfidf/tfall.pickle","rb")
freq=pickle.load(f)
f.close()

sample_list=[]
positive=0
negative=0

for i in range(0,volume):
	words_body=set([])
	with open("/home/ubuntu/thesiswork/data/body"+str(i)+".txt.mentions","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			words_body.add(item[2])

	with open("/home/ubuntu/thesiswork/data/abs"+str(i)+".txt.mentions","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		words_abs=set([])
		for item in reader:
			if item[2]=="ConceptName":
				continue
			if item[2] in words_abs:
				continue
			else:
				tf_idf=tfidf[word_list.index(item[2])][i]
				tfall=freq[item[2]]
				first_position=int(item[4])
				if item[2] in words_body:
					positive+=1
					label=1
				else:
					negative+=1
					label=0
				sample_list.append([tf_idf,tfall,first_position,label])

pn_rate=int(positive/negative)
# n_sample_list=[]

# for sample in sample_list:
# 	if sample[3]==0:
# 		for i in range(0,pn_rate):
# 			n_sample_list.append(sample)

# sample_list.extend(n_sample_list)
random.shuffle(sample_list)

f=open("/home/ubuntu/results/svm/saliency_samples.pickle","wb")
pickle.dump(sample_list,f)
f.close()

print(pn_rate)

# 3500 articles -> 10809 samples, p-n rate 31