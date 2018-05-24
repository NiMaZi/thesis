import sys
import csv
import json

volume=int(sys.argv[1])

f=open("/home/ubuntu/results_new/ontology/stypelist.json",'r',encoding='utf-8')
semantic_list=json.load(f)
f.close()

f=open("/home/ubuntu/results_new/ontology/word_list.json",'r',encoding='utf-8')
word_list=json.load(f)
f.close()

f=open("/home/ubuntu/results_new/ontology/word_dict.json",'r',encoding='utf-8')
word_dict=json.load(f)
f.close()

word_count={}
semantic_count={}

for w in word_list:
	word_count[w]=0.0

for s_type in semantic_list:
	semantic_count[s_type]=0.0

for i in range(0,volume):
	try:
		with open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				word_count[item[1]]+=1.0
				try:
					for s_type in item[3].split('|'):
						semantic_count[s_type]+=1.0
				except:
					continue
	except:
		continue
	try:
		with open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				word_count[item[1]]+=1.0
				try:
					for s_type in item[3].split('|'):
						semantic_count[s_type]+=1.0
				except:
					continue
	except:
		continue
	try:
		with open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				word_count[item[1]]+=1.0
				try:
					for s_type in item[3].split('|'):
						semantic_count[s_type]+=1.0
				except:
					continue
	except:
		continue
	try:
		with open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				word_count[item[1]]+=1.0
				try:
					for s_type in item[3].split('|'):
						semantic_count[s_type]+=1.0
				except:
					continue
	except:
		continue

f=open("/home/ubuntu/results_new/ontology/word_count.json",'w',encoding='utf-8')
json.dump(word_count,f)
f.close()

f=open("/home/ubuntu/results_new/ontology/semantic_count.json",'w',encoding='utf-8')
json.dump(semantic_count,f)
f.close()