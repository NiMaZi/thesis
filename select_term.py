import csv
import sys
import os
import pickle
from ner.ner.annotator.annotator import Annotator
from difflib import SequenceMatcher as Sqm

volume=int(sys.argv[1])
term_num=int(sys.argv[2])

term_dir="/home/ubuntu/.noble/terminologies/"

term_list=[]
term_dict={}

for file in os.listdir(term_dir):
	if file.split(".")[0]:
		term_list.append([file.split(".")[0],0.0])

kf=open("/home/ubuntu/results/saliency/keywords_list","rb")
keywords_list=pickle.load(kf)
kf.close()

if term_num==-1:
	term_num=len(term_list)

annotator_list=[]
for i in range(0,term_num):
	print("buidling annotator with term "+term_list[i][0]+".\n")
	_annotator = Annotator("ner/NobleJar/NobleCoder-1.0.jar","ner/NobleJar/Annotator.java",searchMethod="best-match",terminology=term_list[i][0])
	annotator_list.append(_annotator)

if volume==-1:
	volume=12935

for i in range(0,volume): # max 12935
	keywords=keywords_list[i]
	for j in range(0,len(term_list)):
		
		annotation_set=set()

		_filename = "thesiswork/kdata/abs"+str(i)+".txt"
		annotator_list[j].process(_filename)

		f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".txt.mentions","r",newline='',encoding='utf-8')
		reader=csv.reader(f)
		try:
			for item in reader:
				if item[2]=="ConceptName":
					continue
				annotation_set.add(item[2])
		except:
			pass
		f.close()

		_filename = "thesiswork/kdata/body"+str(i)+".txt"
		annotator_list[j].process(_filename)

		f=open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".txt.mentions","r",newline='',encoding='utf-8')
		reader=csv.reader(f)
		try:
			for item in reader:
				if item[2]=="ConceptName":
					continue
				annotation_set.add(item[2])
		except:
			pass
		f.close()

		term_score=0.0

		for keyword in keywords:
			for annotation in annotation_set:
				term_score+=Sqm(None,keyword.lower(),annotation.lower()).ratio()

		term_score/=len(keywords)
		term_list[j][1]+=term_score

for term in term_list:
	term[1]/=volume
	term_dict[term[0]]=[term[1]]

print(term_dict)

sorted_terms=sorted(term_dict.items(),key=lambda x:x[1])

of=open("/home/ubuntu/select_term_log.txt","w")

for term in sorted_terms:
	of.write(str(term))
	of.write("\n")

of.close()