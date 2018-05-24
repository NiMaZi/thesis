import os
import sys
import csv
import subprocess

# Mention,ConceptCode,ConceptName,Synonyms,Start,End

# Document	Matched Term	Code	Concept Name	Semantic Type	Annotations	Certainty	ContextualAspect	ContextualModality	Degree	Experiencer	Permanence	Polarity	Temporality

fname=['abs','body','title','keywords']

for i in range(0,12935):
	for j in range(0,4):
		subprocess.call(['java','-jar','/home/ubuntu/ner/NobleJar/NobleCoder-1.0.jar','-terminology','NCI_Thesaurus','-input','/home/ubuntu/thesiswork/kdata/'+fname[j]+str(i)+'.txt','-output','/home/ubuntu/thesiswork/kdata/disambiguation','-search','best-match','-selectBestCandidates'])
		f=open('/home/ubuntu/thesiswork/kdata/disambiguation/RESULTS.tsv','r',encoding='utf-8')
		unamb=f.read()
		f.close()
		tmp_list=unamb.split('\n')
		unamb_list=[]
		for item in tmp_list:
			unamb_list.append(item.split('\t'))
		result_list=[]
		result_list.append(['Mention','ConceptCode','ConceptName','SemanticType','Start'])
		unamb_list=unamb_list[1:len(unamb_list)-1]
		for item in unamb_list:
			result_list.append([item[1],item[2],item[3],item[4],item[5].split(',')[0].split('/')[1]])
		f=open('/home/ubuntu/thesiswork/kdata/'+fname[j]+str(i)+'.csv','w',encoding='utf-8')
		wr=csv.writer(f)
		for row in result_list:
			wr.writerow(row)
		f.close()
