import boto3
import os
import json
import csv

def count_on_corpus(source,volume):
	s3=boto3.resource("s3")
	myBucket=s3.Bucket('workspace.scitodate.com')
	homedir=os.environ['HOME']
	valid_count=0
	mention_count=0
	for i in range(0,volume):
		abs_mention=0
		body_mention=0
		myBucket.download_file("yalun/"+source+"/abs"+str(i)+".csv",homedir+"/temp/tmp.csv")
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=='C38054':
					abs_mention=1
					break
		myBucket.download_file("yalun/"+source+"/body"+str(i)+".csv",homedir+"/temp/tmp.csv")
		with open(homedir+"/temp/tmp.csv",'r',encoding='utf-8') as cf:
			rd=csv.reader(cf)
			for item in rd:
				if item[0]=="Mention":
					continue
				if item[1]=='C38054':
					body_mention=1
					break
		if abs_mention==0:
			valid_count+=1
			if body_mention==1:
				mention_count+=1
	return mention_count,valid_count

if __name__=="__main__":
	valid_count=0
	mention_count=0
	m,v=count_on_corpus("kdata",12000)
	valid_count+=v
	mention_count+=m
	m,v=count_on_corpus("annotated_papers",14000)
	valid_count+=v
	mention_count+=m
	m,v=count_on_corpus("annotated_papers_with_txt",13000)
	valid_count+=v
	mention_count+=m
	m,v=count_on_corpus("annotated_papers_with_txt_new",15000)
	valid_count+=v
	mention_count+=m
	m,v=count_on_corpus("annotated_papers_with_txt_new2",95000)
	valid_count+=v
	mention_count+=m
	homedir=os.environ['HOME']
	f=open(homedir+"/results/statistics/count_eeg.txt",'w')
	f.write("%d,%d\n"%(mention_count,valid_count))
	f.close()