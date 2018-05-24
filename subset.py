import csv
import json

volume=12935

word_set=set()
for i in range(0,volume):
	try:
		f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[1]=="ConceptCode":
				continue
			word_set.add(item[1])
		f.close()
	except:
		pass

	try:
		f=open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[1]=="ConceptCode":
				continue
			word_set.add(item[1])
		f.close()
	except:
		pass

	try:
		f=open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[1]=="ConceptCode":
				continue
			word_set.add(item[1])
		f.close()
	except:
		pass

	try:
		f=open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv",'r',encoding='utf-8')
		rd=csv.reader(f)
		for item in rd:
			if item[1]=="ConceptCode":
				continue
			word_set.add(item[1])
		f.close()
	except:
		pass

print(len(word_set))
word_list=list(word_set)

f=open("/home/ubuntu/results_new/ontology/sub_word_list.json",'w',encoding='utf-8')
json.dump(word_list,f)
f.close()