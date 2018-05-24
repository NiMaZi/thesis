import sys
import csv
import pickle
from difflib import SequenceMatcher as Sqm

volume=int(sys.argv[1])
real_volume=volume

f=open("/home/ubuntu/results/saliency/keywords_list","rb")
keywords_list=pickle.load(f)
f.close()

record_abs=0.0
record_body=0.0
record_title=0.0
record_kw=0.0
record_sal=0.0
record_cover=0.0

for i in range(0,volume):
	print("counting on article "+str(i)+".")
	record_list_abs=[]
	record_set_abs_entity=set()
	try:
		with open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				mention=item[0]
				entity=item[2]
				record_list_abs.append([mention,entity])
				record_set_abs_entity.add(entity)
				# record_abs+=1
	except:
		real_volume-=1
		continue
	record_set_body=set()
	try:
		with open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				entity=item[2]
				record_set_body.add(entity)
				# record_body+=1
	except:
		real_volume-=1
		continue
	record_set_title=set()
	try:
		with open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				entity=item[2]
				record_set_title.add(entity)
				# record_title+=1
	except:
		real_volume-=1
		continue
	record_set_kw=set()
	try:
		with open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
			reader=csv.reader(csvfile)
			for item in reader:
				if item[2]=="ConceptName":
					continue
				entity=item[2]
				record_set_kw.add(entity)
				# record_kw+=1
	except:
		real_volume-=1
		continue
	cover_set=set()
	saliency_set=set()
	for record in record_list_abs:
		if record[1] in record_set_body:
			# record_cover+=1
			cover_set.add(record[1])
		if record[1] in record_set_kw or record[1] in record_set_title:
			saliency_set.add(record[1])
			# record_sal+=1
			continue
		for kw in keywords_list[i]:
			if Sqm(None,kw,record[0].lower()).ratio()>=0.5:
				saliency_set.add(record[1])
				# record_sal+=1
				break
	record_abs+=len(record_set_abs_entity)
	record_body+=len(record_set_body)
	record_title+=len(record_set_title)
	record_kw+=len(record_set_kw)
	record_sal+=len(saliency_set)
	record_cover+=len(cover_set)

record_abs/=real_volume
record_body/=real_volume
record_title/=real_volume
record_kw/=real_volume
record_sal/=real_volume
record_cover/=real_volume

print(record_abs,record_body,record_title,record_kw,record_sal,record_sal/record_abs,record_cover/record_abs)