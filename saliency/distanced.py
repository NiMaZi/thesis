import sys
import csv
import json
import pickle

volume=int(sys.argv[1])

distanced_list=[]

# dlf=open("/home/ubuntu/results/saliency/distanced.pkl","rb")
# distanced_list=pickle.load(dlf)
# dlf.close()

# print(len(distanced_list))

# dd=distanced_list[0:6000]

for i in range(0,volume):
	
	record={}

	record['abs']=[]
	f=open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".txt","r",encoding='utf-8')
	doc=f.read()
	length=float(len(doc))
	f.close()
	with open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			mention=item[0]
			entity=item[2]
			start_pos=float(item[4])/length
			record['abs'].append([mention,entity,start_pos])

	record['body']=[]
	f=open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".txt","r",encoding='utf-8')
	doc=f.read()
	length=float(len(doc))
	f.close()
	with open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			mention=item[0]
			entity=item[2]
			start_pos=float(item[4])/length
			record['body'].append([mention,entity,start_pos])

	record['title']=[]
	f=open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".txt","r",encoding='utf-8')
	doc=f.read()
	length=float(len(doc))
	f.close()
	with open("/home/ubuntu/thesiswork/kdata/title"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			mention=item[0]
			entity=item[2]
			start_pos=float(item[4])/length
			record['title'].append([mention,entity,start_pos])

	record['keywords']=[]
	f=open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".txt","r",encoding='utf-8')
	doc=f.read()
	length=float(len(doc))
	f.close()
	with open("/home/ubuntu/thesiswork/kdata/keywords"+str(i)+".csv","r",newline='',encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			mention=item[0]
			entity=item[2]
			start_pos=float(item[4])/length
			record['keywords'].append([mention,entity,start_pos])

	distanced_list.append(record)

print(distanced_list)
print(sys.getsizeof(distanced_list))

ds=json.dumps(distanced_list)

slf=open("/home/ubuntu/results_new/saliency/distanced.json","w")
slf.write(ds)
slf.close()

# dlf=open("/home/ubuntu/results_new/saliency/distanced.pkl","wb")
# pickle.dump(distanced_list,dlf)
# dlf.close()
