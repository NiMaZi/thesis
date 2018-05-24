import csv
import pickle

entity_dict_list=[]

entity_dict={}
entity_dict_abs={}
entity_dict_body={}

for i in range(0,3500):
	entity_dict={}
	entity_dict_abs={}
	entity_dict_body={}

	with open("/home/ubuntu/thesiswork/data/abs"+str(i)+".txt.mentions","r",newline='',encoding='utf-8') as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			if item[2] in entity_dict_abs:
				entity_dict_abs[item[2]]=entity_dict_abs[item[2]]+1
			else:
				entity_dict_abs[item[2]]=1

	with open("/home/ubuntu/thesiswork/data/body"+str(i)+".txt.mentions","r",newline='',encoding='utf-8') as csvfile:
		reader = csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			if item[2] in entity_dict_body:
				entity_dict_body[item[2]]=entity_dict_body[item[2]]+1
			else:
				entity_dict_body[item[2]]=1

	entity_dict["abs"]=entity_dict_abs
	entity_dict["body"]=entity_dict_body

	entity_dict_list.append(entity_dict)

	print("finish building dictionary for article "+str(i)+".\n")


# f=open("/home/ubuntu/results/initDict.txt","w")
# f.write(str(entity_dict_list))
# f.close()

g=open("/home/ubuntu/results/initDict.pickle","wb")
pickle.dump(entity_dict_list,g)
g.close()