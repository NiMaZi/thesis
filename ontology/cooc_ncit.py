import sys
import csv
import json
import time
from scipy.sparse import lil_matrix, csc_matrix, save_npz, load_npz

f=open("/home/ubuntu/results/ontology/c2id.json",'r',encoding='utf-8')
c2id=json.load(f)
f.close()

dev_mat=lil_matrix((len(c2id),len(c2id)))

volume=int(sys.argv[1])
avg_time=0.0

for i in range(0,volume):
	mention_set=set()
	check_point_1=time.time()
	with open("/home/ubuntu/thesiswork/kdata/abs"+str(i)+".csv","r",encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			mention_set.add("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+item[1])
	with open("/home/ubuntu/thesiswork/kdata/body"+str(i)+".csv","r",encoding='utf-8') as csvfile:
		reader=csv.reader(csvfile)
		for item in reader:
			if item[2]=="ConceptName":
				continue
			mention_set.add("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#"+item[1])
	mention_list=list(mention_set)
	check_point_2=time.time()
	for j1 in range(0,len(mention_list)):
		for j2 in range(j1,len(mention_list)):
			mention1=mention_list[j1]
			mention2=mention_list[j2]
			if mention1==mention2:
				continue
			m_index1=c2id[mention1]
			m_index2=c2id[mention2]
			dev_mat[m_index1,m_index2]+=1.0/float(volume)
			dev_mat[m_index2,m_index1]+=1.0/float(volume)
	check_point_4=time.time()
	print(i,check_point_2-check_point_1,check_point_4-check_point_2)
	avg_time+=check_point_4-check_point_1
avg_time/=volume
print(avg_time)

path="/home/ubuntu/results/statistics/cooc_mat_simple.npz"
save_npz(path,dev_mat.tocsc())