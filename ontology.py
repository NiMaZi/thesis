import csv
import sys
import pickle
import numpy as np
from scipy.sparse import lil_matrix

pre_dict={}
prelist=[]
word_list=[]
id2name={}
word_list=["http://www.w3.org/2002/07/owl#Thing"]

def walk(index,length):
	# global max_length
	if index=="http://www.w3.org/2002/07/owl#Thing":
		if length>17:
			# global max_length
			# max_length=length
			print(length)
		return
	if not index in pre_dict.keys():
		# print("out of ontology.\n")
		# print(index)
		return
	# print(length,pre_dict[index][0])
	p_index_list=pre_dict[index][1].split('|')
	for p_index in p_index_list:
		walk(p_index,length+1)



with open("/Users/yalunzheng/Downloads/NCIT.csv","r",newline='',encoding='utf-8') as csvfile:
	reader=csv.reader(csvfile)
	count=0.0
	c_count=[0.0 for i in range(0,209)]
	c_types=['none' for i in range(0,209)]
	semantic_types=set()
	cids=[]
	for item in reader:
		if item[0]=="Class ID":
			continue
		word_list.append(item[0])
		prelist.append([item[0],item[1],item[7]])
		pre_dict[item[0]]=[item[1],item[7]]

ontology_dmat=lil_matrix((133610,133610))
print(type(ontology_dmat))

# ontology_dmat=np.zeros((len(prelist),len(prelist))) # directed
# ontology_nmat=np.zeros((len(prelist),len(prelist))) # non-directed

# miss_count=0
# root=0
for i in range(0,len(prelist)):
	print(i)
	_list=prelist[i][2].split('|')
	for item in _list:
		# print(word_list.index(item),word_list.index(prelist[i][0]))
		ontology_dmat[word_list.index(item),word_list.index(prelist[i][0])]=1
		# ontology_nmat[word_list.index(id2name[item])][word_list.index(prelist[i][1])]=1
		# ontology_nmat[word_list.index(prelist[i][1])][word_list.index(id2name[item])]=1

print(sys.getsizeof(ontology_dmat))

# print(root)

# 	ontology_dmat[word_list.index(id2name[item[2]])][word_list.index(item[1])]=1
# 	ontology_nmat[word_list.index(id2name[item[2]])][word_list.index(item[1])]=1
# 	ontology_nmat[word_list.index(item[1])][word_list.index(id2name[item[2]])]=1

# for i in range(0,len(prelist)):
# 	if ontology_dmat[79][i]==1:
# 		print(word_list[i])

# dist=np.zeros((len(prelist),len(prelist)))

# for i in range(0,len(prelist)):
# 	for j in range(0,len(prelist)):
# 		dist[i][j]=np.inf

# for i in range(0,len(prelist)):
# 	dist[i][i]=0.0

# for i in range(0,len(prelist)):
# 	for j in range(0,len(prelist)):
# 		if ontology_nmat[i][j]==1:
# 			dist[i][j]=1

# for k in range(0,len(prelist)):
# 	for i in range(0,len(prelist)):
# 		for j in range(0,len(prelist)):
# 			if dist[i][j]>(dist[i][k]+dist[k][j]):
# 				dist[i][j]=dist[i][k]+dist[k][j]

# f=open("/Users/yalunzheng/Documents/BioPre/ontology_wordlist.pkl","wb")
# pickle.dump(word_list,f)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/ontology_tree.pkl","wb")
# pickle.dump(ontology_dmat,f)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/ontology_map.pkl","wb")
# pickle.dump(ontology_nmat,f)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/ontology_path.pkl","wb")
# pickle.dump(dist,f)
# f.close()