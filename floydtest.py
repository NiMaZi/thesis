import pickle
import numpy as np

f=open("/Users/yalunzheng/Documents/BioPre/ontology_wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

f=open("/Users/yalunzheng/Documents/BioPre/ontology_map.pkl","rb")
tmat=pickle.load(f)
f.close()

logf=open("/Users/yalunzheng/Documents/BioPre/log.txt","w")

logf.write("logs start.\n")

# dist=[[np.inf for i in range(0,len(word_list))] for i in range(0,len(word_list))]
path=[[-1 for i in range(0,len(word_list))] for i in range(0,len(word_list))]
path_list=[[[] for i in range(0,len(word_list))] for i in range(0,len(word_list))]

def floyd(mat,path):
	for i in range(0,len(mat)):
		for j in range(0,len(mat)):
			for k in range(0,len(mat)):
				print(i,j,k)
				if mat[i][j]>mat[i][k]+mat[k][j]:
					mat[i][j]=mat[i][k]+mat[k][j]
					path[i][j]=k

def get_path(i,j,path):
	if path[i][j]==-1:
		return []
	else:
		path_list=get_path(i,path[i][j],path)
		path_list.append(path[i][j])
		path_list.extend(get_path(path[i][j],j,path))
		return path_list


dist=np.zeros((len(word_list),len(word_list)))

for i in range(0,len(word_list)):
	for j in range(0,len(word_list)):
		dist[i][j]=np.inf

for i in range(0,len(word_list)):
	dist[i][i]=0.0

for i in range(0,len(word_list)):
	for j in range(0,len(word_list)):
		if tmat[i][j]==1:
			dist[i][j]=1.0

for k in range(0,len(word_list)):
	for i in range(0,len(word_list)):
		for j in range(0,len(word_list)):
			if dist[i][j]>(dist[i][k]+dist[k][j]):
				dist[i][j]=dist[i][k]+dist[k][j]
				path[i][j]=k

for i in range(0,len(word_list)):
	for j in range(0,len(word_list)):
		if i==j:
			continue
		try:
			logf.write("("+str(i)+","+str(j)+","+str(dist[i][j])+")\n")
		except:
			pass
		p_list=[i]
		p_list.extend(get_path(i,j,path))
		p_list.append(j)
		path_list[i][j].extend(p_list)
		try:
			logf.write(str(path_list[i][j])+"\n")
		except:
			pass

logf.close()
		

f=open("/Users/yalunzheng/Documents/BioPre/ontology_dist.pkl","wb")
pickle.dump(dist,f)
f.close()

f=open("/Users/yalunzheng/Documents/BioPre/ontology_path.pkl","wb")
pickle.dump(path_list,f)
f.close()