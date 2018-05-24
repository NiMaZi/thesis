import pickle
import numpy as np

max_depth=0
max_leaf=[]

def DFS(root,depth,max_depth,out_depth,name_set):
	max_depth=depth
	if depth==out_depth:
		name_set.add(word_list[root])
		print(word_list[root])
	for i in range(0,len(word_list)):
		if tree[root][i]==1:
			cur_depth,name_set=DFS(i,depth+1,max_depth,out_depth,name_set)
			if cur_depth>max_depth:
				max_depth=cur_depth
	return max_depth,name_set

def get_parent(i,d):
	min_d=d
	min_j=-1
	for j in range(0,len(tree)):
		if tree[j][i]==1 and word2depth[word_list[j]]<d:
			if word2depth[word_list[j]]<min_d:
				min_j=j
				min_d=word2depth[word_list[j]]
	return min_j

f=open("/Users/yalunzheng/Documents/BioPre/ontology_wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

f=open("/Users/yalunzheng/Documents/BioPre/ontology_tree.pkl","rb")
tree=pickle.load(f)
f.close()

f=open("/Users/yalunzheng/Documents/BioPre/ontology_word2depth.pkl","rb")
word2depth=pickle.load(f)
f.close()

f=open("/Users/yalunzheng/Documents/BioPre/ontology_depth2word.pkl","rb")
depth2word=pickle.load(f)
f.close()

tree[295][702]=0

word2tvec={}

for word in word_list:
	ontology_vec=[-1.0 for i in range(0,16)]
	ontology_vec[0]=0.0
	ontology_vec[word2depth[word]]=depth2word[word2depth[word]].index(word)
	word_origin=word
	while True:
		parent_index=get_parent(word_list.index(word),word2depth[word])
		if parent_index==-1:
			break
		ontology_vec[word2depth[word_list[parent_index]]]=depth2word[word2depth[word_list[parent_index]]].index(word_list[parent_index])
		word=word_list[parent_index]
	word2tvec[word_origin]=ontology_vec
	print(word_origin,ontology_vec)


f=open("/Users/yalunzheng/Documents/BioPre/ontology_word2taxonomy.pkl","wb")
pickle.dump(word2tvec,f)
f.close()
# print(word2tvec)

# c=0.0
# for i in range(0,len(word_list)):
# 	for j in range(0,len(word_list)):
# 		c+=tree[i][j]

# print(c/(len(word_list)*len(word_list)))
# print(word_list[295],word_list[702])

# in_degree=[0 for i in range(0,len(word_list))]
# vis=[1 for i in range(0,len(word_list))]
# stack=[]

# for i in range(0,len(word_list)):
# 	for j in range(0,len(word_list)):
# 		in_degree[j]+=tree[i][j]

# for i in range(0,len(in_degree)):
# 	if in_degree[i]==0:
# 		stack.append(i)

# while(stack):
# 	i=stack.pop()
# 	vis[i]=0
# 	for j in range(0,len(word_list)):
# 		if tree[i][j]==1:
# 			tree[i][j]=0
# 			in_degree[j]-=1
# 			if in_degree[j]==0:
# 				stack.append(j)

# for i in range(0,len(vis)):
# 	if vis[i]==1:
# 		print("cyclic")
# 		break

# print(DFS(79,0,0)) # max depth 16
# word2depth={}

# for i in range(0,17):
# 	print("\ndepth="+str(i)+"\n")
# 	max_depth,name_set=DFS(79,0,0,i,set())
# 	for word in set(name_set):
# 		if word in word2depth.keys():
# 			if i<word2depth[word]:
# 				word2depth[word]=i
# 		else:
# 			word2depth[word]=i
# depth2word={}
# for word in word2depth.keys():
# 	if word2depth[word] in depth2word.keys():
# 		depth2word[word2depth[word]].append(word)
# 	else:
# 		depth2word[word2depth[word]]=[word]

# f=open("/Users/yalunzheng/Documents/BioPre/ontology_word2depth.pkl","wb")
# pickle.dump(word2depth,f)
# f.close()

# f=open("/Users/yalunzheng/Documents/BioPre/ontology_depth2word.pkl","wb")
# pickle.dump(depth2word,f)
# f.close()

# print(depth2word)