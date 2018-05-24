import pickle

f=open("/home/ubuntu/results/saliency/simplemat.pkl","rb")
dev_mat=pickle.load(f)
f.close()

count=0

for i in range(0,len(dev_mat)):
	for j in range(0,len(dev_mat)):
		if dev_mat[i][j]>0.0:
			count+=1

print(count,len(dev_mat)*len(dev_mat))