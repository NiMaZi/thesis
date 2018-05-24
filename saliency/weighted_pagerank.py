import sys
import pickle
import math
import numpy as np

mode=int(sys.argv[1])
threshold=float(sys.argv[2])

f=open("/home/ubuntu/results/ontology/ontology_wordlist.pkl","rb")
word_list=pickle.load(f)
f.close()

if mode==1:
	f=open("/home/ubuntu/results/saliency/simplemat.pkl","rb")
	dev_mat=pickle.load(f)
	f.close()
else:
	sys.exit(0)

VR=[float(1.0/(len(word_list))) for i in range(0,len(word_list))]
NVR=[float(1.0/(len(word_list))) for i in range(0,len(word_list))]

epoch=0
d=0.85

while True:
	for i in range(0,len(VR)):
		inw=0.0
		for j in range(0,len(VR)):
			outw=0.0
			for k in range(0,len(VR)):
				outw+=dev_mat[j][k]
			if outw==0.0:
				continue
			inw+=(dev_mat[j][i]/outw)*VR[j]
		NVR[i]=(1.0-d)+d*inw
	delta=0.0
	for i in range(0,len(VR)):
		delta+=(NVR[i]-VR[i])*(NVR[i]-VR[i])
	delta=math.sqrt(delta)
	# print("epoch="+str(epoch)+" delta="+str(delta)+"\n")
	epoch+=1
	for i in range(0,len(VR)):
		VR[i]=NVR[i]
	if delta<threshold:
		break

word_centrality={}
for i in range(0,len(word_list)):
	word_centrality[word_list[i]]=VR[i]


f=open("/home/ubuntu/results/saliency/centrality.pkl","wb")
pickle.dump(word_centrality,f)
f.close()