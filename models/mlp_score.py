import sys
import csv
import json
import numpy as np
from gensim.models import word2vec
from keras.models import load_model


f=open("/home/ubuntu/results/saliency/featured_list_com.json",'r')
featured_list_com=json.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/word_list.json",'r')
word_list=json.load(f)
f.close()

jaccard=np.load("/home/ubuntu/results/ontology/jaccard.npy")
milne_witten=np.load("/home/ubuntu/results/ontology/milne_witten.npy")
adamic_adar=np.load("/home/ubuntu/results/ontology/adamic_adar.npy")
dice=np.load("/home/ubuntu/results/ontology/dice.npy")

f=open("/home/ubuntu/results/ontology/n2id.json",'r')
n2id=json.load(f)
f.close()
f=open("/home/ubuntu/results/statistics/idf.json",'r')
idf=json.load(f)
f.close()
f=open("/home/ubuntu/results/statistics/tf_all.json",'r')
tf_all=json.load(f)
f.close()

f=open("/home/ubuntu/results/ontology/KG_n2v.json",'r')
KG_n2v=json.load(f)
f.close()

KG_e2v=word2vec.Word2Vec.load("/home/ubuntu/results/e2v_sg_e100.model")

cooc_simple=np.load("/home/ubuntu/results/statistics/cooc_simple.npy")

abbl_order=[9,13,7,11,4,6,5,3,10,2,0,12,8,1]

def score_do(model_n):
	path="/home/ubuntu/results_new/models/MLP_abblation_"+str(model_n)+".h5"
	model=load_model(path)
	threshold=0.0
	volume=662
	P_all=0.0
	R_all=0.0
	P_volume=volume
	R_volume=volume
	for i in range(4000,4000+volume):
		_abs=featured_list_com[i]['abs']
		_body=set(featured_list_com[i]['body'].keys())-set(featured_list_com[i]['abs'].keys())
		predictions=set()
		for w in word_list:
			score=0.0
			for a in _abs:
				if a==w:
					continue
				if not a in word_list:
					continue
				jacc=jaccard[n2id[a]][n2id[w]]
				mlwt=milne_witten[n2id[a]][n2id[w]]
				aa=adamic_adar[n2id[a]][n2id[w]]
				di=dice[n2id[a]][n2id[w]]
				cocs=cooc_simple[n2id[a]][n2id[w]]
				idf_a=idf[a]
				idf_b=idf[w]
				tf_all_a=tf_all[a]
				tf_all_b=tf_all[w]
				tfidf_a=tf_all_a*idf_a
				tfidf_b=tf_all_b*idf_b
				nodevec_a=KG_n2v[str(n2id[a])]
				nodevec_b=KG_n2v[str(n2id[w])]
				wordvec_a=list(KG_e2v.wv[a])
				wordvec_b=list(KG_e2v.wv[w])
				term_feature=[_abs[a][0],_abs[a][1],_abs[a][2],jacc,mlwt,aa,di,cocs,idf_a,idf_b,tf_all_a,tf_all_b,tfidf_a,tfidf_b]
				term_embedding=nodevec_a+nodevec_b+wordvec_a+wordvec_b
				term_all=term_feature+term_embedding
				pred=model.predict(np.array([term_all])[:,abbl_order[:model_n]])
				score+=pred[0][0]
			score/=len(_abs)
			if score>threshold:
				predictions.add(w)
		tp=len(predictions&_body)
		fp=len(predictions-_body)
		fn=len(_body-predictions)
		try:
			P=tp/(tp+fp)
		except:
			P=0.0
		try:
			R=tp/(tp+fn)
		except:
			R=0.0
		P_all+=P
		R_all+=R
	P_all/=P_volume
	R_all/=R_volume
	F1=2*P_all*R_all/(P_all+R_all)
	f=open("/home/ubuntu/results_new/mpl_abblation_log.txt",'a')
	f.write("%d,%.3f,%.3f,%.3f\n"%(model_n,P_all,R_all,F1))
	f.close()

for mn in range(1,len(abbl_order)+1):
	score_do(mn)